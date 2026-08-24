"""Evaluacion cualitativa offline de trazas M3 con un LLM-as-a-judge.

El juez analiza una unica dimension: si el agente incorpora el feedback
correctivo de las tools en su siguiente accion relevante. No vuelve a ejecutar
escenarios ni modifica al agente evaluado.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from student_framework.agent import MyAgent, StructuredCallUsage

if TYPE_CHECKING:
    from mia_agents.protocols import LLMClient as LLMClientProtocol


DEFAULT_JUDGE_MODEL_ID = "moonshot.kimi-k2-thinking"
DEFAULT_JUDGE_REGION = "us-east-1"
JUDGE_RESULTS_DIR = PROJECT_ROOT / "eval" / "results" / "judge"

FEEDBACK_RUBRIC = {
    2: (
        "Consistente: incorpora el feedback correctivo en la siguiente accion "
        "relevante en todos los episodios."
    ),
    1: (
        "Parcial: lo incorpora en la siguiente accion relevante en algunos "
        "episodios, pero no en todos."
    ),
    0: (
        "Ausente: no lo incorpora en la siguiente accion relevante en ningun "
        "episodio."
    ),
}

JUDGE_SYSTEM_PROMPT = """
Sos un evaluador independiente de trazas de agentes. Evalua una sola
dimension: USO DEL FEEDBACK.

Definiciones:
- Feedback correctivo: respuesta de una tool que informa que una accion fallo,
  que una precondicion no se cumple, que una combinacion es incompatible o que
  aporta una correccion concreta que deberia cambiar una decision posterior.
- No es feedback correctivo una descripcion neutral del mundo, una accion
  exitosa ni el resultado final deterministico del escenario.
- Siguiente accion relevante: primera accion posterior en la que la correccion
  recibida deberia influir. Si la accion inmediatamente posterior trata otro
  asunto, avanza solo hasta la primera accion relacionada. No saltees una
  accion incompatible para elegir otra posterior que termino funcionando.
- Feedback incorporado: esa siguiente accion relevante respeta la restriccion
  o correccion recibida. Si el agente la aplica muchos pasos despues, pero la
  primera accion relevante la ignoro, el episodio cuenta como no incorporado.

Identifica todos los episodios de feedback correctivo de la traza. Para cada
uno indica los pasos relacionados, describe brevemente la evidencia y decide
si el agente incorporo el feedback. Si no hubo una accion posterior relevante,
usa next_relevant_step=null e incorporated=false.

Puntaje:
- 2: todos los episodios fueron incorporados;
- 1: algunos si y otros no;
- 0: ninguno fue incorporado;
- null: no hubo feedback correctivo y applicable=false.

Evalua solo la evidencia de la traza. No uses goal_achieved ni goal_reason, que
se ocultan durante el juicio. No infieras que el feedback se uso solo porque el
agente finalmente tuvo exito. Los numeros de paso son 1-based.
""".strip()


class FeedbackEpisode(BaseModel):
    """Un feedback correctivo y la primera accion posterior relacionada."""

    feedback_step: int = Field(ge=1)
    feedback_evidence: str
    next_relevant_step: int | None = Field(default=None, ge=1)
    next_action_evidence: str | None = None
    incorporated: bool
    explanation: str


class FeedbackJudgment(BaseModel):
    """Salida estructurada del juez para la dimension uso del feedback."""

    applicable: bool
    score: int | None = Field(default=None, ge=0, le=2)
    justification: str
    episodes: list[FeedbackEpisode] = Field(default_factory=list)


def _trace_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Adapta un resultado de eval/run.py al payload ciego del juez."""
    agent_result = result.get("agent_result") or {}
    steps = [
        {
            "step": index,
            "tool_name": step.get("tool_name"),
            "tool_input": step.get("tool_input"),
            "tool_output": step.get("tool_output"),
            "error": step.get("error"),
        }
        for index, step in enumerate(agent_result.get("steps") or [], start=1)
    ]

    return {
        "scenario": result.get("scenario"),
        "difficulty": result.get("difficulty"),
        "attempt": result.get("attempt"),
        "scenario_instruction": result.get("user_message"),
        "goal": result.get("goal"),
        "agent_final_answer": agent_result.get("answer"),
        "agent_final_error": agent_result.get("error"),
        "steps": steps,
    }


def build_judge_prompt(result: dict[str, Any]) -> str:
    """Combina la rubrica de feedback y la traza observable."""
    rubric = {str(score): anchor for score, anchor in FEEDBACK_RUBRIC.items()}
    return (
        "Evalua el uso del feedback en esta ejecucion.\n\n"
        "RUBRICA:\n"
        + json.dumps(rubric, ensure_ascii=False, indent=2)
        + "\n\nTRAZA:\n"
        + json.dumps(_trace_payload(result), ensure_ascii=False, indent=2)
    )


def _add_usage(
    current: dict[str, int | None],
    usage: StructuredCallUsage,
) -> None:
    """Acumula uso conservando None si el proveedor no informa tokens."""
    for field in ("input_tokens", "output_tokens"):
        value = getattr(usage, field)
        if value is not None:
            current[field] = (current[field] or 0) + value
    current["attempts"] = (current["attempts"] or 0) + usage.attempts


def judge_result(
    result: dict[str, Any],
    llm_client: "LLMClientProtocol",
) -> dict[str, Any]:
    """Juzga una traza mediante una salida estructurada."""
    agent = MyAgent(
        llm_client=llm_client,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )
    usage: dict[str, int | None] = {
        "input_tokens": None,
        "output_tokens": None,
        "attempts": 0,
    }

    structured = agent.structured_call_with_usage(
        prompt=build_judge_prompt(result),
        schema=FeedbackJudgment,
    )
    _add_usage(usage, structured.usage)
    judgment = structured.value

    return {
        "scenario": result.get("scenario"),
        "difficulty": result.get("difficulty"),
        "attempt": result.get("attempt"),
        "source_outcome": {
            "goal_achieved": result.get("goal_achieved"),
            "goal_reason": result.get("goal_reason"),
        },
        "feedback_use": {
            "applicable": judgment.applicable,
            "score": judgment.score,
        },
        "judgment": judgment.model_dump(),
        "judge_usage": usage,
    }


def _judgment_error(result: dict[str, Any], exc: Exception) -> dict[str, Any]:
    raw_usage = getattr(exc, "usage", None)
    if isinstance(raw_usage, StructuredCallUsage):
        judge_usage: dict[str, int | None] = {
            "input_tokens": raw_usage.input_tokens,
            "output_tokens": raw_usage.output_tokens,
            "attempts": raw_usage.attempts,
        }
    elif isinstance(raw_usage, dict):
        judge_usage = {
            "input_tokens": raw_usage.get("input_tokens"),
            "output_tokens": raw_usage.get("output_tokens"),
            "attempts": raw_usage.get("attempts"),
        }
    else:
        judge_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "attempts": None,
        }

    return {
        "scenario": result.get("scenario"),
        "difficulty": result.get("difficulty"),
        "attempt": result.get("attempt"),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "judge_usage": judge_usage,
        "source_outcome": {
            "goal_achieved": result.get("goal_achieved"),
            "goal_reason": result.get("goal_reason"),
        },
    }


def build_judge_summary(
    entries: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
    *,
    requested_traces: int | None = None,
) -> dict[str, Any]:
    """Resume uso del feedback, costo y fallos del juez."""
    errors = errors or []
    requested = (
        requested_traces
        if requested_traces is not None
        else len(entries) + len(errors)
    )

    if entries:
        execution_status = (
            "completed_with_errors" if errors else "completed"
        )
    else:
        execution_status = "failed" if requested else "completed"

    scores = [
        entry["feedback_use"]["score"]
        for entry in entries
        if entry["feedback_use"]["score"] is not None
    ]
    distribution = {str(score): scores.count(score) for score in range(3)}
    episodes = [
        episode
        for entry in entries
        for episode in entry["judgment"]["episodes"]
    ]
    incorporated = sum(episode["incorporated"] for episode in episodes)
    input_tokens = 0
    output_tokens = 0
    has_input_tokens = False
    has_output_tokens = False
    for item in [*entries, *errors]:
        usage = item["judge_usage"]
        if usage["input_tokens"] is not None:
            has_input_tokens = True
            input_tokens += usage["input_tokens"]
        if usage["output_tokens"] is not None:
            has_output_tokens = True
            output_tokens += usage["output_tokens"]

    return {
        "execution_status": execution_status,
        "requested_traces": requested,
        "judged_traces": len(entries),
        "failed_judgments": len(errors),
        "feedback_use": {
            "average_score": sum(scores) / len(scores) if scores else None,
            "score_distribution": distribution,
            "evaluated_traces": len(scores),
            "not_applicable": len(entries) - len(scores),
            "feedback_episodes": len(episodes),
            "incorporated_episodes": incorporated,
            "incorporation_rate": (
                incorporated / len(episodes) if episodes else None
            ),
        },
        "judge_input_tokens": input_tokens if has_input_tokens else None,
        "judge_output_tokens": output_tokens if has_output_tokens else None,
    }


def _judge_run_id(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _filename_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-").lower()
    return slug or "unknown"


def _source_run_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
    return slug or "unknown"


def _scope_slug(
    scenarios: set[str] | None,
    attempts: set[int] | None,
    limit: int | None,
) -> str:
    parts: list[str] = []
    if scenarios:
        if len(scenarios) == 1:
            parts.append(_filename_slug(next(iter(scenarios))))
        else:
            parts.append(f"scenarios-{len(scenarios)}")
    if attempts:
        if len(attempts) == 1:
            parts.append(f"attempt-{next(iter(attempts))}")
        else:
            parts.append(f"attempts-{len(attempts)}")
    if limit is not None:
        parts.append(f"limit-{limit}")
    return "__".join(parts) if parts else "all"


def build_default_output_path(
    *,
    report_path: Path,
    report: dict[str, Any],
    model_id: str,
    timestamp: datetime,
    scenarios: set[str] | None = None,
    attempts: set[int] | None = None,
    limit: int | None = None,
) -> Path:
    """Construye una ruta separada, trazable y no destructiva."""
    source_run_id = (report.get("metadata") or {}).get("run_id")
    source_slug = _source_run_slug(str(source_run_id or report_path.stem))
    filename = "__".join(
        (
            _judge_run_id(timestamp),
            _filename_slug(model_id),
            _scope_slug(scenarios, attempts, limit),
        )
    ) + ".json"
    return JUDGE_RESULTS_DIR / source_slug / filename


def _build_judge_report(
    *,
    source_report: dict[str, Any],
    entries: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    requested_traces: int,
    judge_metadata: dict[str, Any],
    timestamp: datetime,
) -> dict[str, Any]:
    return {
        "metadata": {
            "judge_run_id": _judge_run_id(timestamp),
            "timestamp_utc": timestamp.isoformat(),
            "source_run_id": (source_report.get("metadata") or {}).get(
                "run_id"
            ),
            "judge_provider": judge_metadata,
            "evaluation_dimension": "uso_del_feedback",
            "deterministic_outcome_included_in_prompt": False,
            "source_outcome_attached_after_judgment": True,
            "rubric_scale": "0-2-null",
        },
        "summary": build_judge_summary(
            entries,
            errors,
            requested_traces=requested_traces,
        ),
        "judgments": entries,
        "errors": errors,
    }


def judge_report(
    report: dict[str, Any],
    llm_client: "LLMClientProtocol",
    *,
    judge_metadata: dict[str, Any] | None = None,
    scenarios: set[str] | None = None,
    attempts: set[int] | None = None,
    limit: int | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Evalua las trazas seleccionadas y conserva los fallos por intento."""
    results = report.get("results") or []
    if scenarios is not None:
        results = [
            result
            for result in results
            if result.get("scenario") in scenarios
        ]
    if attempts is not None:
        results = [
            result for result in results if result.get("attempt") in attempts
        ]
    if limit is not None:
        results = results[:limit]

    entries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    report_timestamp = timestamp or datetime.now(timezone.utc)
    metadata = judge_metadata or {"provider": "injected", "model": None}

    for result in results:
        try:
            entries.append(judge_result(result, llm_client))
        except Exception as exc:
            errors.append(_judgment_error(result, exc))

        if checkpoint is not None:
            checkpoint(
                _build_judge_report(
                    source_report=report,
                    entries=entries,
                    errors=errors,
                    requested_traces=len(results),
                    judge_metadata=metadata,
                    timestamp=report_timestamp,
                )
            )

    return _build_judge_report(
        source_report=report,
        entries=entries,
        errors=errors,
        requested_traces=len(results),
        judge_metadata=metadata,
        timestamp=report_timestamp,
    )


def _build_judge_client(
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    region: str = DEFAULT_JUDGE_REGION,
) -> tuple["LLMClientProtocol", dict[str, Any]]:
    """Construye el cliente Bedrock independiente usado por el juez."""
    from mia_agents._env import load_env_files
    from mia_agents.llm_client import BedrockProvider, LLMClient

    load_env_files()
    client = LLMClient(BedrockProvider(model=model_id, region=region))
    metadata = {
        "provider": "bedrock",
        "model_provider": model_id.split(".", maxsplit=1)[0],
        "model": model_id,
        "region": region,
    }
    return client, metadata


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Guarda un checkpoint sin dejar un JSON parcial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_report(
    report: dict[str, Any],
    *,
    report_path: Path,
    model_id: str = DEFAULT_JUDGE_MODEL_ID,
    region: str = DEFAULT_JUDGE_REGION,
    output_path: Path | None = None,
    scenarios: set[str] | None = None,
    attempts: set[int] | None = None,
    limit: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """Ejecuta el juez y guarda su reporte separado del reporte fuente."""
    timestamp = datetime.now(timezone.utc)
    resolved_output_path = output_path or build_default_output_path(
        report_path=report_path,
        report=report,
        model_id=model_id,
        timestamp=timestamp,
        scenarios=scenarios,
        attempts=attempts,
        limit=limit,
    )
    selected_results = report.get("results") or []
    if scenarios is not None:
        selected_results = [
            result
            for result in selected_results
            if result.get("scenario") in scenarios
        ]
    if attempts is not None:
        selected_results = [
            result
            for result in selected_results
            if result.get("attempt") in attempts
        ]
    if limit is not None:
        selected_results = selected_results[:limit]
    judge_metadata = {
        "provider": "bedrock",
        "model_provider": model_id.split(".", maxsplit=1)[0],
        "model": model_id,
        "region": region,
    }

    if not selected_results:
        judged = _build_judge_report(
            source_report=report,
            entries=[],
            errors=[],
            requested_traces=0,
            judge_metadata=judge_metadata,
            timestamp=timestamp,
        )
    else:
        try:
            llm_client, judge_metadata = _build_judge_client(model_id, region)
            judged = judge_report(
                report,
                llm_client,
                judge_metadata=judge_metadata,
                scenarios=scenarios,
                attempts=attempts,
                limit=limit,
                checkpoint=lambda partial: _write_json_atomic(
                    resolved_output_path,
                    partial,
                ),
                timestamp=timestamp,
            )
        except Exception as exc:
            judged = _build_judge_report(
                source_report=report,
                entries=[],
                errors=[
                    _judgment_error(result, exc)
                    for result in selected_results
                ],
                requested_traces=len(selected_results),
                judge_metadata=judge_metadata,
                timestamp=timestamp,
            )

    _write_json_atomic(resolved_output_path, judged)
    return judged, resolved_output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalua uso del feedback en trazas M3 con un LLM judge."
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--attempt", action="append", type=int, dest="attempts")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument("--region", default=DEFAULT_JUDGE_REGION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    scenarios = set(args.scenarios) if args.scenarios else None
    attempts = set(args.attempts) if args.attempts else None
    judged, output_path = evaluate_report(
        report,
        report_path=args.report,
        model_id=args.model,
        region=args.region,
        output_path=args.output,
        scenarios=scenarios,
        attempts=attempts,
        limit=args.limit,
    )

    print(json.dumps(judged["summary"], ensure_ascii=False, indent=2))
    print(f"Juicios guardados en: {output_path}", file=sys.stderr)
    if judged["errors"]:
        print(
            "Advertencia: "
            f"{len(judged['errors'])} evaluaciones quedaron registradas "
            "en errors.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
