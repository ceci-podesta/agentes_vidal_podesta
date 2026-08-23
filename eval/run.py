"""Runner incremental de evaluacion para M3."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mia_agents.llm_client import LLMClient
from mia_world import (
    Scenario,
    check_goal,
    list_scenarios,
    make_world_tools,
)
from student_framework import build_agent
from student_framework.m3_research import (
    ResearchDiagnostics,
    make_research_documents_tool,
)


SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

DIFFICULTY_ORDER = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "extreme": 3,
}

# Corrida oficial para el informe M3: los 5 escenarios del criterio de
# aprobacion (easy/medium/hard) mas extreme-archive, ya validado en
# desarrollo. Los dos escenarios extreme restantes (vault-combination,
# backtracking-vault) no son obligatorios y hoy no se resuelven; quedan
# fuera para no gastar tiempo/costo de Bedrock en la corrida que se cita
# en el informe. Usar None para evaluar el dataset completo (los 8).
DEVELOPMENT_SCENARIOS: list[str] | None = [
    "study-with-key",
    "color-locks",
    "apartment-keys",
    "library-search",
    "office-sequence",
    "extreme-archive",
]

# "final" marca una corrida como evidencia oficial para el informe;
# "development" marca corridas exploratorias/de diagnostico. Es
# independiente del subconjunto de escenarios elegido arriba.
RUN_KIND = "final"

# Repeticiones por escenario para pass@k (ENUNCIADO_M3.md lo sugiere
# explicitamente como metrica valida). Una sola corrida no alcanza para
# saber si el agente resuelve un escenario de forma confiable: el LLM es
# no determinista y ya se observo un caso real (corrida
# 20260822T184341503167Z) donde apartment-keys y library-search fallaron
# pese a haber pasado en pruebas manuales sueltas con la misma config.
REPEATS_PER_SCENARIO = 5

# Umbral de exito para considerar un escenario "resuelto" bajo pass@k.
PASS_AT_K_THRESHOLD = 0.5

M3_AGENT_CONFIG = {
    "max_iterations": 25,
    "max_repeated_failures": 1,
    "max_repeated_observations": 1,
    "observation_tool_names": {"look", "examine", "research_documents"},
    "use_m3_scratchpad": True,
    "system_prompt": """
Sos un agente que resuelve acertijos en un mundo interactivo para cumplir el
objetivo indicado.

Antes de cada tool call, consultá el bloque Scratchpad M3 incluido en este
mensaje. Usalo como estado de trabajo de hechos ya observados.

Si el scratchpad dice "Sin hechos todavía", empezá con look. No adivines IDs,
inventario, ubicación ni salidas: usá sólo valores confirmados por el
scratchpad o por la última observación.

Si un ítem figura como contenido descubierto pero no está en Inventario, usá
take antes de use. Sólo usá un ítem sobre un destino visible en la ubicación
actual.

Usá la observación más reciente de las tools como fuente de verdad sobre el
estado actual del mundo.

Antes de actuar, usá look para observar el entorno y obtener los IDs exactos
de los objetos. Cuando una tool muestra un objeto como [id: ...], usá ese ID
exacto en las siguientes llamadas.

Usá examine para inspeccionar objetos y descubrir contenido o pistas. Para usar
un objeto portable, primero incorporalo al inventario con take. Antes de use,
verificá que el ítem esté en tu inventario y que el objeto destino sea visible
en la sala actual.

Cuando tengas una llave, relacionála con la cerradura o contenedor cuya
descripción indique que es compatible. No pruebes cada llave nueva en la puerta
de salida sin una señal de que encaja.

Para moverte, elegí únicamente entre las salidas indicadas en la observación más
reciente. Si go devuelve un error, corregí la dirección según las salidas que
informa ese error.

Después de cada resultado de una tool, actualizá tu plan. Si una acción falla,
corregí la causa y no repitas exactamente la misma acción sin información nueva.
Evitá volver a inspeccionar objetos cuyo estado no cambió.

Cuando examine revele una coleccion de mas de cinco documentos similares,
llama UNA sola vez a research_documents con TODOS los IDs exactos de esa
coleccion y el objetivo actual. No la llames sobre subconjuntos de la misma
coleccion: la tool administra internamente los lotes y devuelve un reporte
compacto.

Despues usa primero ese reporte para decidir el siguiente paso. No inspecciones
manualmente documentos que research_documents ya informo como investigados,
salvo que el reporte indique una razon concreta para hacerlo. No uses esta tool
para contenedores pequeños.

Continuá usando tools hasta cumplir el objetivo o hasta que no haya más acciones
útiles.
""".strip(),
}


def _llm_provider_metadata() -> dict[str, Any]:
    """Registra que proveedor/modelo resolvio LLMClient.from_env().

    Replica la misma precedencia que `LLMClient.from_env()` (fijo, en
    `mia_agents/llm_client.py`) sin importar sus internals: sirve para dejar
    constancia, dentro del propio reporte, de que la corrida se hizo contra
    el modelo que exige el criterio de aprobacion.
    """
    if os.environ.get("OLLAMA_HOST"):
        return {
            "provider": "ollama",
            "host": os.environ["OLLAMA_HOST"],
            "model": os.environ.get("OLLAMA_MODEL", "llama3.1"),
        }
    if os.environ.get("BEDROCK_MODEL_ID"):
        return {
            "provider": "bedrock",
            "model": os.environ["BEDROCK_MODEL_ID"],
            "region": (
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-1"
            ),
        }
    return {"provider": "unknown", "model": None}


def discover_scenarios(scenarios_dir: Path) -> list[Scenario]:
    """Carga el dataset y devuelve los escenarios en orden de dificultad."""
    scenarios = list_scenarios(scenarios_dir)
    return sorted(
        scenarios,
        key=lambda scenario: (
            DIFFICULTY_ORDER[scenario.difficulty],
            scenario.id,
        ),
    )


def select_scenarios(scenarios: list[Scenario]) -> list[Scenario]:
    """Aplica el filtro temporal de desarrollo, si existe."""
    if DEVELOPMENT_SCENARIOS is None:
        return scenarios

    by_id = {scenario.id: scenario for scenario in scenarios}
    unknown_ids = [
        scenario_id
        for scenario_id in DEVELOPMENT_SCENARIOS
        if scenario_id not in by_id
    ]
    if unknown_ids:
        raise SystemExit(
            "Escenarios de desarrollo inexistentes: "
            + ", ".join(unknown_ids)
        )

    return [by_id[scenario_id] for scenario_id in DEVELOPMENT_SCENARIOS]


def run_scenario(
    scenario: Scenario,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Ejecuta un escenario una vez y devuelve su registro evaluable."""
    world = scenario.initial_world
    llm_client = llm_client or LLMClient.from_env()
    agent = build_agent(
        {
            **M3_AGENT_CONFIG,
            "llm_client": llm_client,
        }
    )

    world_tools = make_world_tools(world)
    for function, schema in world_tools:
        agent.register_tool(function, schema)

    try:
        examine_tool, examine_schema = next(
            (function, schema)
            for function, schema in world_tools
            if schema.name == "examine"
        )
    except StopIteration as exc:
        raise RuntimeError(
            "Los escenarios M3 deben registrar la tool examine."
        ) from exc

    research_diagnostics = ResearchDiagnostics()
    research_tool, research_schema = make_research_documents_tool(
        llm_client=llm_client,
        examine_tool=examine_tool,
        examine_schema=examine_schema,
        diagnostics=research_diagnostics,
    )
    agent.register_tool(research_tool, research_schema)

    started_at = perf_counter()
    result = agent.run(scenario.user_message)
    duration_seconds = perf_counter() - started_at
    goal_achieved, goal_reason = check_goal(world, scenario.goal)

    return {
        "scenario": scenario.id,
        "difficulty": scenario.difficulty,
        "user_message": scenario.user_message,
        "goal": scenario.goal,
        "goal_achieved": goal_achieved,
        "goal_reason": goal_reason,
        "duration_seconds": duration_seconds,
        "agent_result": asdict(result),
        "delegation": research_diagnostics.as_dict(),
    }

def _json_safe(value: Any) -> Any:
    """Convierte configuracion a valores que json puede serializar."""
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, set):
        normalized = [_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
        )

    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]

    return repr(value)


def _step_has_error(step: dict[str, Any]) -> bool:
    """Cuenta errores explicitados por el agente o devueltos por una tool."""
    if step.get("error"):
        return True

    tool_output = step.get("tool_output")
    return (
        isinstance(tool_output, str)
        and tool_output.lstrip().startswith("Error:")
    )


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula metricas del principal y de workers delegados."""
    evaluated_scenarios = len(results)
    achieved_scenarios = sum(
        1 for result in results if result["goal_achieved"]
    )

    tool_calls = 0
    tool_errors = 0
    input_tokens = 0
    output_tokens = 0
    worker_tool_calls = 0
    worker_errors = 0
    worker_input_tokens = 0
    worker_output_tokens = 0
    workers_started = 0
    duration_seconds = 0.0
    by_difficulty: dict[str, dict[str, Any]] = {}

    for result in results:
        difficulty = result["difficulty"]
        difficulty_summary = by_difficulty.setdefault(
            difficulty,
            {
                "evaluated_scenarios": 0,
                "achieved_scenarios": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "workers_started": 0,
                "worker_tool_calls": 0,
                "worker_errors": 0,
                "worker_input_tokens": 0,
                "worker_output_tokens": 0,
                "duration_seconds": 0.0,
            },
        )

        agent_result = result["agent_result"]
        delegation = result.get("delegation") or {}
        steps = agent_result.get("steps", [])

        scenario_tool_calls = len(steps)
        scenario_tool_errors = sum(
            1 for step in steps if _step_has_error(step)
        )
        scenario_input_tokens = agent_result.get("input_tokens") or 0
        scenario_output_tokens = agent_result.get("output_tokens") or 0
        scenario_workers_started = delegation.get("workers_started") or 0
        scenario_worker_tool_calls = delegation.get("worker_tool_calls") or 0
        scenario_worker_errors = len(
            delegation.get("worker_errors") or []
        )
        scenario_worker_input_tokens = delegation.get("input_tokens") or 0
        scenario_worker_output_tokens = delegation.get("output_tokens") or 0
        scenario_duration = result["duration_seconds"]

        tool_calls += scenario_tool_calls
        tool_errors += scenario_tool_errors
        input_tokens += scenario_input_tokens
        output_tokens += scenario_output_tokens
        workers_started += scenario_workers_started
        worker_tool_calls += scenario_worker_tool_calls
        worker_errors += scenario_worker_errors
        worker_input_tokens += scenario_worker_input_tokens
        worker_output_tokens += scenario_worker_output_tokens
        duration_seconds += scenario_duration

        difficulty_summary["evaluated_scenarios"] += 1
        difficulty_summary["achieved_scenarios"] += int(
            result["goal_achieved"]
        )
        difficulty_summary["tool_calls"] += scenario_tool_calls
        difficulty_summary["tool_errors"] += scenario_tool_errors
        difficulty_summary["input_tokens"] += scenario_input_tokens
        difficulty_summary["output_tokens"] += scenario_output_tokens
        difficulty_summary["workers_started"] += scenario_workers_started
        difficulty_summary["worker_tool_calls"] += scenario_worker_tool_calls
        difficulty_summary["worker_errors"] += scenario_worker_errors
        difficulty_summary["worker_input_tokens"] += (
            scenario_worker_input_tokens
        )
        difficulty_summary["worker_output_tokens"] += (
            scenario_worker_output_tokens
        )
        difficulty_summary["duration_seconds"] += scenario_duration

    for difficulty_summary in by_difficulty.values():
        evaluated = difficulty_summary["evaluated_scenarios"]
        difficulty_summary["accuracy"] = (
            difficulty_summary["achieved_scenarios"] / evaluated
            if evaluated
            else 0.0
        )

    return {
        "evaluated_scenarios": evaluated_scenarios,
        "achieved_scenarios": achieved_scenarios,
        "accuracy": (
            achieved_scenarios / evaluated_scenarios
            if evaluated_scenarios
            else 0.0
        ),
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "workers_started": workers_started,
        "worker_tool_calls": worker_tool_calls,
        "worker_errors": worker_errors,
        "worker_input_tokens": worker_input_tokens,
        "worker_output_tokens": worker_output_tokens,
        "total_tool_calls": tool_calls + worker_tool_calls,
        "total_tool_errors": tool_errors + worker_errors,
        "total_input_tokens": input_tokens + worker_input_tokens,
        "total_output_tokens": output_tokens + worker_output_tokens,
        "duration_seconds": duration_seconds,
        "average_duration_seconds": (
            duration_seconds / evaluated_scenarios
            if evaluated_scenarios
            else 0.0
        ),
        "by_difficulty": by_difficulty,
    }

def build_pass_at_k_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrupa corridas repetidas por escenario y calcula su tasa de exito.

    No es el estimador combinatorio clasico de pass@k (Chen et al. 2021):
    es la tasa empirica de exito sobre las `k` corridas efectivamente
    realizadas, que es lo medible con un presupuesto de Bedrock acotado.
    Un escenario se marca `resolved` cuando su tasa alcanza
    `PASS_AT_K_THRESHOLD`.
    """
    by_scenario: dict[str, dict[str, Any]] = {}
    for result in results:
        entry = by_scenario.setdefault(
            result["scenario"],
            {"difficulty": result["difficulty"], "attempts": 0, "achieved": 0},
        )
        entry["attempts"] += 1
        entry["achieved"] += int(result["goal_achieved"])

    for entry in by_scenario.values():
        entry["success_rate"] = entry["achieved"] / entry["attempts"]
        entry["resolved"] = entry["success_rate"] >= PASS_AT_K_THRESHOLD

    return by_scenario


def build_report(
    results: list[dict[str, Any]],
    selected_scenario_ids: list[str],
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Arma un artefacto reproducible de una corrida de evaluacion."""
    timestamp = timestamp or datetime.now(timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)

    return {
        "metadata": {
            "run_id": timestamp.strftime("%Y%m%dT%H%M%S%fZ"),
            "timestamp_utc": timestamp.isoformat(),
            "run_kind": RUN_KIND,
            "selected_scenarios": selected_scenario_ids,
            "repeats_per_scenario": REPEATS_PER_SCENARIO,
            "llm_provider": _llm_provider_metadata(),
            "agent_config": _json_safe(M3_AGENT_CONFIG),
        },
        "summary": build_summary(results),
        "pass_at_k": build_pass_at_k_summary(results),
        "results": results,
    }


def save_report(report: dict[str, Any], output_dir: Path) -> Path:
    """Guarda un reporte JSON y devuelve su ruta."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{report['metadata']['run_id']}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path

def main() -> int:
    """Ejecuta cada escenario REPEATS_PER_SCENARIO veces (pass@k) y guarda evidencia."""
    # Timestamp fijo para todo el run: cada checkpoint pisa el mismo
    # archivo con mas datos, en vez de crear uno nuevo por intento.
    timestamp = datetime.now(timezone.utc)
    output_dir = RESULTS_DIR / RUN_KIND

    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    selected_scenario_ids: list[str] = []
    for attempt in range(1, REPEATS_PER_SCENARIO + 1):
        # `Scenario.initial_world` es un `World` concreto y mutable, no una
        # fabrica: reusar los mismos objetos entre intentos dejaria el mundo
        # con la puerta ya abierta o items ya tomados del intento anterior, e
        # invalidaria pass@k. `load_scenario`/`list_scenarios` si construyen
        # un `World` nuevo desde el JSON en cada llamada, asi que se
        # redescubren los escenarios en cada intento para partir de cero.
        available = discover_scenarios(SCENARIOS_DIR)
        selected = select_scenarios(available)
        selected_scenario_ids = [scenario.id for scenario in selected]

        for scenario in selected:
            result = run_scenario(scenario)
            result["attempt"] = attempt
            results.append(result)

        # Checkpoint incremental: si se corta a mitad de camino (p. ej. el
        # token de AWS vence, como ya paso una vez), no se pierde el
        # progreso de los intentos ya completados.
        report = build_report(
            results=results,
            selected_scenario_ids=selected_scenario_ids,
            timestamp=timestamp,
        )
        report_path = save_report(report, output_dir)
        print(
            f"Checkpoint tras intento {attempt}/{REPEATS_PER_SCENARIO}: {report_path}",
            file=sys.stderr,
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Reporte final guardado en: {report_path}", file=sys.stderr)

    all_resolved = all(
        entry["resolved"] for entry in report["pass_at_k"].values()
    )
    return 0 if all_resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
