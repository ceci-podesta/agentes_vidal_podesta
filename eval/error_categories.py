"""Categorizacion de errores para el analisis de M3 (funciones puras).

Deriva categorias de fallo reproducibles a partir de un reporte generado por
`eval/run.py`, sin volver a llamar al LLM ni depender de juicio manual donde
se puede automatizar. Las categorias replican las siete definidas en
`evaluación_M3.md` y se anclan en los mensajes de error literales que emite
`mia_world/tools.py` (fijo) y en el `goal_reason` de `mia_world.check_goal`
(fijo).

No cubre `delegation.worker_errors` (los rechazos de grounding de
`research_documents`): esos no son fallos del agente principal, son la señal
de que la validación de la delegación funcionó (ver `evolución_agente_M3.md`,
corrida 11).

Uso:
    python eval/error_categories.py eval/results/final/<run_id>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Umbral de tokens de entrada por escenario a partir del cual se marca
# presión de contexto para revisión manual. No mide el límite exacto del
# modelo: es una señal para que un humano confirme la causa en la traza.
CONTEXT_PRESSURE_INPUT_TOKENS = 100_000

ID_INVENTADO_MARKERS = ("no existe ningún objeto con id",)
VIOLACION_ESTADO_MARKERS = (
    "no ves ningún",
    "no es visible o accesible",
    "no llevas ningún",
    "no es algo que puedas llevarte",
    "no hay salida",
    "está bloqueado por",
    "lleva a una sala",
)
REDUNDANTE_MARKERS = (
    "ya llevas",
    # Guardia anti-repetición del propio agente (student_framework/agent.py),
    # no de mia_world/tools.py: mismo fenómeno (acción/observación repetida
    # sin progreso), mensaje distinto.
    "ya falló sin que hubiera progreso",
    "ya fue realizada sin que hubiera",
)


def _step_error_text(step: dict[str, Any]) -> str | None:
    """Extrae el texto de error de un `AgentStep` serializado, si hay."""
    if step.get("error"):
        return str(step["error"])
    tool_output = step.get("tool_output")
    if isinstance(tool_output, str) and tool_output.lstrip().startswith("Error:"):
        return tool_output
    return None


def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def categorize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Devuelve las categorías de fallo detectadas para un escenario."""
    categories: dict[str, list[str]] = {}

    def _flag(category: str, evidence: str) -> None:
        categories.setdefault(category, []).append(evidence)

    agent_result = result.get("agent_result") or {}
    steps = agent_result.get("steps") or []

    previous_signature: tuple[Any, Any] | None = None
    for index, step in enumerate(steps):
        error_text = _step_error_text(step)
        if error_text:
            if _matches_any(error_text, ID_INVENTADO_MARKERS):
                _flag("id_inventado", f"paso {index}: {error_text}")
            elif _matches_any(error_text, VIOLACION_ESTADO_MARKERS):
                _flag("violacion_estado", f"paso {index}: {error_text}")
            elif _matches_any(error_text, REDUNDANTE_MARKERS):
                _flag("accion_redundante", f"paso {index}: {error_text}")

        signature = (step.get("tool_name"), step.get("tool_input"))
        if previous_signature is not None and signature == previous_signature:
            _flag(
                "accion_redundante",
                f"paso {index}: repite {signature!r} sin información nueva",
            )
        previous_signature = signature

    final_error = agent_result.get("error")
    if final_error and "iteraciones" in final_error.lower():
        _flag("limite_iteraciones", final_error)
    elif not result.get("goal_achieved") and not final_error:
        _flag(
            "terminacion_prematura",
            "el agente devolvió una respuesta final sin cumplir el goal y "
            "sin agotar max_iterations",
        )

    goal_reason = result.get("goal_reason") or ""
    if not result.get("goal_achieved") and (
        "no ocurrió en el orden requerido" in goal_reason
        or "faltan condiciones" in goal_reason
    ):
        _flag("planificacion_orden_incorrecto", goal_reason)

    input_tokens = agent_result.get("input_tokens") or 0
    delegation = result.get("delegation") or {}
    worker_input_tokens = delegation.get("input_tokens") or 0
    total_input_tokens = input_tokens + worker_input_tokens
    if (
        not result.get("goal_achieved")
        and total_input_tokens >= CONTEXT_PRESSURE_INPUT_TOKENS
    ):
        _flag(
            "presion_contexto",
            f"input_tokens totales={total_input_tokens} sin alcanzar el "
            "goal (confirmar manualmente en la traza si la causa fue la "
            "ventana de contexto)",
        )

    return {
        "scenario": result.get("scenario"),
        "difficulty": result.get("difficulty"),
        "goal_achieved": result.get("goal_achieved"),
        "categories": categories,
    }


def categorize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Aplica `categorize_result` a cada escenario de un reporte de `eval/run.py`."""
    per_scenario = [
        categorize_result(result) for result in report.get("results", [])
    ]

    totals: dict[str, int] = {}
    for entry in per_scenario:
        for category in entry["categories"]:
            totals[category] = totals.get(category, 0) + 1

    return {"per_scenario": per_scenario, "totals": totals}


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(
            "Uso: python eval/error_categories.py <reporte.json>",
            file=sys.stderr,
        )
        return 1

    report_path = Path(argv[0])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis = categorize_report(report)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
