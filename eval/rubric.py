"""Rúbrica cualitativa manual para M3 (funciones puras).

`evaluación_M3.md` elige una rúbrica manual sobre LLM-as-judge para no sumar
costo, variabilidad ni otra dependencia de modelo. Este módulo no reemplaza
esa lectura humana de la traza: genera la planilla a completar a partir de un
reporte real de `eval/run.py`, valida que los puntajes cargados sean válidos
y calcula los agregados que van al informe.

Flujo:
    python eval/rubric.py template eval/results/final/<run_id>.json \\
        eval/results/final/<run_id>.rubric.json
    # completar a mano "scores" y "justification" en el archivo generado
    python eval/rubric.py summary eval/results/final/<run_id>.rubric.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Copiado literal de la tabla de `evaluación_M3.md`; si esa tabla cambia,
# actualizar acá también para que la planilla no quede desalineada con el
# criterio publicado.
RUBRIC_DIMENSIONS: dict[str, dict[int, str]] = {
    "estado_del_mundo": {
        0: "Ignora IDs, inventario, ubicación o salidas disponibles.",
        1: "Tiene errores recuperables de estado.",
        2: "Usa de forma consistente los hechos observados.",
    },
    "recuperacion": {
        0: "Repite el error o abandona.",
        1: "Corrige tras uno o más intentos extra.",
        2: "Corrige en el siguiente paso útil.",
    },
    "planificacion_y_eficiencia": {
        0: "No llega a una secuencia útil o se desvía repetidamente.",
        1: "Llega con exploración o acciones redundantes.",
        2: "Sigue una secuencia próxima al óptimo y evita pasos innecesarios.",
    },
}

VALID_SCORES = (0, 1, 2)


def build_rubric_template(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Arma una entrada vacía por escenario del reporte, lista para completar."""
    entries = []
    for result in report.get("results", []):
        entries.append(
            {
                "scenario": result.get("scenario"),
                "difficulty": result.get("difficulty"),
                "goal_achieved": result.get("goal_achieved"),
                "scores": {dimension: None for dimension in RUBRIC_DIMENSIONS},
                "justification": {dimension: "" for dimension in RUBRIC_DIMENSIONS},
            }
        )
    return entries


def validate_rubric_entries(entries: list[dict[str, Any]]) -> None:
    """Falla ruidosamente si falta completar o hay un puntaje fuera de rango."""
    for entry in entries:
        scenario = entry.get("scenario")
        scores = entry.get("scores") or {}
        missing_dimensions = set(RUBRIC_DIMENSIONS) - set(scores)
        if missing_dimensions:
            raise ValueError(
                f"{scenario!r}: faltan dimensiones {sorted(missing_dimensions)}"
            )
        for dimension, score in scores.items():
            if score not in VALID_SCORES:
                raise ValueError(
                    f"{scenario!r}: puntaje inválido {score!r} en "
                    f"{dimension!r} (debe ser 0, 1 o 2; None = sin completar)"
                )


def summarize_rubric(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula promedios por dimensión, total por escenario y por dificultad."""
    validate_rubric_entries(entries)

    per_scenario = []
    dimension_totals = {dimension: 0 for dimension in RUBRIC_DIMENSIONS}
    by_difficulty: dict[str, dict[str, Any]] = {}

    for entry in entries:
        scores = entry["scores"]
        total_score = sum(scores.values())
        per_scenario.append(
            {
                "scenario": entry["scenario"],
                "difficulty": entry["difficulty"],
                "scores": scores,
                "total_score": total_score,
                "max_score": len(RUBRIC_DIMENSIONS) * max(VALID_SCORES),
            }
        )

        for dimension, score in scores.items():
            dimension_totals[dimension] += score

        difficulty = entry["difficulty"]
        difficulty_summary = by_difficulty.setdefault(
            difficulty, {"scenarios": 0, "average_total_score": 0.0}
        )
        difficulty_summary["scenarios"] += 1
        difficulty_summary["average_total_score"] += total_score

    scenario_count = len(entries) or 1
    for difficulty_summary in by_difficulty.values():
        difficulty_summary["average_total_score"] /= difficulty_summary[
            "scenarios"
        ]

    return {
        "scenarios_scored": len(entries),
        "average_by_dimension": {
            dimension: total / scenario_count
            for dimension, total in dimension_totals.items()
        },
        "by_difficulty": by_difficulty,
        "per_scenario": per_scenario,
    }


def main(argv: list[str] | None = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) < 2 or argv[0] not in {"template", "summary"}:
        print(
            "Uso:\n"
            "  python eval/rubric.py template <reporte.json> <salida.json>\n"
            "  python eval/rubric.py summary <planilla_completada.json>",
            file=sys.stderr,
        )
        return 1

    command, input_path, *rest = argv

    if command == "template":
        if not rest:
            print("Falta la ruta de salida.", file=sys.stderr)
            return 1
        report = json.loads(Path(input_path).read_text(encoding="utf-8"))
        template = build_rubric_template(report)
        output_path = Path(rest[0])
        output_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Planilla generada en: {output_path}", file=sys.stderr)
        return 0

    entries = json.loads(Path(input_path).read_text(encoding="utf-8"))
    summary = summarize_rubric(entries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
