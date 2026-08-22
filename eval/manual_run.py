"""Corre un solo escenario con la configuracion real de M3 y muestra su traza.

A diferencia de `python -m mia_world.cli run --scenario ...`, este script usa
`eval.run.run_scenario`: la misma configuracion (`M3_AGENT_CONFIG`, con
scratchpad, guardia anti-repeticion y `research_documents`) que corre
`eval/run.py`. Sirve para inspeccionar un escenario puntual sin pagar el
costo de correr el dataset completo.

Uso:
    python eval/manual_run.py --scenario office-sequence
    python eval/manual_run.py --scenario scenarios/03-hard-library-search.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run import SCENARIOS_DIR, run_scenario  # noqa: E402
from mia_world.scenarios import list_scenarios, load_scenario  # noqa: E402


def _resolve_scenario(spec: str, scenarios_dir: Path):
    path = Path(spec)
    if path.is_file():
        return load_scenario(path)

    available = list_scenarios(scenarios_dir)
    by_id = {sc.id: sc for sc in available}
    if spec in by_id:
        return by_id[spec]

    by_difficulty = [sc for sc in available if sc.difficulty == spec]
    if by_difficulty:
        return by_difficulty[0]

    options = ", ".join(sorted(by_id)) or "(ninguno)"
    raise SystemExit(f"No se encontró el escenario {spec!r}. Disponibles: {options}.")


def print_trace(result: dict) -> None:
    print(f"# Escenario: {result['scenario']} ({result['difficulty']})")
    print()
    for index, step in enumerate(result["agent_result"]["steps"], start=1):
        marker = " [ERROR]" if step.get("error") else ""
        print(f"{index:>2}. {step['tool_name']}({step['tool_input']}){marker}")
        print(f"    -> {step['tool_output']}")
    print()
    print(f"Respuesta final del agente: {result['agent_result']['answer']!r}")
    agent_error = result["agent_result"].get("error")
    if agent_error:
        print(f"Error final del AgentResult: {agent_error}")
    print(f"goal_achieved: {result['goal_achieved']} — {result['goal_reason']}")
    delegation = result.get("delegation") or {}
    if delegation.get("workers_started"):
        print(
            f"Delegación: {delegation['workers_started']} workers, "
            f"{delegation.get('worker_tool_calls', 0)} tool calls de worker, "
            f"{len(delegation.get('worker_errors') or [])} worker_errors"
        )
    tokens_in = result["agent_result"].get("input_tokens")
    tokens_out = result["agent_result"].get("output_tokens")
    print(f"Tokens principal: entrada={tokens_in} salida={tokens_out}")
    print(f"Duración: {result['duration_seconds']:.2f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        required=True,
        help="Id, dificultad o path a un JSON de scenarios/.",
    )
    parser.add_argument(
        "--scenarios-dir",
        default=str(SCENARIOS_DIR),
        help="Directorio donde buscar escenarios (por defecto: scenarios/).",
    )
    args = parser.parse_args(argv)

    scenario = _resolve_scenario(args.scenario, Path(args.scenarios_dir))
    result = run_scenario(scenario)
    print_trace(result)

    return 0 if result["goal_achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
