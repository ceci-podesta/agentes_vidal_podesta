"""Runner incremental de evaluacion para M3."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mia_world import (
    Scenario,
    check_goal,
    list_scenarios,
    make_world_tools,
)
from student_framework import build_agent


SCENARIOS_DIR = PROJECT_ROOT / "scenarios"

DIFFICULTY_ORDER = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
    "extreme": 3,
}

# Durante el desarrollo, agregar IDs aqui habilita escenarios adicionales.
# Para la corrida final, usar None para evaluar todo el dataset descubierto.
DEVELOPMENT_SCENARIOS: list[str] | None = [
    "study-with-key",
     "color-locks",
    # "apartment-keys",
    # "library-search",
    # "office-sequence",
    # "extreme-archive",
    # "vault-combination",
    # "backtracking-vault",
]


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


def run_scenario(scenario: Scenario) -> dict[str, Any]:
    """Ejecuta un escenario una vez y devuelve su registro evaluable."""
    world = scenario.initial_world
    agent = build_agent()

    for function, schema in make_world_tools(world):
        agent.register_tool(function, schema)

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
    }


def main() -> int:
    """Ejecuta y evalua todos los escenarios habilitados."""
    available = discover_scenarios(SCENARIOS_DIR)
    selected = select_scenarios(available)

    results = [run_scenario(scenario) for scenario in selected]
    print(
        json.dumps(
            {
                "evaluated_scenarios": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if all(result["goal_achieved"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
