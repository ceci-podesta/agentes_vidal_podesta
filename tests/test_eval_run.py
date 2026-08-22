from datetime import datetime, timezone
import json

from eval.run import build_report, build_summary, save_report


def _result(
    scenario: str,
    difficulty: str,
    goal_achieved: bool,
    duration_seconds: float,
    input_tokens: int | None,
    output_tokens: int | None,
    steps: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "difficulty": difficulty,
        "goal_achieved": goal_achieved,
        "duration_seconds": duration_seconds,
        "agent_result": {
            "steps": steps,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def test_build_summary_aggregates_metrics_and_difficulties() -> None:
    results = [
        _result(
            scenario="easy-case",
            difficulty="easy",
            goal_achieved=True,
            duration_seconds=1.5,
            input_tokens=10,
            output_tokens=5,
            steps=[
                {"tool_output": "OK", "error": None},
                {"tool_output": "OK", "error": None},
            ],
        ),
        _result(
            scenario="hard-case",
            difficulty="hard",
            goal_achieved=False,
            duration_seconds=2.5,
            input_tokens=None,
            output_tokens=7,
            steps=[
                {"tool_output": "Error: ID inexistente.", "error": None},
                {
                    "tool_output": "Error: llamada bloqueada.",
                    "error": "Llamada repetida.",
                },
            ],
        ),
    ]

    summary = build_summary(results)

    assert summary["evaluated_scenarios"] == 2
    assert summary["achieved_scenarios"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["tool_calls"] == 4
    assert summary["tool_errors"] == 2
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 12
    assert summary["duration_seconds"] == 4.0
    assert summary["average_duration_seconds"] == 2.0
    assert summary["by_difficulty"]["easy"]["accuracy"] == 1.0
    assert summary["by_difficulty"]["hard"]["accuracy"] == 0.0


def test_report_is_json_serializable_and_can_be_saved(tmp_path) -> None:
    results = [
        _result(
            scenario="easy-case",
            difficulty="easy",
            goal_achieved=True,
            duration_seconds=1.0,
            input_tokens=3,
            output_tokens=2,
            steps=[],
        )
    ]
    report = build_report(
        results=results,
        selected_scenario_ids=["easy-case"],
        timestamp=datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
    )

    report_path = save_report(report, tmp_path)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.name == "20260821T120000000000Z.json"
    assert saved_report["metadata"]["selected_scenarios"] == ["easy-case"]
    assert saved_report["metadata"]["agent_config"][
        "observation_tool_names"
    ] == ["examine", "look", "research_documents"]
    assert saved_report["summary"]["accuracy"] == 1.0

def test_build_summary_includes_delegated_worker_metrics() -> None:
    result = _result(
        scenario="extreme-case",
        difficulty="extreme",
        goal_achieved=True,
        duration_seconds=3.0,
        input_tokens=20,
        output_tokens=10,
        steps=[{"tool_output": "OK", "error": None}],
    )
    result["delegation"] = {
        "workers_started": 2,
        "worker_tool_calls": 6,
        "input_tokens": 40,
        "output_tokens": 12,
        "worker_errors": ["un reporte invalido"],
    }

    summary = build_summary([result])

    assert summary["workers_started"] == 2
    assert summary["worker_tool_calls"] == 6
    assert summary["worker_errors"] == 1
    assert summary["worker_input_tokens"] == 40
    assert summary["worker_output_tokens"] == 12
    assert summary["total_tool_calls"] == 7
    assert summary["total_tool_errors"] == 1
    assert summary["total_input_tokens"] == 60
    assert summary["total_output_tokens"] == 22
    assert summary["by_difficulty"]["extreme"]["workers_started"] == 2

