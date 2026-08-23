import pytest

from eval.rubric import (
    RUBRIC_DIMENSIONS,
    build_rubric_template,
    summarize_rubric,
    validate_rubric_entries,
)


def test_build_rubric_template_covers_every_scenario_and_dimension() -> None:
    report = {
        "results": [
            {"scenario": "study-with-key", "difficulty": "easy", "goal_achieved": True},
            {"scenario": "color-locks", "difficulty": "medium", "goal_achieved": True},
        ]
    }

    template = build_rubric_template(report)

    assert [entry["scenario"] for entry in template] == [
        "study-with-key",
        "color-locks",
    ]
    for entry in template:
        assert set(entry["scores"]) == set(RUBRIC_DIMENSIONS)
        assert all(score is None for score in entry["scores"].values())


def _scored_entry(scenario: str, difficulty: str, scores: dict[str, int]) -> dict[str, object]:
    return {
        "scenario": scenario,
        "difficulty": difficulty,
        "goal_achieved": True,
        "scores": scores,
        "justification": {dimension: "ok" for dimension in scores},
    }


def test_validate_rubric_entries_rejects_missing_dimension() -> None:
    entries = [
        {
            "scenario": "case",
            "scores": {"estado_del_mundo": 2},
        }
    ]

    with pytest.raises(ValueError, match="faltan dimensiones"):
        validate_rubric_entries(entries)


def test_validate_rubric_entries_rejects_out_of_range_score() -> None:
    entries = [_scored_entry("case", "easy", {dim: 5 for dim in RUBRIC_DIMENSIONS})]

    with pytest.raises(ValueError, match="puntaje inválido"):
        validate_rubric_entries(entries)


def test_summarize_rubric_computes_averages_and_totals() -> None:
    entries = [
        _scored_entry(
            "study-with-key",
            "easy",
            {"estado_del_mundo": 2, "recuperacion": 2, "planificacion_y_eficiencia": 2},
        ),
        _scored_entry(
            "library-search",
            "hard",
            {"estado_del_mundo": 1, "recuperacion": 0, "planificacion_y_eficiencia": 1},
        ),
    ]

    summary = summarize_rubric(entries)

    assert summary["scenarios_scored"] == 2
    assert summary["average_by_dimension"]["estado_del_mundo"] == 1.5
    assert summary["average_by_dimension"]["recuperacion"] == 1.0
    assert summary["by_difficulty"]["easy"]["average_total_score"] == 6
    assert summary["by_difficulty"]["hard"]["average_total_score"] == 2
    assert summary["per_scenario"][0]["total_score"] == 6
    assert summary["per_scenario"][0]["max_score"] == 6
