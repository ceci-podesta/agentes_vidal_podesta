"""Tests propios del LLM-as-a-judge; no modifican conformidad docente."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from eval.llm_judge import (
    build_default_output_path,
    build_judge_prompt,
    judge_report,
)
from mia_agents.testing import MockLLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import LLMResponse, ToolCall


def _trace(
    *,
    scenario: str = "demo",
    attempt: int = 1,
    goal_achieved: bool = True,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "difficulty": "easy",
        "attempt": attempt,
        "user_message": "Abri la puerta.",
        "goal": {"type": "item_open", "item": "puerta"},
        "goal_achieved": goal_achieved,
        "goal_reason": "OUTCOME_SECRETO",
        "plan_block": "PLAN_SECRETO_NO_ES_EVIDENCIA",
        "agent_result": {
            "answer": "Listo.",
            "error": None,
            "steps": [
                {
                    "tool_name": "use",
                    "tool_input": '{"item":"llave_roja","target":"puerta"}',
                    "tool_output": "Error: la llave roja no encaja.",
                    "error": None,
                },
                {
                    "tool_name": "use",
                    "tool_input": '{"item":"llave_azul","target":"puerta"}',
                    "tool_output": "La puerta se abre.",
                    "error": None,
                },
            ],
        },
    }


def _judgment_arguments(
    *,
    score: int = 2,
    incorporated: bool = True,
) -> dict[str, object]:
    return {
        "applicable": True,
        "score": score,
        "justification": "Incorporo la incompatibilidad en la accion siguiente.",
        "episodes": [
            {
                "feedback_step": 1,
                "feedback_evidence": "Error: la llave roja no encaja.",
                "next_relevant_step": 2,
                "next_action_evidence": '"item":"llave_azul"',
                "incorporated": incorporated,
                "explanation": "Cambio la llave despues del feedback.",
            }
        ],
    }


def _response(
    arguments: dict[str, object],
    call_id: str = "judge-1",
) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=FINAL_RESULT_TOOL_NAME,
                arguments=json.dumps(arguments),
            )
        ],
        input_tokens=100,
        output_tokens=20,
    )


def test_prompt_hides_outcome_and_unverified_plan() -> None:
    prompt = build_judge_prompt(_trace())

    assert '"goal_achieved"' not in prompt
    assert '"goal_reason"' not in prompt
    assert "OUTCOME_SECRETO" not in prompt
    assert '"plan_block"' not in prompt
    assert "PLAN_SECRETO_NO_ES_EVIDENCIA" not in prompt
    assert '"attempt": 1' in prompt
    assert '"goal"' in prompt
    assert '"steps"' in prompt


def test_judge_report_returns_feedback_outcome_metadata_and_usage() -> None:
    mock = MockLLMClient([_response(_judgment_arguments())])
    report = {"metadata": {"run_id": "run-demo"}, "results": [_trace()]}

    judged = judge_report(
        report,
        mock,
        judge_metadata={"provider": "mock", "model": "judge-demo"},
    )

    assert judged["metadata"]["evaluation_dimension"] == "uso_del_feedback"
    assert judged["metadata"]["rubric_scale"] == "0-2-null"
    assert judged["metadata"]["deterministic_outcome_included_in_prompt"] is False
    assert judged["summary"]["requested_traces"] == 1
    assert judged["summary"]["judged_traces"] == 1
    assert judged["summary"]["failed_judgments"] == 0
    assert judged["summary"]["feedback_use"] == {
        "average_score": 2.0,
        "score_distribution": {"0": 0, "1": 0, "2": 1},
        "evaluated_traces": 1,
        "not_applicable": 0,
        "feedback_episodes": 1,
        "incorporated_episodes": 1,
        "incorporation_rate": 1.0,
    }
    assert judged["judgments"][0]["source_outcome"] == {
        "goal_achieved": True,
        "goal_reason": "OUTCOME_SECRETO",
    }
    assert judged["judgments"][0]["judge_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "attempts": 1,
    }
    assert judged["errors"] == []


def test_no_feedback_is_not_applicable() -> None:
    trace = _trace()
    agent_result = trace["agent_result"]
    assert isinstance(agent_result, dict)
    agent_result["steps"] = [
        {
            "tool_name": "look",
            "tool_input": "{}",
            "tool_output": "Ves una puerta cerrada.",
            "error": None,
        }
    ]
    judgment = {
        "applicable": False,
        "score": None,
        "justification": "No hubo feedback correctivo.",
        "episodes": [],
    }
    mock = MockLLMClient([_response(judgment)])

    judged = judge_report(
        {"metadata": {"run_id": "run-demo"}, "results": [trace]},
        mock,
    )

    assert judged["summary"]["feedback_use"]["evaluated_traces"] == 0
    assert judged["summary"]["feedback_use"]["not_applicable"] == 1


def test_one_failed_judgment_does_not_discard_the_following_trace() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(
                content="texto libre",
                input_tokens=100,
                output_tokens=20,
            ),
            LLMResponse(
                content="texto libre",
                input_tokens=100,
                output_tokens=20,
            ),
            LLMResponse(
                content="texto libre",
                input_tokens=100,
                output_tokens=20,
            ),
            _response(_judgment_arguments(), "valid-2"),
        ]
    )
    checkpoints: list[dict[str, object]] = []
    report = {
        "metadata": {"run_id": "run-two"},
        "results": [
            _trace(scenario="first", attempt=1),
            _trace(scenario="second", attempt=1),
        ],
    }

    judged = judge_report(
        report,
        mock,
        checkpoint=lambda partial: checkpoints.append(partial),
    )

    assert len(checkpoints) == 2
    assert judged["summary"]["requested_traces"] == 2
    assert judged["summary"]["execution_status"] == "completed_with_errors"
    assert judged["summary"]["judged_traces"] == 1
    assert judged["summary"]["failed_judgments"] == 1
    assert judged["judgments"][0]["scenario"] == "second"
    assert judged["errors"][0]["scenario"] == "first"
    assert judged["errors"][0]["error_type"] == "StructuredCallError"
    assert judged["errors"][0]["judge_usage"] == {
        "input_tokens": 300,
        "output_tokens": 60,
        "attempts": 3,
    }
    assert judged["summary"]["judge_input_tokens"] == 400
    assert judged["summary"]["judge_output_tokens"] == 80


def test_filters_by_scenario_and_attempt_before_judging() -> None:
    mock = MockLLMClient([_response(_judgment_arguments())])
    report = {
        "metadata": {"run_id": "run-filter"},
        "results": [
            _trace(scenario="color-locks", attempt=1),
            _trace(scenario="color-locks", attempt=2),
            _trace(scenario="office-sequence", attempt=2),
        ],
    }

    judged = judge_report(
        report,
        mock,
        scenarios={"color-locks"},
        attempts={2},
    )

    assert mock.call_count == 1
    assert judged["summary"]["requested_traces"] == 1
    assert judged["judgments"][0]["scenario"] == "color-locks"
    assert judged["judgments"][0]["attempt"] == 2


def test_default_output_path_is_separate_traceable_and_automatic() -> None:
    timestamp = datetime(2026, 8, 23, 23, 15, tzinfo=timezone.utc)
    path = build_default_output_path(
        report_path=Path("eval/results/final/source.json"),
        report={"metadata": {"run_id": "20260823T201948569061Z"}},
        model_id="mistral.mistral-large-3-675b-instruct",
        timestamp=timestamp,
        scenarios={"office-sequence"},
        attempts={2},
    )

    assert path.parent.name == "20260823T201948569061Z"
    assert path.parent.parent.name == "judge"
    assert path.name == (
        "20260823T231500000000Z__"
        "mistral-mistral-large-3-675b-instruct__"
        "office-sequence__attempt-2.json"
    )


def test_global_judge_failure_is_saved_without_raising(
    monkeypatch,
    tmp_path,
) -> None:
    import eval.llm_judge as judge_module

    def fail_client(model_id, region):
        raise RuntimeError("credenciales vencidas")

    monkeypatch.setattr(
        judge_module,
        "_build_judge_client",
        fail_client,
    )

    source_report = {
        "metadata": {"run_id": "run-global-failure"},
        "results": [
            {
                "scenario": "study-with-key",
                "difficulty": "easy",
                "attempt": 1,
                "goal_achieved": False,
                "goal_reason": "puerta cerrada",
            }
        ],
    }
    output_path = tmp_path / "judge-failed.json"

    judged, saved_path = judge_module.evaluate_report(
        source_report,
        report_path=tmp_path / "source.json",
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()
    assert judged["summary"]["execution_status"] == "failed"
    assert judged["summary"]["requested_traces"] == 1
    assert judged["summary"]["failed_judgments"] == 1
    assert judged["errors"][0]["error_type"] == "RuntimeError"


def test_cli_returns_zero_when_judgments_have_errors(
    monkeypatch,
    tmp_path,
) -> None:
    import json
    import eval.llm_judge as judge_module

    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "metadata": {"run_id": "run-partial"},
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "judge.json"
    partial_report = {
        "summary": {
            "execution_status": "completed_with_errors",
            "failed_judgments": 1,
        },
        "errors": [
            {
                "error_type": "StructuredCallError",
            }
        ],
    }

    monkeypatch.setattr(
        judge_module,
        "evaluate_report",
        lambda *args, **kwargs: (partial_report, output_path),
    )

    exit_code = judge_module.main(
        [str(source_path), "--output", str(output_path)]
    )

    assert exit_code == 0
