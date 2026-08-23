"""Tests propios para el planificador inicial opcional de M3."""

from __future__ import annotations

import json

from mia_agents.testing import MockLLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import LLMResponse, ToolCall

from student_framework import build_agent
from student_framework.m3_planner import Plan, render_plan


def _final_result_response(arguments: dict[str, object], *, call_id: str = "plan-1") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id=call_id, name=FINAL_RESULT_TOOL_NAME, arguments=json.dumps(arguments)),
        ],
    )


def _tool_names(tools) -> list[str]:
    return [tool.name if hasattr(tool, "name") else tool["name"] for tool in (tools or [])]


def test_render_plan_numbers_steps_and_adds_disclaimer() -> None:
    plan = Plan(steps=["observar la sala", "tomar la llave", "abrir la puerta"])

    rendered = render_plan(plan)

    assert "1. observar la sala" in rendered
    assert "2. tomar la llave" in rendered
    assert "3. abrir la puerta" in rendered
    assert "priorizá siempre lo que confirmen las tools" in rendered


def test_plan_requires_at_least_one_step() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Plan(steps=[])


def test_agent_without_planner_does_not_request_a_plan() -> None:
    """Por default (use_m3_planner=False) no hay llamada extra de planificación."""
    mock = MockLLMClient([LLMResponse(content="listo")])
    agent = build_agent({"llm_client": mock})

    agent.run("objetivo")

    assert mock.call_count == 1
    assert FINAL_RESULT_TOOL_NAME not in _tool_names(mock.calls[0]["tools"])


def test_agent_with_planner_requests_plan_before_first_tool_turn() -> None:
    """Con use_m3_planner=True, la primera llamada fuerza final_result con el plan."""
    mock = MockLLMClient(
        [
            _final_result_response({"steps": ["observar la sala", "tomar la llave dorada"]}),
            LLMResponse(content="listo"),
        ]
    )
    agent = build_agent({"llm_client": mock, "use_m3_planner": True})

    agent.run("salí de la sala")

    assert mock.call_count == 2
    assert FINAL_RESULT_TOOL_NAME in _tool_names(mock.calls[0]["tools"])

    second_system = mock.calls[1]["system"]
    assert "Plan inicial" in second_system
    assert "1. observar la sala" in second_system
    assert "2. tomar la llave dorada" in second_system


def test_agent_with_planner_and_scratchpad_combines_both_blocks() -> None:
    """El plan y el scratchpad coexisten en el mismo system prompt."""
    def look() -> str:
        return "Estás en Cocina.\nVes:\n  - llave [id: llave]\nSalidas: oeste."

    from mia_agents.types import ToolSchema

    look_schema = ToolSchema.from_callable(look)
    mock = MockLLMClient(
        [
            _final_result_response({"steps": ["mirar alrededor"]}),
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="l1", name=look_schema.name, arguments="{}")],
            ),
            LLMResponse(content="listo"),
        ]
    )
    agent = build_agent(
        {"llm_client": mock, "use_m3_planner": True, "use_m3_scratchpad": True}
    )
    agent.register_tool(look, look_schema)

    agent.run("salí de la sala")

    third_system = mock.calls[2]["system"]
    assert "Plan inicial" in third_system
    assert "Ubicación actual: Cocina" in third_system
