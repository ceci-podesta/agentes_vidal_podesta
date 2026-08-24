"""Tests de métricas para structured_call_with_usage."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from mia_agents.testing import MockLLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import LLMResponse, ToolCall
from student_framework import build_agent
from student_framework.agent import StructuredCallError


class Answer(BaseModel):
    result: int


def _final_result_response(
    arguments: dict[str, object],
    *,
    call_id: str,
    input_tokens: int | None,
    output_tokens: int | None,
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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_structured_call_with_usage_returns_value_and_tokens() -> None:
    mock = MockLLMClient(
        [
            _final_result_response(
                {"result": 42},
                call_id="final-1",
                input_tokens=11,
                output_tokens=7,
            )
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.structured_call_with_usage(
        prompt="devolve el resultado",
        schema=Answer,
    )

    assert result.value == Answer(result=42)
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.attempts == 1


def test_structured_call_with_usage_sums_repair_attempts() -> None:
    mock = MockLLMClient(
        [
            _final_result_response(
                {"result": "no es un entero"},
                call_id="final-1",
                input_tokens=10,
                output_tokens=3,
            ),
            _final_result_response(
                {"result": 42},
                call_id="final-2",
                input_tokens=20,
                output_tokens=4,
            ),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.structured_call_with_usage(
        prompt="devolve un entero",
        schema=Answer,
    )

    assert result.value == Answer(result=42)
    assert result.usage.input_tokens == 30
    assert result.usage.output_tokens == 7
    assert result.usage.attempts == 2


def test_structured_call_error_keeps_usage_after_all_repairs_fail() -> None:
    mock = MockLLMClient(
        [
            LLMResponse(content="texto libre", input_tokens=10, output_tokens=1),
            LLMResponse(content="texto libre", input_tokens=20, output_tokens=2),
            LLMResponse(content="texto libre", input_tokens=30, output_tokens=3),
        ]
    )
    agent = build_agent({"llm_client": mock})

    with pytest.raises(StructuredCallError) as caught:
        agent.structured_call_with_usage(
            prompt="devolve un entero",
            schema=Answer,
            max_repair_attempts=2,
        )

    assert caught.value.usage.input_tokens == 60
    assert caught.value.usage.output_tokens == 6
    assert caught.value.usage.attempts == 3


def test_structured_call_pairs_invalid_tool_call_with_tool_result() -> None:
    mock = MockLLMClient(
        [
            _final_result_response(
                {"result": "no es un entero"},
                call_id="invalid-final",
                input_tokens=10,
                output_tokens=3,
            ),
            _final_result_response(
                {"result": 42},
                call_id="valid-final",
                input_tokens=20,
                output_tokens=4,
            ),
        ]
    )
    agent = build_agent({"llm_client": mock})

    agent.structured_call_with_usage(
        prompt="devolve un entero",
        schema=Answer,
    )

    repair_messages = mock.calls[1]["messages"]
    assert repair_messages[-2]["role"] == "tool"
    assert repair_messages[-2]["tool_call_id"] == "invalid-final"
    assert "Error de validacion" in repair_messages[-2]["content"]
    assert repair_messages[-1]["role"] == "user"

