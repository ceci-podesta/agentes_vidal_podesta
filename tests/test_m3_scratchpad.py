"""Tests propios para el scratchpad determinista de M3."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field

from mia_agents.testing import MockLLMClient
from mia_agents.types import AgentStep, LLMResponse, ToolCall, ToolSchema

from student_framework import build_agent
from student_framework.m3_scratchpad import M3Scratchpad


def test_scratchpad_extracts_world_facts() -> None:
    """El scratchpad conserva ubicación, IDs, inventario y errores recientes."""
    scratchpad = M3Scratchpad()

    scratchpad.record(
        AgentStep(
            "look",
            "{}",
            "Estás en Cocina.\n"
            "Ves:\n"
            "  - cajón [id: cajon]\n"
            "Salidas: oeste.",
        )
    )
    scratchpad.record(
        AgentStep(
            "examine",
            json.dumps({"target": "cajon"}),
            "cajón: Contiene:\n  - llave dorada [id: llave_oro]",
        )
    )
    scratchpad.record(
        AgentStep(
            "take",
            json.dumps({"item": "llave_oro"}),
            "Tomas llave dorada.",
        )
    )
    scratchpad.record(
        AgentStep(
            "go",
            json.dumps({"direction": "sur"}),
            "Error: no hay salida 'sur' desde aquí. Salidas disponibles: oeste.",
        )
    )

    rendered = scratchpad.render()

    assert "Ubicación actual: Cocina" in rendered
    assert "Salidas conocidas: oeste" in rendered
    assert "Inventario: llave_oro" in rendered
    assert "cajon: llave_oro" in rendered
    assert "no hay salida 'sur'" in rendered


def test_agent_sends_scratchpad_in_following_chat_calls() -> None:
    """Cada llamada posterior al LLM recibe los hechos ya observados."""
    def look() -> str:
        return (
            "Estás en Cocina.\n"
            "Ves:\n"
            "  - cajón [id: cajon]\n"
            "Salidas: oeste."
        )

    def take(
        item: Annotated[str, Field(description="ID del ítem a tomar")],
    ) -> str:
        return "Tomas llave dorada."

    look_schema = ToolSchema.from_callable(look)
    take_schema = ToolSchema.from_callable(take)
    mock = MockLLMClient([
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id="l1", name=look_schema.name, arguments="{}")
            ],
        ),
        LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name=take_schema.name,
                    arguments=json.dumps({"item": "llave_oro"}),
                )
            ],
        ),
        LLMResponse(content="listo"),
    ])
    agent = build_agent({
        "llm_client": mock,
        "use_m3_scratchpad": True,
    })
    agent.register_tool(look, look_schema)
    agent.register_tool(take, take_schema)

    result = agent.run("explorá")

    second_system = mock.calls[1]["system"]
    third_system = mock.calls[2]["system"]

    assert "Ubicación actual: Cocina" in second_system
    assert "Salidas conocidas: oeste" in second_system
    assert "Inventario: llave_oro" in third_system
    assert result.answer == "listo"
