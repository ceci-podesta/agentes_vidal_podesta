"""Tests propios para las mejoras del agente en M3."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import Field

from mia_agents.testing import MockLLMClient
from mia_agents.types import LLMResponse, ToolCall, ToolSchema

from student_framework import build_agent


def _response(call_id: str, name: str, arguments: dict[str, str]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments),
            )
        ],
    )


def test_repeated_failure_is_blocked_after_observation() -> None:
    """Una repetición idéntica no vuelve a ejecutar la tool que ya falló."""
    calls = 0

    def use(
        item: Annotated[str, Field(description="Ítem a usar")],
    ) -> str:
        nonlocal calls
        calls += 1
        return "Error: falta tomar el ítem primero."

    def look() -> str:
        return "La habitación sigue igual."

    use_schema = ToolSchema.from_callable(use)
    look_schema = ToolSchema.from_callable(look)
    mock = MockLLMClient([
        _response("u1", use_schema.name, {"item": "llave"}),
        _response("l1", look_schema.name, {}),
        _response("u2", use_schema.name, {"item": "llave"}),
        LLMResponse(content="cambio de estrategia"),
    ])
    agent = build_agent({
        "llm_client": mock,
        "max_repeated_failures": 1,
        "max_repeated_observations": 1,
        "observation_tool_names": {"look"},
    })
    agent.register_tool(use, use_schema)
    agent.register_tool(look, look_schema)

    result = agent.run("probá una herramienta")

    assert calls == 1
    assert result.answer == "cambio de estrategia"
    assert result.steps[2].error is not None
    assert "ya falló" in result.steps[2].error


def test_successful_action_resets_failed_call_guard() -> None:
    """Un take exitoso habilita reintentar un use que falló antes."""
    has_key = False
    use_calls = 0

    def use(
        item: Annotated[str, Field(description="Ítem a usar")],
    ) -> str:
        nonlocal use_calls
        use_calls += 1
        if not has_key:
            return "Error: falta tomar el ítem primero."
        return "La puerta se abre."

    def take(
        item: Annotated[str, Field(description="Ítem a tomar")],
    ) -> str:
        nonlocal has_key
        has_key = True
        return "Tomás el ítem."

    use_schema = ToolSchema.from_callable(use)
    take_schema = ToolSchema.from_callable(take)
    mock = MockLLMClient([
        _response("u1", use_schema.name, {"item": "llave"}),
        _response("t1", take_schema.name, {"item": "llave"}),
        _response("u2", use_schema.name, {"item": "llave"}),
        LLMResponse(content="puerta abierta"),
    ])
    agent = build_agent({
        "llm_client": mock,
        "max_repeated_failures": 1,
    })
    agent.register_tool(use, use_schema)
    agent.register_tool(take, take_schema)

    result = agent.run("abrí la puerta")

    assert use_calls == 2
    assert result.steps[2].tool_output == "La puerta se abre."
    assert result.answer == "puerta abierta"


def test_progress_observation_resets_failed_call_guard() -> None:
    """Un examine que revela estado nuevo habilita reintentar un take bloqueado.

    Reproduce el bug real de office-sequence: `take` falla porque el ítem
    todavía no es visible, después un `examine` abre el contenedor y lo
    revela, pero como `examine` es una observation tool, sin
    `progress_observation_tools` el `take` seguiría bloqueado por la guarda
    aunque el mundo ya haya cambiado.
    """
    revealed = False
    take_calls = 0

    def take(
        item: Annotated[str, Field(description="Ítem a tomar")],
    ) -> str:
        nonlocal take_calls
        take_calls += 1
        if not revealed:
            return "Error: no es visible o accesible desde aquí."
        return "Tomás el ítem."

    def examine(
        target: Annotated[str, Field(description="Objeto a examinar")],
    ) -> str:
        nonlocal revealed
        revealed = True
        return "Contiene: llave_maestra [id: llave_maestra]."

    take_schema = ToolSchema.from_callable(take)
    examine_schema = ToolSchema.from_callable(examine)
    mock = MockLLMClient([
        _response("t1", take_schema.name, {"item": "llave_maestra"}),
        _response("e1", examine_schema.name, {"target": "caja_fuerte"}),
        _response("t2", take_schema.name, {"item": "llave_maestra"}),
        LLMResponse(content="tomé la llave"),
    ])
    agent = build_agent({
        "llm_client": mock,
        "max_repeated_failures": 1,
        "max_repeated_observations": 1,
        "observation_tool_names": {"examine"},
        "progress_observation_tools": {"examine"},
    })
    agent.register_tool(take, take_schema)
    agent.register_tool(examine, examine_schema)

    result = agent.run("conseguí la llave maestra")

    assert take_calls == 2
    assert result.steps[2].tool_output == "Tomás el ítem."
    assert result.answer == "tomé la llave"


def test_repeated_observation_is_blocked_without_progress() -> None:
    """Dos look idénticos sin progreso ejecutan el callable una sola vez."""
    calls = 0

    def look() -> str:
        nonlocal calls
        calls += 1
        return "La habitación sigue igual."

    schema = ToolSchema.from_callable(look)
    mock = MockLLMClient([
        _response("l1", schema.name, {}),
        _response("l2", schema.name, {}),
        LLMResponse(content="voy a hacer otra cosa"),
    ])
    agent = build_agent({
        "llm_client": mock,
        "max_repeated_observations": 1,
        "observation_tool_names": {"look"},
    })
    agent.register_tool(look, schema)

    result = agent.run("observá")

    assert calls == 1
    assert result.steps[1].error is not None
    assert "observación ya fue realizada" in result.steps[1].error
