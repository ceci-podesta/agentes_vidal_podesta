"""Escenarios de prueba propios del grupo para el Milestone 1.

A diferencia de `tests/conformance/test_m1.py` (FIJO), estos escenarios
los escribimos nosotras para ejercitar el agente de punta a punta en
situaciones realistas, en particular el requisito del enunciado de que el
agente use **al menos dos herramientas** en una misma corrida.

Usan `MockLLMClient`, así que son deterministas y **no consumen créditos
de API**. El mock devuelve, en orden, las respuestas que un LLM real
produciría: primero los `tool_calls` y al final la respuesta de texto.

    pytest tests/test_scenarios_m1.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from mia_agents.testing import MockLLMClient
from mia_agents.types import LLMResponse, ToolCall

from student_framework import build_agent
from student_framework.agent import MyAgent
from student_framework.tools.calculator import calculator, calculator_schema
from student_framework.tools.word_counter import word_counter, word_counter_schema


def _tool_call(call_id: str, name: str, **arguments: object) -> ToolCall:
    """Crea un ToolCall con argumentos serializados como JSON (igual que un LLM real)."""
    return ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))


# ---------------------------------------------------------------------------
# Escenario 1: dos herramientas en una misma corrida (calculadora + contador).
# ---------------------------------------------------------------------------
def test_scenario_dos_herramientas_calculadora_y_contador() -> None:
    """El agente calcula 8 * 3 y luego cuenta las palabras de una frase.

    Demuestra el bucle multi-turno: tool_call -> resultado -> tool_call ->
    resultado -> respuesta final de texto.
    """
    mock = MockLLMClient(
        [
            # Turno 1: el modelo pide la calculadora.
            LLMResponse(
                content=None,
                tool_calls=[
                    _tool_call("c1", "calculator", left_operand=8, right_operand=3, operator="*")
                ],
            ),
            # Turno 2: con el resultado "24", pide contar palabras.
            LLMResponse(
                content=None,
                tool_calls=[
                    _tool_call("c2", "word_counter", text="hola que tal")
                ],
            ),
            # Turno 3: respuesta final de texto (sin tool_calls -> termina).
            LLMResponse(content="8 por 3 es 24 y la frase tiene 3 palabras."),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("Calculá 8 * 3 y contá las palabras de 'hola que tal'.")

    # Se usaron exactamente dos herramientas, en orden.
    assert len(result.steps) == 2
    assert result.steps[0].tool_name == "calculator"
    assert result.steps[0].tool_output == "24"
    assert result.steps[0].error is None
    assert result.steps[1].tool_name == "word_counter"
    assert result.steps[1].tool_output == "3"
    assert result.steps[1].error is None

    # Respuesta final y número total de llamadas al LLM (2 tools + 1 final).
    assert result.answer == "8 por 3 es 24 y la frase tiene 3 palabras."
    assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Escenario 2: leer un archivo real y contar sus palabras (file_reader + contador).
# ---------------------------------------------------------------------------
def test_scenario_leer_archivo_y_contar_palabras() -> None:
    """El agente lee un archivo permitido dentro de sample_files y cuenta sus palabras."""
    sample_dir = Path("sample_files")
    sample_dir.mkdir(exist_ok=True)
    archivo = sample_dir / "notas_test_m1.txt"
    archivo.write_text("uno dos tres cuatro", encoding="utf-8")

    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    _tool_call("c1", "file_reader", path="notas_test_m1.txt")
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    _tool_call("c2", "word_counter", text="uno dos tres cuatro")
                ],
            ),
            LLMResponse(content="El archivo tiene 4 palabras."),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("Leé notas_test_m1.txt y contá sus palabras.")

    assert len(result.steps) == 2
    assert result.steps[0].tool_name == "file_reader"
    assert result.steps[0].tool_output == "uno dos tres cuatro"
    assert result.steps[0].error is None
    assert result.steps[1].tool_name == "word_counter"
    assert result.steps[1].tool_output == "4"
    assert result.steps[1].error is None
    assert result.answer == "El archivo tiene 4 palabras."



# ---------------------------------------------------------------------------
# Escenario 3: el resultado de la herramienta llega al LLM en la 2da llamada.
# ---------------------------------------------------------------------------
def test_scenario_resultado_se_realimenta_al_llm() -> None:
    """Tras ejecutar una tool, su salida aparece en los mensajes de la 2da llamada."""
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    _tool_call("c1", "calculator", left_operand=10, right_operand=4, operator="+")
                ],
            ),
            LLMResponse(content="El resultado es 14."),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("¿Cuánto es 10 + 4?")

    # La salida "14" debe estar volcada en los mensajes de la segunda llamada.
    segunda_llamada = mock.calls[1]["messages"]
    contenidos = [str(m.get("content")) for m in segunda_llamada]
    assert any("14" in c for c in contenidos)
    assert result.answer == "El resultado es 14."


# ---------------------------------------------------------------------------
# Escenario 4: robustez ante una herramienta inexistente (alucinación).
# ---------------------------------------------------------------------------
def test_scenario_herramienta_desconocida_no_rompe() -> None:
    """Si el LLM alucina una tool inexistente, el paso queda con error y run() no rompe."""
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[_tool_call("c1", "herramienta_fantasma", x=1)],
            ),
            LLMResponse(content="Disculpá, no pude completar la tarea."),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("Usá una herramienta que no existe.")

    assert len(result.steps) == 1
    assert result.steps[0].tool_name == "herramienta_fantasma"
    assert result.steps[0].error is not None
    assert result.steps[0].tool_output is None
    assert result.answer == "Disculpá, no pude completar la tarea."


# ---------------------------------------------------------------------------
# Escenario 5: terminación por max_iterations (sin bucle infinito).
# ---------------------------------------------------------------------------
def test_scenario_corta_por_max_iterations() -> None:
    """Si el LLM nunca deja de pedir tools, el agente corta en max_iterations."""
    # Construimos el agente directamente para fijar un tope chico (3 llamadas).
    respuestas = [
        LLMResponse(
            content=None,
            tool_calls=[
                _tool_call(f"c{i}", "calculator", left_operand=1, right_operand=1, operator="+")
            ],
        )
        for i in range(3)
    ]
    mock = MockLLMClient(respuestas)
    agent = MyAgent(llm_client=mock, max_iterations=3)
    agent.register_tool(calculator, calculator_schema)
    agent.register_tool(word_counter, word_counter_schema)

    result = agent.run("Entrá en bucle.")

    # Llamó al LLM exactamente max_iterations veces y devolvió un resultado válido.
    assert mock.call_count == 3
    assert result.error is not None
    assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# Escenario 6: respuesta directa sin herramientas (un solo turno).
# ---------------------------------------------------------------------------
def test_scenario_respuesta_directa_sin_tools() -> None:
    """Pregunta que el LLM responde sin usar herramientas: una sola llamada."""
    mock = MockLLMClient([LLMResponse(content="¡Hola! ¿En qué puedo ayudarte?")])
    agent = build_agent({"llm_client": mock})

    result = agent.run("hola")

    assert result.answer == "¡Hola! ¿En qué puedo ayudarte?"
    assert result.steps == []
    assert mock.call_count == 1
