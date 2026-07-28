"""Tests propios del grupo para el Milestone 2.

Cubren los comportamientos implementados por María:
  - Reintentos ante fallos transitorios del cliente LLM (punto 7).
  - Errores recuperables en calculator, file_reader y word_counter (punto 6).
  - Tests adicionales de sliding window y structured_call sugeridos por Ceci.

    pytest tests/test_scenarios_m2.py -v
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest
from pydantic import Field

from mia_agents.testing import MockLLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import LLMResponse, ToolCall

from student_framework import build_agent
from student_framework.agent import MyAgent
from student_framework.tools.calculator import calculator
from student_framework.tools.file_reader import file_reader
from student_framework.tools.word_counter import word_counter


# ===========================================================================
# Punto 7 — Reintentos ante fallos transitorios
# ===========================================================================


def test_retry_on_transient_llm_error_succeeds() -> None:
    """Un timeout del cliente LLM se reintenta y la ejecución termina con éxito.

    El MockLLMClient lanza TimeoutError en la primera llamada y devuelve una
    respuesta válida en la segunda. El agente debe reintentar y completar.
    """
    mock = MockLLMClient(
        [
            TimeoutError("simulated timeout"),
            LLMResponse(content="respuesta tras reintento"),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("algo simple")

    assert result.answer == "respuesta tras reintento"
    assert mock.call_count == 2, (
        "el agente debería haber llamado al LLM dos veces: 1 fallida + 1 exitosa"
    )


def test_retry_on_connection_error_succeeds() -> None:
    """Un ConnectionError también es transitorio y debe reintentarse."""
    mock = MockLLMClient(
        [
            ConnectionError("connection refused"),
            LLMResponse(content="éxito tras reconexión"),
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.run("ping")

    assert result.answer == "éxito tras reconexión"
    assert mock.call_count == 2


def test_no_retry_on_non_transient_llm_error() -> None:
    """Un error definitivo (no transitorio) no debe reintentarse; se propaga de inmediato."""
    mock = MockLLMClient([ValueError("esquema de autenticación inválido")])
    agent = build_agent({"llm_client": mock})

    with pytest.raises(ValueError):
        agent.run("algo")

    assert mock.call_count == 1, "un error definitivo no debe generar reintentos"


def test_retry_exhaustion_re_raises() -> None:
    """Cuando se agotan todos los reintentos, la excepción transitoria se propaga."""
    mock = MockLLMClient(
        [TimeoutError("timeout")] * 4  # más intentos que _MAX_LLM_RETRIES
    )
    agent = build_agent({"llm_client": mock})

    with pytest.raises(TimeoutError):
        agent.run("algo")

    # 1 intento inicial + 3 reintentos = 4 llamadas al LLM.
    assert mock.call_count == 4


def test_permission_error_is_not_retried() -> None:
    """PermissionError hereda de OSError pero es definitivo: no debe reintentarse."""
    from student_framework.agent import _is_transient_error

    assert not _is_transient_error(PermissionError("acceso denegado"))
    assert not _is_transient_error(FileNotFoundError("no existe"))
    assert not _is_transient_error(IsADirectoryError("es un directorio"))
    # OSError genérico sí es transitorio
    assert _is_transient_error(OSError("connection reset"))


def test_tool_retries_on_transient_exception() -> None:
    """Una tool que lanza TimeoutError debe reintentarse antes de devolver error."""
    from mia_agents.types import ToolSchema

    call_count = 0

    def flaky_tool(
        text: Annotated[str, Field(description="texto de prueba")],
    ) -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise TimeoutError("timeout simulado")
        return "resultado tras reintento"

    schema = ToolSchema.from_callable(flaky_tool)

    mock = MockLLMClient([
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=schema.name, arguments=json.dumps({"text": "x"}))],
        ),
        LLMResponse(content="listo"),
    ])
    agent = build_agent({"llm_client": mock})
    agent.register_tool(flaky_tool, schema)

    result = agent.run("usá la tool")

    assert call_count == 2, "la tool debería haberse llamado dos veces: 1 fallida + 1 exitosa"
    assert result.steps[0].error is None
    assert result.steps[0].tool_output == "resultado tras reintento"


# ===========================================================================
# Punto 6a — Errores recuperables: calculadora
# ===========================================================================


def test_calculator_non_numeric_left_operand() -> None:
    """Un left_operand no numérico devuelve un mensaje accionable."""
    result = calculator(left_operand="cuarenta", right_operand=2, operator="+")

    assert "left_operand" in result
    assert "cuarenta" in result
    assert result.startswith("Error")


def test_calculator_non_numeric_right_operand() -> None:
    """Un right_operand no numérico devuelve un mensaje accionable."""
    result = calculator(left_operand=10, right_operand="dos", operator="*")

    assert "right_operand" in result
    assert "dos" in result
    assert result.startswith("Error")


def test_calculator_unsupported_operator_lists_valid_ones() -> None:
    """Un operador no soportado devuelve los operadores permitidos."""
    result = calculator(left_operand=4, right_operand=2, operator="/")

    assert result.startswith("Error")
    # El mensaje debe listar los operadores válidos.
    for op in ("+", "-", "*", "%"):
        assert op in result, f"operador {op!r} debería aparecer en el mensaje de error"


def test_calculator_modulo_by_zero_is_actionable() -> None:
    """Módulo por cero devuelve un mensaje que explica la restricción."""
    result = calculator(left_operand=10, right_operand=0, operator="%")

    assert result.startswith("Error")
    assert "cero" in result.lower()
    # No debe ser un mensaje genérico de Python como ZeroDivisionError.
    assert "ZeroDivisionError" not in result


def test_calculator_happy_path_unchanged() -> None:
    """Las operaciones válidas siguen funcionando correctamente."""
    assert calculator(3, 4, "+") == "7"
    assert calculator(10, 3, "-") == "7"
    assert calculator(6, 7, "*") == "42"
    assert calculator(10, 3, "%") == "1"
    # Resultado decimal cuando corresponde.
    assert calculator(7, 2, "%") == "1"
    assert calculator(10.5, 3, "%") == "1.5"


# ===========================================================================
# Punto 6b — Errores recuperables: file_reader
# ===========================================================================


def test_file_reader_empty_path_is_actionable() -> None:
    """Una ruta vacía devuelve un mensaje que explica cómo debe ser una ruta válida."""
    result = file_reader("")

    assert result.startswith("Error")
    assert "vacía" in result or "vacia" in result.lower()
    # Debe sugerir cómo construir una ruta válida.
    assert "sample_files" in result


def test_file_reader_absolute_path_is_actionable() -> None:
    """Una ruta absoluta devuelve un mensaje que explica la regla."""
    result = file_reader("/etc/passwd")

    assert result.startswith("Error")
    assert "absoluta" in result
    assert "relativa" in result


def test_file_reader_dotdot_traversal_is_actionable() -> None:
    """Una ruta con '..' devuelve un mensaje que explica la restricción."""
    result = file_reader("../secreto.txt")

    assert result.startswith("Error")
    assert ".." in result


def test_file_reader_nonexistent_file_lists_available() -> None:
    """Archivo inexistente: si el directorio contenedor existe, lista los archivos disponibles."""
    result = file_reader("archivo_que_no_existe_xyz_abc.txt")

    assert result.startswith("Error")
    # El mensaje debe listar archivos disponibles para que el LLM pueda corregir.
    assert "disponibles" in result or "disponible" in result


def test_file_reader_directory_lists_contents() -> None:
    """Cuando la ruta apunta a un directorio, se indica y se lista el contenido."""
    # "." resuelve a _ALLOWED_DIR, que es sample_files/ (un directorio válido dentro del sandbox).
    result = file_reader(".")

    assert result.startswith("Error")
    assert "directorio" in result


def test_file_reader_valid_file_reads_correctly() -> None:
    """El happy path sigue funcionando: un archivo válido se lee sin error."""
    result = file_reader("hola_mundo.md")

    # Debe devolver contenido, no un mensaje de error.
    assert not result.startswith("Error")
    assert len(result) > 0


# ===========================================================================
# Tests adicionales de sliding window (sugeridos por Ceci)
# ===========================================================================


def test_latest_user_message_survives_sliding_window() -> None:
    """Con un presupuesto muy chico, el último mensaje del usuario siempre llega al LLM."""
    mock = MockLLMClient(
        [LLMResponse(content=f"respuesta {i}") for i in range(10)]
    )
    # max_history_messages=2 es muy pequeño: solo caben 2 mensajes.
    agent = build_agent({"llm_client": mock, "max_history_messages": 2})

    for i in range(10):
        agent.run(f"turno {i}: mensaje único #{i}")

    for i, call in enumerate(mock.calls):
        messages = call["messages"]
        assert len(messages) <= 2, f"llamada {i}: el presupuesto fue superado"
        roles = [m.get("role") for m in messages]
        assert "user" in roles, f"llamada {i}: el mensaje del usuario no está en la ventana"


def test_zero_history_limit_raises_before_calling_llm() -> None:
    """max_history_messages=0 debe levantar ValueError antes de llamar al LLM."""
    mock = MockLLMClient([LLMResponse(content="nunca debería llegar aquí")])
    agent = MyAgent(llm_client=mock, max_history_messages=0)

    with pytest.raises(ValueError):
        agent.run("algo")

    assert mock.call_count == 0, "el LLM no debe recibir ninguna llamada con presupuesto 0"


# ===========================================================================
# Test adicional de structured_call (sugerido por Ceci)
# ===========================================================================


def test_structured_call_repair_includes_previous_error_in_messages() -> None:
    """En el segundo intento de reparación, los mensajes incluyen la respuesta fallida anterior."""
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    def _final_result_response(arguments: dict, *, call_id: str = "fr-1") -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(id=call_id, name=FINAL_RESULT_TOOL_NAME, arguments=json.dumps(arguments))
            ],
        )

    mock = MockLLMClient(
        [
            _final_result_response({"value": "no_es_un_int"}),       # falla: tipo incorrecto
            _final_result_response({"value": 42}, call_id="fr-2"),   # éxito
        ]
    )
    agent = build_agent({"llm_client": mock})

    result = agent.structured_call(prompt="dame un entero", schema=Answer)

    assert isinstance(result, Answer)
    assert result.value == 42
    assert mock.call_count == 2

    # Los mensajes de la segunda llamada deben contener la respuesta fallida anterior.
    second_call_messages = str(mock.calls[1]["messages"])
    assert "no_es_un_int" in second_call_messages or FINAL_RESULT_TOOL_NAME in second_call_messages, (
        "la segunda llamada debe incluir el contexto del error anterior para guiar la reparación"
    )


# ===========================================================================
# Punto 6c — Errores recuperables: word_counter
# ===========================================================================


def test_word_counter_none_input_is_actionable() -> None:
    """text=None devuelve un mensaje accionable en lugar de crashear."""
    result = word_counter(None)

    assert result.startswith("Error")
    assert "text" in result
    assert "nulo" in result or "None" in result


def test_word_counter_non_string_input_is_actionable() -> None:
    """text con tipo incorrecto (ej: número) devuelve un mensaje accionable."""
    result = word_counter(42)

    assert result.startswith("Error")
    assert "text" in result
    # Debe mencionar el tipo o el valor recibido para que el LLM pueda corregir.
    assert "int" in result or "42" in result


def test_word_counter_happy_path_unchanged() -> None:
    """El happy path sigue funcionando correctamente."""
    assert word_counter("hola mundo") == "2"
    assert word_counter("") == "0"
    assert word_counter("una sola") == "2"


# ===========================================================================
# Tests adicionales de tokens (sugeridos por Ceci, excluyendo el caso None)
# ===========================================================================


def test_input_only_tokens_treats_output_as_zero() -> None:
    """Si el proveedor solo reporta input_tokens, output_tokens debe ser 0."""
    mock = MockLLMClient([
        LLMResponse(content="respuesta", input_tokens=100),
    ])
    agent = build_agent({"llm_client": mock})
    result = agent.run("algo")

    assert result.input_tokens == 100
    assert result.output_tokens == 0


def test_explicit_zero_tokens_are_not_none() -> None:
    """El proveedor reporta explícitamente 0 y 0: el resultado debe ser 0, no None."""
    mock = MockLLMClient([
        LLMResponse(content="respuesta", input_tokens=0, output_tokens=0),
    ])
    agent = build_agent({"llm_client": mock})
    result = agent.run("algo")

    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.input_tokens is not None
    assert result.output_tokens is not None


def test_tokens_reset_for_each_run() -> None:
    """Cada run() reporta solo los tokens de su propio turno; no se acumulan entre runs."""
    mock = MockLLMClient([
        LLMResponse(content="primera respuesta", input_tokens=100, output_tokens=50),
        LLMResponse(content="segunda respuesta", input_tokens=200, output_tokens=30),
    ])
    agent = build_agent({"llm_client": mock})

    result1 = agent.run("primer turno")
    result2 = agent.run("segundo turno")

    assert result1.input_tokens == 100
    assert result1.output_tokens == 50
    assert result2.input_tokens == 200, (
        f"esperado 200, obtuvo {result2.input_tokens!r} — los tokens no deben acumularse entre runs"
    )
    assert result2.output_tokens == 30


def test_missing_tokens_are_none_when_provider_does_not_report_usage() -> None:
    """Sin reporte del proveedor, los tokens quedan como dato no disponible."""
    mock = MockLLMClient([
        LLMResponse(content="respuesta sin métricas de tokens"),
    ])
    agent = build_agent({"llm_client": mock})

    result = agent.run("algo")

    assert result.input_tokens is None
    assert result.output_tokens is None


# ===========================================================================
# Test adicional de historial con tool interaction (sugerido por Ceci)
# ===========================================================================


def test_history_persists_tool_interaction() -> None:
    """Después de un run con tool, el turno siguiente conserva el contexto completo.

    Verifica que en la segunda llamada al LLM (segundo run) el historial incluya
    tanto el mensaje del assistant con tool_calls como el resultado de la tool
    (role='tool'), para que el LLM tenga contexto completo de lo que pasó.
    """
    from mia_agents.testing import make_recording_tool

    tool, schema = make_recording_tool(return_value="resultado_de_la_tool")
    mock = MockLLMClient([
        # Primer run — turno 1: LLM pide la tool.
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name=schema.name, arguments=json.dumps({"text": "x"}))],
        ),
        # Primer run — turno 2: LLM da respuesta final.
        LLMResponse(content="listo, usé la tool"),
        # Segundo run: respuesta directa.
        LLMResponse(content="respuesta del segundo turno"),
    ])
    agent = build_agent({"llm_client": mock})
    agent.register_tool(tool, schema)

    agent.run("primer turno: usá la tool")
    agent.run("segundo turno: ¿qué hiciste antes?")

    # La tercera llamada al LLM (primer call del segundo run) debe ver en su
    # historial el resultado de la tool del turno anterior.
    third_call_messages = str(mock.calls[2]["messages"])
    assert "resultado_de_la_tool" in third_call_messages, (
        "el resultado de la tool del run anterior debe estar en el historial del siguiente run"
    )
