"""Implementación de su agente.

Completen `register_tool` y `run` para el Milestone 1.
En el Milestone 2 amplíen `MyAgent` para que sea estatal y respete
`max_history_messages`.

Los tests de conformidad en `tests/conformance/test_m1.py` y
`test_m2.py` describen con precisión qué comportamientos deben funcionar
— léanlos antes de empezar.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, ToolCall, ToolSchema


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            System prompt por defecto.
        max_iterations : int
            Tope de iteraciones del bucle del agente (M1).
        max_history_messages : int
            Número máximo de mensajes que se permiten en la lista
            `messages` enviada al LLM en una única llamada. En M1 este
            valor es ignorado; el agente sólo necesita aceptarlo en su
            constructor. En M2 deben respetarlo: la longitud de la
            lista de mensajes pasada a `self._llm.chat(...)` no puede
            superar este número en ninguna llamada, sin importar la
            estrategia de memoria que elijan.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}

        # TODO (M1): inicializa el estado interno para las herramientas registradas.
        # TODO (M2): inicializa la estructura de historial conversacional.

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """Registra una herramienta callable junto a su esquema.

        El esquema suele obtenerse con `ToolSchema.from_callable(fn)`. En
        `run`, pasá `tools=list(self._schemas.values())`; el cliente LLM
        aplica `to_llm_spec()` al llamar al proveedor.

        El callable se invoca con kwargs que coinciden con la firma.
        Debe devolver una cadena.
        """
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema


    def run(self, user_message: str) -> AgentResult:
        """Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        Comportamiento esperado (consulta tests/conformance/test_m1.py
        para el contrato exacto del M1):
          - Llama a `self._llm.chat(..., tools=list(self._schemas.values()))`.
          - Si la respuesta contiene tool_calls, ejecuta cada uno y vuelca
            los resultados en la siguiente llamada al chat.
          - Si la respuesta solo contiene texto (sin `tool_calls`),
            devuélvelo en `AgentResult.answer`. En M1 no uses la tool
            sintética `final_result`; ese patrón es de M2 (ver README y
            ENUNCIADO_M2.md).
          - Limita el bucle a `self._max_iterations` y termina de forma
            limpia cuando se alcance.
          - Registra cada invocación de herramienta como un `AgentStep`
            dentro de `result.steps`.

        En el M2, además, llamadas sucesivas sobre la misma instancia
        deben continuar la conversación, y la longitud de la lista de
        mensajes enviada al LLM no debe superar `self._max_history_messages`.
        Acumula los tokens de entrada/salida reportados por los
        `LLMResponse` y exponlos en `AgentResult.input_tokens` /
        `AgentResult.output_tokens`.
        """
        # Historial de la conversación para esta llamada a `run`. En M1 cada
        # `run` es independiente (sin estado entre llamadas).
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        steps: list[AgentStep] = []
        total_input_tokens: int | None = None
        total_output_tokens: int | None = None

        # Tope de llamadas al LLM: el bucle nunca llama a `chat` más de
        # `max_iterations` veces, evitando bucles infinitos.
        for _ in range(self._max_iterations):
            response = self._llm.chat(
                messages=messages,
                tools=list(self._schemas.values()) if self._schemas else None,
                system=self._system,
            )

            # Acumular tokens reportados por el proveedor (None -> 0). Si
            # ninguna respuesta reporta tokens, el total queda en None.
            if response.input_tokens is not None:
                total_input_tokens = (total_input_tokens or 0) + response.input_tokens
            if response.output_tokens is not None:
                total_output_tokens = (total_output_tokens or 0) + response.output_tokens

            # Condición de parada (M1): el LLM devuelve texto sin tool_calls.
            if not response.tool_calls:
                return AgentResult(
                    answer=response.content or "",
                    steps=steps,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )

            # El modelo pidió herramientas: registramos su turno (incluidos
            # los tool_calls) en el historial antes de ejecutarlas.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            # Ejecutar cada herramienta y volcar su resultado al historial,
            # de modo que aparezca en la siguiente llamada a `chat`.
            for tool_call in response.tool_calls:
                step, tool_output = self._execute_tool_call(tool_call)
                steps.append(step)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )

        # Se alcanzó `max_iterations` sin una respuesta final de texto.
        # Devolvemos igualmente un AgentResult válido, registrando el corte.
        return AgentResult(
            answer="",
            steps=steps,
            error=(
                f"Se alcanzó el máximo de iteraciones ({self._max_iterations}) "
                "sin que el modelo produjera una respuesta final."
            ),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        )

    def _execute_tool_call(self, tool_call: ToolCall) -> tuple[AgentStep, str]:
        """Ejecuta un único `tool_call` y devuelve su `AgentStep` y la salida.

        Robustez (contrato M1): nunca lanza excepción. Ante argumentos JSON
        inválidos, herramienta inexistente o fallo del callable, devuelve un
        `AgentStep` con `error` no nulo. La cadena devuelta es lo que se
        vuelca como mensaje `role: "tool"` para el LLM.
        """
        name = tool_call.name
        raw_arguments = tool_call.arguments or ""

        # 1. Parsear los argumentos JSON emitidos por el LLM.
        try:
            kwargs = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError as exc:
            error = f"Argumentos JSON inválidos para '{name}': {exc}"
            return (
                AgentStep(name, raw_arguments, None, error=error),
                error,
            )

        # 2. Buscar la herramienta registrada (robustez ante alucinaciones).
        tool = self._tools.get(name)
        if tool is None:
            error = f"Herramienta desconocida: '{name}'."
            return (
                AgentStep(name, raw_arguments, None, error=error),
                error,
            )

        # 3. Ejecutar el callable con los kwargs parseados.
        try:
            output = tool(**kwargs)
        except Exception as exc:  # noqa: BLE001 - cualquier fallo se reporta
            error = f"Error al ejecutar '{name}': {exc}"
            return (
                AgentStep(name, raw_arguments, None, error=error),
                error,
            )

        output_str = output if isinstance(output, str) else str(output)
        return (
            AgentStep(name, raw_arguments, output_str, error=None),
            output_str,
        )

    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema` (M2).

        Obligatorio: herramienta sintética `final_result` (ver
        `mia_agents.final_result_tool_schema` / `FINAL_RESULT_TOOL_NAME`).
        El agente ofrece esa tool al LLM, valida los `arguments` del
        `tool_call` y reintenta con contexto de reparación si el modelo
        responde con texto libre o con argumentos inválidos.

        Implementa esto en el M2:
          - Pasa `tools=[final_result_tool_schema(schema)]` en cada
            llamada a `chat` dentro de este método.
          - Termina solo cuando llega un `tool_call` a `final_result`
            cuyos argumentos validan con `schema.model_validate(...)`.
          - Reintenta hasta `max_repair_attempts` incluyendo el fallo en
            los mensajes (respuesta previa, mensaje `tool`, o user de
            reparación).
          - Si tras los reintentos sigue fallando, levanta una excepción
            limpia (no devuelvas valores parciales ni `None` sin avisar).

        El M1 deja esto como stub; los tests de M2 verifican el contrato.
        """
        raise NotImplementedError("M2: implementa salida estructurada con reparación")
