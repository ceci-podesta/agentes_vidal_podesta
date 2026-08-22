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
from dataclasses import dataclass
from typing import Any, Callable
from pydantic import ValidationError


from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, AgentStep, LLMResponse, ToolCall, ToolSchema
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME, final_result_tool_schema

from .m3_scratchpad import M3Scratchpad


_MAX_LLM_RETRIES = 3
_TRANSIENT_KEYWORDS = ("timeout", "timed out", "rate limit", "throttl", "503", "502", "504")


@dataclass
class StructuredCallUsage:
    """Uso de LLM consumido por una llamada estructurada."""

    input_tokens: int | None
    output_tokens: int | None
    attempts: int


@dataclass
class StructuredCallResult:
    """Valor Pydantic validado y métricas de su obtención."""

    value: Any
    usage: StructuredCallUsage


class StructuredCallError(ValueError):
    """Error de salida estructurada que conserva el consumo realizado."""

    def __init__(self, message: str, usage: StructuredCallUsage) -> None:
        super().__init__(message)
        self.usage = usage


def _is_transient_error(exc: Exception) -> bool:
    """True cuando la excepción es probablemente transitoria y vale la pena reintentar."""
    # Subtipos de OSError que son definitivos: excluirlos antes del check general,
    # ya que PermissionError, FileNotFoundError e IsADirectoryError heredan de OSError
    # pero no se resuelven reintentando.
    if isinstance(exc, (PermissionError, FileNotFoundError, IsADirectoryError)):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)



class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
        max_repeated_failures: int | None = None,
        max_repeated_observations: int | None = None,
        observation_tool_names: set[str] | None = None,
        use_m3_scratchpad: bool = False,
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
        for option_name, option_value in (
            ("max_repeated_failures", max_repeated_failures),
            ("max_repeated_observations", max_repeated_observations),
        ):
            if option_value is not None and option_value < 1:
                raise ValueError(f"{option_name} debe ser mayor que cero.")

        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._max_repeated_failures = max_repeated_failures
        self._max_repeated_observations = max_repeated_observations
        self._observation_tool_names = set(observation_tool_names or set())
        self._use_m3_scratchpad = use_m3_scratchpad
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        self._history: list[dict[str, Any]] = []#M2: Este atributo pertenece a la instancia y sobrevive entre llamadas sucesivas a run()
                                                #Es donde vamos a almacenar el historial de la conversación

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


    def _chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSchema] | None,
        system: str | None,
    ) -> LLMResponse:
        """Llama a self._llm.chat() reintentando ante errores transitorios.

        Errores transitorios (timeout, rate limit, 5xx de red) se reintentan
        hasta _MAX_LLM_RETRIES veces. Errores definitivos (argumentos inválidos,
        autenticación, etc.) se propagan inmediatamente sin reintentar.
        """
        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                return self._llm.chat(messages=messages, tools=tools, system=system)
            except Exception as exc:
                is_last_attempt = attempt >= _MAX_LLM_RETRIES
                if is_last_attempt or not _is_transient_error(exc):
                    raise
        # Nunca se alcanza; satisface al type checker.
        raise RuntimeError("unreachable")

    def _build_sliding_window(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Devuelve una vista acotada del historial para enviar al LLM.

        Conserva los mensajes recientes. Si el ultimo mensaje del user quedo
        fuera de esa ventana, lo incorpora y usa los lugares restantes para
        los mensajes mas recientes.
        """
        limit = self._max_history_messages

        #M2 exige que el ultimo mensaje del user esté en la ventana.
        #Con max_history_messages definido eso seria imposible, por lo que levantamos un error. (Caso de borde)
        if limit < 1:
            raise ValueError("max_history_messages debe ser mayor que cero.")

        # Si todo el historial entra, devolvemos una copia sin recortar.
        if len(messages) <= limit:
            return list(messages)

        # Primera aproximacion: los ultimos N mensajes, por recencia.
        window = messages[-limit:]

        # Buscamos desde el final el ultimo mensaje emitido por el usuario.
        # run() siempre agrega un user_message antes de llamar a este metodo.
        latest_user = next(
            message
            for message in reversed(messages)
            if message.get("role") == "user"
        )

        # Si el ultimo mensaje del user ya esta en la ventana reciente, no hace falta modificarla.
        if latest_user in window:
            return list(window)

        #Si el ultimo mensaje del user no esta en la ventana, lo agregamos y completamos con los mensajes 
        #mas recientes hasta llegar al limite.
        #Caso de borde: Si limit (= max_history_messages) es 1, no tomamos mensajes adicionales al último del user
        #Y como messages[-0:] devolveria toda la lista, trabajamos con la condición [] para limit = 1
        recent_messages = messages[-(limit - 1):] if limit > 1 else []

        # La ventana final conserva el ultimo mensaje del user y completa el "presupuesto" con los mensajes mas recientes.
        return [latest_user, *recent_messages]




    @staticmethod
    def _tool_call_signature(tool_call: ToolCall) -> tuple[str, str]:
        """Devuelve una firma estable para comparar llamadas equivalentes."""
        raw_arguments = tool_call.arguments or ""
        try:
            parsed_arguments = json.loads(raw_arguments) if raw_arguments else {}
        except json.JSONDecodeError:
            return tool_call.name, raw_arguments

        if not isinstance(parsed_arguments, dict):
            return tool_call.name, raw_arguments

        normalized_arguments = json.dumps(
            parsed_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return tool_call.name, normalized_arguments

    @staticmethod
    def _is_failed_tool_step(step: AgentStep) -> bool:
        """Reconoce errores de ejecución y el formato textual de M3."""
        return step.error is not None or (
            step.tool_output or ""
        ).startswith("Error:")

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

        #M2
        #list(...) crea una lista nueva en la que guarda los elementos del "historial persistente" entre runs (que vive en self._history).
        #Al armarlo así, estamos creando una copia del historial previo y no editandolo directamente.
        messages = list(self._history)
        
        #El mensaje nuevo (user_message) se agrega despues del historial anterior.
        messages.append({"role": "user", "content": user_message})

        # Como en M1, steps y los contadores siguen perteneciendo solamente al run actual.
        steps: list[AgentStep] = []
        failed_call_counts: dict[tuple[str, str], int] = {}
        observation_counts: dict[tuple[str, str], int] = {}
        scratchpad = M3Scratchpad() if self._use_m3_scratchpad else None


        total_input_tokens = 0
        total_output_tokens = 0
        has_token_usage = False


        # Tope de llamadas al LLM: el bucle nunca llama a `chat` más de
        # `max_iterations` veces, evitando bucles infinitos.
        for _ in range(self._max_iterations):
            
            messages_for_llm = self._build_sliding_window(messages)

            system_for_llm = self._system
            if scratchpad is not None:
                system_for_llm = f"{self._system}\n\n{scratchpad.render()}"

            response = self._chat_with_retry(
                messages=messages_for_llm,
                tools=list(self._schemas.values()) if self._schemas else None,
                system=system_for_llm,
            )

            #Acumular tokens reportados por el proveedor.
            #Si el proveedor informa al menos uno de los dos valores, 
            #los campos ausentes de otras respuestas cuentan 0.

            if (
                response.input_tokens is not None
                or response.output_tokens is not None
            ):
                has_token_usage = True


            total_input_tokens += response.input_tokens or 0
            total_output_tokens += response.output_tokens or 0


            # Condición de parada (M1): el LLM devuelve texto sin tool_calls.
            #if not response.tool_calls:
            #    return AgentResult(
            #        answer=response.content or "",
            #        steps=steps,
            #        input_tokens=total_input_tokens,
            #        output_tokens=total_output_tokens,
            #    )

            if not response.tool_calls:
                #Es decir, si la respuesta del LLM (response) no contiene pedidos para ejecutar tools,
                #sabemos que response.content es la respuesta final y la appendeamos a messages con role "assistant"

                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content or "",
                    }
                )

                #Llegado este punto, actualizamos el contenido de self._history con el historial (messages)
                #respetando max_history_messages.
                #es decir, la info que traía inicialmente más la que se sumó de esta nueva llamada a run()
                self._history = messages

                return AgentResult(
                    answer=response.content or "",
                    steps=steps,
                    # M2: 0 significa "el proveedor reportó cero"; None,
                    # "el proveedor no informó uso de tokens".
                    input_tokens=(
                        total_input_tokens if has_token_usage else None
                    ),
                    output_tokens=(
                        total_output_tokens if has_token_usage else None
                    ),
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
                signature = self._tool_call_signature(tool_call)
                is_observation = (
                    tool_call.name in self._observation_tool_names
                )

                failure_count = failed_call_counts.get(signature, 0)
                observation_count = observation_counts.get(signature, 0)

                if (
                    self._max_repeated_failures is not None
                    and failure_count >= self._max_repeated_failures
                ):
                    error = (
                        "La llamada ya falló sin que hubiera progreso. "
                        "Elegí una acción diferente."
                    )
                    tool_output = f"Error: {error}"
                    step = AgentStep(
                        tool_call.name,
                        tool_call.arguments,
                        tool_output,
                        error=error,
                    )
                elif (
                    is_observation
                    and self._max_repeated_observations is not None
                    and observation_count >= self._max_repeated_observations
                ):
                    error = (
                        "La observación ya fue realizada sin que hubiera "
                        "progreso. Elegí una acción diferente."
                    )
                    tool_output = f"Error: {error}"
                    step = AgentStep(
                        tool_call.name,
                        tool_call.arguments,
                        tool_output,
                        error=error,
                    )
                else:
                    step, tool_output = self._execute_tool_call(tool_call)

                    if self._is_failed_tool_step(step):
                        failed_call_counts[signature] = failure_count + 1
                    elif is_observation:
                        observation_counts[signature] = observation_count + 1
                    else:
                        failed_call_counts.clear()
                        observation_counts.clear()

                steps.append(step)
                if scratchpad is not None:
                    scratchpad.record(step)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )

        
        #Si se alcanza el limite `max_iterations` sin una respuesta final de texto (salió del for), 
        #conservamos la ventana disponoble
        
        self._history = self._build_sliding_window(messages)

        #Y Devolvemos un AgentResult válido, que indique la situación (corte por alcanzar el límite de iteraciones).
        return AgentResult(
            answer="",
            steps=steps,
            error=(
                f"Se alcanzó el máximo de iteraciones ({self._max_iterations}) "
                "sin que el modelo produjera una respuesta final."
            ),
            # M2: aplicamos la misma convención si el bucle terminó por
            # max_iterations en vez de terminar con una respuesta final.
            input_tokens=total_input_tokens if has_token_usage else None,
            output_tokens=total_output_tokens if has_token_usage else None,
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
            if not isinstance(kwargs, dict):
                raise ValueError("Los argumentos de la tool deben ser un objeto JSON.")
        except json.JSONDecodeError as exc:
            error = f"Argumentos JSON inválidos para '{name}': {exc}"
            return (
                AgentStep(name, raw_arguments, None, error=error),
                error,
            )
        except ValueError as exc:
            error = f"Argumentos inválidos para '{name}': {exc}"
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

        # 3. Ejecutar el callable con retry ante errores transitorios.
        for attempt in range(_MAX_LLM_RETRIES + 1):
            try:
                output = tool(**kwargs)
                break
            except Exception as exc:
                is_last_attempt = attempt >= _MAX_LLM_RETRIES
                if is_last_attempt or not _is_transient_error(exc):
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
        """Mantiene el contrato M2: devuelve solamente el valor validado."""
        return self.structured_call_with_usage(
            prompt=prompt,
            schema=schema,
            max_repair_attempts=max_repair_attempts,
        ).value

    def structured_call_with_usage(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> StructuredCallResult:
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
        
        final_result_schema = final_result_tool_schema(schema)


        # "Conversacion" interna a structured_call (No guarda relación con self._history y self._max_history_messages).
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]


        # Guarda el ultimo problema para informar una reparacion o el fallo definitivo tras agotar los intentos.
        #(se inicializa en None)
        last_error: str | None = None

        # M3: este metodo tiene su propio loop de llamadas al LLM. Igual que
        # run(), conserva None si el proveedor no informa uso en ningun intento.
        total_input_tokens = 0
        total_output_tokens = 0
        has_token_usage = False

        # Intento inicial + cantidad maxima de reparaciones permitidas.
        for attempt in range(max_repair_attempts + 1):
            response = self._chat_with_retry(
                messages=messages,
                # En cada intento ofrecemos solamente final_result (pero messages va guardando el historial).
                tools=[final_result_schema],
                system=self._system,
            )

            if (
                response.input_tokens is not None
                or response.output_tokens is not None
            ):
                has_token_usage = True

            total_input_tokens += response.input_tokens or 0
            total_output_tokens += response.output_tokens or 0

            # Buscar el tool_call obligatorio que cierra la respuesta.
            final_call = next(
                (
                    tool_call
                    for tool_call in response.tool_calls
                    if tool_call.name == FINAL_RESULT_TOOL_NAME
                ),
                None,
            )


            if final_call is None:
                # Texto libre o una tool distinta: no es una salida valida.
                last_error = (
                    "El modelo no invocó la tool final_result. "
                    "Debe usarla para devolver la respuesta."
                )
            else:
                try:
                    # Pasar de la cadena JSON emitida por el LLM a dict Python.
                    arguments = json.loads(final_call.arguments or "{}")


                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Los argumentos de final_result deben ser un "
                            "objeto JSON."
                        )


                    # Si valida, devolvemos la instancia Pydantic y las
                    # metricas acumuladas en los intentos de esta llamada.
                    return StructuredCallResult(
                        value=schema.model_validate(arguments),
                        usage=StructuredCallUsage(
                            input_tokens=(
                                total_input_tokens
                                if has_token_usage
                                else None
                            ),
                            output_tokens=(
                                total_output_tokens
                                if has_token_usage
                                else None
                            ),
                            attempts=attempt + 1,
                        ),
                    )
                except (
                    json.JSONDecodeError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    # JSON invalido, estructura incorrecta o tipos Pydantic
                    # invalidos: se guarda el detalle para reparar.
                    last_error = str(exc)


            # No se genera otro intento despues de usar la ultima oportunidad.
            if attempt == max_repair_attempts:
                break


            # Conservar la respuesta defectuosa antes del pedido de reparacion.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                            },
                        }
                        for tool_call in response.tool_calls
                    ],
                }
            )


            # Instruccion temporal para que el LLM corrija el formato.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "La respuesta anterior no cumplió el formato requerido. "
                        f"Error: {last_error}. "
                        "Corrige la respuesta invocando la tool final_result."
                    ),
                }
            )


        # Respuesta cuando se agotan intentos (asegura que no se devuelve None ni un objeto parcial).
        raise StructuredCallError(
            "No se pudo obtener una respuesta estructurada válida "
            f"después de {max_repair_attempts + 1} intentos. "
            f"Último error: {last_error}",
            usage=StructuredCallUsage(
                input_tokens=(
                    total_input_tokens if has_token_usage else None
                ),
                output_tokens=(
                    total_output_tokens if has_token_usage else None
                ),
                attempts=max_repair_attempts + 1,
            ),
        )
