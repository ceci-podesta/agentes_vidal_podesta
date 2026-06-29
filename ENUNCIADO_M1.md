# Milestone 1 — Bucle del agente y herramientas

## Objetivo

Construyan un agente funcional capaz de invocar herramientas para
responder preguntas que no podría responder solo con su conocimiento.

## Alcance

En M1 trabajen sobre el núcleo del framework: cliente LLM, registro de
herramientas y bucle agente. No hace falta resolver memoria, salida
estructurada, reintentos sofisticados ni evaluación sobre el mundo de M3.

## Lo que deben construir

- **Uso del cliente LLM provisto.** Usen `mia_agents.llm_client.LLMClient`
  como punto de acceso al modelo. El bucle del agente debe depender del
  protocolo `chat(...)`
- **Una interfaz y un registro de herramientas.** Cada herramienta es un
  callable con tipos en la firma (`Annotated` + `Field` de Pydantic para
  describir cada argumento). El esquema para el LLM se genera con
  `ToolSchema.from_callable(fn)` — **no escriban JSON Schema a mano**. En
  `run`, pasen `tools=list(self._schemas.values())`; el `LLMClient` fijo
  traduce cada `ToolSchema` al formato de Bedrock/Ollama.
- **Tres herramientas obligatorias** (detalle abajo).
- **Un bucle del agente.** Dado un mensaje del usuario, el agente debe
  razonar, decidir si invoca una herramienta, ejecutarla, observar el
  resultado y continuar hasta producir una respuesta final. Debe terminar
  sin bucles infinitos (deben definir un máximo de pasos)

  **Condición de parada en M1:** el bucle termina cuando el LLM devuelve
  **texto sin `tool_calls`** (ese texto es `AgentResult.answer`) o cuando
  se excede el presupuesto maximo de pasos

## Herramientas obligatorias (M1)

Implementen **exactamente estas tres** (pueden elegir nombres de función
distintos si el `ToolSchema.name` sigue siendo claro para el LLM; la
referencia usa `calculator`, `file_reader` y una tercera libre).

### 1. Calculadora simple (cómputo puro)

- **Entrada:** dos operandos numéricos y un operador.
- **Operadores soportados:** `+`, `-`, `*`, `%` (módulo).
- **Salida:** el resultado de la operación, como `str`.
- Sin `eval`, sin expresiones arbitrarias: solo la operación binaria
  indicada.

### 2. Lector de archivos (E/S restringida)

- **Entrada:** una ruta a un archivo.
- **Comportamiento:** leer y **mostrar el contenido** del archivo (solo
  archivos de **texto**; codificación UTF-8 recomendada).


### 3. Herramienta libre (a su elección)

- Cualquier utilidad que demuestre el mismo patrón
  (`callable` + `ToolSchema.from_callable` + registro).
- Ejemplos: contador de palabras, conversor de unidades, búsqueda en un
  JSON local, etc.

## Patrón de herramientas (referencia)

### 1. Definir la herramienta (`student_framework/tools/`)

```python
from __future__ import annotations
from typing import Annotated
from pydantic import Field
from mia_agents.types import ToolSchema


def reverse_string(
    text: Annotated[str, Field(description="El texto a invertir.")],
) -> str:
    """Invierte la cadena indicada y devuelve el resultado."""
    return text[::-1]

reverse_string_schema = ToolSchema.from_callable(reverse_string)
```

- El **docstring** de la función es la descripción de la herramienta para el LLM.
- Los **tipos** y `Field(description=...)` definen el JSON Schema de argumentos.

### 2. Registrar en el agente

```python
def register_tool(self, tool: Callable[..., str], schema: ToolSchema) -> None:
```


### 3. Exponer al LLM en `run`

```python
resp = self._llm.chat(
    messages=messages,
    tools=list(self._schemas.values()) if self._schemas else None,
    system=self._system,
)
```

Si el LLM devuelve `tool_calls`, parseen `arguments` (JSON), ejecuten
`self._tools[nombre](**kwargs)` y agreguen un mensaje `role: "tool"` con
el resultado antes de la siguiente llamada a `chat`.

Ver también `student_framework/tools/example.py`

## Contrato que deben implementar

M1: Diseñen su implementación contra este contrato. La
corrección lo verifica de forma automática y determinista (inyectando un
cliente LLM programable vía `config["llm_client"]`, sin claves de API), por lo
que cualquier lógica que dependa de un proveedor concreto en lugar del cliente
inyectado no será evaluable.

### Creacion y tipo de retorno

- `build_agent(config)` existe y devuelve un objeto que satisface el
  protocolo `Agent` (`isinstance(agent, Agent)` es `True`).
- Si `config["llm_client"]` está presente, el agente **debe** usar ese
  cliente. No instancien un cliente propio cuando les pasan uno.
- `agent.run(user_message)` devuelve siempre un `AgentResult`.

### Respuesta sin herramientas (un solo turno)

Cuando el LLM responde con texto y **sin** `tool_calls`:

- `result.answer` es exactamente ese texto (`content` de la respuesta).
- `result.steps` es una lista vacía: no se registran pasos si no hubo
  llamadas a herramientas.
- Se invoca al cliente LLM **una sola vez**.

### Registro y exposición de herramientas

- `register_tool(callable, ToolSchema)` acepta esa firma exacta.
- En cada `run`, el agente pasa los esquemas al cliente vía
  `chat(tools=...)`. En la primera llamada, `tools` **no** puede ser
  `None` y el `ToolSchema.name` registrado debe aparecer en esa lista.

### Ejecución de una herramienta

Cuando el LLM emite un `tool_call`:

- El agente parsea `arguments` (JSON) y ejecuta la llamada con esos
  argumentos (p. ej. la tool recibe recibe `{"text": "hola"}`).
- Tras una respuesta final de texto, `result.answer` es ese texto.
- Se registra **exactamente un** `AgentStep` por cada herramienta
  invocada.
- El `tool_name` del paso coincide con el nombre del esquema.

### Realimentación del resultado al LLM

- Ejecutar una herramienta provoca una **segunda** llamada al cliente
  LLM: una con el `tool_call` y otra con la respuesta final.
- El valor devuelto por la herramienta debe aparecer en los `messages`
  de esa segunda llamada (el resultado se vuelca como contexto antes de
  volver a invocar `chat`).

### Campos del `AgentStep`

Para una herramienta que devuelve `"42"`:

- `step.tool_name` = nombre del esquema.
- `step.tool_output == "42"` (el valor exacto que devolvió el callable).
- `step.error is None` cuando la ejecución fue exitosa.

### Terminación (sin bucles infinitos)

- Ante una ejecucion fallida que loopea, el agente debe respetar `max_iterations` 
  y dejar de llamar al LLM cuando llega a esa cantidad de llamadas. El
  scaffold trae `max_iterations=10`; cualquier límite razonable sirve.
- Aun al cortar por límite, `run` devuelve un `AgentResult` válido.

### Herramienta desconocida (robustez)

- Si el LLM alucina un nombre de herramienta que no existe, el agente
  **no** debe romperse: `run` siempre devuelve un `AgentResult`.
- Esa invocación fallida queda registrada como un `AgentStep` con su
  campo `error` no nulo.

### Entradas básicas

- `run` no debe lanzar excepciones con mensajes como `"hola"`,
  `"¿cuánto es 2+2?"` ni con la cadena vacía `""`.

## Informe obligatorio

1. Diagrama de arquitectura (cajas y flechas).
2. Diseño de la interfaz de herramientas — `ToolSchema.from_callable`,
   `Annotated`/`Field`, qué guardan en `register_tool`, qué pasan en
   `chat(tools=...)` y qué hace el `LLMClient` fijo con cada esquema.
3. Cómo termina el bucle y qué pasa cuando se alcanza el limite de mensajes.
4. Limitaciones conocidas.
