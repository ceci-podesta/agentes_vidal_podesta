# Informe — Milestone 1

Framework mínimo de agentes (`mia_agents`). Implementación del grupo en
`student_framework/`.

---

## 1. Diagrama de arquitectura

```
                    ┌──────────────────────────────────────────────┐
                    │                  build_agent()                 │
                    │            (student_framework/__init__.py)     │
                    │  - crea MyAgent con el LLMClient inyectado     │
                    │  - registra las 3 herramientas                 │
                    └───────────────────────┬──────────────────────┘
                                            │
                              user_message  │  AgentResult
                                            ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │                            MyAgent.run()                              │
   │                        (bucle del agente, M1)                        │
   │                                                                       │
   │   messages = [{"role":"user", "content": user_message}]              │
   │   repetir (hasta max_iterations):                                     │
   │     1. resp = self._llm.chat(messages, tools=schemas, system)  ──────┼──┐
   │     2. ¿resp.tool_calls vacío?                                        │  │
   │          SÍ  -> return AgentResult(answer=resp.content, steps)        │  │
   │          NO  -> por cada tool_call:                                   │  │
   │                  - _execute_tool_call() -> AgentStep + salida         │  │
   │                  - append {"role":"tool", content: salida}            │  │
   └───────────────────┬─────────────────────────────────────────────────┘  │
                       │                                                       │
       register_tool   │                                          chat(...)   │
                       ▼                                                       ▼
        ┌──────────────────────────────┐                  ┌──────────────────────────────┐
        │   self._tools   (name->fn)    │                  │          LLMClient            │
        │   self._schemas (name->schema)│                  │   (FIJO, mia_agents)          │
        └──────────────┬───────────────┘                  │  to_llm_spec() por cada tool  │
                       │                                   │  Bedrock (Converse) / Ollama  │
                       │ tool(**kwargs)                    │  / MockLLMClient en tests     │
                       ▼                                   └──────────────────────────────┘
        ┌──────────────────────────────┐
        │  Herramientas (callables)     │
        │  - calculator                 │
        │  - file_reader                │
        │  - word_counter               │
        └──────────────────────────────┘
```

**Flujo en una frase:** `build_agent` arma el agente y le registra las
herramientas; `run` mantiene la conversación con el LLM a través del
`LLMClient`, y cuando el modelo pide una herramienta el agente la ejecuta
localmente y le devuelve el resultado, hasta que el modelo responde con
texto final.

---

## 2. Diseño de la interfaz de herramientas

### Definición de una herramienta

Cada herramienta es un **callable de Python** con tipos en la firma y un
docstring. Ejemplo (`student_framework/tools/word_counter.py`):

```python
def word_counter(
    text: Annotated[str, Field(description="El texto cuyas palabras se desean contar.")],
) -> str:
    """Cuenta la cantidad de palabras en un texto y devuelve el resultado."""
    return str(len(text.split()))

word_counter_schema = ToolSchema.from_callable(word_counter)
```

- El **docstring** se convierte en la `description` de la herramienta para el LLM.
- Los **tipos** + `Annotated[..., Field(description=...)]` definen el JSON
  Schema de los argumentos. **No escribimos `parameters={...}` a mano:**
  `ToolSchema.from_callable(fn)` lo deriva de la firma.

### Qué guarda `register_tool`

```python
def register_tool(self, tool, schema) -> None:
    self._tools[schema.name] = tool       # name -> callable a ejecutar
    self._schemas[schema.name] = schema   # name -> ToolSchema a exponer al LLM
```

Dos diccionarios indexados por `schema.name`: uno con el callable (para
**ejecutar**) y otro con el `ToolSchema` (para **describirle** la
herramienta al LLM). La clave compartida es lo que permite, cuando llega
un `tool_call`, encontrar el callable correcto por su nombre.

### Qué se pasa en `chat(tools=...)`

En cada llamada, `run` expone los esquemas registrados:

```python
resp = self._llm.chat(
    messages=messages,
    tools=list(self._schemas.values()) if self._schemas else None,
    system=self._system,
)
```

Se pasan los objetos `ToolSchema` directamente (no dicts).

### Qué hace el `LLMClient` fijo con cada esquema

El `LLMClient` (FIJO) traduce cada `ToolSchema` al formato nativo del
proveedor: llama `to_llm_spec()` (que devuelve `name`/`description`/
`parameters`) y lo envuelve según corresponda —
`{"toolSpec": {"name", "description", "inputSchema": {"json": ...}}}` para
Bedrock (API Converse) o `{"type": "function", "function": {...}}` para
Ollama. **El agente nunca sabe qué proveedor hay detrás**: solo depende
del método `chat(...)`. Por eso los tests pueden inyectar un
`MockLLMClient` y el código del agente funciona sin cambios (y por eso no
hay conflicto si una integrante usa Bedrock y la otra Ollama).

---

## 3. Cómo termina el bucle

El bucle de `run` tiene **dos condiciones de salida**:

1. **Respuesta final (caso normal).** Cuando el LLM devuelve texto **sin**
   `tool_calls`, ese `content` es la respuesta y se devuelve
   inmediatamente en `AgentResult.answer`. Si no hubo herramientas,
   `steps` queda vacío y el LLM se llamó una sola vez.

2. **Tope `max_iterations` (corte de seguridad).** El `for _ in
   range(self._max_iterations)` garantiza que **nunca** se llame a `chat`
   más de `max_iterations` veces (por defecto 10). Si el modelo se queda
   pidiendo herramientas indefinidamente, al agotar las iteraciones el
   bucle sale y `run` devuelve igualmente un `AgentResult` **válido** con
   `answer=""`, los `steps` acumulados y un `error` explicando que se
   alcanzó el límite. Así se evitan los bucles infinitos sin lanzar
   excepciones.

En cada vuelta en la que el modelo pide herramientas:
- Se agrega el turno del asistente (con sus `tool_calls`) al historial.
- Por cada `tool_call` se ejecuta el callable y se registra **un**
  `AgentStep` (`tool_name`, `tool_input`, `tool_output`, `error`).
- El resultado se vuelca como mensaje `role: "tool"`, de modo que aparece
  en la **siguiente** llamada a `chat` (realimentación al LLM).

**Robustez:** `_execute_tool_call` nunca lanza excepción. Ante argumentos
JSON inválidos, una herramienta inexistente (alucinación del modelo) o un
fallo del callable, devuelve un `AgentStep` con `error` no nulo y el bucle
continúa. `run` siempre devuelve un `AgentResult`.

---

## 4. Limitaciones conocidas

- **Sin estado entre llamadas (por diseño en M1).** Cada `run` arranca con
  un historial nuevo; no hay memoria conversacional multiturno. Eso llega
  en M2 (junto con `max_history_messages`, que en M1 se acepta pero se
  ignora).
- **`structured_call` no implementado.** Queda como stub
  (`NotImplementedError`); la salida estructurada con `final_result` y la
  reparación con reintentos son parte de M2.
- **Sin reintentos ante fallos transitorios del proveedor.** Si `chat`
  lanza una excepción (red, API), `run` no la captura: eso es alcance de
  M2.
- **`file_reader` lee el archivo completo en memoria.** No hay límite de
  tamaño ni streaming; para archivos muy grandes podría ser costoso. Solo
  soporta texto UTF-8.
- **La calculadora opera sobre dos operandos y un operador** (`+`, `-`,
  `*`, `%`). No evalúa expresiones arbitrarias (decisión deliberada: sin
  `eval`, por seguridad).
- **Validación de argumentos limitada.** Se parsea el JSON del `tool_call`
  y se invoca el callable con esos kwargs; no se validan los tipos contra
  el esquema antes de ejecutar (la validación fuerte llega con
  `structured_call` en M2).
