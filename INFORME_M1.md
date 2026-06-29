# Informe M1 — Borrador Validado

## 1. Alcance del Milestone 1

En este milestone implementamos el núcleo de un agente con herramientas:

- construcción del agente desde `build_agent(config)`;
- inyección de un cliente LLM real o mockeado;
- registro de herramientas mediante `register_tool`;
- exposición de herramientas al LLM mediante `ToolSchema`;
- ejecución local de herramientas solicitadas por el LLM;
- realimentación del resultado al modelo;
- corte seguro por `max_iterations`.

El alcance de M1 no incluye memoria persistente entre llamadas, salida estructurada con `final_result`, reintentos ante fallos transitorios del proveedor ni evaluación sobre el mundo simulado de M3.

## 2. Diagrama de arquitectura

```text
┌────────────────────────────────────────────┐
│ **Constructor del agente**                 │
│ Crea una instancia de MyAgent y registra   │
│ las tools disponibles.                     │
│                                            │
│ build_agent(config)                        │
│ (student_framework/__init__.py)            │
└─────────────────────┬──────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────────┐
│ **Agente**                                                    │
│ Coordina el registro de tools y ejecuta                       │
│ el bucle principal del M1.                                    │
│                                                               │
│ MyAgent                                                       │
│ (student_framework/agent.py)                                  │
└───────────────┬──────────────────────────────────────┬────────┘
                │                                      │
                │                                      │
                ▼                                      ▼
┌────────────────────────────────┐   ┌────────────────────────────────┐
│ **Registro de schemas**        │   │ **Registro de tools**          │
│ Guarda las descripciones que   │   │ Guarda las tools registradas   │
│ se exponen al LLM.             │   │ para ejecutarlas por nombre.   │
│                                │   │                                │
│ self._schemas                  │   │ self._tools                    │
│ (student_framework/agent.py)   │   │ (student_framework/agent.py)   │
└───────────────┬────────────────┘   └───────────────┬────────────────┘
                │                                    │
                ▼                                    ▼
┌────────────────────────────────┐   ┌────────────────────────────────┐
│ **Cliente LLM**                │   │ **Tools**                      │
│ Recibe mensajes, schemas y     │   │ Código de las herramientas     │
│ system prompt; adapta el       │   │ disponibles.                   │
│ pedido al proveedor elegido.   │   │                                │
│ (Ollama / Bedrock /            │   │                                │
│   MockLLMClient)               │   │                                │
│                                │   │ calculator / file_reader /     │
│ LLMClient.chat(...)            │   │ word_counter                   │
│ (mia_agents/llm_client.py)     │   │ (student_framework/tools/*)    │
└────────────────────────────────┘   └────────────────────────────────┘
     
```

El esquema separa dos caminos que el agente necesita coordinar para poder usar tools.

Por un lado está el camino de los **schemas**. Cuando se registran las tools, el agente guarda en `self._schemas` las descripciones estructuradas de cada herramienta: nombre, descripción y parámetros esperados. Esos schemas se envían al cliente LLM en `chat(tools=...)` para que el modelo conozca qué tools existen y cómo debe pedir su ejecución.

Por otro lado está el camino de las **tools ejecutables**. El agente también guarda en `self._tools` las funciones Python reales asociadas a cada nombre de tool. Cuando el LLM responde con un `tool_call`, el agente usa el nombre de esa llamada para buscar la función correspondiente en `self._tools` y ejecutarla con los argumentos recibidos.

La idea central es que el LLM no ejecuta código directamente. El LLM solo ve los schemas y, a partir de ellos, puede solicitar una tool. El agente recibe esa solicitud, ejecuta la función Python real y luego devuelve el resultado al LLM para que continúe el bucle o produzca una respuesta final.


## 3. Diseño de la interfaz de herramientas

En M1, una herramienta se modela como una función Python común, pero con una firma suficientemente descriptiva para que el framework pueda exponerla al LLM como una tool invocable.

### 3.1 Herramientas como funciones Python

Cada herramienta vive en `student_framework/tools/` y debe ser un callable que devuelva `str`.

Ejemplo simplificado (`word_counter`):

```python
def word_counter(
    text: Annotated[str, Field(description="El texto cuyas palabras se desean contar.")],
) -> str:
    """Cuenta la cantidad de palabras en un texto y devuelve el resultado."""
    return str(len(text.split()))
```

La función es la implementación real que ejecuta el agente cuando el LLM solicita esa herramienta.

### 3.2 Uso de `Annotated` y `Field`

Los parámetros usan type hints y `Annotated[..., Field(description=...)]`.

```python
text: Annotated[
    str,
    Field(description="El texto cuyas palabras se desean contar."),
]
```

Esto cumple dos objetivos:

- indicar el tipo esperado del argumento;
- describir el significado del argumento para el LLM.

Estas descripciones influyen en la capacidad del modelo para elegir la herramienta correcta y construir argumentos válidos.

### 3.3 Generación del schema con `ToolSchema.from_callable`

Cada herramienta define también su schema:

```python
word_counter_schema = ToolSchema.from_callable(word_counter)
```

`ToolSchema.from_callable` inspecciona la función y deriva:

- `name`: el nombre de la función;
- `description`: el docstring;
- `parameters`: un JSON Schema generado a partir de la firma, los tipos y los `Field`.

De esta forma no escribimos JSON Schema manualmente. El contrato de la herramienta queda derivado del código Python.

### 3.4 Registro de herramientas

El agente mantiene dos diccionarios internos:

```python
def register_tool(self, tool, schema) -> None:
    self._tools[schema.name] = tool
    self._schemas[schema.name] = schema
```

`self._tools` guarda las funciones ejecutables:

```text
"calculator" -> calculator
"file_reader" -> file_reader
"word_counter" -> word_counter
```

`self._schemas` guarda los schemas que se le muestran al LLM:

```text
"calculator" -> calculator_schema
"file_reader" -> file_reader_schema
"word_counter" -> word_counter_schema
```

Ambos diccionarios usan `schema.name` como clave. Cuando el LLM devuelve un `tool_call` con un nombre, el agente usa esa clave para encontrar el callable correspondiente.

### 3.5 Exposición de herramientas al LLM

En cada vuelta del loop, `run` llama al cliente LLM pasando los schemas registrados:

```python
response = self._llm.chat(
    messages=messages,
    tools=list(self._schemas.values()),
    system=self._system,
)
```

El agente no le pasa funciones Python al LLM. Le pasa descripciones estructuradas (`ToolSchema`) para que el modelo sepa qué herramientas existen, cuándo usarlas y qué argumentos debe emitir.

### 3.6 Traducción del schema en `LLMClient`

`LLMClient` pertenece al framework fijo (`mia_agents`) y adapta cada `ToolSchema` al proveedor configurado.

Internamente, cada schema puede convertirse con:

```python
schema.to_llm_spec()
```

Eso produce una estructura con:

```text
name
description
parameters
```

Luego el provider la traduce al formato correspondiente:

- Ollama usa una estructura tipo `{"type": "function", "function": ...}`;
- Bedrock Converse usa una estructura tipo `{"toolSpec": ...}`;
- en tests, `MockLLMClient` recibe los schemas directamente.

Gracias a esta separación, `MyAgent` no depende de un proveedor concreto. El agente solo sabe llamar a `chat(...)` y ejecutar herramientas cuando recibe `tool_calls`.

## 4. Terminación del bucle y manejo de errores

El método `run` implementa el bucle del agente para una única interacción. En M1 no hay memoria persistente entre llamadas: cada ejecución arranca con un historial nuevo que contiene el mensaje del usuario.

El bucle termina en dos casos:

1. **Respuesta final del LLM.** Si `LLMClient.chat(...)` devuelve una respuesta sin `tool_calls`, el agente toma `response.content` como respuesta final y devuelve un `AgentResult`.
2. **Límite de iteraciones.** Si el modelo sigue pidiendo herramientas y no produce una respuesta final, el agente corta al alcanzar `max_iterations`. En ese caso devuelve un `AgentResult` válido con los `steps` acumulados y un error global.

Los errores de herramientas se registran a nivel de `AgentStep.error`. Por ejemplo: argumentos JSON inválidos, una herramienta inexistente o una excepción dentro del callable. El agente no rompe el flujo por estos casos; devuelve la observación de error al LLM como mensaje `role="tool"` para que el modelo pueda responder o intentar reparar.

`AgentResult.error` queda reservado para errores del flujo completo, como alcanzar el máximo de iteraciones sin respuesta final.

## 5. Validación

Además de los tests de conformidad provistos por el scaffold, agregamos escenarios propios en `tests/test_scenarios_m1.py`.

Estos escenarios validan:

- uso de más de una herramienta en una misma corrida;
- realimentación del resultado de una tool al LLM;
- lectura de archivos permitidos con `file_reader`;
- robustez frente a herramientas inexistentes;
- corte por `max_iterations`;
- respuesta directa sin herramientas.

Comandos utilizados:

```bash
python -m pytest tests/conformance/test_m1.py -v
python -m pytest tests/test_tool_schema.py -v
python -m pytest tests/test_scenarios_m1.py -v
```

## 6. Limitaciones conocidas

- **Sin estado entre llamadas.** Cada `run` arranca con un historial nuevo; no hay memoria conversacional multiturno. Eso queda para M2 junto con `max_history_messages`.
- **`structured_call` no implementado.** La salida estructurada con la herramienta sintética `final_result` y la reparación con reintentos forman parte de M2.
- **Sin reintentos ante fallos transitorios del proveedor.** Si `chat` lanza una excepción por red, API o credenciales, `run` no implementa todavía una política de retry.
- **`file_reader` tiene acceso acotado.** Por seguridad y para respetar la consigna de E/S restringida, solo lee archivos dentro de `sample_files/`. Esto evita que el agente lea rutas arbitrarias del sistema o archivos personales.
- **`file_reader` lee archivos completos en memoria.** No hay streaming ni límite explícito de tamaño. Está pensado para archivos de texto pequeños en UTF-8.
- **La calculadora opera sobre dos operandos y un operador.** No evalúa expresiones arbitrarias ni usa `eval`, por seguridad.
- **Validación básica de argumentos.** Se verifica que `arguments` sea JSON válido y que represente un objeto, pero no se validan tipos contra el schema antes de ejecutar la herramienta.
