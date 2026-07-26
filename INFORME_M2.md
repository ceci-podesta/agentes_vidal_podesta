# Informe Milestone 2 — Memoria, prompting y robustez

**Grupo:** Vidal – Podestá  
**Materia:** Agentes Autónomos — Maestría en Inteligencia Artificial

---

## 1. Estrategia de memoria

### Implementación: sliding window con prioridad a recencia

El agente mantiene un historial persistente en `self._history: list[dict]`, que sobrevive entre llamadas sucesivas a `run()`. Cada `run()` trabaja sobre una copia local de ese historial, agrega el nuevo mensaje del usuario y los turnos del LLM, y al finalizar persiste el resultado de vuelta en `self._history`.

La acotación se implementa en `_build_sliding_window(messages)` y funciona así:

1. Si la cantidad de mensajes no supera `max_history_messages`, se devuelve la lista completa sin recortar.
2. Si la supera, se toman los últimos `N` mensajes (ventana deslizante por recencia).
3. Se verifica que el último mensaje del usuario esté dentro de esa ventana. Si por algún motivo no lo está (borde: `max_history_messages=1` con historial muy largo), se lo incorpora explícitamente y se completan las posiciones restantes con los mensajes más recientes.

**Invariante garantizada:** el mensaje de usuario más reciente siempre aparece en la siguiente llamada al LLM, sin importar el tamaño del historial.

**Decisión de diseño — qué se descarta:** se priorizan los mensajes más recientes porque representan el estado más actualizado de la conversación. Los mensajes más antiguos se descartan cuando el presupuesto se agota. Conservar automáticamente el primer mensaje fue descartado: en muchas conversaciones el primero es simplemente un saludo ("Hola"), que consume presupuesto durante toda la sesión sin aportar contexto relevante. El system prompt se pasa por separado (parámetro `system=` de `chat()`), por lo que no ocupa slots del historial.

**Problema encontrado:** cuando `max_history_messages=0`, la función levanta `ValueError` antes de llamar al LLM. Este caso se considera una configuración inválida y se falla explícitamente en lugar de silenciarlo.

### Tradeoffs

| Aspecto | Decisión | Alternativa descartada |
|---|---|---|
| Qué conservar | Recencia (últimos N mensajes) | Primer + últimos (no siempre el primero es valioso) |
| Dónde aplicar el límite | En cada llamada a `chat()` | Solo al persistir (podría superar el límite dentro de un run) |
| Estrategias alternativas | No implementadas | Summarization o offload/retrieve: más ricas pero fuera del alcance del M2 |

---

## 2. Salida estructurada (`structured_call`)

### Cómo se ofrece `final_result` al LLM

`structured_call(prompt, schema, max_repair_attempts)` construye una herramienta sintética con `final_result_tool_schema(schema)` y la pasa como única tool disponible en cada llamada a `chat()`. Esto fuerza al LLM a responder invocando esa herramienta en lugar de emitir texto libre.

### Validación de argumentos

En cada iteración del loop:

1. Se busca un `tool_call` cuyo nombre sea `FINAL_RESULT_TOOL_NAME` (`"final_result"`).
2. Si el LLM respondió con texto libre o con otra tool, se clasifica como inválido.
3. Si invocó `final_result`, se deserializa el JSON de argumentos y se valida con `schema.model_validate(arguments)`.
4. Si la validación de Pydantic pasa, se devuelve la instancia validada.

### Reparación de fallos de validación

Cuando falla (texto libre, JSON malformado, tipos incorrectos), el error se agrega al historial interno de `structured_call` de esta forma:

```
messages = [
    {role: user, content: <prompt original>},
    {role: assistant, tool_calls: [<respuesta fallida>]},   ← se conserva
    {role: user, content: "La respuesta anterior no cumplió el formato ... Error: <detalle>"}  ← pedido de corrección
]
```

Incluir la respuesta fallida en el historial permite que el LLM vea exactamente qué salió mal y pueda corregirlo sin perder el contexto del prompt original.

### Cuando se agotan los reintentos

Tras `max_repair_attempts + 1` intentos sin éxito, se levanta `ValueError` con el detalle del último error. No se devuelve `None` ni un objeto parcial.

**Observación de Ceci:** `structured_call` mantiene su propio historial interno, independiente de `self._history`. Esto es intencional: las llamadas estructuradas son operaciones atómicas que no deben contaminar el historial conversacional general.

---

## 3. Errores recuperables en herramientas

### Calculadora (`calculator.py`)

La calculadora detecta y devuelve mensajes accionables para los siguientes casos:

| Error | Mensaje devuelto |
|---|---|
| `left_operand` no numérico | Indica el parámetro, el valor recibido y cómo debe verse un valor válido |
| `right_operand` no numérico | Ídem para el segundo operando |
| Operador no soportado | Lista los operadores permitidos: `'+'`, `'-'`, `'*'`, `'%'` |
| Módulo por cero | Explica que `right_operand` debe ser distinto de cero |

**Motivación:** los tipos en la firma (`Annotated[float, ...]`) son anotaciones para la generación del schema JSON, pero Python no los coerciona en tiempo de ejecución. Si el LLM pasa `"cuarenta"` como operando, el código anterior fallaba con un `TypeError` genérico capturado en `_execute_tool_call`. Ahora la calculadora valida explícitamente con `float()` y devuelve un mensaje útil.

**Ejemplo de recuperación:** el LLM envía `left_operand="cuarenta"`. La calculadora responde:

```
Error: el parámetro 'left_operand' recibió el valor 'cuarenta', que no es numérico.
Se esperaba un número entero o decimal, por ejemplo: 3, 2.5 o -10.
```

El LLM recibe ese mensaje como resultado de la herramienta y puede corregir la llamada en el siguiente turno.

### Lector de archivos (`file_reader.py`)

El lector detecta y devuelve mensajes accionables para:

| Error | Mensaje devuelto |
|---|---|
| Ruta vacía | Explica formato válido + lista archivos disponibles en `sample_files` |
| Ruta absoluta | Explica que solo se aceptan rutas relativas dentro de `sample_files` |
| Ruta con `..` | Explica la restricción de traversal antes de intentar resolver |
| Escape del sandbox (tras `resolve()`) | Explica el directorio permitido |
| Ruta es un directorio | Indica el error y lista el contenido del directorio |
| Archivo inexistente | Lista los archivos disponibles en el directorio contenedor (si existe y es válido) |

**Ejemplo de recuperación (archivo inexistente):** el LLM pide `"notas.txt"` pero el archivo se llama `"notas_test_m1.txt"`. La respuesta:

```
Error: el archivo 'notas.txt' no existe.
Archivos disponibles en 'sample_files': 'hola_mundo.md', 'lorem_ipsum.txt', 'notas_test_m1.txt'.
```

El LLM puede elegir la ruta correcta y reintentar en el siguiente turno.

### Contador de palabras (`word_counter.py`)

Aunque el enunciado menciona calculadora y lector de archivos como ejemplos, aplicamos el mismo criterio a la tercera herramienta del framework.

| Error | Mensaje devuelto |
|---|---|
| `text` es `None` | Indica que el parámetro es nulo y muestra cómo debe verse un valor válido |
| `text` no es string (ej: número) | Indica el tipo recibido y el valor, y aclara que se esperaba texto |

**Motivación:** igual que con `float`, Python no coerciona el tipo `str` en tiempo de ejecución. Si el LLM pasa `null` o un número, `text.split()` lanzaría un `AttributeError` genérico. Ahora se devuelve un mensaje accionable.

**Ejemplo de recuperación:** el LLM envía `text=null`. El contador responde:

```
Error: el parámetro 'text' es nulo. Se esperaba una cadena de texto, por ejemplo: 'hola mundo'.
```

---

## 4. Reintentos ante fallos transitorios

### Implementación

El agente implementa `_chat_with_retry()` que envuelve todas las llamadas a `self._llm.chat()` (tanto en `run()` como en `structured_call()`). La lógica es:

```
para attempt en 0..MAX_LLM_RETRIES:
    intentar self._llm.chat(...)
    si éxito → devolver resultado
    si excepción es transitoria y quedan intentos → reintentar
    si excepción no es transitoria o no quedan intentos → propagar
```

**Clasificación de errores transitorios** (`_is_transient_error`):

- Por tipo: `ConnectionError`, `TimeoutError`, `OSError`
- Por mensaje (case-insensitive): `"timeout"`, `"timed out"`, `"rate limit"`, `"throttl"`, `"503"`, `"502"`, `"504"`

Esto cubre los casos más comunes con AWS Bedrock: `ThrottlingException` (rate limit), errores de red y timeouts. Los errores definitivos (`ValidationError`, `PermissionError`, credenciales incorrectas) se propagan inmediatamente sin reintentar.

**Número máximo de reintentos:** 3 (constante `_MAX_LLM_RETRIES`). Total de intentos: 1 inicial + 3 reintentos = 4 llamadas antes de fallar.

**Sin delay entre reintentos:** para este contexto educativo se optó por reintentar sin espera. En un sistema productivo se agregaría backoff exponencial (ej: `time.sleep(base * 2**attempt)`) para respetar los límites del proveedor.

### Scope: LLM vs. herramientas

Los reintentos están implementados para las llamadas al cliente LLM, que es donde ocurren los fallos transitorios más comunes (red, rate limit, timeouts de Bedrock). Las herramientas locales (`calculator`, `file_reader`, `word_counter`) realizan operaciones síncronas en memoria o disco: sus fallos son definitivos (argumento inválido, archivo inexistente) y ya quedan capturados en `_execute_tool_call` como mensajes de error accionables para el LLM.

---

## 5. Modos de fallo dentro vs. fuera del alcance

### Dentro del alcance (manejados)

| Situación | Comportamiento |
|---|---|
| Conversación que supera el presupuesto de contexto | Sliding window la acota; `run()` sigue devolviendo `AgentResult` válido |
| LLM responde con texto libre en `structured_call` | Se reintenta con mensaje de reparación |
| Argumentos de `final_result` no validan el schema | Se reintenta con el detalle del error de Pydantic |
| Se agotan los reintentos de `structured_call` | Se levanta `ValueError` limpio |
| Timeout o rate limit del cliente LLM | Se reintenta hasta 3 veces |
| Tool con argumentos inválidos (tipo o valor) | Mensaje accionable devuelto al LLM; la conversación continúa |
| Herramienta desconocida (alucinación) | `AgentStep` con `error` no nulo; el bucle continúa |
| LLM en bucle infinito de tool calls | Se corta en `max_iterations` con `AgentResult.error` descriptivo |

### Fuera del alcance (no manejados deliberadamente)

| Situación | Razón |
|---|---|
| Backoff exponencial entre reintentos LLM | Complejidad innecesaria para el contexto educativo; el SDK de boto3 ya implementa reintentos a nivel HTTP |
| Reintentos de herramientas externas | Las herramientas actuales son locales; no tienen fallos de red |
| Summarization del historial | Más complejo que sliding window; fuera del alcance de M2 |
| Detección semántica de errores transitorios | Se usa clasificación por tipo/mensaje; suficiente para Bedrock |
| `max_history_messages=0` graceful | Se considera configuración inválida; se falla con `ValueError` explícito |
