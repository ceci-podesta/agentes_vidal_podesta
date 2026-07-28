# Informe Milestone 2 — Memoria, prompting y robustez

**Grupo:** Vidal – Podestá  
**Materia:** Agentes Autónomos — Maestría en Inteligencia Artificial

---

## 1. Estrategia de memoria

En M1 cada `run()` era independiente: el agente no recordaba nada entre llamadas. En M2, el agente pasa a tener estado: las llamadas sucesivas sobre la misma instancia continúan la misma conversación. Para esto, el constructor acepta `max_history_messages` y la lista de mensajes enviada al LLM en cada `chat()` nunca puede superar ese tope. La estrategia elegida es sliding window, con una invariante que no puede romperse: el mensaje de usuario más reciente siempre debe aparecer en la siguiente llamada al LLM. El agente también debe sobrevivir a conversaciones largas —decenas de turnos con mensajes grandes— sin romperse ni degradar la respuesta.

### Cómo fue implementada

Implementamos **sliding window con prioridad a recencia**.

El agente mantiene un historial persistente en `self._history: list[dict]`, que sobrevive entre llamadas sucesivas a `run()`. Cada `run()` trabaja sobre una copia local de ese historial, agrega el nuevo mensaje del usuario y los turnos del LLM, y al finalizar persiste el resultado de vuelta en `self._history`.

La acotación se implementa en `_build_sliding_window(messages)`, que se ejecuta antes de cada llamada a `chat()`:

1. Si la cantidad de mensajes no supera `max_history_messages`, se devuelve la lista completa sin recortar.
2. Si la supera, se toman los últimos `N` mensajes (ventana deslizante por recencia).
3. Se verifica que el último mensaje del usuario esté dentro de esa ventana y, si no lo está, se lo incorpora explícitamente completando el resto con los mensajes más recientes.

**Invariante garantizada:** el mensaje de usuario más reciente siempre aparece en la siguiente llamada al LLM, sin importar el tamaño del historial.

**Decisión de diseño — qué se descarta:** se priorizan los mensajes más recientes porque representan el estado más actualizado de la conversación. Una estrategia alternativa común es conservar siempre el primer mensaje del usuario, bajo la hipótesis de que contiene el objetivo principal del agente. Decidimos no adoptarla por dos razones: primero, no hay garantía de que el primer mensaje contenga el goal (podría ser simplemente "Hola"); segundo, el objetivo del usuario puede cambiar o refinarse a lo largo de los turnos, con lo cual conservar el primer mensaje no aportaría nada útil. El system prompt se pasa por separado (parámetro `system=` de `chat()`), por lo que no ocupa slots del historial.

### Problemas encontrados

El único caso problemático detectado fue `max_history_messages=0`: con ese valor sería imposible incluir el último mensaje del usuario (que es obligatorio por invariante). En lugar de comportarse de forma silenciosa o inesperada, la función levanta un `ValueError` explícito antes de llamar al LLM. Este caso se considera una configuración inválida.

Un caso de borde más sutil ocurre cuando hay múltiples llamadas a tools dentro de un mismo run que generan muchos mensajes `role="tool"` y `role="assistant"`. Si esos mensajes superan el límite, el último mensaje del usuario podría quedar fuera de la ventana. El paso 3 de `_build_sliding_window` lo previene garantizando que ese mensaje siempre esté presente.

### Tradeoffs

| Aspecto | Decisión | Alternativa descartada |
|---|---|---|
| Qué conservar | Recencia (últimos N mensajes) | Primer + últimos (el primero no siempre es valioso) |
| Dónde aplicar el límite | En cada llamada a `chat()` | Solo al persistir (podría superar el límite dentro de un run) |
| Estrategias alternativas | No implementadas | Summarization o offload/retrieve: más ricas pero fuera del alcance del M2 |

---

## 2. Salida estructurada (`structured_call`)

El enunciado requiere que `structured_call` exija al LLM una respuesta estructurada mediante la herramienta sintética `final_result`. El agente debe ofrecer al LLM un schema Pydantic, validar los argumentos del `tool_call` resultante y reintentar con un prompt de reparación si la validación falla o si el modelo responde con texto libre. Debe definirse un número máximo de reintentos y una estrategia de fallo limpia cuando no se pueda obtener el formato deseado.

`structured_call(prompt, schema, max_repair_attempts=2)` es el método que implementa ese contrato. Mantiene su propio historial interno, independiente de `self._history`: las llamadas estructuradas son operaciones atómicas que no deben contaminar el historial conversacional general.

### Cómo se ofrece `final_result` al LLM

`structured_call` construye una herramienta sintética llamada `final_result` a partir del schema recibido, usando `final_result_tool_schema(schema)`. Esa tool es la **única** que se le ofrece al LLM en cada llamada a `chat()`:

```python
response = self._chat_with_retry(
    messages=messages,
    tools=[final_result_schema],   # solo esta tool, nada más
    system=self._system,
)
```

Al no tener otras opciones, el LLM debe invocar `final_result` para responder. Si responde con texto libre, se trata como un fallo.

### Cómo se validan los argumentos

En cada iteración del loop se aplican estas validaciones en orden:

1. **¿El LLM invocó `final_result`?** Se busca un `tool_call` con ese nombre exacto. Si respondió con texto libre o con otra tool, es inválido.
2. **¿El JSON es parseable?** Se deserializa con `json.loads()`. Un JSON malformado es inválido.
3. **¿Los argumentos son un objeto?** Se verifica que el resultado sea un `dict`. Un array u otro tipo primitivo es inválido.
4. **¿Validan contra el schema Pydantic?** Se llama a `schema.model_validate(arguments)`. Tipos incorrectos, campos faltantes o valores fuera de rango son inválidos.

Si todas las validaciones pasan, `structured_call` termina devolviendo la instancia Pydantic validada.

### Cómo se reparan los fallos de validación

Cuando alguna validación falla, el error se guarda en `last_error` y se construye un pedido de corrección que se agrega al historial interno:

```
messages = [
    {role: user,      content: <prompt original>},
    {role: assistant, tool_calls: [<respuesta fallida>]},          ← se conserva
    {role: user,      content: "Error: <detalle>. Corregí la respuesta invocando final_result."}
]
```

Incluir la respuesta fallida en el historial permite que el LLM vea exactamente qué salió mal y pueda corregirlo en el siguiente intento sin perder el contexto del prompt original.

### Qué pasa cuando se agotan los reintentos

El loop corre `max_repair_attempts + 1` veces (1 intento inicial + N reparaciones). Con el valor por defecto de `max_repair_attempts=2`, hay hasta 3 intentos en total.

Si se agotan todos sin éxito, se levanta `ValueError` con el detalle del último error:

```python
raise ValueError(
    f"No se pudo obtener una respuesta válida después de {max_repair_attempts + 1} intentos. "
    f"Último error: {last_error}"
)
```

No se devuelve `None` ni un objeto parcial. El llamador recibe una excepción clara que describe por qué falló.

---

## 3. Errores recuperables en herramientas

El enunciado pide mejorar el manejo de errores de las herramientas de M1 —calculadora, lector de archivos y, por extensión, el contador de palabras— distinguiendo qué fallos son recuperables (el LLM puede corregir sus argumentos e intentar de nuevo) y devolviendo un mensaje accionable en lugar de un error genérico de Python.

Un error es **recuperable** cuando el LLM puede corregir sus propios argumentos y reintentar la llamada. En lugar de que el agente crashee o devuelva un error genérico de Python, la herramienta devuelve un mensaje que explica exactamente qué salió mal y cómo corregirlo. Ese mensaje llega al LLM como resultado de la tool call, dentro del historial de la conversación, y el LLM puede actuar sobre él en el siguiente turno.

La motivación técnica es que las anotaciones de tipo en la firma (`Annotated[float, ...]`, `Annotated[str, ...]`) solo sirven para generar el schema JSON. Python no las coerciona en tiempo de ejecución, por lo que si el LLM pasa `"cuarenta"` donde se espera un número, el código falla con un error genérico. En M2 cada herramienta valida sus argumentos explícitamente y devuelve mensajes útiles.

### Calculadora (`calculator.py`)

**Errores recuperables detectados:**

| Error | Qué información devuelve al LLM |
|---|---|
| `left_operand` no numérico | Nombre del parámetro, valor recibido, ejemplo de valor válido |
| `right_operand` no numérico | Ídem para el segundo operando |
| Operador no soportado | Lista completa de operadores permitidos: `'+'`, `'-'`, `'*'`, `'%'` |
| Módulo por cero | Nombre del parámetro (`right_operand`) y la restricción |

**Ejemplo concreto de recuperación:**

El LLM intenta calcular "veinte más cinco" y envía `left_operand="veinte"`. La calculadora responde:

```
Error: el parámetro 'left_operand' recibió el valor 'veinte', que no es numérico.
Se esperaba un número entero o decimal, por ejemplo: 3, 2.5 o -10.
```

El LLM recibe ese texto como resultado de la tool call, entiende que debe pasar un número, y en el siguiente turno corrige la llamada a `left_operand=20`.

### Lector de archivos (`file_reader.py`)

**Errores recuperables detectados:**

| Error | Qué información devuelve al LLM |
|---|---|
| Ruta vacía | Explica el formato válido y lista los archivos disponibles en `sample_files` |
| Ruta absoluta | Explica que solo se aceptan rutas relativas e indica cómo debe verse una válida |
| Ruta con `..` | Explica la restricción de traversal antes de intentar resolver la ruta |
| Escape del sandbox (tras `resolve()`) | Indica el directorio permitido |
| Ruta es un directorio | Indica el error y lista el contenido del directorio |
| Archivo inexistente | Lista los archivos disponibles en el directorio contenedor |

**Ejemplo concreto de recuperación:**

El LLM pide `"notas.txt"` pero el archivo se llama `"notas_test_m1.txt"`. El lector responde:

```
Error: el archivo 'notas.txt' no existe.
Archivos disponibles en 'sample_files': 'hola_mundo.md', 'lorem_ipsum.txt', 'notas_test_m1.txt'.
```

El LLM ve la lista, identifica el nombre correcto y en el siguiente turno corrige la llamada a `path="notas_test_m1.txt"`.

### Contador de palabras (`word_counter.py`)

La tercera herramienta es de elección libre del grupo. Aplicamos el mismo criterio que a las anteriores.

**Errores recuperables detectados:**

| Error | Qué información devuelve al LLM |
|---|---|
| `text` es `None` | Indica que el parámetro es nulo y muestra cómo debe verse un valor válido |
| `text` no es string (ej: número) | Indica el tipo recibido, el valor, y que se esperaba una cadena |

**Ejemplo concreto de recuperación:**

El LLM envía `text=null`. El contador responde:

```
Error: el parámetro 'text' es nulo. Se esperaba una cadena de texto, por ejemplo: 'hola mundo'.
```

El LLM corrige la llamada en el siguiente turno pasando el texto como string.

---

## 4. Reintentos ante fallos transitorios

El enunciado requiere que el agente envuelva sus llamadas al cliente LLM y a las herramientas de forma que los fallos transitorios —timeouts, errores 5xx, rate limits, excepciones de red— se reintenten automáticamente, y que los errores definitivos afloren de forma limpia sin reintentos innecesarios.

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

Toda implementación tiene límites. Esta sección documenta qué escenarios de fallo decidimos manejar explícitamente y cuáles dejamos fuera, explicando en cada caso el razonamiento detrás de la decisión.

### Dentro del alcance — manejados deliberadamente

Estos son los fallos que el agente detecta y resuelve sin que el llamador tenga que intervenir:

| Situación | Comportamiento | Por qué lo incluimos |
|---|---|---|
| Conversación que supera el presupuesto de contexto | Sliding window acota el historial; `run()` sigue devolviendo `AgentResult` válido | Requisito explícito del enunciado |
| LLM responde con texto libre en `structured_call` | Se reintenta con mensaje de reparación | Requisito explícito del enunciado |
| Argumentos de `final_result` no validan el schema | Se reintenta con el detalle del error de Pydantic | Requisito explícito del enunciado |
| Se agotan los reintentos de `structured_call` | Se levanta `ValueError` limpio con el último error | Evitar devolver `None` o un objeto parcial sin avisar |
| Timeout o rate limit del cliente LLM | Se reintenta hasta 3 veces | Requisito explícito del enunciado; común con AWS Bedrock |
| Error definitivo del cliente LLM (credenciales, schema inválido) | Se propaga inmediatamente sin reintentar | Reintentar no ayudaría; el llamador necesita saber |
| Tool con argumentos inválidos (tipo o valor) | Mensaje accionable devuelto al LLM; la conversación continúa | Requisito explícito del enunciado |
| Herramienta desconocida (alucinación del LLM) | `AgentStep` con `error` no nulo; el bucle continúa | Requisito de robustez de M1, preservado en M2 |
| LLM en bucle infinito de tool calls | Se corta en `max_iterations` con `AgentResult.error` descriptivo | Requisito de M1, preservado en M2 |

### Fuera del alcance — excluidos deliberadamente

Estos son fallos que reconocemos como reales pero que decidimos no manejar, con la justificación de cada decisión:

| Situación | Por qué lo dejamos fuera |
|---|---|
| Backoff exponencial entre reintentos LLM | Complejidad innecesaria para el contexto educativo. El SDK de boto3 ya implementa reintentos con backoff a nivel HTTP. En producción sería la mejora natural. |
| Reintentos de herramientas externas | Las tres herramientas son locales (memoria y disco). Sus fallos son definitivos por naturaleza: no tiene sentido reintentar un archivo que no existe. Una tool que haga llamadas de red requeriría su propio retry. |
| Summarization del historial | Alternativa más rica que sliding window, pero fuera del alcance de M2. Requeriría una llamada extra al LLM en cada turno para resumir el contexto descartado. |
| Detección semántica de errores transitorios | La clasificación por tipo de excepción y keywords en el mensaje es suficiente para Bedrock. Una detección semántica requeriría parsear payloads de error de cada proveedor, lo que acoplaría el agente a implementaciones específicas. |
| `max_history_messages=0` con comportamiento graceful | Se considera una configuración inválida por definición: con 0 slots es imposible cumplir la invariante del último mensaje del usuario. Fallamos explícitamente con `ValueError` en lugar de silenciarlo. |

---

## 6. Suite de tests

**Total: 51 tests — todos pasan.**

### Tests de conformidad (provistos por los profesores)

No se modificaron. Se ejecutan en cada iteración para garantizar que el M2 no rompe el M1.

| Archivo | Tests | Qué verifican |
|---|---|---|
| `tests/conformance/test_m1.py` | 5 | Contrato público de M1: `build_agent`, `register_tool`, `run`, ejecución de tools |
| `tests/conformance/test_m2.py` | 7 | Contrato público de M2: statefulness, sliding window, `structured_call`, tracking de tokens |
| `tests/test_tool_schema.py` | 8 | Generación correcta de schemas JSON a partir de callables |

### Tests propios del grupo

| Archivo | Tests | Qué verifican |
|---|---|---|
| `tests/test_scenarios_m1.py` | 6 | Escenarios de punta a punta de M1: dos tools en un run, realimentación al LLM, robustez ante alucinaciones, corte por `max_iterations` |
| `tests/test_scenarios_m2.py` | 25 | Features de M2 — detalle abajo |

**Desglose de `test_scenarios_m2.py` por feature:**

| Feature | Tests | Casos cubiertos |
|---|---|---|
| Reintentos ante fallos transitorios | 4 | Éxito tras retry, retry por `ConnectionError`, no-retry ante error definitivo, agotamiento de reintentos |
| Errores recuperables — calculadora | 5 | `left_operand` no numérico, `right_operand` no numérico, operador inválido, módulo por cero, happy path |
| Errores recuperables — file_reader | 6 | Ruta vacía, absoluta, con `..`, archivo inexistente con listado, directorio, happy path |
| Errores recuperables — word_counter | 3 | `text=None`, `text` de tipo incorrecto, happy path |
| Memoria y sliding window | 3 | Último mensaje del usuario siempre presente, `max_history_messages=0` falla antes del LLM, historial persiste tras interacción con tool |
| Tracking de tokens | 3 | Solo input tokens trata output como 0, ceros explícitos no son `None`, tokens se resetean entre runs |
| Salida estructurada | 1 | El segundo intento de reparación incluye la respuesta fallida y el detalle del error |
