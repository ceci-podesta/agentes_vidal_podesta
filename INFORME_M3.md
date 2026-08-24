# Informe Milestone 3 — Evaluación sobre un problema objetivo (sala de escape)

**Grupo:** Vidal – Podestá
**Materia:** Agentes Autónomos — Maestría en Inteligencia Artificial

**Resultado logrado:** se resolvieron los 5 escenarios obligatorios
(`study-with-key`, `color-locks`, `apartment-keys`, `library-search`,
`office-sequence`, de `easy` a `hard`) más 1 escenario `extreme`
(`extreme-archive`). Los otros dos `extreme` (`vault-combination`,
`backtracking-vault`) no se resolvieron — ver Sección 5, limitación d.

**Reproducir los resultados** (desde la raíz de este repo,
`agentes_vidal_podesta-m3-world-evaluation/`):

```bash
# Setup inicial (solo la primera vez)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configurar Bedrock (cada vez que abrís una terminal nueva)
export AWS_REGION="us-east-2"
export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
export AWS_ACCESS_KEY_ID="tu-access-key"
export AWS_SECRET_ACCESS_KEY="tu-secret-key"
export AWS_SESSION_TOKEN="tu-session-token"   

# Correr la suite de tests (196 tests)
pytest -q

# Correr la evaluación oficial (pass@k, los 6 escenarios)
python eval/run.py

# Categorizar errores de una corrida ya generada
python eval/error_categories.py eval/results/final/<run_id>.json
# python eval/error_categories.py eval/results/final/20260823T201948569061Z.json
```

Los resultados quedan en `eval/results/final/<run_id>.json` (un archivo por
corrida, nombrado con el timestamp de esa corrida). Los dos archivos que
cita este informe (Sección 3.1):

- Con scratchpad (resultado oficial), accuracy global 26/30 (0.87):
  `eval/results/final/20260823T201948569061Z.json`
- Sin scratchpad (ablation, Experimento 6), accuracy global 27/30 (0.90):
  `eval/results/final/20260823T221012501252Z.json` — para reproducir esta
  corrida hay que poner `"use_m3_scratchpad": False` en `M3_AGENT_CONFIG`
  (`eval/run.py`) antes de ejecutar el comando de arriba; no hay un flag de
  línea de comandos para esto.

> **Nota de estado.** Este informe se apoya en la **corrida final oficial**
> (`pass@k`, k=5, run_id `20260823T201948569061Z`) contra
> `amazon.nova-lite-v1:0`, con el planner explícito y el scratchpad M3
> activados. Los 6 escenarios evaluados quedan resueltos (`pass@k ≥ 0.5`).
> La dimensión cualitativa (LLM-as-judge, Sección 2) ya está implementada y
> documentada, con una limitación conocida sobre su confiabilidad (Sección
> 5, ítem i). No se inventan números: donde falta evidencia, se dice que
> falta.

---

## 1. Aproximación

### Contexto: qué es M3 respecto a M1 y M2

En M1 y M2 no se resolvía un problema puntual: se construía la
infraestructura del agente. El resultado de esas dos etapas es un framework
propio escrito en Python puro — sin apoyarse en SDKs de agentes de terceros
(paquetes que ya traen resuelto el loop de razonar/llamar herramientas, el
manejo de contexto, los reintentos, etc.). Es decir, esa lógica de base se
implementó a mano, hablando directo con la API de Bedrock.

- **M1 — Loop básico.** `LLMClient` sobre Bedrock, `ToolRegistry` con schema
  autogenerado y un loop ReAct con condición de corte: el agente razona,
  llama herramientas y devuelve un resultado.
- **M2 — Robustez.** Manejo de contexto (sliding window / summarization),
  structured outputs con Pydantic más retry, cálculo de tokens y manejo de
  errores ante tools que fallan.
- **M3 — Aplicación + evaluación.** Ahora se trata de aplicar ese framework a
  un problema concreto, medirlo con rigor y explicar qué se encontró. La
  máquina se pone a prueba.

### El problema: sala de escape

- El mundo vive en `mia_world/` y es fijo: no se toca.
- El dataset son los escenarios en `scenarios/`, de dificultad creciente
  (easy → medium → hard → extreme).
- Mismos verbos genéricos (`look`, `examine`, `take`, `use`, `go`) en todos
  los escenarios.
- La meta se verifica con `mia_world.check_goal` sobre el estado del mundo,
  no sobre el texto que devuelve el agente — esto da una métrica fiable
  (el agente puede "decir" que resolvió el escenario sin haberlo hecho, pero
  el estado del mundo no miente). La mayoría de los escenarios pide que
  `puerta_principal` quede con `open_state == "open"`; los escenarios
  multi-sala (navegación) le suman condiciones adicionales — por ejemplo
  estar en una sala determinada o haber pasado por los eventos en un orden
  específico (`mia_world/goals.py`, tipos `agent_in_room` y `sequence`).

### Qué agregamos nosotros

El framework de M1+M2 no se modificó en su contrato público: `build_agent`
(`student_framework/__init__.py`), y los métodos `register_tool`/`run` de la
clase `Agent` (`student_framework/agent.py`) mantienen la misma interfaz
declarada en `mia_agents/protocols.py`. Lo que M3 agrega es específico del
problema, no un cambio de arquitectura del agente:

#### 1. Registro de tools del mundo

**Lo que pedía el enunciado.** El TP da armados los cinco verbos del mundo
(`mia_world/tools.py`: `look`, `examine`, `take`, `use`, `go`) y pide
conectarlos al agente de M1+M2 registrándolos con `agent.register_tool(...)`
— es el primer paso explícito para pasar de "framework genérico" a "agente
que puede jugar la sala de escape".

**Qué hicimos.** Escribimos `eval/run.py::run_scenario`, la función que arma
un escenario y se lo entrega al agente: crea el mundo, crea el agente, y
registra ahí las tools correspondientes (las cuatro siempre, más `go` solo si
el escenario tiene varias salas). No modificamos ninguna tool del mundo, solo
el "cableado" que las conecta al agente en cada corrida.

**Más detalladamente.** El agente no nace sabiendo hacer nada en el mundo:
hay que decirle explícitamente qué acciones tiene disponibles. Eso es
"registrar una tool". `run_scenario` hace, en orden:

1. Crea el mundo del escenario — el estado inicial, con sus cuartos y
   objetos.
2. Crea el agente, todavía "vacío".
3. Llama a `make_world_tools(world)` (`mia_world/tools.py:312`), que
   devuelve los verbos ya conectados a ese mundo puntual, agregando `go`
   solo si hay más de un cuarto.
4. Recorre esa lista dándolas de alta una por una con
   `agent.register_tool(...)`.

La aclaración de que "ninguna tool del mundo fue modificada" importa porque
marca la diferencia con los aportes propios de M3 (como `research_documents`,
subsección 2): acá no inventamos un verbo nuevo, solo tomamos los que ya
venían dados y se los entregamos al agente para cada escenario.

#### 2. Una tool propia: `research_documents`

(`student_framework/m3_research.py`). A diferencia de las tools de la
subsección 1, esta no viene del scaffold: es la única tool nueva que
diseñamos desde cero para M3. Se registra siempre, sin flag de on/off, en
todos los escenarios (`eval/run.py:263-269`), pero solo es relevante cuando
`examine` revela una colección de más de cinco documentos similares — el
caso de `extreme-archive` (20 expedientes, ~16K tokens), que no entra en el
contexto principal.

Delega la inspección en lotes de cinco a un sub-agente aislado, que solo
puede observar (no puede `take`/`use`/`go`), y devuelve un reporte compacto
validado contra un schema Pydantic — el código verifica que lo que reporta
el sub-agente coincida con lo que realmente se observó, descartando
cualquier objeto u ID que el sub-agente afirme haber encontrado sin que
`examine` lo haya revelado. El agente principal nunca ve la prosa completa
de los 20 expedientes, solo ese reporte ya verificado. Esta tool se retoma
más adelante en el Experimento 3 (Sección 4), donde se compara contra una
versión anterior sin la validación descrita arriba.

#### 3. Scratchpad M3

(`student_framework/m3_scratchpad.py`). Opcional vía el flag
`use_m3_scratchpad` (`student_framework/agent.py:93`, activado en
`eval/run.py:89`). Un bloque de estado de trabajo (IDs observados,
inventario, ubicación, salidas) que se reconstruye determinísticamente a
partir de las tool outputs y se inyecta en el mensaje de sistema antes de
cada llamada al LLM. No es memoria aprendida ni resumen: es una proyección
determinística del estado ya observado, pensada para escenarios donde un
único `look` no alcanza (`apartment-keys`, `office-sequence`).

Surge de una limitación del sliding window de M2: cuando el historial se
recorta por longitud, una observación hecha varios turnos atrás (por
ejemplo, en otro cuarto) puede quedar fuera de la ventana visible para el
modelo. El scratchpad reinyecta ese estado en cada turno, sin depender de
que siga presente en el historial recortado. Incluye además un aviso
determinístico de objetos ya revelados en un contenedor chico (≤5 ítems) que
todavía no están en el inventario — ver Experimento 5.

#### 4. Planner explícito

(`student_framework/m3_planner.py`). Opcional (`use_m3_planner`), activado
por default en la config oficial tras el Experimento 4.

**Por qué existe:** nace de dos fallos concretos observados en escenarios
previos, no de una idea genérica de "vamos a agregar un planner":

- `color-locks` — a veces el LLM, apenas recibe el mensaje inicial (que ya
  describe bastante la escena), directamente devuelve una narración de texto
  completa sin llamar a ninguna tool. Es el fallo más grave: el agente
  "contesta" en vez de actuar.
- `office-sequence` — tiene un goal de tipo `sequence` (hay que hacer las
  cosas en un orden específico). Un agente que reacciona turno a turno, sin
  plan, tiende a perder ese orden.

**Cómo funciona, en dos partes:**

1. Antes de que el agente observe nada, una única llamada forzada a
   `structured_call` (el mismo mecanismo de M2 que obliga a llamar
   `final_result`) le pide al LLM un `Plan` (modelo Pydantic: lista de
   `steps` en lenguaje natural, mínimo un paso) a partir del mensaje del
   usuario. Al forzar esa salida estructurada, el modelo no puede devolver
   solo prosa — tiene que producir el objeto `Plan` sí o sí, lo que ataca
   directo el fallo de `color-locks`. Además, el prompt de planificación es
   explícito en dos restricciones: no puede nombrar tools ni IDs (todavía no
   se observó el mundo real), y no puede inventar qué llave abre qué
   cerradura si el mensaje no lo dice — tiene que proponer algo genérico
   ("probar cada llave en el objeto compatible") en vez de adivinar, para
   que el plan no meta un supuesto falso que el agente después siga a
   ciegas.
2. Ese plan se renderiza como texto (`render_plan`) y se inyecta en el
   system prompt junto al scratchpad, con una advertencia final explícita:
   puede tener detalles incompletos o equivocados, y el agente debe
   priorizar siempre lo que confirmen las tools por sobre el plan si hay
   contradicción. Es una guía de orden y alcance, no un script que el agente
   ejecuta ciegamente: el loop ReAct sigue intacto, tool call por tool call.

Los resultados reales de activar el planner sobre `color-locks` y
`office-sequence` están en el Experimento 4 (Sección 4).

#### 5. Guardas anti-repetición

Opcionales (`max_repeated_failures`, `max_repeated_observations` —
`student_framework/agent.py:90-91`, ambas en `None` por defecto), activadas
en la config oficial con valor `1` (`eval/run.py:86-87`). Cortan una tool
call si repite exactamente la misma acción fallida o la misma observación
sin información nueva — evita que el agente entre en bucle cuando el LLM no
actualiza su plan.

El agente identifica una "misma llamada" por firma (`_tool_call_signature`:
nombre de la tool + argumentos, `agent.py:229`), no por resultado — dos
`examine` sobre targets distintos no cuentan como repetición. Con el valor
`1` de la config oficial: si una tool call ya falló una vez con esa firma
exacta, el siguiente intento idéntico se bloquea antes de llamar al modelo;
lo mismo para observaciones (`look`, `examine`, `research_documents`, según
`observation_tool_names` en `eval/run.py:88`) repetidas sin haber tomado
ninguna acción nueva de por medio. En ambos casos, en vez de ejecutar la
tool, se le devuelve al agente un error explícito ("elegí una acción
diferente") para forzarlo a probar algo distinto.

#### 6. Fix a un mecanismo heredado de M2 (retry ante errores transitorios)

**Contexto.** M2 ya incluía un mecanismo de reintentos —
`_chat_with_retry` (`student_framework/agent.py`) — para que la corrida no
abortara ante errores de red pasajeros (timeouts, límites de tasa, errores
5xx): reintenta la llamada al modelo unas pocas veces
(`_is_transient_error` decide qué mensajes de error cuentan como
"pasajeros") antes de dejar que el error se propague.

**Qué encontramos.** Al correr la evaluación oficial de M3 contra Bedrock
apareció un error puntual que ese mecanismo no reconocía:
`ModelErrorException` ("invalid sequence as part of ToolUse"). No es un bug
de lo que nosotros le mandamos al modelo — es Bedrock/Nova Lite generando
ocasionalmente, por azar del muestreo, un bloque de uso de tool mal
formado. Como `_is_transient_error` no tenía ese mensaje en su lista de
errores "pasajeros", la excepción se propagaba sin reintentar y abortaba
toda la corrida `pass@k` en el medio — se perdían los 5 intentos de todos
los escenarios ya evaluados por un error de un solo intento.

**Qué se cambió.** Se agregaron dos frases a la lista de errores
reconocidos como transitorios (`modelerrorexception`,
`invalid sequence as part of tooluse`), para que este caso entre por el
mismo camino de reintento que ya existía, en vez de crear un mecanismo
nuevo. Con esto, la corrida oficial completa (30 intentos) pasó a terminar
sin abortar por este motivo.

#### 7. Herramienta de diagnóstico: `eval/manual_run.py`

`python eval/run.py` corre los 6 escenarios × 5 intentos (`pass@k`) cada
vez, contra Bedrock — caro en tiempo y en tokens para iterar mientras se
prueba un cambio puntual. Para eso construimos `eval/manual_run.py`: corre
**un solo escenario, una sola vez**, con la misma configuración real
(`M3_AGENT_CONFIG`) que usa `eval/run.py`, e imprime la traza completa en
la terminal en vez de guardar un JSON. Se usó para diagnosticar fallos
puntuales y para las "pruebas puntuales repetidas" que se citan en el
Experimento 5 (Sección 4).

```bash
python eval/manual_run.py --scenario office-sequence --planner
```

(`--scenario` acepta un id de escenario, una dificultad, o un path a un
JSON de `scenarios/`; `--planner` es opcional y activa el plan inicial sin
tocar la config oficial). Al terminar imprime un resumen con este formato:

```text
Respuesta final del agente: '...'
goal_achieved: True — <razón que da check_goal>
Tokens principal: entrada=<N> salida=<N>
Duración: <segundos> s
```

(antes de este resumen imprime la traza completa, paso a paso: cada tool
call con su resultado).

#### 8. Tests nuevos para los componentes de M3

Cada pieza agregada en esta entrega tiene sus propios tests, además de los
ya existentes de M1/M2 y los de conformidad (`tests/conformance/`, fijos,
no se tocan): `tests/test_m3_planner.py`, `tests/test_m3_research.py`,
`tests/test_m3_scratchpad.py`, `tests/test_scenarios_m3.py` (escenarios de
integración propios para M3), `tests/test_error_categories.py`,
`tests/test_eval_run.py` y `tests/test_llm_judge.py`. En total suman **40
tests nuevos**, dentro de una suite completa de 196 tests.

```bash
pytest -q
```

---

**Mismo agente en los tres niveles.** El criterio de aprobación pide el mismo
agente y el mismo system prompt en `easy`/`medium`/`hard`. `M3_AGENT_CONFIG`
en `eval/run.py` es una única configuración (un `system_prompt`, un
`max_iterations`, una guardia) que se aplica a todos los escenarios
seleccionados en la misma corrida; no hay ramas por escenario ni prompts
alternativos.

Ese único `system_prompt` no salió escrito de una sola vez: se fue iterando
a medida que se identificaban fallos puntuales en escenarios concretos —las
instrucciones sobre consultar el scratchpad (Experimento 1), no adivinar
qué llave abre qué cerradura y recordar `look` después de `go` (Experimento
4), y tomar todo lo revelado en un contenedor antes de salir de la sala
(Experimento 5) se fueron agregando así. Cada ajuste se probó primero contra
el escenario puntual que lo motivaba con `eval/manual_run.py` (subsección 7)
—mucho más barato que correr el `pass@k` completo— y solo después se
confirmó con una corrida oficial que no rompiera el resto de los
escenarios, manteniendo siempre un único prompt compartido.

**Qué no se especializó:** las tools del mundo, el motor de estado
(`mia_world/`) y el dataset (`scenarios/`) son fijos y no se tocaron.

---

## 2. Métricas

### Cuantitativa — accuracy de goals

```text
accuracy = escenarios con goal_achieved == True / escenarios evaluados
```

Se calcula con `mia_world.check_goal` sobre el **estado del mundo**, no sobre
lo que el agente narra. Es la métrica primaria por lo mismo que argumenta el
paper de ALFWorld/ReAct: un agente puede declarar éxito sin haber cambiado el
estado real. `check_goal` además verifica orden temporal (`event_log`) para
goals de tipo `sequence`, como el de `office-sequence`.

Se reporta accuracy global y por dificultad, más las métricas operativas que
la acompañan (no la reemplazan, la explican): tool calls, tool errors,
tokens de entrada/salida, latencia (`duration_seconds` por intento y
agregada), y exceso de tool calls sobre el óptimo publicado en
`ENUNCIADO_M3.md` por escenario. Tokens de entrada/salida se usan como proxy
de coste (en vez de un cálculo en USD): evita atar el informe a un pricing de
Bedrock que puede cambiar, y tokens ya es la unidad que determina si un
escenario entra o no en la ventana de contexto — la variable que de verdad
importa para este problema.

### Cualitativa — LLM-as-judge

El enunciado pide rúbrica **o** LLM-as-judge, no ambas. Se había diseñado
primero una rúbrica manual de 3 dimensiones (`estado_del_mundo`,
`recuperacion`, `planificacion_y_eficiencia`, puntaje 0-2 cada una), pero
requiere que una persona lea cada traza y complete los puntajes a mano —
decidimos no hacer esa lectura humana. Por eso el grupo pasa a
**LLM-as-judge** como la dimensión cualitativa del informe.

**Qué mide.** Una única dimensión: **uso del feedback** — ante un resultado
de tool que corrige o contradice la acción del agente (un error, una
precondición no cumplida, una combinación incompatible), ¿la siguiente
acción relevante incorpora esa corrección? Escala:

| Puntaje | Significado |
|---|---|
| `2` | consistente — incorpora el feedback en todos los episodios |
| `1` | parcial — lo incorpora en algunos episodios, no en todos |
| `0` | ausente — no lo incorpora en ningún episodio |
| `null` | no aplicable — la traza no tuvo feedback correctivo que evaluar |

Aplicar la corrección muchos pasos después no cuenta como "incorporado": la
rúbrica exige que sea la primera acción relevante posterior, para no
confundir aprendizaje real con que el agente haya llegado por prueba y
error. Se descartaron otras dimensiones candidatas (cumplimiento, cantidad
de errores, repeticiones, tokens, límite de iteraciones) porque ya tienen
fuente determinística (Sección "Análisis de errores" más abajo); agregarlas
como juicio del LLM hubiera sumado costo y variabilidad sin información
nueva. El juez no recibe `goal_achieved` ni `goal_reason` en el prompt —
esos campos se adjuntan después de que el juez ya falló, para no sesgar su
lectura de la traza.

**Qué modelo se usó y por qué.** Se calibraron dos modelos sobre la misma
traza (`office-sequence`, intento 2, que tiene 7 episodios de feedback
identificados manualmente):

| Modelo | Episodios detectados (de 7) | Cobertura |
|---|---:|---:|
| `mistral.mistral-large-3-675b-instruct` | 3 | 42,9 % |
| `moonshot.kimi-k2-thinking` | 6 | 85,7 % |

(archivos: `eval/results/judge/20260823T201948569061Z/20260824T001758662830Z__mistral-mistral-large-3-675b-instruct__office-sequence__attempt-2.json`
y `.../20260824T003843076617Z__moonshot-kimi-k2-thinking__office-sequence__attempt-2.json`)

Se adoptó **Kimi K2 Thinking** como juez por default (`DEFAULT_JUDGE_MODEL_ID`
en `eval/llm_judge.py`) por la mejor cobertura observada — no se lo considera
perfecto (omitió un episodio incluso en esta calibración).

**Validación de la implementación** (`20260823T201948569061Z`, la misma
corrida de 30 intentos/6 escenarios que cita el resto del informe, evaluada
por ambos modelos para conservar la comparación como evidencia adicional de
la elección). Se considera una validación de la implementación, no el
resultado final del juez, porque no incluye `vault-combination` ni
`backtracking-vault`:

| Métrica | Mistral Large 3 | Kimi K2 Thinking |
|---|---:|---:|
| Juicios válidos (de 30) | 24 | 25 |
| Trazas con feedback detectado | 16 | 13 |
| Episodios detectados | 43 | 30 |
| Episodios incorporados | 13 | 22 |
| Tasa de incorporación (sobre lo detectado) | 30,2 % | 73,3 % |
| Puntaje medio (0-2) | 1,125 | 1,385 |

(archivos: `eval/results/judge/20260823T201948569061Z/20260824T004905427025Z__mistral-mistral-large-3-675b-instruct__all.json`
y `.../20260824T005103809632Z__moonshot-kimi-k2-thinking__all.json`)

Los juicios fallidos (6 de Mistral, 5 de Kimi) fueron por no invocar
`final_result` tras los reintentos, o por citar evidencia que no coincidía
literalmente con el paso indicado — quedan registrados en el JSON, no
detienen el resto de la corrida. La tasa de incorporación usa como
denominador solo los episodios que cada juez detectó, por lo que **no es
comparable en términos absolutos entre sí** — ver limitación en la
Sección 5.

**Resultado final del juez.** La evaluación cualitativa "de producción" —ya
con Kimi K2 Thinking como único juez, sin comparar modelos— se corrió
sobre los ocho escenarios del dataset completo (`DEVELOPMENT_SCENARIOS=None`),
no sobre los seis de la corrida que este informe usa como evidencia
cuantitativa oficial (Sección 3.1). Es una corrida distinta del agente,
`20260824T022515999713Z` (40 intentos: los mismos 5 escenarios obligatorios
más los 3 `extreme`), con la misma config oficial pero otro resultado de
`pass@k` por no-determinismo del LLM (`color-locks` da 3/5 en esta corrida
en vez de 5/5).

| Métrica | Kimi K2 Thinking (40 intentos, 8 escenarios) |
|---|---:|
| Juicios válidos | 40/40 (sin fallos) |
| Trazas con feedback detectado | 26 de 40 |
| Episodios detectados | 96 |
| Episodios incorporados | 64 |
| Tasa de incorporación (sobre lo detectado) | 66,7 % |
| Puntaje medio (0-2) | 1,38 |
| Tokens de entrada / salida del juez | 177.911 / 112.651 |

(archivo: `eval/results/judge/20260824T022515999713Z/20260824T023928831731Z__moonshot-kimi-k2-thinking__all.json`)

Esta tasa de 66,7 % es la que se audita en la limitación de la Sección 5:
no debe leerse como "el agente incorpora el 66,7 % del feedback real", sino
como lo que el juez logró detectar y clasificar sobre lo que detectó.

**Cómo se corre.** Es parte del mismo comando reproducible, pero
desactivada por default:

```bash
# En eval/run.py: RUN_LLM_JUDGE = False (por defecto)
python eval/run.py               # no corre el juez, solo el pass@k

# Para correr el juez sobre un reporte ya generado, sin repetir escenarios:
python eval/llm_judge.py eval/results/final/20260823T201948569061Z.json
```

Se deja en `False` por defecto para que reproducir el criterio de
aprobación (`pass@k`) no le imponga a quien corrija el costo y tiempo
extra de una segunda evaluación con LLM sobre cada intento. Los resultados
completos de ambos modelos quedan igual disponibles en
`eval/results/judge/20260823T201948569061Z/`.

### Análisis de errores — categorías

`eval/error_categories.py` toma el reporte JSON que ya generó `eval/run.py`
(no vuelve a correr al agente) y clasifica cada intento en una o más de
siete categorías de fallo, buscando frases fijas dentro de los mensajes de
error que devuelven las tools (`mia_world/tools.py`) y dentro del
`goal_reason` que explica por qué `check_goal` no dio por cumplido el
objetivo:

| Categoría | A qué corresponde |
|---|---|
| `id_inventado` | el agente usó un ID de objeto que no existe en el mundo |
| `violacion_estado` | intentó una acción que el estado actual no permite (agarrar algo que no ve, salir por donde no hay salida, etc.) |
| `accion_redundante` | repitió la misma tool call o la misma observación sin que hubiera novedad |
| `limite_iteraciones` | se cortó por agotar `max_iterations` sin terminar |
| `terminacion_prematura` | el agente dio una respuesta final sin haber cumplido el goal y sin haber agotado las iteraciones |
| `planificacion_orden_incorrecto` | en un goal de tipo `sequence` (pasos en orden obligatorio), lo resolvió pero en el orden equivocado |
| `presion_contexto` | no llegó al goal y acumuló muchísimos tokens de entrada — señal de que puede haberse quedado sin espacio de contexto (se marca para revisar la traza a mano, no es un límite exacto del modelo) |

Como la clasificación se basa en texto literal ya fijo (los mensajes de
`mia_world/tools.py` no cambian) y en el `goal_reason` que ya calcula
`check_goal`, no hay interpretación subjetiva de por medio: es reproducible.

**Una exclusión deliberada:** no cuenta los `worker_errors` que puede generar
la delegación de `research_documents` en `extreme-archive`. Esos errores
ocurren cuando el sub-agente de investigación afirma algo que la observación
real no respalda, y el código lo rechaza — es la validación (Sección 1,
subsección 2) funcionando como se espera, no una falla del agente principal.
Contarlos junto a las 7 categorías de arriba mezclaría dos cosas distintas:
errores del agente vs. rechazos correctos de un chequeo interno (ver
Experimento 3).

---

## 3. Resultados

### 3.1 Corrida final (criterio de aprobación)

Esta sección presenta dos corridas oficiales, una al lado de la otra, que
solo difieren en `use_m3_scratchpad`:

```bash
python eval/run.py
```

Ambas evaluaron los 5 escenarios obligatorios más `extreme-archive`, con
`amazon.nova-lite-v1:0` (`us-east-2`), `pass@k` con k=5 (cada escenario
corrido 5 veces), y comparten el resto de la configuración:

- `use_m3_planner=True` (planner explícito, Experimento 4).
- `max_iterations=25` (hasta 25 iteraciones por intento).
- `max_history_messages=200` (ventana de historial de hasta 200 mensajes).
- `max_repeated_failures=max_repeated_observations=1` (guardas
  anti-repetición, cortan en la primera repetición).

#### Con scratchpad (`use_m3_scratchpad=True`) — resultado oficial

`eval/results/final/20260823T201948569061Z.json` — **este es el JSON que se
toma como resultado final** (el que cuenta para el criterio de aprobación).

**`pass@k` por escenario (umbral de resolución: 0.5):**

| Escenario | Dificultad | Éxitos/Intentos | `pass@k` | ¿Resuelto? |
|---|---|---:|---:|:---:|
| `study-with-key` | easy | 5/5 | 1.00 | ✅ |
| `color-locks` | medium | 5/5 | 1.00 | ✅ |
| `apartment-keys` | medium | 4/5 | 0.80 | ✅ |
| `library-search` | hard | 4/5 | 0.80 | ✅ |
| `office-sequence` | hard | 4/5 | 0.80 | ✅ |
| `extreme-archive` | extreme | 4/5 | 0.80 | ✅ |

Los 6 escenarios quedan resueltos.

**Accuracy global y por dificultad** (30 evaluaciones = 6 escenarios × 5
intentos):

| | Evaluados | Logrados | Accuracy |
|---|---:|---:|---:|
| Global | 30 | 26 | 0.87 |
| `easy` | 5 | 5 | 1.00 |
| `medium` | 10 | 9 | 0.90 |
| `hard` | 10 | 8 | 0.80 |
| `extreme` | 5 | 4 | 0.80 |

**Costo y latencia:** 428 tool calls del agente principal (63 con error),
más 88 tool calls de workers delegados (8 con error, solo en
`extreme-archive`); 1.647.536 tokens de entrada / 34.920 de salida del
principal; duración total 643 s (~10,7 min), promedio 21,4 s por intento.

**Análisis de errores** (`eval/error_categories.py` sobre esta misma corrida,
30 intentos evaluados; un intento puede caer en más de una categoría):

| Categoría | Intentos afectados |
|---|---:|
| `violacion_estado` (ítem/destino/dirección inválidos) | 14 |
| `accion_redundante` (acción u observación repetida sin progreso) | 8 |
| `limite_iteraciones` | 7 |
| `planificacion_orden_incorrecto` | 1 |
| `presion_contexto` | 1 |
| `terminacion_prematura` | 1 |
| `id_inventado` | 0 |

`violacion_estado` y `accion_redundante` son, en su gran mayoría,
**recuperables**: aparecen en intentos que igual llegaron a
`goal_achieved: true` (p. ej. `apartment-keys` probando `use` en un cuarto
sin la puerta visible, `office-sequence` fallando una dirección de `go` y
corrigiendo con la salida que devuelve el error). Son ineficiencia, no
fallo de objetivo — coherente con la dimensión "recuperación" que también
usa la rúbrica cualitativa descartada (Sección 2).

`limite_iteraciones` (7/30) concentra el costo real: aparece en las 3 fallas
de `office-sequence` (dos con `planificacion_orden_incorrecto` combinado) y
en la falla de `apartment-keys`, siempre por agotar el presupuesto en
reintentos bloqueados por la guardia anti-repetición o navegación perdida,
no por falta de información. La única falla de `library-search` es la
excepción: `terminacion_prematura` — el agente devolvió una respuesta final
sin haber conseguido `llave_grabada`, sin agotar `max_iterations`.

La única falla de `extreme-archive` es cualitativamente distinta: intentó
`use`/`take` directamente sobre `expediente_1042`, `expediente_1387`, etc.
(los IDs de la estantería) en vez de llamar a `research_documents` sobre la
colección completa como indica el prompt — 12 `violacion_estado` + 20
`accion_redundante` en un solo intento, y termina con `presion_contexto`
(116.808 tokens de entrada sin alcanzar el goal). Es el único intento donde
el modelo no siguió la instrucción de delegar sobre colecciones grandes.

#### Sin scratchpad (`use_m3_scratchpad=False`) — ablation

`eval/results/final/20260823T221012501252Z.json`. Mismo comando, misma
config, con una única diferencia respecto a la corrida de arriba:
`use_m3_scratchpad=False`.

**`pass@k` por escenario:**

| Escenario | Dificultad | Éxitos/Intentos | `pass@k` | ¿Resuelto? |
|---|---|---:|---:|:---:|
| `study-with-key` | easy | 5/5 | 1.00 | ✅ |
| `color-locks` | medium | 5/5 | 1.00 | ✅ |
| `apartment-keys` | medium | 5/5 | 1.00 | ✅ |
| `library-search` | hard | 3/5 | 0.60 | ✅ |
| `office-sequence` | hard | 5/5 | 1.00 | ✅ |
| `extreme-archive` | extreme | 4/5 | 0.80 | ✅ |

**Accuracy global:** 27/30 (0.90) — no se recalculó el desglose por
dificultad para esta corrida.

**Costo y latencia:** 1.968.058 tokens de entrada, duración total 794 s
(~13,2 min). No se corrió `eval/error_categories.py` sobre esta corrida —
el desglose por categoría de error de arriba es específico de la corrida
con scratchpad.

**Comparación directa:** apagar el scratchpad dio accuracy global
*levemente mejor* (0.90 vs. 0.87) y resolvió mejor `office-sequence` y
`apartment-keys`, pero empeoró `library-search` (0.80 → 0.60) y costó más
tokens (+320.522) y más tiempo (+151 s). El scratchpad **no** es la
configuración oficial por ser estrictamente mejor en accuracy — es la que
se usa para el criterio de aprobación, y el análisis completo de este
trade-off (por qué ayuda en unos escenarios y no en otros) está en el
Experimento 6 (Sección 4).

---

## 4. Experimentos

Los Experimentos 1 y 2 se corrieron sobre un checkpoint intermedio del
desarrollo: una versión anterior del system prompt (antes de agregar
`research_documents` y sus instrucciones), con el scratchpad y la guardia
anti-repetición ya activos. Punto de partida (un run por escenario,
`max_iterations=25`):

| Escenario | Dificultad | Goal | Pasos | Óptimo | Exceso | Duración | Tokens in/out |
|---|---|---|---:|---:|---:|---:|---:|
| `study-with-key` | easy | ✅ | 6 | 3 | +3 | 6,95 s | 13.551 / 351 |
| `color-locks` | medium | ✅ | 17 | 11 | +6 | 18,38 s | 50.471 / 1.069 |
| `apartment-keys` | medium | ✅ | 10 | 7 | +3 | 7,39 s | 18.974 / 452 |
| `library-search` | hard | ✅ (con `max_iterations=25`) | 17 | 7 | +10 | 22,12 s | 69.027 / 1.174 |
| `office-sequence` | hard | ✅ (con `max_iterations=25`) | 22 | 13 | +9 | 22,22 s | 66.844 / 1.116 |

Con `max_iterations=20` (misma config), `library-search` y `office-sequence`
fallaban por límite de iteraciones, no por falta de información ni de
ítems: ya tenían el ítem correcto y el destino correcto en
inventario/vista, el límite cortaba antes del `use` final — es exactamente
lo que compara el Experimento 2.

### Experimento 1 — Instrucción explícita de consultar el scratchpad

**Contexto:** el scratchpad (Sección 1, subsección 3) ya estaba construido y
presente en el system prompt, pero eso no garantiza que el modelo lo use —
puede seguir adivinando IDs, inventario o ubicación en vez de mirar los
hechos ya confirmados que tiene justo ahí.

**Qué se cambió:** una única línea del system prompt, encendida/apagada —
pedirle explícitamente al modelo que consulte el `Scratchpad M3` antes de
cada tool call y que no adivine IDs/inventario/ubicación cuando el
scratchpad no los confirma. El scratchpad en sí estaba presente y activo en
ambas corridas; lo único que cambia es si el prompt se lo recuerda.

**Qué se buscaba mejorar:** que el agente dejara de repetir acciones basadas
en suposiciones (por ejemplo, usar un objeto que cree tener sin haberlo
tomado todavía) cuando la información real ya estaba disponible.

**Qué se mejoró** (`color-locks`, `apartment-keys`, un run por condición):

| Condición | `color-locks` | `apartment-keys` |
|---|---|---|
| Sin la instrucción (corrida 5) | ❌ (20 pasos, se traba repitiendo `use` de `llave_roja` sin haberla tomado) | ✅ (13 pasos) |
| Con la instrucción (corrida 6) | ✅ (13 pasos) | ✅ (12 pasos) |

`color-locks` pasó de fallar a resolverse; `apartment-keys` ya resolvía en
ambos casos, pero con menos pasos. Conclusión: tener el estado disponible no
alcanza, el prompt tiene que decirle al modelo *cuándo y para qué* usarlo —
es la intervención de mayor impacto observada por línea de prompt agregada.

**Limitación:** un run por condición, LLM no determinista. Evidencia
favorable, no causalidad confirmada.

### Experimento 2 — Presupuesto de iteraciones

**Contexto:** cada intento del agente tiene un tope de turnos
(`max_iterations`) antes de cortarse por límite. Un tope muy ajustado puede
cortar a un agente que está razonando correctamente, solo porque gastó
turnos en errores recuperables (reintentos, inspecciones de más) antes de
llegar al final.

**Qué se cambió:** `max_iterations`: `20` → `25`. Nada más — mismo prompt,
mismo scratchpad, misma guardia anti-repetición.

**Qué se buscaba mejorar:** que los dos escenarios `hard`
(`library-search`, `office-sequence`), que necesitan más pasos por tener más
objetos y salas intermedias, tuvieran margen suficiente para terminar.

**Qué se mejoró:**

| Escenario | `max_iterations=20` (corrida 7) | `max_iterations=25` (corrida 8) |
|---|---|---|
| `library-search` | ❌ — llega con `llave_grabada` en inventario y la puerta identificada, pero el límite corta antes de `use(llave_grabada, puerta_principal)` | ✅ — 17 pasos |
| `office-sequence` | ❌ — cumple el orden `documento antes que puerta`, pero el límite corta antes del `use` final | ✅ — 22 pasos |

Ambos escenarios pasaron de fallar a resolverse. En los dos casos el fallo
con `20` no era de razonamiento ni de estado — el agente ya tenía el ítem
correcto y el destino correcto, era puramente de presupuesto. Cinco
iteraciones extra alcanzaron sin degradar los tres escenarios más simples.

**Aumentar `max_iterations` no es una solución general.** Solo ayuda cuando
el agente está bien encarado y le faltan pasos para llegar — no cuando el
enfoque está mal desde el principio. El único intento fallido de
`extreme-archive` en la corrida oficial (Sección 3.1, análisis de errores)
es el ejemplo contrario: el agente ignoró la instrucción de usar
`research_documents` e intentó `use`/`take` directamente sobre los IDs de
la estantería desde el arranque — ahí más iteraciones no habrían ayudado,
solo habría gastado más tokens repitiendo el mismo enfoque equivocado (y de
hecho terminó cortado por `presion_contexto`, no por `max_iterations`).

**Limitación:** mismo caveat de n=1 por condición. Además, `25` fue elegido
por prueba y ajuste, no por un análisis de cuántas iteraciones "recuperables"
necesita en el peor caso cada mecánica del dataset.

### Experimento 3 — Delegar la lectura de documentos a un sub-agente (`extreme-archive`)

**Contexto:** `extreme-archive` tiene 20 expedientes. Leerlos todos con
`examine` desde el loop principal ocupa ~16K tokens, que no entran cómodos
en el contexto de un modelo chico junto con el resto de la conversación.
`research_documents` (Sección 1, subsección 2) se construyó para que un
sub-agente aislado lea por lotes y devuelva solo un resumen compacto.

**Qué se cambió:** disponibilidad de la tool `research_documents`: `False`
(el agente principal debe usar `examine` sobre cada expediente, uno por
uno) → `True` (puede delegar la lectura de la colección completa en una
sola llamada).

**Qué se buscaba mejorar:** que el agente pudiera encontrar
`llave_archivo` (mencionada en uno de los 20 expedientes) sin agotar el
contexto ni las iteraciones re-leyendo documentos de a uno.

**Qué se mejoró:**

| Configuración | Resultado |
|---|---|
| Sin `research_documents` | ❌ — el modelo decide terminar antes de agotar `max_iterations`, no llega a examinar los 20 expedientes ni a encontrar `llave_archivo` |
| Con `research_documents` | ✅ — 5 pasos del principal, 1 llamada a `research_documents`, 4 workers, `llave_archivo` correctamente encontrada y usada |

Detalle de la corrida exitosa:

| Objetivo | Pasos principal | Llamadas a `research_documents` | Workers | Duración | Tokens totales (principal + workers) |
|---|---:|---:|---:|---:|---:|
| ✅ | 5 | 1 | 4 | 18,79 s | 32.596 / 1.847 |

Óptimo publicado: 4 tool calls del principal (no cuenta el costo de la
delegación, porque el enunciado no define un peor caso de fuerza bruta que
entre en 16K tokens de contexto).

**Nota de desarrollo:** la primera versión de esta tool tenía un bug — el
principal la llamaba con subconjuntos parciales de IDs (5, 2 y 1) en vez de
la colección completa, y se le pedía a cada worker cerrar con JSON libre;
varios cerraron con `Invalid JSON` sin hallazgos, y el principal terminó
re-inspeccionando a mano y agotando `max_iterations`. El rediseño (arriba)
separó dos responsabilidades que esa primera versión mezclaba: (1) la tool
obtiene determinísticamente el `examine` de cada ID del lote autorizado, sin
depender de que el worker decida bien qué inspeccionar, y (2) un LLM
sintetiza esas observaciones ya obtenidas mediante
`structured_call(..., ResearchReport)`, con el mismo mecanismo de
validación Pydantic + reintentos de M2.

**Conclusión:** la delegación por lotes es la diferencia entre resolver y no
resolver este escenario con este agente. El costo es real: la versión
rediseñada sumó 20 observaciones de worker (16.143 tokens de entrada de
workers) además de los tokens del principal — más caro y más lento que un
escenario de tamaño normal, pero evita el corte por contexto.

**Limitaciones conocidas de la delegación:**

- Los workers son de solo observación pero comparten `world.revealed`, así
  que un `examine` de un worker sí puede revelar un objeto que el principal
  después toma — es el comportamiento buscado, pero exige que ningún worker
  ejecute acciones que muten el mundo.
- Los tokens de workers no forman parte del `AgentResult` del principal; el
  runner los reporta por separado (`worker_input_tokens`, etc.) y hay que
  sumarlos a mano para el costo total.
- El lote fijo de 5 documentos reduce el contexto de cada worker, pero no
  garantiza que cualquier documento individual entre en el modelo elegido.
- La estrategia reduce texto y preserva hechos observados, pero no sustituye
  razonamiento: no garantiza que el principal elija bien entre pistas
  ambiguas dentro de lo que el worker reporta.

### Experimento 4 — Planificación explícita antes de actuar

**Contexto:** dos fallas distintas motivaron este experimento. En
`color-locks`, el modelo a veces respondía con una narración de texto
completa en vez de llamar a una tool, porque el mensaje inicial ya describe
la escena en detalle. En `office-sequence`, un goal con orden obligatorio,
el agente sin plan tendía a perder ese orden reaccionando turno a turno.

**Qué se cambió:** `use_m3_planner`: `False` → `True` — antes de observar el
mundo, se fuerza una única llamada estructurada (`structured_call`) que
genera un plan de pasos en lenguaje natural, inyectado en el system prompt.
(Junto con el flag se agregaron dos líneas de prompt menores relacionadas —
ver limitación.)

**Qué se buscaba mejorar:** que el modelo respondiera siempre con una acción
estructurada desde el primer turno en vez de texto libre (`color-locks`), y
que tener un plan ordenado ayudara a respetar la secuencia obligatoria de
`office-sequence`.

**Qué se mejoró** (`pass@k` oficial, k=5, mismos 6 escenarios, antes/después
de activar el planner):

| Escenario | Sin planner (baseline) | Con planner |
|---|---:|---:|
| `color-locks` | 2/5 (0.40) ❌ no resuelto | 3/5 (0.60) ✅ resuelto |
| `office-sequence` | 2/5 (0.40) ❌ no resuelto | 2/5 (0.40) ❌ no resuelto |

El planner resolvió `color-locks`: forzar una respuesta estructurada desde
el primer turno saca al modelo del modo "responder con narración". No
alcanzó para `office-sequence` — mismo `pass@k` antes y después. Diagnosticar
por qué (el agente examina la caja fuerte del archivo, ve dos ítems y toma
solo uno) llevó al Experimento 5.

**Limitación:** no se puede atribuir el resultado de `color-locks`
exclusivamente al planner en sí — junto con el flag se sumaron también dos
líneas de prompt sin relación directa con la planificación (instrucción
anti-adivinanza sobre qué llave abre qué cerradura, recordatorio de hacer
`look` después de cada `go`). No se ablacionó cada cambio por separado.

### Experimento 5 — Aviso determinístico de objetos pendientes en el scratchpad

**Contexto:** el Experimento 4 no resolvió `office-sequence`. Diagnóstico:
el agente examina la caja fuerte del archivo, ve dos objetos revelados
juntos (`documento_confidencial` y `llave_maestra`), y sigue camino
llevándose solo uno. Pedirle por prompt que "tome todo lo revelado" no
bastó de forma sostenida entre corridas repetidas — el modelo podía dejar
de aplicarlo bajo presión de iteraciones.

**Qué se cambió:** `_pending_items()` en el scratchpad (una función nueva en
`student_framework/m3_scratchpad.py`): `False` (sin aviso) → `True` (con
aviso) — a partir de las tool outputs ya observadas, detecta objetos
revelados en un contenedor chico (≤5 ítems) que todavía no están en el
inventario, y lo reinyecta como advertencia en el system prompt en cada
turno hasta que se toman. Ambas condiciones ya tenían el planner activo y
las mismas instrucciones de prompt sobre tomar objetos revelados; la
diferencia es únicamente si además existe este aviso determinístico de
código.

**Qué se buscaba mejorar:** que el agente no "se olvidara" de tomar uno de
dos objetos revelados juntos, de forma sostenida entre corridas repetidas
(no solo una vez).

**Qué se mejoró** (`pass@k` oficial, k=5, planner activado en ambas
corridas):

| Escenario | Sin el aviso | Con el aviso |
|---|---:|---:|
| `office-sequence` | 2/5 (0.40) ❌ no resuelto | 4/5 (0.80) ✅ resuelto |
| resto de escenarios | resueltos, sin cambios | resueltos, sin cambios |

Las instrucciones de prompt por sí solas no habían sostenido el
comportamiento de forma confiable (se probaron primero sin este cambio y el
olvido reapareció); el aviso determinístico —reinyectado en cada turno
hasta que el ítem se toma, en vez de una instrucción de una sola vez— sí lo
sostuvo, en pruebas puntuales repetidas (`manual_run.py`, 3/3) y en la
corrida oficial.

**Limitación:** el aviso se acompañó de tres instrucciones de prompt
adicionales (volver a `examine` tras abrir un contenedor con `use`; tomar
todo lo revelado antes de salir de la sala; no repetir una llamada ya
bloqueada por la guardia); no se aisló cuánto aportó cada una por separado.
La falla restante de `office-sequence` en la corrida final (1/5) ya no es
por este motivo: el agente terminó tomando ambos ítems pero agotó las
iteraciones en reintentos de `go` bloqueados por la guardia antes de volver
a la puerta principal (Sección 3.1, análisis de errores).

### Experimento 6 — Scratchpad activado vs. desactivado (ablation completo)

**Contexto:** con el resto de la configuración ya definida (planner,
guardas, tope de iteraciones), quedaba una pregunta abierta: el scratchpad
agrega tokens en cada turno — ¿aporta lo suficiente en todos los escenarios
como para justificar ese costo, o solo en los que motivaron su diseño?

**Qué se cambió:** `use_m3_scratchpad`: `True` → `False` — el bloque de
estado determinístico deja de generarse e inyectarse por completo (a
diferencia del Experimento 1, que solo variaba si el prompt *pedía*
consultarlo, sin apagar el mecanismo en sí). Corrida `pass@k` oficial
completa (k=5, 6 escenarios) en ambas condiciones.

**Qué se buscaba mejorar (o, en este caso, verificar):** si apagar el
scratchpad degradaba la accuracy general, dado su costo en tokens — es
decir, confirmar que vale la pena mantenerlo activo en la config oficial.

**Qué se mejoró (y qué no):** tabla completa (`pass@k` por escenario,
accuracy global, tokens, duración) en la Sección 3.1, "Con scratchpad vs.
sin scratchpad". Resumen: apagar el scratchpad dio accuracy global
*levemente mejor*, y `office-sequence` — el escenario que motivó el
Experimento 5 — mejoró (4/5 → 5/5) sin él. Al mismo tiempo, `library-search`
empeoró (4/5 → 3/5) y el costo subió: sin el resumen compacto, el modelo
depende del historial crudo completo (`max_history_messages=200`), más
verboso por turno que el bloque de scratchpad.

Las dos fallas nuevas de `library-search` comparten una causa concreta:
en ambas, el agente nunca examinó los libros de la estantería uno por uno
(el paso que revela `llave_caja` dentro de `libro_sermones`) — en una
probó `use` de cada libro directamente sobre la puerta sin haberlo tomado
ni examinado, en la otra se quedó repitiendo observaciones ya hechas sin
avanzar a examinar ningún libro individual. Es la misma mecánica que el
scratchpad, cuando está activo, ayuda a sostener (recordar en qué punto de
una secuencia de exploración larga se está parado).

**Conclusión:** el scratchpad no es una mejora estrictamente positiva para
todos los escenarios de este dataset con este modelo — ayuda en exploración
secuencial larga de un solo lugar (`library-search`, 8 objetos a revisar en
la misma sala) y es neutral o levemente negativo en escenarios donde el
historial crudo ya alcanza (multi-sala con pocos objetos por sala,
`apartment-keys`/`office-sequence`). El costo en tokens de mantenerlo
siempre activo es real y medible.

**Limitación:** una sola corrida `pass@k` (k=5) por condición; varias de las
diferencias (`apartment-keys` y `office-sequence`, 4/5 vs 5/5) son de un solo
intento y están dentro del rango esperable de ruido de un LLM no
determinista — no alcanza para afirmar causalidad en esos dos casos
específicos. La diferencia en `library-search` (4/5 vs 3/5, con una causa
identificada y consistente en las dos fallas) es la más interpretable de
las tres.

---

## 5. Limitaciones y qué construirían a continuación

**Limitaciones actuales:**

**a. Manejo de la ventana de contexto — bug real, workaround, no fix de raíz.**

`_build_sliding_window` (M2, `student_framework/agent.py`) recorta el
historial por cantidad de mensajes, sin respetar que un turno del LLM con
`tool_calls` y sus `toolResult` correspondientes son un grupo atómico para
Bedrock. Con `max_iterations=25`, el historial puede acercarse al límite
por defecto de `max_history_messages=50`, y si el recorte cae en medio de
ese grupo, Bedrock devuelve
`ValidationException: "toolResult blocks... exceeds... toolUse blocks"`
y la corrida completa aborta sin producir ningún resultado. Se mitigó
subiendo `max_history_messages` a 200 en `M3_AGENT_CONFIG` (da margen
amplio para que el recorte prácticamente nunca se dispare con este
dataset), pero es un workaround: no reescribe `_build_sliding_window` para
que respete los grupos atómicos, y con escenarios de más pasos que los de
este dataset podría volver a aparecer. Separado de esto —y sí resuelto de
raíz—, `extreme-archive` (~16K tokens en 20 expedientes) no entra en el
historial principal por volumen, no por recorte: para eso está la
delegación a `research_documents` (Sección 1, Experimento 3), que evita
que esa prosa llegue al contexto del agente principal.

**b. Resultado contraintuitivo del ablation de scratchpad (Experimento 6).**

El ablation completo dio accuracy global levemente mejor sin el scratchpad,
aunque `library-search` empeoró — con una sola corrida `pass@k` por
condición. Varias de las diferencias por escenario (`apartment-keys`,
`office-sequence`) son de un solo intento y no alcanzan para afirmar
causalidad; solo la de `library-search` tiene una causa identificada y
consistente entre sus dos fallas.

**c. Dos modos de fallo residuales sin resolver.**

Identificados en el análisis de errores (Sección 3.1) pero no resueltos: en
`office-sequence`, un intento falló por agotar iteraciones en reintentos de
`go` bloqueados por la guardia anti-repetición después de haber conseguido
ambos ítems (no es un problema de estado ni de plan, es de
recuperación/navegación); en `extreme-archive`, un intento ignoró por
completo la instrucción de usar `research_documents` y probó `use`/`take`
uno por uno sobre los 20 IDs de la estantería, terminando por presión de
contexto — la instrucción de delegar no se sigue el 100% de las veces.

**d. `vault-combination` y `backtracking-vault` sin resolver.**

Los dos escenarios `extreme` no obligatorios no se resuelven. No se
investigó si el límite es del framework (prompt, presupuesto de
iteraciones) o del modelo elegido (Nova Lite): no se probó con Nova Pro
para aislar esa variable.

**e. La delegación de `research_documents` no se generalizó.**

Es específica para "colecciones de más de cinco documentos similares" vía
instrucción de prompt; no hay una heurística en código que decida cuándo
delegar.

**f. El costo de tokens de los workers delegados no se compensa con nada.**

La estrategia evita el desborde de contexto a costa de más llamadas al LLM
en total.

**g. El agente a veces no se detiene apenas cumple el goal.**

`check_goal` verifica el estado del mundo, no lo que dice el agente
(Sección 2) — el agente no recibe una señal explícita de "ya ganaste"
apenas abre la puerta principal. En algunos intentos, después de cumplir la
condición de éxito real, el agente sigue dando pasos de más (observando o
moviéndose) antes de emitir su respuesta final, en vez de detenerse en el
momento exacto en que el mundo ya cumple el objetivo. Esto no afecta
`goal_achieved` (que se calcula sobre el estado final, no sobre cuándo se
alcanzó), pero sí infla el conteo de pasos y tokens de esos intentos.
`[PENDIENTE: agregar un ejemplo concreto de una traza real con este
patrón.]`

**h. El agente base (sin planner) tiene dificultad para planificar tareas
de varios pasos.**

En `color-locks` y `office-sequence`, sin el planner explícito, el agente
fallaba por no descomponer el objetivo en pasos — en `color-locks`,
narrando en texto en vez de llamar tools; en `office-sequence`, perdiendo
el orden obligatorio del goal (ver Experimento 4, Sección 4). El planner
mitigó esto lo suficiente para que ambos escenarios queden resueltos en la
config oficial (Sección 3.1), pero es una limitación del agente base, no
algo resuelto en el loop ReAct en sí: sin esta pieza agregada, la
dificultad para planificar reaparece.

**i. El LLM-as-judge no es una medición exacta — hay que leer su porcentaje
con cuidado.**

Se auditó la salida de Kimi K2 Thinking sobre el "resultado final del juez"
(Sección 2: 40 intentos, los 8 escenarios) contra una revisión más profunda
hecha con un modelo más potente (GPT-5.6 Sol) — archivo
`eval/results/judge/20260824T022515999713Z/20260824T023928831731Z__moonshot-kimi-k2-thinking__all.json`.
Nota: esta corrida del agente es distinta de la que este informe cita como
evidencia cuantitativa oficial en la Sección 3.1 (30 intentos, 6 escenarios) —
ver la aclaración en Sección 2. Los hallazgos de esta auditoría son sobre el
comportamiento del juez, no del agente, así que se generalizan más allá de
esta corrida puntual:

- **Buena precisión, cobertura incompleta:** cuando el juez marcó un
  episodio como feedback, casi siempre lo era (96,9 % de precisión), pero
  omitió 58 de 151 episodios reales identificados por la referencia
  (61,6 % de cobertura) — sobre todo en secuencias largas de errores
  repetidos, donde tiende a resumir en vez de listar cada uno.
- **Falsos positivos puntuales:** en 3 de 40 trazas interpretó como
  feedback correctivo un reporte interno de `research_documents` que no
  iba dirigido a corregir la siguiente acción del agente — e interpretó
  salidas equivalentes de forma inconsistente entre trazas.
- **Al menos una inconsistencia interna:** en una traza marcó los 7
  episodios como incorporados pero el puntaje agregado no correspondía
  (según la rúbrica, debía ser el máximo).
- **El % que reporta el juez tiene un denominador parcial:** la tasa de
  incorporación se calcula solo sobre los episodios que el juez detectó,
  no sobre los reales — con la cobertura incompleta de arriba, esa tasa
  puede quedar sistemáticamente distorsionada respecto de una referencia
  más completa (66,7 % informado por Kimi vs. 81,46 % con la referencia
  más exhaustiva, sobre esa corrida de 40 intentos).

**Conclusión:** el juez sirve como diagnóstico orientativo — para encontrar
ejemplos concretos de feedback ignorado y comparar patrones entre
escenarios — pero no como ground truth, y su porcentaje agregado no debería
citarse sin esta aclaración.

**Qué construiríamos a continuación:**

1. Repetir esta auditoría de calidad del juez sobre la corrida oficial de
   30 intentos (la que cita el resto del informe), no solo sobre la de 40,
   para saber si estos mismos patrones de error aparecen ahí.
2. Repetir el Experimento 6 con más corridas por condición para separar señal
   de ruido en `apartment-keys` y `office-sequence`, y decidir si conviene
   activar el scratchpad solo para escenarios con muchos objetos en una
   misma sala (`library-search`, `extreme-archive`) en vez de siempre.
3. Investigar los dos modos de fallo residuales de la sección de
   limitaciones (el bloqueo de `go` en `office-sequence`, el bypass de
   `research_documents` en `extreme-archive`) — ninguno de los dos necesita
   una intervención nueva de diseño, probablemente alcance con ajustar la
   guardia anti-repetición y reforzar la instrucción de delegación.
4. Reescribir `_build_sliding_window` para recortar por grupos atómicos
   (turno de `tool_calls` + sus `toolResult`), no por cantidad de mensajes —
   arreglo de raíz del bug de Bedrock en vez del workaround de
   `max_history_messages=200`.
5. Repetir los Experimentos 1-3 con 3-5 corridas por condición, y ablacionar
   por separado cada componente de los paquetes de los Experimentos 4 y 5,
   para poder hablar de tendencia y de causalidad por pieza, no de una sola
   muestra ni de un paquete agregado.
6. Probar `vault-combination` y `backtracking-vault` con Nova Pro para
   separar "no lo resuelve este framework" de "no lo resuelve este modelo".
7. Una función que compare métricas entre corridas guardadas
   (`eval/results/**/*.json`) para no tener que leer JSON a mano al iterar
   sobre el prompt.
