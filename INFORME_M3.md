# Informe Milestone 3 — Evaluación sobre un problema objetivo (sala de escape)

**Grupo:** Vidal – Podestá
**Materia:** Agentes Autónomos — Maestría en Inteligencia Artificial

> **Nota de estado.** Este informe combina evidencia real de corridas de
> desarrollo (documentadas en `evolución_agente_M3.md`) con una **corrida
> final pendiente de ejecutar** contra `amazon.nova-lite-v1:0` desde la
> máquina con credenciales AWS (WSL). Las secciones que dependen de esa
> corrida están marcadas explícitamente como `[PENDIENTE]`, con el comando
> exacto para completarlas. No se inventan números: donde falta evidencia, se
> dice que falta.

---

## 1. Aproximación

El framework de M1+M2 no se modificó en su contrato público: `build_agent`,
`register_tool` y `run` mantienen la misma interfaz. Lo que M3 agrega es
específico del problema, no un cambio de arquitectura del agente:

- **Registro de tools del mundo.** `eval/run.py::run_scenario` instancia un
  `World` por escenario y registra las tools fijas de `mia_world/tools.py`
  (`look`, `examine`, `take`, `use`, y `go` cuando el escenario es multi-sala)
  con `agent.register_tool(...)`, tal como indica el enunciado. Ninguna tool
  del mundo fue modificada.
- **Una tool propia: `research_documents`** (`student_framework/m3_research.py`).
  Se registra en todos los escenarios, pero solo es relevante cuando `examine`
  revela una colección de más de cinco documentos similares — el caso de
  `extreme-archive` (20 expedientes, ~16K tokens), que no entra en el
  contexto principal. Delega la inspección en lotes de cinco a un
  sub-agente aislado, que solo puede observar (no puede `take`/`use`/`go`),
  y devuelve un reporte compacto validado contra un schema Pydantic. El
  agente principal nunca ve la prosa completa de los 20 expedientes.
- **Scratchpad M3** (`student_framework/m3_scratchpad.py`). Un bloque de
  estado de trabajo (IDs observados, inventario, ubicación, salidas) que se
  reconstruye determinísticamente a partir de las tool outputs y se inyecta
  en el mensaje de sistema antes de cada llamada al LLM. No es memoria
  aprendida ni resumen: es una proyección determinística del estado ya
  observado, pensada para escenarios donde un único `look` no alcanza
  (`apartment-keys`, `office-sequence`).
- **Guardas anti-repetición.** `max_repeated_failures` y
  `max_repeated_observations` cortan una tool call si repite exactamente la
  misma acción fallida o la misma observación sin información nueva —
  evita que el agente entre en bucle cuando el LLM no actualiza su plan.

**Mismo agente en los tres niveles.** El criterio de aprobación pide el mismo
agente y el mismo system prompt en `easy`/`medium`/`hard`. `M3_AGENT_CONFIG`
en `eval/run.py` es una única configuración (un `system_prompt`, un
`max_iterations`, una guardia) que se aplica a todos los escenarios
seleccionados en la misma corrida; no hay ramas por escenario ni prompts
alternativos.

**Qué no se especializó:** las tools del mundo, el motor de estado
(`mia_world/`) y el dataset (`scenarios/`) son fijos y no se tocaron (ver
diff contra `tp_mia_agentes_2026-main` en el diagnóstico previo a este
informe — cero diferencias en `mia_world/`, `scenarios/`, `mia_agents/` y
`tests/conformance/`).

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
tokens de entrada/salida, latencia, y exceso de tool calls sobre el óptimo
publicado en `ENUNCIADO_M3.md` por escenario.

### Cualitativa — rúbrica manual (no LLM-as-judge)

Se optó por una rúbrica manual de 3 dimensiones (`estado_del_mundo`,
`recuperacion`, `planificacion_y_eficiencia`, puntaje 0-2 cada una) en lugar
de LLM-as-judge. Razón: LLM-as-judge suma costo de Bedrock, variabilidad (el
juez también es no determinista) y una dependencia de modelo adicional que
el enunciado no exige. La rúbrica completa, con la descripción de cada
puntaje, está en `evaluación_M3.md`.

Para que aplicarla no dependa de leer JSON a mano, se construyó
`eval/rubric.py`: genera la planilla vacía a partir de un reporte real
(`python eval/rubric.py template <reporte.json> <planilla.json>`), valida que
los puntajes cargados sean 0/1/2 y calcula promedios por dimensión y por
dificultad (`python eval/rubric.py summary <planilla.json>`). La lectura y el
puntaje de cada traza siguen siendo humanos; la herramienta solo evita errores
de transcripción y hace el cálculo reproducible.

### Análisis de errores — categorías

Siete categorías (definidas en `evaluación_M3.md`): ID inventado, violación
de estado, acción/observación redundante, límite de iteraciones,
terminación prematura, presión de contexto, y planificación/orden incorrecto
en un goal compuesto. `eval/error_categories.py` las aplica de forma
determinística sobre un reporte ya generado, anclándose en los mensajes de
error literales que emite `mia_world/tools.py` (fijo, no interpretación) y en
el `goal_reason` de `check_goal` (p. ej. `"no ocurrió en el orden requerido"`
para fallos de secuencia). No cubre los `worker_errors` de la delegación de
`extreme-archive`: esos son rechazos de grounding del sintetizador, una señal
de que la validación funciona, no un fallo del agente principal (ver
Experimento 3).

---

## 3. Resultados

### 3.1 Evidencia de desarrollo (prompts previos al final, documentada en `evolución_agente_M3.md`)

Estas corridas usaron versiones **anteriores** del system prompt (antes de
agregar `research_documents` y sus instrucciones), por lo que no son la
evidencia que cita el criterio de aprobación — se incluyen porque muestran
la trayectoria y porque son la base de los Experimentos 1 y 2 (sección 4).

| Escenario | Dificultad | Goal | Pasos | Óptimo | Exceso | Duración | Tokens in/out |
|---|---|---|---:|---:|---:|---:|---:|
| `study-with-key` | easy | ✅ | 6 | 3 | +3 | 6,95 s | 13.551 / 351 |
| `color-locks` | medium | ✅ | 17 | 11 | +6 | 18,38 s | 50.471 / 1.069 |
| `apartment-keys` | medium | ✅ | 10 | 7 | +3 | 7,39 s | 18.974 / 452 |
| `library-search` | hard | ✅ (con `max_iterations=25`) | 17 | 7 | +10 | 22,12 s | 69.027 / 1.174 |
| `office-sequence` | hard | ✅ (con `max_iterations=25`) | 22 | 13 | +9 | 22,22 s | 66.844 / 1.116 |

Con `max_iterations=20` (corrida 7), `library-search` y `office-sequence`
fallaban por límite de iteraciones, no por falta de información ni de ítems
(ver Experimento 2). Los tres escenarios `easy`/`medium` se sostienen sin
regresión en todas las corridas del checkpoint.

`extreme-archive` (corrida 11, ya con `research_documents` integrado):

| Objetivo | Pasos principal | Llamadas a `research_documents` | Workers | Duración | Tokens totales (principal + workers) |
|---|---:|---:|---:|---:|---:|
| ✅ | 5 | 1 | 4 | 18,79 s | 32.596 / 1.847 |

Óptimo publicado: 4 tool calls del principal (no cuenta el costo de la
delegación, porque el enunciado no define un peor caso de fuerza bruta que
entre en 16K tokens de contexto).

### 3.2 Corrida final (criterio de aprobación) — `[PENDIENTE]`

No existe todavía una corrida nativa de `eval/run.py` con la configuración
**final** (el `M3_AGENT_CONFIG` vigente, con `research_documents` y sus
instrucciones) sobre los 5 escenarios obligatorios. Es la brecha más
importante a cerrar antes de entregar: sin esto no hay evidencia fresca de
que "el mismo agente, en los tres niveles" (como pide el criterio de
aprobación) siga funcionando con el prompt que terminó evolucionando para
resolver `extreme-archive`.

`eval/run.py` ya quedó configurado para esto (`DEVELOPMENT_SCENARIOS` con los
5 escenarios obligatorios + `extreme-archive`, `RUN_KIND="final"`, y el
reporte ahora registra `metadata.llm_provider` para dejar constancia del
modelo usado). Falta ejecutarlo:

```bash
export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
export AWS_REGION="us-east-1"
# + credenciales AWS
cd scaffold/   # o la raíz del repo, según donde quede clonado en WSL
python eval/run.py
```

El reporte queda en `eval/results/final/<run_id>.json`. Con eso:

1. Completar la tabla de arriba con `run_kind=final` y `metadata.llm_provider.model == "amazon.nova-lite-v1:0"`.
2. `python eval/error_categories.py eval/results/final/<run_id>.json` → pegar `totals` acá.
3. `python eval/rubric.py template eval/results/final/<run_id>.json eval/results/final/<run_id>.rubric.json`, completar a mano leyendo la traza, y `python eval/rubric.py summary ...` → pegar `average_by_dimension` acá.

### 3.3 Fuera del criterio de aprobación

`vault-combination` y `backtracking-vault` (`extreme`, no obligatorios según
lo indicado en clase) no se resuelven hoy: última corrida, 0/2, con 53 tool
calls y 12 errores de tool combinados, sin usar delegación (no disparan la
condición de >5 documentos similares). No se investigó más a fondo porque no
son requisito de aprobación; quedan en Limitaciones (sección 5).

---

## 4. Experimentos

### Experimento 1 — Scratchpad: disponible vs. instrucción explícita de consultarlo

**Qué se cambió:** una única línea de diferencia en el system prompt: pedirle
al modelo que consulte el bloque `Scratchpad M3` antes de cada tool call y
que no adivine IDs/inventario/ubicación cuando el scratchpad no los confirma.
El scratchpad en sí (la construcción determinística del estado) estaba
presente en ambas corridas.

**Qué pasó** (`color-locks`, `apartment-keys`, un run por condición):

| Condición | `color-locks` | `apartment-keys` |
|---|---|---|
| Scratchpad sin instrucción (corrida 5) | ❌ (20 pasos, se traba repitiendo `use` de `llave_roja` sin haberla tomado) | ✅ (13 pasos, un `use` a distancia) |
| Scratchpad con instrucción (corrida 6) | ✅ (13 pasos) | ✅ (12 pasos) |

**Conclusión:** tener el estado disponible no alcanza; el prompt tiene que
decirle al modelo *cuándo y para qué* usarlo. Es la intervención de mayor
impacto observado por línea de prompt agregada.

**Limitación:** un run por condición, LLM no determinista. Evidencia
favorable, no causalidad confirmada.

### Experimento 2 — Presupuesto de iteraciones: `max_iterations=20` vs `25`

**Qué se cambió:** solo el tope de iteraciones, sin tocar prompt, scratchpad
ni guardia. Escenarios observados: `library-search`, `office-sequence`
(los dos `hard`).

**Qué pasó:**

| Escenario | `max_iterations=20` (corrida 7) | `max_iterations=25` (corrida 8) |
|---|---|---|
| `library-search` | ❌ — llega con `llave_grabada` en inventario y la puerta identificada, pero el límite corta antes de `use(llave_grabada, puerta_principal)` | ✅ — 17 pasos |
| `office-sequence` | ❌ — cumple el orden `documento antes que puerta`, pero el límite corta antes del `use` final | ✅ — 22 pasos |

**Conclusión:** en ambos casos el fallo con `20` no era de razonamiento ni de
estado — el agente ya tenía el ítem correcto y el destino correcto. Era
puramente de presupuesto: los errores recuperables intermedios (reintentos de
`use` sin tener el ítem, inspecciones de más) consumen iteraciones que además
cuentan turnos de LLM, no solo acciones de mundo. Cinco iteraciones extra
alcanzaron sin degradar los tres escenarios más simples.

**Limitación:** mismo caveat de n=1 por condición. Además, `25` fue elegido
por prueba y ajuste, no por un análisis de cuántas iteraciones "recuperables"
necesita en el peor caso cada mecánica del dataset.

### Experimento 3 — Investigación delegada vs. loop principal solo (`extreme-archive`)

**Qué se cambió:** en vez de que el agente principal use `examine` sobre cada
uno de los 20 expedientes (~16K tokens, no entra en el contexto de la mayoría
de modelos chicos), se agregó `research_documents`: delega la inspección en
lotes de 5 a un sub-agente de solo observación, que devuelve un reporte
compacto y validado.

**Qué pasó:**

| Configuración | Resultado |
|---|---|
| Loop principal solo (corrida 9, baseline) | ❌ — el modelo decide terminar antes de agotar `max_iterations`, no llega a examinar los 20 expedientes ni a encontrar `llave_archivo` |
| Delegación, primera integración (corrida 10) | Ver hallazgos en `evolución_agente_M3.md`; llevó a una corrección de guardia y de grounding |
| Delegación con guardia + grounding revisado (corrida 11) | ✅ — 5 pasos del principal, 1 llamada a `research_documents`, 4 workers, `llave_archivo` correctamente encontrada y usada |

**Conclusión:** la delegación por lotes es la diferencia entre resolver y no
resolver este escenario con este agente. El costo es real: la corrida 11
sumó 20 observaciones de worker (16.143 tokens de entrada de workers) además
de los tokens del principal — más caro y más lento que un escenario de
tamaño normal, pero evita el corte por contexto.

**Limitación conocida** (documentada en `evolución_agente_M3.md`): los
workers son de solo observación pero comparten `world.revealed`, así que un
`examine` de un worker sí puede revelar un objeto que el principal después
toma — es el comportamiento buscado, pero exige que ningún worker ejecute
acciones que muten el mundo. Los tokens de workers no forman parte del
`AgentResult` del principal; el runner los reporta por separado
(`worker_input_tokens`, etc.) y hay que sumarlos a mano para el costo total.

---

## 5. Limitaciones y qué construirían a continuación

**Limitaciones actuales:**

- No hay, al momento de escribir este informe, una corrida final consolidada
  de los 5 escenarios obligatorios con el prompt definitivo (sección 3.2).
  Es la limitación más importante y la primera en resolverse.
- Los tres experimentos son de una sola corrida por condición: el LLM es no
  determinista y no hay margen de error reportado. Una conclusión más sólida
  requeriría 3-5 corridas por condición, que no se hicieron por presupuesto
  de créditos de Bedrock (ver `nota_ceci.md`).
- La rúbrica cualitativa es manual y de un solo evaluador; no se midió
  acuerdo entre evaluadores (inter-rater reliability), algo relevante dado
  que las tres dimensiones dejan margen de interpretación.
- `vault-combination` y `backtracking-vault` (los dos `extreme` no
  obligatorios) no se resuelven. No se investigó si el límite es del
  framework (prompt, presupuesto de iteraciones) o del modelo elegido
  (Nova Lite): no se probó con Nova Pro para aislar esa variable.
- La delegación de `extreme-archive` no se generalizó: es específica para
  "colecciones de más de cinco documentos similares" vía instrucción de
  prompt; no hay una heurística en código que decida cuándo delegar.
- El costo de tokens de los workers delegados no se compensa con nada: la
  estrategia evita el desborde de contexto a costa de más llamadas al LLM en
  total.

**Qué construiríamos a continuación:**

1. Cerrar la corrida final y, con esos datos, decidir si `office-sequence`
   necesita algo más que ReAct + scratchpad — el enunciado sugiere
   explícitamente un experimento *planner explícito vs. ReAct puro* para
   goals compuestos, que no llegamos a hacer.
2. Repetir los experimentos con 3-5 corridas por condición para poder hablar
   de tendencia y no de una sola muestra.
3. Probar `vault-combination` y `backtracking-vault` con Nova Pro para
   separar "no lo resuelve este framework" de "no lo resuelve este modelo".
4. Una función que compare métricas entre corridas guardadas
   (`eval/results/**/*.json`) para no tener que leer JSON a mano al iterar
   sobre el prompt.
