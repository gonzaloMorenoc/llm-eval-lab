# Diseño: página Quality Gate en el dashboard

**Fecha:** 2026-08-15
**Estado:** pendiente de revisión
**Entrega:** 2 de 4 del bloque de front (UX del Run → **Quality Gate** → tendencias y
exportación → pulido visual)

## 1. Resumen

El motor del gate de regresión (`src/gate/`) existe desde la PR #11, pero solo se puede
usar desde la terminal. En el dashboard su única presencia es una tabla estadística cruda
al final de Compare Runs (`3_compare.py:411-459`): sin veredicto, sin baselines y sin
política. Quien use la herramienta por su interfaz no ve la funcionalidad que la
diferencia de cualquier otro cuaderno de evaluación.

Esta entrega añade una página que responde a tres preguntas: **¿este run pasa el gate y
por qué?**, **¿qué baselines tengo y cómo creo uno?** y **¿bajo qué reglas se está
juzgando?**

No añade lógica de evaluación: consume `load_baseline`, `build_baseline`, `evaluate_gate`
y `GatePolicy` tal cual, por el mismo motivo por el que la sección estadística de Compare
llama al motor en vez de recalcular — dos implementaciones podrían discrepar.

## 2. Objetivos y no-objetivos

### Objetivos

- Emitir el veredicto del gate para un run ya ejecutado contra un baseline elegido, con
  el motivo del fallo en lenguaje llano.
- Listar los baselines existentes y crear uno nuevo a partir de uno o varios runs.
- Mostrar la política vigente y permitir simular otros umbrales sin escribir en disco.
- Avisar cuando el baseline se creó con una versión distinta de los casos de prueba
  (`dataset_hash`), hoy escrito pero nunca leído.
- Presentar los estados de error del motor como estados explicados, no como excepciones.

### No-objetivos

- **Ejecutar evaluaciones.** La página no lanza runs; para eso está Run Evaluation, que
  además ya sabe hacer varias muestras. El equivalente exacto de `check` (ejecutar y
  comparar) sigue siendo cosa de la CLI y del CI.
- **Escribir `config/gate.yaml`.** La simulación es efímera. El dashboard no modifica un
  fichero versionado que decide si pasan las builds.
- **Bloquear el veredicto por deriva de dataset.** El aviso informa; no cambia el
  resultado. Hacerlo bloqueante obligaría a decidir antes si la CLI debe comportarse
  igual, y eso es una decisión de la CLI, no del dashboard.
- Borrar o renombrar baselines desde la interfaz.
- Sustituir la sección estadística de Compare Runs (ver §7).

## 3. Estructura de la página

`src/dashboard/pages/5_gate.py`, secciones apiladas en este orden:

1. **Selección** — desplegables de baseline y de run, más el estado de compatibilidad
   (deriva de dataset, métricas no comparables).
2. **Veredicto** — PASS/FAIL grande, número de muestras y lista de motivos de bloqueo.
3. **Tabla de métricas** — línea base, actual, regresión, IC 95%, p-valor, si está
   gateada y su veredicto individual. Mismas columnas que `render_gate_console`, para que
   quien vea el informe de CI reconozca la tabla.
4. **Política vigente** (expander) — umbrales actuales, su origen, y sliders de
   simulación.
5. **Crear baseline** (expander) — selección de runs, nombre y guardado.

El veredicto se ve sin ningún clic más allá de elegir baseline y run. La política y la
creación de baselines quedan debajo porque son apoyo, no el camino principal.

## 4. Módulos

Toda la lógica va en `src/dashboard/components/gate_view.py`, funciones puras sin
importar Streamlit, para que sean testeables directamente — igual que
`components/stability.py`. La página solo pinta.

| Función | Responsabilidad |
|---|---|
| `list_baselines(dir) -> list[BaselineSummary]` | Nombre, fichero, fecha, runs de origen, nº de casos, muestras. Ignora ficheros ilegibles registrando el motivo. |
| `dataset_drift(baseline, run_cases) -> DriftReport` | Si los casos del run difieren de aquellos con los que se construyó el baseline (§5). |
| `verdict_rows(verdict) -> list[dict]` | Filas de la tabla de métricas, ya formateadas. |
| `blocking_reasons(verdict) -> list[str]` | Por qué falla el gate, en frases legibles. |

`BaselineSummary` y `DriftReport` son modelos Pydantic pequeños en ese mismo módulo.

### `blocking_reasons`

El `GateVerdict` expone tres causas de fallo por separado y con semánticas distintas que
la página debe mantener distinguibles, porque en CI significan cosas distintas:

- `hard_rule_violations` — regla dura (nuevo fallo crítico, flakiness excesiva).
- `comparisons` con `breaches=True` — regresión significativa y relevante en una métrica
  gateada. Se redacta con sus números: «pass_rate cae 0.08, por encima del límite 0.05».
- `missing_gated_metrics` — **no es una regresión**: es un error de configuración que en
  CI produce exit 2, y fue el fallo que en la PR #11 dejaba pasar un PASS falso cuando un
  evaluador desaparecía. Se muestra aparte y con ese lenguaje.

## 5. Deriva de dataset

`build_baseline` calcula `dataset_hash` sobre **los casos de ese run**
(`baseline.py:94`), no sobre el dataset completo. De ahí se siguen dos cosas.

**Contra qué se compara.** No contra los ficheros de `datasets/`, sino contra los casos
con los que se ejecutó el run que se está juzgando, que viajan dentro de su propio
`report.json` (`TestResult.test_case`). Esa es la comparación que responde a la pregunta
real: el baseline se construyó con unos casos y el run con otros; si alguno cambió por el
medio, el veredicto compara cosas distintas. Leer `datasets/` respondería a otra pregunta
—si el disco cambió desde entonces— y marcaría como obsoleto todo baseline hecho con un
subconjunto de datasets, que es un uso normal: el aviso saltaría casi siempre y se
aprendería a ignorarlo, que es peor que no tenerlo.

**Cómo se calcula.** Se toman los ids del baseline, se buscan esos casos entre los del
run y se les aplica `compute_dataset_hash`. Si el resultado difiere del `dataset_hash`
guardado, algún caso cambió de contenido conservando su id.

Si el run no contiene todos los ids del baseline, el hash **no es comparable**: el del
baseline se calculó sobre el conjunto completo y un hash parcial no se le puede enfrentar.
En ese caso el informe dice exactamente eso —no comparable— en lugar de afirmar que hubo
deriva. Los casos ausentes ya los reporta `pair_cases` como `removed_case_ids`, y los
nuevos como `new_case_ids`; el aviso de deriva no los duplica.

## 6. Errores

Cada uno tiene su propio estado en la interfaz, con qué pasó y qué hacer:

| Situación | Origen | Presentación |
|---|---|---|
| Runs o baselines incomparables | `CompatibilityError` | Qué no cuadra (conjunto de métricas, modo del chatbot) y por qué impide juzgar |
| Métrica gateada no comparable | `verdict.missing_gated_metrics` | Error de configuración, no regresión; en CI sería exit 2 |
| Baseline ilegible o inválido | `BaselineError` | Fichero y motivo; el resto de baselines sigue disponible |
| Política inválida | `PolicyError` | Se indica y se usan los valores por defecto integrados |
| Directorio de baselines inexistente | — | Estado vacío que explica cómo crear el primero |
| Sin runs guardados | — | Estado vacío con enlace a Run Evaluation |
| Baseline de una muestra | `verdict.samples == 1` | Aviso de baja potencia estadística, como ya hace el reporter de consola |

Ninguno de estos casos debe llegar al usuario como traceback ni como el texto crudo de
una excepción.

## 7. Relación con Compare Runs

Compare Runs conserva su sección estadística. Equivale a `compare`: dos runs, sin
veredicto. Esta página equivale a `check`: baseline, con veredicto. La CLI mantiene los
dos verbos por la misma razón, y unificarlos obligaría a elegir un run como baseline
implícito, que es justo la ambigüedad que el fichero de baseline elimina.

Se añade un enlace en ambos sentidos.

## 8. Simulación de política

Los sliders construyen un `GatePolicy` nuevo con los valores elegidos y se vuelve a
llamar a `evaluate_gate`. No se toca disco. Se muestra el veredicto simulado junto al
real, dejando claro cuál es cuál y que la simulación no afecta al CI.

Se simulan `significance_level`, `min_effect_size` y el `max_regression` de las métricas
ya gateadas. Añadir o quitar métricas de la política no se simula: cambia la forma de la
política y su sitio natural es el YAML.

Coste: `n_resamples` por defecto es 10 000 y el bootstrap está sembrado (`seed`), así que
recomputar es determinista y del orden de décimas de segundo para los tamaños de dataset
actuales. Si resultara lento, la salida se memoiza por
`(baseline, run, valores de política)`.

## 9. Estrategia de tests (TDD)

`tests/test_dashboard_gate_view.py`, con `tests/gate_helpers.py` para los datos
sintéticos:

- `list_baselines`: lee varios, ordena, ignora el ilegible sin perder los demás,
  directorio inexistente devuelve lista vacía.
- `dataset_drift`: sin cambios no hay deriva; un caso con el mismo id y distinto texto sí
  la produce; **un baseline hecho con un subconjunto de datasets no la produce** (la
  regresión que motiva §5); un run al que le falta algún id del baseline se reporta como
  no comparable, no como deriva.
- `verdict_rows`: una fila por comparación, marca de gateada, regresión con signo.
- `blocking_reasons`: distingue las tres causas; un veredicto que pasa no da motivos;
  `missing_gated_metrics` no se redacta como regresión.

La página en sí (`5_gate.py`) no lleva tests unitarios, como el resto de páginas, y se
verifica conduciendo el dashboard en marcha: veredicto PASS, veredicto FAIL, aviso de
deriva, y creación de un baseline con sobrescritura.

## 10. Riesgos

- **Escribir en `baselines/`.** Son ficheros que se commitean y gobiernan el CI.
  Sobrescribir uno existente exige confirmación explícita en la interfaz, con el nombre
  del fichero afectado.
- **Un aviso de deriva ruidoso se ignora.** Mitigado por §5; el test del subconjunto de
  datasets es la garantía de que no vuelve.
- **Duplicar el motor.** No se replica ninguna decisión del gate: si el veredicto de la
  página y el de la CLI difirieran alguna vez, sería un fallo, no una diferencia de
  criterio.
