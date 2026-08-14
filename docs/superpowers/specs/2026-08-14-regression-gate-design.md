# Diseño: quality gate de regresión para CI/CD

**Fecha:** 2026-08-14
**Estado:** aprobado en conversación, pendiente de plan de implementación
**Fase:** 1 de 3 (gate → ingesta de trazas → evaluación de agentes)

## 1. Resumen

LLM Eval Lab pasa de ser un laboratorio de evaluación que se ejecuta manualmente a una
herramienta que impide que la calidad de un sistema LLM regrese sin que nadie lo note.
Caso de uso objetivo: un equipo cambia un prompt, un modelo o su pipeline RAG, abre un
PR, y el gate compara la calidad contra un baseline versionado y rompe el build si hay
una regresión estadísticamente significativa y relevante.

El problema técnico central que resuelve: dos runs de un LLM nunca producen los mismos
números, así que comparar medias sueltas genera falsas alarmas o deja pasar regresiones
reales. El gate usa muestreo múltiple y comparación estadística pareada por caso de
prueba.

## 2. Objetivos y no-objetivos

### Objetivos (v1)

- CLI con subcomandos (`run`, `baseline save`, `check`, `compare`) y códigos de salida
  aptos para CI.
- Baselines versionados en el repo del usuario, pequeños y diffeables.
- Motor estadístico: bootstrap pareado por caso, muestreo múltiple, métrica de
  flakiness, reglas duras para safety.
- Política de gate configurable en YAML.
- Salidas: tabla en consola (rich), Markdown para comentario de PR,
  `$GITHUB_STEP_SUMMARY`.
- GitHub Action compuesta reutilizable desde otros repos (`uses:
  gonzaloMorenoc/llm-eval-lab@v1`).
- Capa estadística añadida a la página Compare Runs del dashboard.

### No-objetivos (v1)

- Caché de respuestas o de juicios de evaluadores.
- Publicación en PyPI (paso posterior, tras estabilizar la CLI).
- Comentario automático en PRs vía API de GitHub (el usuario pega el Markdown o usa una
  action de terceros).
- Detección de mejoras (el gate solo vigila regresiones).
- Ingesta de trazas y evaluación de agentes (fases 2 y 3, cada una con su propio ciclo
  de diseño).

## 3. CLI

Nueva CLI construida con **Typer** (encaja con el uso existente de `rich`). El entry
point `python -m src` y el script `llm-eval-lab` se conservan; sin subcomando se
comportan como `run` (compatibilidad hacia atrás).

```
llm-eval-lab run [--samples N] [--evaluators rule_based,safety] [--datasets functional,safety]
llm-eval-lab baseline save <run_id> [--name main]
llm-eval-lab check --baseline main [--samples N] [--policy config/gate.yaml]
llm-eval-lab compare <run_a> <run_b>
```

- `run`: comportamiento actual + flags que sustituyen (sin eliminar) las variables de
  entorno `ACTIVE_PROVIDER`, `CHATBOT_MODE`, `USE_*`. `--samples N` ejecuta cada caso
  N veces (default 1).
- `baseline save`: convierte `results/<run_id>/report.json` en
  `baselines/<name>.json`.
- `check`: ejecuta la evaluación (equivale a `run`), compara contra el baseline y
  aplica la política. Códigos de salida: `0` pasa, `1` regresión detectada, `2` error
  de ejecución (fallo de API, baseline incompatible, config inválida).
- `compare`: comparación estadística entre dos runs guardados, sin veredicto ni código
  de salida especial. Reutiliza el mismo motor.

## 4. Baselines

### Formato

Un baseline es un `RunSummary` recortado, serializado a JSON en el repo del usuario:

```
baselines/
└── main.json
```

Contenido:

- Metadatos: `run_id` de origen, timestamp, provider, modelo, `chatbot_mode`,
  `dataset_hash` (hash SHA-256 del contenido concatenado de los JSONL usados),
  `metric_set` (lista de evaluadores/métricas activos), `samples` (nº de muestras por
  caso).
- Por caso: `id` del test case, severidad, categoría, scores por evaluador/métrica
  (media entre muestras), varianza entre muestras, pass/fail, latencia media.
- Sin respuestas completas: innecesarias para comparar y mantienen el archivo pequeño
  y diffeable. (Las respuestas persistidas en `results/` ya pasan por
  `redaction.py`; el baseline ni siquiera las incluye.)

### Ciclo de vida

1. Al mergear a `main`, el equipo ejecuta `run` + `baseline save --name main` (a mano
   o en un workflow) y committea el archivo.
2. Los PRs ejecutan `check --baseline main` contra ese archivo.

### Validación de compatibilidad (en frontera)

Antes de comparar, `check` valida baseline contra run actual:

- `dataset_hash` distinto → los casos se parean por `id`; los casos solo presentes en
  un lado se reportan como "sin baseline" / "eliminados" y quedan fuera de la
  estadística. El comportamiento ante casos nuevos lo decide la política
  (`new_cases`).
- `chatbot_mode` o `metric_set` incompatibles → exit `2` con mensaje claro. Nunca se
  comparan métricas que no existen en ambos lados.
- Baseline corrupto o con esquema desconocido → exit `2`. Validación con Pydantic.

## 5. Motor estadístico

Módulo nuevo `src/gate/` con funciones puras y datos inmutables:

```
src/gate/
├── __init__.py
├── models.py       # BaselineFile, BaselineCase, MetricComparison, GateVerdict (Pydantic)
├── statistics.py   # bootstrap pareado, flakiness — funciones puras sobre arrays
├── comparison.py   # pareo de casos entre runs, construcción de deltas
└── policy.py       # aplica la política y emite GateVerdict
```

### Diseño estadístico

- **Unidad de comparación: el test case**, presente en ambos runs (diseño pareado).
  Se comparan deltas por caso, no medias globales: mayor potencia estadística con
  datasets pequeños (43 casos hoy).
- **Muestreo múltiple:** con `--samples N`, el score de un caso es la media de sus N
  muestras; la varianza alimenta la métrica de **flakiness** por caso (fracción de
  muestras cuyo pass/fail difiere de la mayoría).
- **Bootstrap pareado como único método**, para scores continuos y para pass-rate:
  remuestreo con reemplazo de los deltas por caso → delta medio, intervalo de
  confianza al 95% y p-valor unilateral (solo interesa la regresión). Un solo método:
  fácil de explicar, de testear con seed fija, y sin supuestos de normalidad que los
  scores de LLM no cumplen. Implementación con numpy (ya disponible como dependencia
  transitiva); seed explícita en la firma para reproducibilidad.
- **Reglas duras que puentean la estadística:** un fallo nuevo en un caso de severidad
  `critical` (categoría safety o cualquier otra) rompe el gate directamente, haya o no
  significancia.

### Firmas orientativas

```python
def paired_bootstrap(
    deltas: Sequence[float], *, n_resamples: int = 10_000, seed: int
) -> BootstrapResult:  # mean_delta, ci_low, ci_high, p_value
    ...

def case_flakiness(sample_passes: Sequence[bool]) -> float:
    ...
```

## 6. Política de gate

Archivo YAML propio (`config/gate.yaml`) referenciable con `--policy`; si no existe,
se usan defaults empaquetados. Se mantiene separado de `config/config.yaml` porque la
política pertenece al repo del *usuario* del gate, no a la configuración de providers.

```yaml
gate:
  significance_level: 0.05
  min_effect_size: 0.05        # regresiones menores se ignoran aunque sean significativas
  n_resamples: 10000
  metrics:
    pass_rate:        {max_regression: 0.05}
    answer_relevancy: {max_regression: 0.10}
  hard_rules:
    no_new_critical_failures: true
    max_flakiness: 0.3          # flakiness media del run; por encima, exit 1
  new_cases: report_only        # report_only | fail
```

Una métrica rompe el gate cuando su regresión es a la vez estadísticamente
significativa (`p < significance_level`), mayor que `min_effect_size` y mayor que su
`max_regression`. El `min_effect_size` evita que con muchas muestras cualquier delta
minúsculo "significativo" rompa el build.

Dos reglas de interpretación:

- **Métricas no listadas en `metrics:`** se comparan y se reportan, pero nunca rompen
  el gate. Solo lo listado explícitamente es bloqueante.
- **Dirección de la métrica:** "regresión" respeta la semántica de cada métrica. Para
  métricas donde más es mejor (answer_relevancy, faithfulness…) regresión = bajada;
  para métricas donde menos es mejor (toxicity, hallucination, bias) regresión =
  subida. La dirección se deriva de la misma semántica de umbrales que ya usa la
  configuración existente (`>` vs `<` en los thresholds de RAGAS/DeepEval).

## 7. Salidas y GitHub Action

- **Consola:** tabla `rich` por métrica: delta, IC 95%, p-valor, significativo sí/no,
  veredicto; sección aparte para reglas duras y casos sin baseline.
- **Markdown:** `results/<run_id>/gate_report.md`, pensado para pegarse como
  comentario de PR. Nuevo `src/reporting/gate_reporter.py` siguiendo el patrón de los
  reporters existentes.
- **GitHub Actions:** si `GITHUB_STEP_SUMMARY` está definido, el Markdown se escribe
  también ahí.
- **Action compuesta** (`action.yml` en la raíz): inputs `provider`, `mode`,
  `baseline`, `samples`, `evaluators`, `policy`; las API keys llegan como env/secrets.
  Instala el paquete con pip y ejecuta `check`.
- **Control de coste:** el preset por defecto en CI usa solo evaluadores gratuitos
  (`rule_based`, `safety`, `consistency`); los LLM-based se activan explícitamente.
  Antes de lanzar, `check` imprime una estimación (nº de llamadas = casos × muestras,
  más llamadas de evaluadores LLM si están activos).

## 8. Dashboard

La página **Compare Runs** (`src/dashboard/pages/3_compare.py`) incorpora la capa
estadística reutilizando `src/gate/statistics.py` y `comparison.py`: IC y
significancia por métrica, flakiness por caso. No se crea página nueva.

## 9. Estrategia de tests (TDD)

- **Unitarios (`tests/test_gate_statistics.py`):** funciones puras con seed fija y
  distribuciones sintéticas. Casos: regresión clara → detectada; ruido puro → no
  detectada; efecto pequeño con muchas muestras → filtrado por `min_effect_size`;
  flakiness 0 y 1 en los extremos.
- **Unitarios (`tests/test_gate_policy.py`, `tests/test_gate_comparison.py`):**
  pareo de casos, validación de compatibilidad, reglas duras, veredictos.
- **Integración:** `MockChatbot` gana un modo de variabilidad configurable (rotación
  determinista entre variantes de respuesta) para simular no-determinismo y testear
  `check` de punta a punta sin API keys.
- **E2E (dogfooding):** el CI del propio repo ejecuta `llm-eval-lab check` con el
  provider mock contra un baseline fixture (`tests/fixtures/baseline_mock.json`),
  usando la propia Action.
- Cobertura: se mantiene el gate de ≥80% existente.

## 10. Hoja de ruta posterior

- **Fase 2 — ingesta de trazas:** generalizar la entrada del motor de comparación
  para aceptar resultados importados de trazas reales (OpenTelemetry/OpenInference,
  Langfuse) además de los generados por el runner: mismo motor, datos de producción.
- **Fase 3 — evaluación de agentes:** las trazas de agentes contienen trayectorias;
  los evaluadores de trayectoria (elección de tools, recuperación tras error, coste
  por tarea) entran como evaluadores normales sobre ese modelo extendido.

Cada fase tendrá su propio documento de diseño.

## 11. Decisiones y riesgos

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| Typer para la CLI | argparse | Mejor UX y ayuda generada; dependencia madura y ligera |
| Bootstrap pareado único | t-test/McNemar por tipo de métrica | Un solo método explicable y testeable; sin supuestos de normalidad |
| Baseline commiteado en git | Almacenamiento externo | Cero infraestructura; diffs visibles en PRs |
| Action en el mismo repo | Repo separado | Menos mantenimiento hasta tener usuarios |
| `gate.yaml` separado | Sección en `config.yaml` | La política pertenece al repo del usuario del gate, no a la config de providers |

**Riesgo principal:** coste por run en CI con métricas LLM activas. Mitigación: preset
gratuito por defecto y estimador de coste previo al lanzamiento. **Riesgo secundario:**
con `samples: 1` (default) la potencia estadística es baja; el reporte lo advierte y
recomienda `--samples 3` o más cuando detecta que ningún delta alcanza significancia
por falta de muestras.
