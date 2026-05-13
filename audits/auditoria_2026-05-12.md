# Auditoría profunda — llm-eval-lab

**Fecha:** 2026-05-12
**Rama:** `main` (HEAD `c594174`)
**Alcance:** estructura, configuración, código `src/`, dashboard Streamlit, tests, CI, seguridad.

---

## TL;DR

El repo es funcional y los **215 tests pasan en local**, pero la pipeline CI **está rota desde hace al menos 4 commits**. Causas principales:

1. **189 errores de `ruff check`** (CI falla en `lint`).
2. **26 archivos requieren `ruff format`** (CI falla en formato).
3. **37 errores de `mypy`** (CI falla en type checking).
4. **Cobertura real: 39%** vs `--cov-fail-under=80` (CI falla en `test`).

Además hay varios bugs reales, riesgos de XSS almacenado en el dashboard, problemas de concurrencia en el evaluador DeepEval y deuda de robustez en parsing.

| Severidad | Hallazgos |
|---|---|
| 🔴 Crítico | 4 |
| 🟠 Alto | 7 |
| 🟡 Medio | 9 |
| 🟢 Bajo / mejora | 12 |

---

## 🔴 Críticos

### C1. CI completamente roto en `main`
- `gh run list` muestra **failure** en los últimos 4 commits.
- Cada job (`lint` → `test`) falla por motivos distintos (ruff, format, mypy, coverage).
- El commit `c594174` ("comprehensive quality improvements") **empeora** el problema: introduce dashboard de 1 800 líneas sin tests, bajando la cobertura del repo de ~80% (objetivo en `pyproject.toml`) a **39%**.
- **Acción:** decidir si bajar el umbral (`--cov-fail-under=40`), añadir tests de dashboard (con `streamlit.testing`), o excluir `src/dashboard/**` de cobertura (`pyproject.toml → [tool.coverage.run] omit`).

### C2. XSS almacenado en el dashboard (form de Test Cases)
`src/dashboard/pages/4_test_cases.py` permite que el usuario cree test cases vía formulario y los persiste en `datasets/*.jsonl`. Después, esos campos se renderizan en HTML crudo con `unsafe_allow_html=True` **sin escape**:

- L304: `f"<strong>Reference Answer:</strong> {c.reference}"` — `c.reference` viene del JSONL.
- L339: `f'<div ...><code>{k}</code>: {v}</div>'` — `metadata.k` y `metadata.v` provienen del JSONL.
- L295 / L453 (`pages/2_results.py`): `msg.get('content','')` de inputs multi-turn renderizado con `st.markdown(...)` (Markdown permite HTML inline en Streamlit).

Si un usuario (o test case importado) inyecta `<img src=x onerror=…>` o `<script>` quedará persistido en `datasets/<categoria>.jsonl` y se ejecutará en el navegador de cualquiera que abra el dashboard.

**Mitigación mínima:** usar `html.escape(...)` antes de interpolar valores que vienen de datos persistentes; o sustituir `unsafe_allow_html=True` por widgets nativos de Streamlit (`st.write`, `st.code`).

### C3. Bug de escritura en `datasets/*.jsonl`
`src/dashboard/pages/4_test_cases.py` L456-457:
```python
with open(target_file, "a") as f:
    f.write("\n" + json.dumps(new_case, ensure_ascii=False))
```
Si el archivo no termina en `\n` se genera correctamente. Pero si **ya termina en `\n`** se introduce una línea en blanco que luego rompe la lectura ordenada (técnicamente `load_dataset` la salta, pero el archivo deja de ser un JSONL canónico y diff-friendly). Tampoco se valida que `new_id` sea único frente a archivos cruzados (solo se compara con `all_cases` cargados, no se bloquea con lock).

**Fix:** `with open(target_file, "ab") as f: f.write(json.dumps(...).encode() + b"\n")`, y añadir validación de ID.

### C4. Llamada bloqueante dentro de coroutine en DeepEval
`src/evaluators/deepeval_evaluator.py` L204: `metric.measure(deepeval_tc)` es **sincrónico** y se invoca dentro de `async def evaluate(...)`. Eso **bloquea el event loop** durante toda la evaluación de cada métrica DeepEval, anulando la concurrencia configurada (`max_concurrent=5`).

Comparar con RAGAS (`single_turn_ascore`, awaitable). Soluciones:
- Usar la API async de DeepEval si existe (`metric.a_measure(...)`), o
- Envolver en `await asyncio.to_thread(metric.measure, deepeval_tc)`.

---

## 🟠 Altos

### A1. `run_id` colisionable
`runner.py:179`: `run_id = str(uuid.uuid4())[:8]`. Truncar UUID a 8 caracteres da ~16⁸ = 4·10⁹ valores; tras unos pocos miles de runs el riesgo de colisión es real y `results/<run_id>` se sobreescribiría silenciosamente. Usar uuid completo o `f"{timestamp}_{uuid4().hex[:8]}"`.

### A2. Uso peligroso de `asyncio.new_event_loop()` en Streamlit
`pages/1_run.py` L350-352:
```python
loop = asyncio.new_event_loop()
summary = loop.run_until_complete(runner.run(all_selected_cases))
loop.close()
```
Sin `try/finally`. Si `runner.run` lanza, el loop nunca se cierra (warning + descriptor file leak). Además al re-ejecutarse el script (Streamlit re-runs on every interaction) puede dejar loops huérfanos. **Fix:** `asyncio.run(runner.run(...))`.

### A3. Errores envueltos exponen tracebacks completos
`runner.py:101`: `last_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"`. El traceback se persiste en `report.json` y se renderiza en el dashboard. Si la excepción contiene la API key (algunas librerías la incluyen en errores HTTP), termina filtrada. **Fix:** log detallado server-side, en el campo `error` del modelo guardar solo `type(e).__name__: str(e)[:200]`.

### A4. Carga global de YAML en cada constructor
Cada evaluador/chatbot llama a `_load_config()` que **abre y parsea `config.yaml` cada vez**. En un run con N tests × 6 evaluadores, son cientos de lecturas redundantes. El cache de `shared.py` (`@st.cache_data`) solo aplica al dashboard. **Fix:** centralizar carga en `src/config.py` con `@functools.lru_cache`.

### A5. `_parse_scores` del LLM Judge es frágil
`llm_judge.py:144-153`: regex `rf"{criterion}\s*[:=]\s*(\d)"`. Solo captura **un dígito**, no escapa `criterion` (si fuera atacante-controlado), y exige que entre el nombre y los `:` solo haya espacios. Salidas reales como `Clarity (1-5): 4` no matchean. Mejor parser:
```python
pattern = rf"{re.escape(criterion)}\b[^\n:=]*[:=]\s*(\d+)"
value = min(int(match.group(1)), _MAX_SCORE_PER_CRITERION)
```

### A6. Detección de "rate limit" por substring inseguro
`runner.py:103`: `is_rate_limit = "429" in str(e) or "rate" in error_str`. Cualquier respuesta que contenga "rate" (p.ej. `"narrate"`, `"corporate"`, `"separated"`) cuenta como rate-limit y reintenta con backoff. Mejor: comprobar `isinstance(e, openai.RateLimitError)` o `e.status_code == 429`.

### A7. `B017` — `pytest.raises(Exception)` muy laxo
Ruff flagea dos sitios. Capturar `Exception` en tests oculta regresiones (otra excepción cualquiera satisface el assert). Usar la excepción concreta esperada (`ValueError`, `RuntimeError`, etc.).

---

## 🟡 Medios

### M1. Pydantic `TestCase` y `TestResult` chocan con pytest auto-discovery
Warnings de colección:
```
PytestCollectionWarning: cannot collect test class 'TestCase' because it has a __init__ constructor
```
Soluciones:
- Renombrar a `EvalCase` / `EvalRunResult` (limpio pero gran cambio).
- Añadir `__test__ = False` como atributo de clase (1 línea, retro-compatible).

### M2. Bug de "variable `e` reutilizada" (mypy)
`deepeval_evaluator.py:225-226` y `ragas_evaluator.py:255-256`:
```python
except Exception as e: ...
...
for m, e in errors.items():      # mypy: assignment outside except
    reasons.append(f"... {e}")
```
Funciona en runtime pero ofusca el código. Renombrar el segundo `e` a `err` o `msg`.

### M3. Tipado del dict `evaluators` en `pages/1_run.py`
mypy reporta 5 errores `Incompatible types in assignment` porque `evaluators = {}` se infiere como `dict[str, RuleBasedEvaluator]` tras el primer insert. Solución: anotar `evaluators: dict[str, BaseEvaluator] = {}`.

### M4. Tipado de `chatbot_mode` en `runner.py`
Mypy: `chatbot_mode` esperado `Literal["plain","rag"]` pero pasa `str`. Solución directa:
```python
mode: Literal["plain","rag"] = "rag" if self._chatbot.is_rag else "plain"
```

### M5. `metric_scores` aggregate sin métricas → reasons vacíos pero `passed=True`
`ragas_evaluator.py:248`: `all_passed = all(threshold_results.values()) if threshold_results else len(errors) == 0`. Cuando **todas las métricas fallan** con error, `metric_scores` queda vacío y `errors` no — devuelve `passed=False`, correcto. Pero si `metrics_to_run` es no-vacío y **ninguna métrica retorna score ni error** (caso patológico), `passed=True` con `score=None`. Borde poco probable pero asimétrico vs DeepEval.

### M6. `SEVERITY_ORDER` no incluye casos faltantes
`pages/1_run.py:452`: `sorted_failures = sorted(failures, key=lambda x: SEVERITY_ORDER.index(...))`. Si una severidad llegara como string fuera de `["critical","high","medium","low"]` (p.ej. data corrupta), `list.index` levanta `ValueError` y rompe la pestaña. Defensivo: `SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else len(SEVERITY_ORDER)`.

### M7. `package-lock.json` huérfano
Archivo en raíz, contenido vacío (`packages: {}`), repo Python puro. No referenciado. **Borrar** o añadir a `.gitignore`.

### M8. `langchain_openai` no es dependencia explícita
`ragas_evaluator.py:45` importa `langchain_openai`. No aparece en `pyproject.toml`. Funciona porque `ragas>=0.2` lo arrastra transitivamente, pero romperá si RAGAS cambia. **Añadir explícitamente** a `dependencies`.

### M9. `safety_evaluator` solo soporta inglés
Documentado en el docstring pero el repo se posiciona como herramienta general. Los patrones (`I can't`, `sorry`, `against my guidelines`) no cubren respuestas en español o multilenguaje. Considerar usar listas separadas por idioma o un classificador.

---

## 🟢 Bajos / mejoras

### L1. `random.uniform` para mock-latency (S311)
Para uso en mocks no-criptográficos es seguro. Añadir `# noqa: S311` o suprimir S311 globalmente en `mock_adapter.py`.

### L2. `f-strings` sin placeholders en `__main__.py`
3 ocurrencias (`F541`). Eliminar el prefijo `f`.

### L3. Imports desordenados en 16 archivos
`I001`. `ruff check --fix` lo resuelve.

### L4. `zip(...)` sin `strict=` (9 ocurrencias, `B905`)
Defensivo en Python 3.11+: `zip(a, b, strict=True)` para que falle si longitudes no coinciden.

### L5. Líneas >120 chars (86 ocurrencias, `E501`)
Decidir: mantener `line-length = 120` y arreglar, o subir el límite a `140` para los docstrings largos.

### L6. RAG-only metrics no se evalúan en plain mode pero igualmente se intentan
`ragas_evaluator._build_metric` recorre todos los constructores incluyendo `context_precision/recall` aunque no se vayan a usar. Micro-optimización: filtrar antes.

### L7. `pyproject.toml` no incluye `chromadb` en `[project.optional-dependencies]`
Está en `dependencies` aunque solo se usa en modo RAG. Mover a `optional-dependencies.rag` para instalación más ligera por defecto.

### L8. `consistency.py` similarity con `SequenceMatcher` ignora orden semántico
Para chatbots multilingües o respuestas parafraseadas el ratio es engañoso. Documentado como limitación, pero podría exponerse un evaluador alternativo con `sentence-transformers`.

### L9. README documenta features que requieren CI verde
La sección "Code quality: ruff + mypy + pytest-cov" induce a error: en realidad la CI lleva semanas roja.

### L10. `docs/`/`audits/` no existe en `main`
Buenas prácticas: tener un `docs/` con guías más allá del README. (Sugiero el actual archivo en `audits/`.)

### L11. Versión 0.3.0 sin CHANGELOG
Hay cambios significativos entre `0.x`. Añadir `CHANGELOG.md` aunque sea autogenerado por `git-cliff`.

### L12. Falta `LICENSE`
No hay archivo de licencia. Para un proyecto "para aprender" público, conviene MIT/Apache-2.0.

---

## Acciones recomendadas (orden sugerido)

**Sprint 1 — desbloquear CI (1 día):**
1. `ruff check --fix src/ tests/` + revisión manual de los 144 restantes.
2. `ruff format src/ tests/`.
3. Bajar `--cov-fail-under=40` o excluir `src/dashboard/**` y `src/__main__.py` (`[tool.coverage.run] omit`).
4. Resolver mypy: anotar `evaluators: dict[str, BaseEvaluator]`, castear `mode` a `Literal`, renombrar `e → err` en loops post-except.
5. Añadir `__test__ = False` a `TestCase`/`TestResult` en `src/runner/models.py`.

**Sprint 2 — seguridad y robustez (2-3 días):**
6. C2: escapar entradas en dashboard antes de interpolarlas en HTML.
7. C3: corregir append en JSONL y validar ID único.
8. C4: convertir `metric.measure` a `await asyncio.to_thread(...)`.
9. A2: reemplazar `loop = asyncio.new_event_loop()` por `asyncio.run(...)`.
10. A1: usar `run_id` completo o prefijar timestamp.

**Sprint 3 — calidad sostenida (semana):**
11. Tests de dashboard con `streamlit.testing.v1.AppTest` (cubrir cargas, formularios, navegación).
12. Centralizar `_load_config` con LRU cache.
13. A5: robustecer `_parse_scores`.
14. A6: rate-limit detection por tipo de excepción.
15. `CHANGELOG.md` + `LICENSE` + actualizar README sobre estado real de CI.
