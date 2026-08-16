# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Pass rate trend on the home page.** Every run is a point, oldest to newest,
  with one line per category plus an overall line — the same data the run history
  table already showed, in the shape that answers "are we getting better or
  worse?".
- **Downloads from the dashboard.** Results serves a run's `report.json` and
  `report.md` straight from disk plus a generated CSV (one row per test case);
  Compare Runs exports its statistical table. Both were previously reachable only
  by digging through `results/` in a file manager.
- `src/dashboard/components/export.py` — CSV builders. Uses the `csv` module
  rather than string joins because test case text contains commas, quotes and
  newlines, any of which would otherwise shift columns.

- **Quality Gate page in the dashboard.** The gate engine had been CLI-only: its
  only trace in the UI was an unlabelled statistics table at the bottom of
  Compare Runs. The page judges a stored run against a committed baseline and
  explains the verdict, keeping the three failure causes distinct — a hard rule,
  a breaching metric, and a gated metric that cannot be compared. That last one
  is a configuration error (exit 2 in CI), not a quality regression, and it is
  the case that produced a false PASS in the gate's first release; wording it
  like a regression in the UI would repeat that mistake. Baselines can be listed
  and created from the page, and the policy can be simulated — moving a threshold
  re-evaluates the verdict in memory and never writes `config/gate.yaml`.
- **`dataset_hash` finally has a reader.** Written into every baseline since the
  gate landed but never read, so a test case whose text changed while keeping its
  id was silently compared against a stale baseline entry. The page now warns.
  The hash is recomputed over the baseline's own case ids against the cases
  stored in the run's report — not against `datasets/` on disk, which would flag
  every baseline built from a subset of datasets and produce an alarm so frequent
  it would be ignored.
- **Multi-sample runs from the dashboard.** A `Muestras` slider in the sidebar
  runs the suite N times, and the Run page then reports *stability*: which cases
  passed in some samples and failed in others, with their per-sample pattern
  (`✅❌✅`). A case that contradicts itself is the one that breaks a build
  without anyone changing anything, and it was previously only visible by
  reading `--samples` output in a terminal.
- `src/runner/execution.py` — `run_samples()`, the single implementation of "run
  the suite N times and persist each report", now shared by the CLI's
  `--samples` and the dashboard so the two cannot drift apart.
- `src/dashboard/components/stability.py` — pure helpers behind the stability
  view (`unstable_case_rows`, `sample_pattern`, `stability_headline`).

### Changed
- **The dashboard's progress bar reflects real progress.** It was created at 0%
  and set to 100% after `asyncio.run(...)` returned, so it never moved while 43
  test cases ran. `EvalRunner.run()` now takes an optional
  `on_progress(completed, total)` callback — fired once per finished case, and
  counted across every sample — which the Run page renders as
  `Evaluando… 12/129 casos`. A callback that raises is logged and the run
  continues: a broken progress display must not discard an evaluation whose API
  calls are already paid for.
- **`list_runs()` no longer re-parses every report on each rerun.** Each page
  called it on every interaction and each `report.json` carries its full
  `results` array. Report parsing is now memoised on `(path, mtime)`; directory
  scanning stays uncached, so a new run still appears immediately and an
  overwritten report reloads at once — which a TTL cache could not do without
  either serving stale data or re-reading unchanged files.

### Fixed
- **`category_trend_chart` had never worked.** It passed `height=380` alongside
  `**_LAYOUT_DEFAULTS`, which already sets `height`, so it raised `TypeError` on
  any non-empty input. Nothing ever called it, so nothing ever noticed. It now
  also sorts chronologically itself: `list_runs()` returns newest first, so the
  obvious way to call it would have drawn time backwards and shown decline as
  improvement.
- **DeepEval dropped `answer_relevancy` whenever a test case had no
  `reference`.** The metric scores input-vs-output and never reads
  `expected_output`, so it was listed in `_REFERENCE_REQUIRED` by mistake. All
  12 safety cases lack a reference, which left them measured by bias/toxicity
  alone. Only `g_eval` genuinely needs a reference. The test that asserted the
  old behaviour (`test_resolve_metrics_no_reference_skips_relevancy`) has been
  inverted.
- **`runner.timeout_ms` was configured but never applied.** It sat in
  `config.yaml` and had a dashboard slider, yet no code read it, so a hung
  connection held its semaphore slot indefinitely. Chatbot calls are now
  wrapped in `asyncio.wait_for`; a non-positive value disables the deadline.
- **`load_config()` handed out the cached dict itself.** Any consumer that
  tweaked a value rewrote the configuration for the whole process — and
  `pages/1_run.py` does exactly that when overriding `max_concurrent`. Each
  caller now gets a deep copy.
- **Retry backoff slept after the final attempt.** With the default settings
  every exhausted call wasted 4s sleeping on its way out of the loop.
- **LLM judge persisted un-redacted exception text.** It catches its own
  errors, so it bypassed the runner's redaction and wrote raw provider
  messages — which can echo back the failing API key — into `report.json`.
- **Wrapping SDK errors in `RuntimeError` broke rate-limit detection.**
  `_is_rate_limit()` reads `status_code`, which the wrapper discarded. Adapters
  now raise `ChatbotAPIError` (preserving the status code) and the retry
  heuristics walk the `__cause__` chain.

- **Requested-but-unregistered evaluators vanished without a trace.** A case
  asking for `ragas` without `OPENAI_API_KEY` was scored by whatever remained,
  and a case whose evaluators were *all* missing was reported as a plain
  failure. `TestResult.skipped_evaluators` and `RunSummary.skipped_evaluators` /
  `unevaluated` now record it, and the run prints a warning. On the bundled
  datasets this surfaces that 31 of 43 cases request RAGAS.

### Added
- `src/redaction.py` — secret redaction and error summarization extracted from
  the runner so evaluators can reuse them without a circular import.
- `src/chatbots/errors.py` — `ChatbotAPIError` and `wrap_api_error()`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and pull
  request templates, and `dependabot.yml`.
- `llm-eval-lab` console entry point, plus `[project.urls]`, `authors`,
  `keywords` and `classifiers` in `pyproject.toml`.
- 13 tests covering the request deadline, backoff behaviour, wrapped-error
  classification, config copy semantics, and judge redaction.

- **The CI test job had been failing on `main` since `2da2a53`** — the commit
  titled "fix CI pipeline". It installed `.[dev]`, but
  `tests/test_dashboard_shared.py` imports `src.dashboard.components.shared`,
  which imports `streamlit` from the `dashboard` extra, so the job died during
  collection on every Python version. Now installs `.[dashboard,dev]`, which
  also means dashboard import errors are finally caught by CI.
- **A fresh install had a dead RAGAS evaluator.** `ragas` 0.4.3 (its latest)
  does `from langchain_community.chat_models.vertexai import ChatVertexAI`, a
  module `langchain-community` 0.4.2 removed, so `import ragas` raised
  `ModuleNotFoundError` and all 24 RAGAS tests errored during setup. Existing
  checkouts were unaffected only because they had 0.4.1 installed from before
  the upstream release. Pinned `langchain-community<0.4.2` until ragas drops
  the import.

### Changed
- CI matrix extended to Python 3.13, with pip caching, `cancel-in-progress`
  concurrency, and a read-only `permissions` block.
- `.gitignore` now covers `.DS_Store`, editor directories, and the remaining
  tool caches.
- `RuleBasedEvaluator` thresholds moved to a `rule_based` section in
  `config.yaml` (`min_response_length`, `max_latency_ms`) and are overridable
  per instance. The module docstring promised this was configurable while both
  values were hardcoded. The latency budget is deliberately separate from
  `runner.timeout_ms`: one marks a slow answer as failed, the other aborts the
  request.
- `DemoRAGChatbot._load_knowledge_base()` fetches existing document ids once
  instead of querying ChromaDB per line, so load time no longer scales with
  the size of the knowledge base.
- Dropped the unused `mode` parameter from `__main__._build_evaluators()`.

### Housekeeping
- 

### Added
- Regression quality gate for CI/CD: `src/gate/` (paired bootstrap statistics,
  baseline build/load, case pairing, policy + verdict), `llm-eval-lab` CLI with
  `run` / `baseline save` / `check` / `compare` subcommands (exit codes 0/1/2),
  `gate_report.md` + `$GITHUB_STEP_SUMMARY` output, composite GitHub Action
  (`action.yml`) with a mock-baseline dogfooding job in CI, statistical section
  in the Compare Runs dashboard page, and `config/gate.yaml` policy reference.
- `MockChatbot`/`MockRAGChatbot` accept an optional `transform` callable to
  simulate response drift in gate tests.
- `src/config.py` — process-wide LRU-cached YAML loader. Replaces 8 duplicated
  `_load_config()` helpers and avoids re-parsing `config.yaml` on every
  evaluator/chatbot/report constructor.
- `src/dashboard/components/shared.safe()` — HTML escape helper for any value
  that originates from user input, a persisted dataset, or a model response and
  is later interpolated into `unsafe_allow_html=True` Markdown.
- `src/dashboard/components/shared.append_jsonl()` — safe append-only writer
  that always produces canonical JSONL, regardless of whether the existing
  file already ends with a newline.
- `src/runner/runner._redact_secrets()` / `_summarize_error()` — strip
  recognisable API keys (`sk-…`, `gsk_…`, `AIza…`, `Bearer …`) and truncate
  exception messages before persisting them in `results/<run_id>/report.json`.
- `src/runner/runner._is_rate_limit()` / `_is_network_error()` — type-aware
  retry classifiers based on exception class names and `status_code`, with
  defensive substring matches as a fallback.
- New test modules: `test_config.py`, `test_dashboard_shared.py`,
  `test_runner_safety_helpers.py` (39 tests).
- `audits/auditoria_2026-05-12.md` — deep audit report with severity-tagged
  findings.
- `CHANGELOG.md`, `LICENSE`.

### Changed
- `run_id` is now `yyyymmddTHHMMSS_<8hex>` (sortable, ~3×10¹⁰ namespace)
  instead of `uuid4()[:8]` which had a real birthday-paradox collision risk on
  the `results/<id>/` directory layout.
- `DeepEvalEvaluator.evaluate()` runs `metric.measure(...)` via
  `asyncio.to_thread(...)` so it no longer blocks the event loop and the
  runner's `max_concurrent` limit applies as documented.
- `pages/1_run.py` swaps `loop = asyncio.new_event_loop(); …; loop.close()`
  for `asyncio.run(...)` so a failing run cannot leak an event loop on
  Streamlit re-runs.
- `LLMJudgeEvaluator._parse_scores()` now accepts `Clarity = 4`,
  `Clarity (1-5): 4`, `Clarity - 4`, and multi-digit scores; clamps to the
  `[0, max]` range.
- `pages/4_test_cases.py` uses `append_jsonl()` (see above) and escapes
  reference answers and metadata via `safe()`. Free-form text fields
  (`input`, `expected_behavior`, multi-turn `content`) now render through
  `st.text(...)` instead of `st.markdown(...)`.
- `pages/2_results.py` and `pages/1_run.py` likewise route untrusted text
  (model responses, dataset inputs) through `st.text(...)`.
- `pyproject.toml`: `line-length` raised to 140, `RUF001`/`RUF003`/`E501`
  globally ignored (em-dashes in didactic docstrings), `langchain-openai>=0.2`
  added to runtime deps, dashboard and `__main__` excluded from coverage.
- Centralised secret handling: `ragas_evaluator.py` now wraps the OpenAI key
  in `pydantic.SecretStr` for the LangChain wrappers.

### Fixed
- **CI**: full pipeline (ruff lint, ruff format, mypy, pytest with 80%
  coverage gate) is green for the first time since the dashboard was merged.
- **CVE-class stored XSS** in the Test Cases dashboard form: malicious values
  in `reference`, `metadata`, multi-turn `content`, etc. are no longer
  rendered as raw HTML.
- Malformed JSONL produced by the "Add New Test Case" form when the existing
  file already ended in `\n` (extra blank line) or did not (missing trailing
  newline on the appended entry).
- `EvalRunner._call_chatbot` no longer persists full tracebacks (which can
  contain API keys) to `report.json`; only a redacted summary survives.
- False positives in retry logic for messages containing `"separated"`,
  `"narrate"`, etc., which were being treated as rate-limit errors and
  retried unnecessarily.
- Pytest collection warnings about Pydantic `TestCase` / `TestResult`
  (added `__test__ = False`).
- Sorting fallback in `pages/1_run.py` when an unknown severity slips
  through (now sorts last instead of crashing the page).

### Removed
- 8 duplicated `_load_config()` helpers (kept as thin shims for backwards
  compatibility with tests that monkeypatch the symbols).
- Empty `package-lock.json` (the project is pure Python).

## [0.3.0] — 2026-03-16

- Dashboard UX redesign with design system, wizard UI, and educational
  content.
- DeepEval integration (hallucination, bias, toxicity, GEval).
- Consistency evaluator.
- Expanded test suite (215 tests) and CI/CD workflow.

## [0.2.0]

- Initial Streamlit dashboard for visual evaluation management.

## [0.1.0]

- Initial release: chatbot QA framework with RAGAS evaluation.
