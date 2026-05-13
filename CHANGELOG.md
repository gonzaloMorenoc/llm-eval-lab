# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
