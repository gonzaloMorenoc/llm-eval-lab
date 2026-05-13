"""Tests for the dashboard shared utilities — focused on the security-critical
``safe()`` HTML escaper and the ``list_runs()`` filesystem reader.

These pieces previously had zero coverage and shipped two real bugs (stored
XSS via metadata, silent ``except: pass`` hiding load failures).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dashboard.components.shared import append_jsonl, safe


class TestSafe:
    def test_escapes_html_tags(self) -> None:
        assert safe("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

    def test_escapes_attribute_quotes(self) -> None:
        # Required so that ``f'<div title="{safe(value)}">'`` cannot be broken out of.
        assert safe('"><img src=x onerror=alert(1)>') == "&quot;&gt;&lt;img src=x onerror=alert(1)&gt;"

    def test_escapes_ampersand(self) -> None:
        assert safe("foo & bar") == "foo &amp; bar"

    def test_stringifies_non_strings(self) -> None:
        assert safe(42) == "42"
        assert safe(3.14) == "3.14"

    def test_none_becomes_empty(self) -> None:
        assert safe(None) == ""

    def test_idempotent_on_safe_text(self) -> None:
        assert safe("plain ASCII text") == "plain ASCII text"


class TestListRuns:
    @pytest.fixture
    def fake_results_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect ``RESULTS_DIR`` to a sandbox and clear Streamlit session state."""
        # Streamlit caches the import; reload to keep test isolation clean.
        from src.dashboard.components import shared

        monkeypatch.setattr(shared, "RESULTS_DIR", str(tmp_path))
        # ``st.session_state`` is process-wide; we don't touch the real one.
        import streamlit as st

        st.session_state.clear()
        return tmp_path

    def _write_run(self, root: Path, run_id: str, payload: dict) -> None:
        run_dir = root / run_id
        run_dir.mkdir()
        (run_dir / "report.json").write_text(json.dumps(payload))

    def test_returns_empty_when_no_runs(self, fake_results_dir: Path) -> None:
        from src.dashboard.components.shared import list_runs

        assert list_runs() == []

    def test_lists_runs_newest_first(self, fake_results_dir: Path) -> None:
        from src.dashboard.components.shared import list_runs

        self._write_run(fake_results_dir, "20260101T100000_aaaaaaaa", {"run_id": "old", "pass_rate": 0.5})
        self._write_run(fake_results_dir, "20260601T100000_bbbbbbbb", {"run_id": "new", "pass_rate": 0.9})
        runs = list_runs()
        assert [r["run_id"] for r in runs] == ["new", "old"]

    def test_skips_runs_without_report(self, fake_results_dir: Path) -> None:
        from src.dashboard.components.shared import list_runs

        (fake_results_dir / "empty_run").mkdir()  # no report.json inside
        self._write_run(fake_results_dir, "20260601T100000_bbbbbbbb", {"run_id": "good"})
        runs = list_runs()
        assert [r["run_id"] for r in runs] == ["good"]

    def test_corrupt_report_does_not_crash(self, fake_results_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
        from src.dashboard.components.shared import list_runs

        bad = fake_results_dir / "20260601T100000_xxxxxxxx"
        bad.mkdir()
        (bad / "report.json").write_text("{not json")
        self._write_run(fake_results_dir, "20260601T120000_yyyyyyyy", {"run_id": "good"})

        with caplog.at_level("WARNING"):
            runs = list_runs()

        # The bad run is silently skipped; the good one comes through; a warning was logged.
        assert [r["run_id"] for r in runs] == ["good"]
        assert any("Failed to load run" in rec.message for rec in caplog.records)


class TestAppendJsonl:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "new.jsonl"
        append_jsonl(str(target), {"id": "func_001", "category": "functional"})

        content = target.read_text()
        assert content == '{"id": "func_001", "category": "functional"}\n'

    def test_appends_to_file_ending_in_newline(self, tmp_path: Path) -> None:
        target = tmp_path / "data.jsonl"
        target.write_text('{"id": "a"}\n')
        append_jsonl(str(target), {"id": "b"})

        lines = target.read_text().splitlines()
        assert lines == ['{"id": "a"}', '{"id": "b"}']

    def test_appends_to_file_missing_trailing_newline(self, tmp_path: Path) -> None:
        # Regression: previous bug wrote ``"\\n" + json.dumps(...)`` which
        # always added a leading newline. If the file already ended in '\\n'
        # the result was a blank line; if it didn't end in '\\n', the new
        # entry never got its own terminating newline.
        target = tmp_path / "data.jsonl"
        target.write_text('{"id": "a"}')  # no trailing newline
        append_jsonl(str(target), {"id": "b"})

        lines = target.read_text().splitlines()
        assert lines == ['{"id": "a"}', '{"id": "b"}']

    def test_preserves_unicode(self, tmp_path: Path) -> None:
        target = tmp_path / "data.jsonl"
        append_jsonl(str(target), {"input": "¿Qué es el aprendizaje automático?"})

        # ensure_ascii=False is mandatory so Spanish text round-trips legibly.
        content = target.read_text(encoding="utf-8")
        assert "¿Qué es el aprendizaje automático?" in content

    def test_multiple_appends_produce_valid_jsonl(self, tmp_path: Path) -> None:
        target = tmp_path / "data.jsonl"
        for i in range(5):
            append_jsonl(str(target), {"id": f"row_{i}", "value": i})

        # Every line must parse independently as JSON — that's the JSONL contract.
        parsed = [json.loads(line) for line in target.read_text().splitlines()]
        assert [row["id"] for row in parsed] == [f"row_{i}" for i in range(5)]
