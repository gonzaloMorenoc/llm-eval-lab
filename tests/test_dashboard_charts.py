"""Tests for the chart components that carry logic beyond drawing.

Most chart builders are pure presentation and are verified by looking at the
dashboard. ``category_trend_chart`` is different: it renders time, so the order
of its input decides whether the picture tells the truth.
"""

from __future__ import annotations

from src.dashboard.components.charts import category_trend_chart

_CATS = {"functional": {"pass_rate": 0.5}, "safety": {"pass_rate": 0.5}}


def _run(run_id: str, timestamp: str, pass_rate: float, by_category: dict | None = None) -> dict:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "pass_rate": pass_rate,
        "by_category": by_category if by_category is not None else _CATS,
    }


def _series(fig, name: str) -> list[float]:
    return next(list(trace.y) for trace in fig.data if trace.name == name)


class TestCategoryTrendChart:
    def test_plots_oldest_first_whatever_the_input_order(self) -> None:
        """``list_runs()`` returns newest first, so a trend chart fed straight
        from it would draw time backwards. The ordering is guaranteed here so no
        caller can get it wrong."""
        newest_first = [
            _run("run_c", "2026-03-03T10:00:00+00:00", 0.9, {"functional": {"pass_rate": 0.9}}),
            _run("run_b", "2026-02-02T10:00:00+00:00", 0.5, {"functional": {"pass_rate": 0.5}}),
            _run("run_a", "2026-01-01T10:00:00+00:00", 0.1, {"functional": {"pass_rate": 0.1}}),
        ]

        fig = category_trend_chart(newest_first)

        assert _series(fig, "Functional") == [10.0, 50.0, 90.0]

    def test_already_chronological_input_is_left_alone(self) -> None:
        oldest_first = [
            _run("run_a", "2026-01-01T10:00:00+00:00", 0.1, {"functional": {"pass_rate": 0.1}}),
            _run("run_b", "2026-02-02T10:00:00+00:00", 0.5, {"functional": {"pass_rate": 0.5}}),
        ]

        fig = category_trend_chart(oldest_first)

        assert _series(fig, "Functional") == [10.0, 50.0]

    def test_includes_an_overall_line_next_to_the_categories(self) -> None:
        runs = [
            _run("run_a", "2026-01-01T10:00:00+00:00", 0.25),
            _run("run_b", "2026-02-02T10:00:00+00:00", 0.75),
        ]

        fig = category_trend_chart(runs)

        assert _series(fig, "Global") == [25.0, 75.0]

    def test_x_axis_shows_dates_not_run_ids(self) -> None:
        runs = [_run("20260101T100000_deadbeef", "2026-01-01T10:00:00+00:00", 0.5)]

        fig = category_trend_chart(runs)

        label = next(iter(fig.data[0].x))
        assert "deadbeef" not in label
        assert "01-01" in label

    def test_no_runs_produces_an_empty_figure(self) -> None:
        fig = category_trend_chart([])

        assert len(fig.data) == 0

    def test_a_run_missing_its_timestamp_does_not_crash(self) -> None:
        runs = [
            {"run_id": "no_ts", "pass_rate": 0.4, "by_category": {}},
            _run("run_b", "2026-02-02T10:00:00+00:00", 0.6),
        ]

        fig = category_trend_chart(runs)

        assert _series(fig, "Global") == [40.0, 60.0]
