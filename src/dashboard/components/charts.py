"""Reusable chart components built with Plotly."""

from __future__ import annotations

import plotly.graph_objects as go

# Consistent color palette
COLORS = {
    "primary": "#6366f1",
    "success": "#4ade80",
    "danger": "#f87171",
    "warning": "#facc15",
    "info": "#38bdf8",
    "muted": "#6b7280",
    "bg": "#1e1e2e",
    "bg_card": "#2d2d44",
    "border": "#3d3d5c",
    "text": "#e0e0e0",
    "text_muted": "#a0a0b0",
}

CATEGORY_COLORS = {
    "functional": "#6366f1",
    "safety": "#f87171",
    "regression": "#4ade80",
    "multi_turn": "#38bdf8",
}

SEVERITY_COLORS = {
    "critical": "#dc2626",
    "high": "#f97316",
    "medium": "#eab308",
    "low": "#22c55e",
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    height=350,
)


def pass_rate_bar_chart(categories: dict[str, dict]) -> go.Figure:
    """Horizontal bar chart of pass rate by category with pass/fail counts."""
    cats = sorted(categories.keys())
    rates = [categories[c].get("pass_rate", 0) * 100 for c in cats]
    passed = [categories[c].get("passed", 0) for c in cats]
    total = [categories[c].get("total", 0) for c in cats]
    colors = [CATEGORY_COLORS.get(c, COLORS["primary"]) for c in cats]
    labels = [c.replace("_", " ").title() for c in cats]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=rates,
            orientation="h",
            marker_color=colors,
            text=[f"{r:.0f}% ({p}/{t})" for r, p, t in zip(rates, passed, total)],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<br>Passed: %{customdata[0]}/%{customdata[1]}<extra></extra>",
            customdata=list(zip(passed, total)),
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Pass Rate by Category", font=dict(size=14)),
        xaxis=dict(range=[0, 115], title="Pass Rate (%)", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["border"]),
    )
    return fig


def metrics_radar_chart(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
    title: str = "Evaluation Metrics",
) -> go.Figure:
    """Radar chart for evaluation metrics with threshold overlay."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title=title)
        return fig

    names = list(metrics.keys())
    values = list(metrics.values())
    names_closed = names + [names[0]]
    values_closed = values + [values[0]]
    theta = [n.replace("_", " ").title() for n in names_closed]

    fig = go.Figure()

    # Threshold band
    if thresholds:
        t_values = [thresholds.get(n, 0.65) for n in names] + [thresholds.get(names[0], 0.65)]
        fig.add_trace(
            go.Scatterpolar(
                r=t_values,
                theta=theta,
                fill="toself",
                fillcolor="rgba(248, 113, 113, 0.08)",
                line=dict(color=COLORS["danger"], width=1, dash="dash"),
                name="Threshold",
                hovertemplate="%{theta}: %{r:.2f}<extra>Threshold</extra>",
            )
        )

    # Actual scores
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=theta,
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.2)",
            line=dict(color=COLORS["primary"], width=2.5),
            name="Score",
            hovertemplate="%{theta}: %{r:.3f}<extra>Score</extra>",
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=14)),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"], tickfont=dict(size=10)),
            angularaxis=dict(gridcolor=COLORS["border"]),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=400,
    )
    return fig


def latency_histogram(latencies: list[float]) -> go.Figure:
    """Histogram of response latencies with percentile markers."""
    import statistics

    fig = go.Figure(
        go.Histogram(
            x=latencies,
            nbinsx=25,
            marker_color=COLORS["info"],
            marker_line=dict(color=COLORS["border"], width=1),
            opacity=0.85,
            hovertemplate="Range: %{x}ms<br>Count: %{y}<extra></extra>",
        )
    )

    # Add percentile lines
    if len(latencies) >= 3:
        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95_idx = int(len(sorted_lat) * 0.95)
        p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

        for pval, label, color in [(p50, "P50", COLORS["success"]), (p95, "P95", COLORS["warning"])]:
            fig.add_vline(x=pval, line_dash="dash", line_color=color, line_width=1.5,
                          annotation_text=f"{label}: {pval:.0f}ms", annotation_font_color=color)

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Latency Distribution", font=dict(size=14)),
        xaxis=dict(title="Latency (ms)", gridcolor=COLORS["border"], zeroline=False),
        yaxis=dict(title="Count", gridcolor=COLORS["border"], zeroline=False),
    )
    return fig


def severity_pie_chart(severity_counts: dict[str, int]) -> go.Figure:
    """Donut chart of failure severity distribution."""
    labels = list(severity_counts.keys())
    values = list(severity_counts.values())
    colors = [SEVERITY_COLORS.get(s, COLORS["muted"]) for s in labels]

    fig = go.Figure(
        go.Pie(
            labels=[s.title() for s in labels],
            values=values,
            marker=dict(colors=colors, line=dict(color=COLORS["bg"], width=2)),
            hole=0.45,
            textinfo="label+value",
            textfont=dict(size=12),
            hovertemplate="%{label}: %{value} failures<br>%{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Failures by Severity", font=dict(size=14)),
        height=320,
        showlegend=False,
    )
    return fig


def comparison_bar_chart(
    run_a_metrics: dict[str, float],
    run_b_metrics: dict[str, float],
    label_a: str = "Run A",
    label_b: str = "Run B",
) -> go.Figure:
    """Grouped bar chart comparing two runs."""
    all_metrics = sorted(set(list(run_a_metrics.keys()) + list(run_b_metrics.keys())))
    names = [m.replace("_", " ").title() for m in all_metrics]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name=label_a,
            x=names,
            y=[run_a_metrics.get(m, 0) for m in all_metrics],
            marker_color=COLORS["primary"],
            hovertemplate="%{x}: %{y:.3f}<extra>" + label_a + "</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name=label_b,
            x=names,
            y=[run_b_metrics.get(m, 0) for m in all_metrics],
            marker_color=COLORS["info"],
            hovertemplate="%{x}: %{y:.3f}<extra>" + label_b + "</extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Metric Comparison", font=dict(size=14)),
        barmode="group",
        xaxis=dict(gridcolor=COLORS["border"]),
        yaxis=dict(range=[0, 1.05], title="Score", gridcolor=COLORS["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return fig


def evaluator_scores_chart(results: list[dict]) -> go.Figure:
    """Grouped bar chart showing average score per evaluator."""
    evaluator_scores: dict[str, list[float]] = {}
    for r in results:
        for ev in r.get("evaluations", []):
            name = ev.get("evaluator", "?")
            score = ev.get("score")
            if score is not None:
                evaluator_scores.setdefault(name, []).append(score)

    if not evaluator_scores:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="Evaluator Scores")
        return fig

    names = sorted(evaluator_scores.keys())
    avgs = [sum(evaluator_scores[n]) / len(evaluator_scores[n]) for n in names]
    counts = [len(evaluator_scores[n]) for n in names]

    eval_colors = {
        "rule_based": "#6366f1",
        "safety": "#f87171",
        "ragas": "#4ade80",
        "deepeval": "#38bdf8",
        "consistency": "#facc15",
        "llm_judge": "#a78bfa",
    }
    colors = [eval_colors.get(n, COLORS["muted"]) for n in names]

    fig = go.Figure(
        go.Bar(
            x=[n.replace("_", " ").title() for n in names],
            y=avgs,
            marker_color=colors,
            text=[f"{a:.2f}" for a in avgs],
            textposition="outside",
            hovertemplate="%{x}: %{y:.3f}<br>Tests: %{customdata}<extra></extra>",
            customdata=counts,
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Average Score by Evaluator", font=dict(size=14)),
        yaxis=dict(range=[0, 1.1], title="Avg Score", gridcolor=COLORS["border"]),
        xaxis=dict(gridcolor=COLORS["border"]),
    )
    return fig


def score_distribution_chart(results: list[dict]) -> go.Figure:
    """Box plot of score distribution by category."""
    category_scores: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("test_case", {}).get("category", "?")
        score = r.get("overall_score")
        if score is not None:
            category_scores.setdefault(cat, []).append(score)

    fig = go.Figure()
    for cat in sorted(category_scores.keys()):
        fig.add_trace(
            go.Box(
                y=category_scores[cat],
                name=cat.replace("_", " ").title(),
                marker_color=CATEGORY_COLORS.get(cat, COLORS["muted"]),
                boxmean=True,
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Score Distribution by Category", font=dict(size=14)),
        yaxis=dict(range=[0, 1.05], title="Score", gridcolor=COLORS["border"]),
        showlegend=False,
    )
    return fig


def category_trend_chart(runs: list[dict]) -> go.Figure:
    """Line chart showing pass rate trend across runs per category."""
    if not runs:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title="Pass Rate Trend")
        return fig

    # Collect data per category across runs
    all_cats = set()
    for r in runs:
        all_cats.update(r.get("by_category", {}).keys())

    fig = go.Figure()
    run_labels = [r.get("run_id", "?")[:8] for r in runs]

    for cat in sorted(all_cats):
        rates = []
        for r in runs:
            cat_stats = r.get("by_category", {}).get(cat, {})
            rates.append(cat_stats.get("pass_rate", 0) * 100)

        fig.add_trace(
            go.Scatter(
                x=run_labels,
                y=rates,
                mode="lines+markers",
                name=cat.replace("_", " ").title(),
                line=dict(color=CATEGORY_COLORS.get(cat, COLORS["muted"]), width=2),
                marker=dict(size=8),
                hovertemplate="%{x}: %{y:.1f}%<extra>" + cat + "</extra>",
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text="Pass Rate Trend Across Runs", font=dict(size=14)),
        xaxis=dict(title="Run ID", gridcolor=COLORS["border"]),
        yaxis=dict(title="Pass Rate (%)", range=[0, 105], gridcolor=COLORS["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=380,
    )
    return fig
