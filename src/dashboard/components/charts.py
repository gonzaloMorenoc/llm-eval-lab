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

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    height=350,
)


def pass_rate_bar_chart(categories: dict[str, dict]) -> go.Figure:
    """Horizontal bar chart of pass rate by category."""
    cats = sorted(categories.keys())
    rates = [categories[c].get("pass_rate", 0) * 100 for c in cats]
    colors = [CATEGORY_COLORS.get(c, COLORS["primary"]) for c in cats]

    fig = go.Figure(
        go.Bar(
            y=[c.replace("_", " ").title() for c in cats],
            x=rates,
            orientation="h",
            marker_color=colors,
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Pass Rate by Category",
        xaxis=dict(range=[0, 105], title="Pass Rate (%)", gridcolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"]),
    )
    return fig


def metrics_radar_chart(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
    title: str = "Metrics Overview",
) -> go.Figure:
    """Radar chart for evaluation metrics."""
    if not metrics:
        fig = go.Figure()
        fig.update_layout(**_LAYOUT_DEFAULTS, title=title)
        return fig

    names = list(metrics.keys())
    values = list(metrics.values())
    # Close the polygon
    names_closed = names + [names[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=[n.replace("_", " ").title() for n in names_closed],
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.15)",
            line=dict(color=COLORS["primary"], width=2),
            name="Score",
        )
    )

    if thresholds:
        t_values = [thresholds.get(n, 0.65) for n in names] + [thresholds.get(names[0], 0.65)]
        fig.add_trace(
            go.Scatterpolar(
                r=t_values,
                theta=[n.replace("_", " ").title() for n in names_closed],
                line=dict(color=COLORS["danger"], width=1, dash="dash"),
                name="Threshold",
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=title,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=COLORS["border"]),
            angularaxis=dict(gridcolor=COLORS["border"]),
        ),
        showlegend=True,
        height=400,
    )
    return fig


def latency_histogram(latencies: list[float]) -> go.Figure:
    """Histogram of response latencies."""
    fig = go.Figure(
        go.Histogram(
            x=latencies,
            nbinsx=20,
            marker_color=COLORS["info"],
            marker_line=dict(color=COLORS["border"], width=1),
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Latency Distribution",
        xaxis=dict(title="Latency (ms)", gridcolor=COLORS["border"]),
        yaxis=dict(title="Count", gridcolor=COLORS["border"]),
    )
    return fig


def severity_pie_chart(severity_counts: dict[str, int]) -> go.Figure:
    """Pie chart of failure severity distribution."""
    severity_colors = {
        "critical": "#dc2626",
        "high": "#f97316",
        "medium": "#eab308",
        "low": "#22c55e",
    }
    labels = list(severity_counts.keys())
    values = list(severity_counts.values())
    colors = [severity_colors.get(s, COLORS["muted"]) for s in labels]

    fig = go.Figure(
        go.Pie(
            labels=[s.title() for s in labels],
            values=values,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo="label+value",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Failures by Severity",
        height=300,
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
        )
    )
    fig.add_trace(
        go.Bar(
            name=label_b,
            x=names,
            y=[run_b_metrics.get(m, 0) for m in all_metrics],
            marker_color=COLORS["info"],
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title="Metric Comparison",
        barmode="group",
        xaxis=dict(gridcolor=COLORS["border"]),
        yaxis=dict(range=[0, 1.05], title="Score", gridcolor=COLORS["border"]),
    )
    return fig
