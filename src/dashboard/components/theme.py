"""Single source of truth for dashboard colour.

Every colour used more than once in the dashboard lives here. Two consumers
read it, which is why the palette is a plain Python dict rather than CSS custom
properties: Plotly cannot read CSS, so charts need the values in Python anyway,
and a second mechanism would enable nothing.

Colours used exactly once stay inline at their call site — naming them would
invent vocabulary for shades that are not part of the system. The guard tests
in ``tests/test_dashboard_theme.py`` keep both halves of that rule honest.
"""

from __future__ import annotations

PALETTE: dict[str, str] = {
    # Text
    "text": "#e2e8f0",  # primary copy on dark surfaces
    "text_soft": "#94a3b8",  # secondary copy, labels
    "text_muted": "#64748b",  # captions, hints
    "text_faint": "#6b7280",  # disabled, tick labels
    "text_dim": "#9ca3af",  # muted body copy inside cards
    "text_ghost": "#374151",  # footers, barely-there separators
    # Surfaces
    "bg_page": "#0f0f1a",  # the app canvas behind everything
    "bg": "#1a1a2e",  # page/card background
    "bg_raised": "#22223d",  # gradient partner for bg
    "bg_sunken": "#13132b",  # inputs, wells
    "border": "#2d2d44",  # default hairline
    "border_strong": "#4d4d6e",  # hover/active hairline
    # Brand
    "accent": "#6366f1",  # indigo — primary accent
    "accent_bright": "#8b5cf6",  # gradient partner for accent
    "accent_soft": "#a78bfa",  # lighter accent for badges
    "accent_pale": "#a5b4fc",  # accent text on tinted backgrounds
    # Status
    "success": "#22c55e",
    "success_bright": "#4ade80",  # success text/plot lines
    "success_deep": "#16a34a",  # gradient partner for success
    "danger": "#ef4444",
    "danger_soft": "#f87171",  # danger text/plot lines
    "danger_pale": "#fca5a5",  # danger text on tinted backgrounds
    "warning": "#f59e0b",
    "warning_bright": "#facc15",  # warning plot lines
    "warning_deep": "#d97706",  # warning borders
    "warning_pale": "#fde68a",  # warning text on tinted backgrounds
    "info": "#38bdf8",
    "info_deep": "#0ea5e9",  # gradient partner for info
}

CATEGORY_COLORS: dict[str, str] = {
    "functional": PALETTE["accent"],
    "safety": PALETTE["danger_soft"],
    "regression": PALETTE["success_bright"],
    "multi_turn": PALETTE["info"],
}

SEVERITY_COLORS: dict[str, str] = {
    "critical": "#dc2626",
    "high": "#f97316",
    "medium": "#eab308",
    "low": PALETTE["success"],
}
