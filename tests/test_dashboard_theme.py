"""Guard tests keeping colour in one place.

The dashboard once carried 328 hardcoded hex literals across 11 files, 301 of
them repetitions of ~15 colours. Extracting them is easy; *keeping* them
extracted is what these tests are for. Without a guard, the literals creep back
one "just this once" at a time.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from src.dashboard.components.theme import PALETTE

_DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "dashboard"
_HEX = re.compile(r"#[0-9a-fA-F]{6}\b")
# theme.py is where colour is allowed to be written out literally.
_EXEMPT = {"theme.py"}


def _literals_by_file() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(_DASHBOARD.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        matches = [m.lower() for m in _HEX.findall(path.read_text(encoding="utf-8"))]
        if matches:
            found[str(path.relative_to(_DASHBOARD))] = matches
    return found


def _where(colour: str) -> list[str]:
    return sorted(name for name, colours in _literals_by_file().items() if colour in colours)


class TestColourLiterals:
    def test_no_colour_is_written_out_more_than_once(self) -> None:
        """A repeated colour is a system colour, and system colours live in
        ``theme.py``. One-off shades are allowed to stay inline."""
        counts = Counter(colour for colours in _literals_by_file().values() for colour in colours)
        repeated = {colour: n for colour, n in counts.items() if n > 1}

        assert not repeated, "Repeated colour literals — move them to theme.PALETTE:\n" + "\n".join(
            f"  {colour} × {n} in {', '.join(_where(colour))}" for colour, n in sorted(repeated.items(), key=lambda kv: -kv[1])
        )

    def test_no_palette_colour_is_hardcoded(self) -> None:
        """Catches the likeliest slip the repetition rule misses: writing the
        accent colour's hex by hand, once, when it already has a name."""
        named = {value.lower(): key for key, value in PALETTE.items()}
        offenders = {colour: named[colour] for colours in _literals_by_file().values() for colour in colours if colour in named}

        assert not offenders, "Palette colours hardcoded — use PALETTE instead:\n" + "\n".join(
            f"  {colour} is PALETTE[{key!r}], found in {', '.join(_where(colour))}" for colour, key in sorted(offenders.items())
        )
