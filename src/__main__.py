"""Entry point: python -m src / llm-eval-lab console script."""

from __future__ import annotations

import sys

from src.cli import app


def main() -> None:
    """Invoke the CLI; with no arguments, default to the ``run`` command."""
    args = sys.argv[1:]
    app(args or ["run"])


def cli() -> None:
    """Backwards-compatible alias for the old console-script entry point."""
    main()


if __name__ == "__main__":
    main()
