"""JSON reporter — serializes RunSummary to a JSON file."""

from __future__ import annotations

import json
import os

from src.runner.models import RunSummary


def generate_json_report(summary: RunSummary, output_dir: str) -> str:
    """Write the full RunSummary as JSON. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report.json")
    with open(path, "w") as f:
        json.dump(summary.model_dump(), f, indent=2, default=str)
    return path
