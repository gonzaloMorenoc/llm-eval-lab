"""Load the gate policy and produce the final verdict."""

from __future__ import annotations

import yaml

from src.gate.comparison import compare_metrics, pair_cases, validate_compatibility
from src.gate.models import BaselineFile, GatePolicy, GateVerdict


class PolicyError(Exception):
    """Raised when the gate policy file is missing or invalid."""


def load_policy(path: str | None) -> GatePolicy:
    """Load a gate policy YAML (top-level ``gate:`` key optional). None -> built-in defaults."""
    if path is None:
        return GatePolicy()
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except OSError as e:
        raise PolicyError(f"Cannot read policy file {path}: {e}") from e
    try:
        return GatePolicy.model_validate(data.get("gate", data))
    except Exception as e:
        raise PolicyError(f"Invalid policy file {path}: {e}") from e


def evaluate_gate(baseline: BaselineFile, current: BaselineFile, policy: GatePolicy) -> GateVerdict:
    """Full gate evaluation. Raises CompatibilityError for non-comparable runs."""
    validate_compatibility(baseline, current)
    pairs, new_ids, removed_ids = pair_cases(baseline, current)
    comparisons = compare_metrics(baseline, current, policy)

    compared = {c.metric for c in comparisons}
    missing_gated = sorted(m for m in policy.metrics if m not in compared)

    violations: list[str] = []
    if policy.hard_rules.no_new_critical_failures:
        broken = [b.id for b, c in pairs if b.severity == "critical" and b.passed and not c.passed]
        if broken:
            violations.append(f"New critical failures: {', '.join(broken)}")
    mean_flakiness = sum(c.flakiness for c in current.cases) / len(current.cases) if current.cases else 0.0
    if mean_flakiness > policy.hard_rules.max_flakiness:
        violations.append(f"Mean flakiness {mean_flakiness:.2f} exceeds limit {policy.hard_rules.max_flakiness:.2f}")
    if new_ids and policy.new_cases == "fail":
        violations.append(f"Cases without baseline: {', '.join(new_ids)}")

    passed = not violations and not missing_gated and not any(c.breaches for c in comparisons)
    return GateVerdict(
        passed=passed,
        comparisons=comparisons,
        hard_rule_violations=violations,
        missing_gated_metrics=missing_gated,
        new_case_ids=new_ids,
        removed_case_ids=removed_ids,
        mean_flakiness=round(mean_flakiness, 4),
        samples=current.samples,
    )
