"""Pure statistical functions for the regression gate: paired bootstrap and flakiness."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.gate.models import BootstrapResult


def paired_bootstrap(deltas: Sequence[float], *, n_resamples: int = 10_000, seed: int) -> BootstrapResult:
    """Bootstrap the mean of paired per-case regression deltas (positive = worse).

    Returns the observed mean, a 95% percentile CI and a one-sided p-value for
    H1 "mean regression > 0", computed as the fraction of resampled means <= 0.
    """
    if len(deltas) == 0:
        raise ValueError("paired_bootstrap requires at least one delta")
    arr = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    return BootstrapResult(
        mean_delta=float(arr.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        p_value=float(np.mean(means <= 0.0)),
    )


def case_flakiness(sample_passes: Sequence[bool]) -> float:
    """Fraction of samples whose pass/fail disagrees with the majority (an even split scores 0.5)."""
    if len(sample_passes) == 0:
        raise ValueError("case_flakiness requires at least one sample")
    n_pass = sum(1 for p in sample_passes if p)
    minority = min(n_pass, len(sample_passes) - n_pass)
    return minority / len(sample_passes)
