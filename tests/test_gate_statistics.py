"""Tests for the gate's pure statistical functions (seeded, deterministic)."""

from __future__ import annotations

import numpy as np
import pytest

from src.gate.statistics import case_flakiness, paired_bootstrap


class TestPairedBootstrap:
    def test_clear_regression_is_significant(self):
        rng = np.random.default_rng(7)
        deltas = rng.normal(loc=0.2, scale=0.05, size=40).tolist()
        result = paired_bootstrap(deltas, seed=42)
        assert result.p_value < 0.05
        assert 0.15 < result.mean_delta < 0.25
        assert result.ci_low > 0.0

    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(7)
        raw = rng.normal(loc=0.0, scale=0.1, size=40)
        deltas = (raw - raw.mean()).tolist()  # centered: mean exactly 0, immune to seed luck
        result = paired_bootstrap(deltas, seed=42)
        assert result.p_value > 0.05

    def test_all_zero_deltas_yield_p_value_one(self):
        result = paired_bootstrap([0.0] * 20, seed=42)
        assert result.mean_delta == 0.0
        assert result.p_value == 1.0

    def test_same_seed_is_reproducible(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.07]
        a = paired_bootstrap(deltas, seed=123)
        b = paired_bootstrap(deltas, seed=123)
        assert a == b

    def test_different_seed_changes_resamples_not_mean(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.07]
        a = paired_bootstrap(deltas, seed=1)
        b = paired_bootstrap(deltas, seed=2)
        assert a.mean_delta == b.mean_delta
        assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)

    def test_empty_deltas_raise(self):
        with pytest.raises(ValueError, match="at least one delta"):
            paired_bootstrap([], seed=42)

    def test_ci_bounds_are_ordered(self):
        result = paired_bootstrap([0.1, 0.2, -0.1, 0.05], seed=42)
        assert result.ci_low <= result.mean_delta <= result.ci_high


class TestCaseFlakiness:
    def test_all_samples_agree_pass(self):
        assert case_flakiness([True, True, True]) == 0.0

    def test_all_samples_agree_fail(self):
        assert case_flakiness([False, False]) == 0.0

    def test_one_dissenter_of_five(self):
        assert case_flakiness([True, True, True, True, False]) == pytest.approx(0.2)

    def test_even_split_is_half(self):
        assert case_flakiness([True, False, True, False]) == pytest.approx(0.5)

    def test_single_sample_is_zero(self):
        assert case_flakiness([True]) == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            case_flakiness([])
