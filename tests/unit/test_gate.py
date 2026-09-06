"""Gate math: normal tails, moments, deflated Sharpe, bootstraps and the six gate rules.

Every expected value here is produced independently of the function under test: either a
closed-form constant (Phi-inverse at 0.975), or arithmetic written out from the defining
formula with `math`/`numpy` primitives. A test that called the implementation to build its
own expectation would only prove the code is self-consistent.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pytest
from pydantic import ValidationError

from swingforge.lab.gate import (
    bootstrap_difference_p5,
    cagr,
    deflated_sharpe,
    equity_curve_from_r,
    evaluate,
    excess_kurtosis,
    mar,
    max_drawdown,
    norm_cdf,
    norm_ppf,
    sharpe,
    skewness,
    stationary_bootstrap_indices,
    stationary_bootstrap_p5,
)

# --- norm_cdf / norm_ppf ------------------------------------------------------


def test_norm_cdf_at_zero_is_one_half() -> None:
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-15)


def test_norm_cdf_matches_erf_definition() -> None:
    for x in (-3.5, -1.0, -0.25, 0.0, 0.25, 1.0, 3.5):
        expected = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert norm_cdf(x) == pytest.approx(expected, abs=1e-15)


def test_norm_cdf_is_symmetric() -> None:
    for x in (0.1, 0.9, 2.4, 4.0):
        assert norm_cdf(-x) == pytest.approx(1.0 - norm_cdf(x), abs=1e-15)


def test_norm_cdf_known_tail_values() -> None:
    # Textbook values, independent of any implementation here.
    assert norm_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-9)
    assert norm_cdf(-1.0) == pytest.approx(0.15865525393145707, abs=1e-12)


def test_norm_cdf_is_relatively_accurate_in_the_far_lower_tail() -> None:
    # `1 + erf(-x/sqrt2)` cancels the leading digits away out here; `erfc` never forms the 1.
    # `abs=0.0` matters: approx's default absolute tolerance of 1e-12 would swallow the tail.
    assert norm_cdf(-7.0) == pytest.approx(0.5 * math.erfc(7.0 / math.sqrt(2.0)), rel=1e-14, abs=0.0)
    assert norm_cdf(-10.0) == pytest.approx(0.5 * math.erfc(10.0 / math.sqrt(2.0)), rel=1e-14, abs=0.0)


def test_norm_ppf_known_quantiles() -> None:
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert norm_ppf(0.95) == pytest.approx(1.644854, abs=1e-6)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_norm_ppf_inverts_norm_cdf_to_machine_precision() -> None:
    for p in (1e-10, 1e-4, 0.02424, 0.02426, 0.1, 0.5, 0.9, 0.97575, 0.9999, 1 - 1e-10):
        assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-12, abs=1e-15)


def test_norm_ppf_is_antisymmetric() -> None:
    for p in (0.001, 0.02, 0.3, 0.45):
        assert norm_ppf(p) == pytest.approx(-norm_ppf(1.0 - p), rel=1e-12, abs=1e-12)


def test_norm_ppf_is_machine_precise_at_the_two_tournament_quantiles() -> None:
    # `statistics.NormalDist.inv_cdf` (Wichura AS241) is the independent reference.
    reference = NormalDist()
    for p in (1.0 - 1.0 / 2016.0, 1.0 - 1.0 / (2016.0 * math.e)):
        assert norm_ppf(p) == pytest.approx(reference.inv_cdf(p), rel=1e-13)
    assert norm_cdf(norm_ppf(1.0 - 1.0 / 2016.0)) == 1.0 - 1.0 / 2016.0  # exact round trip


def test_norm_ppf_degrades_in_the_far_upper_tail_as_documented() -> None:
    reference = NormalDist()
    upper = abs(norm_ppf(1.0 - 1e-12) - reference.inv_cdf(1.0 - 1e-12))
    lower = abs(norm_ppf(1e-12) - reference.inv_cdf(1e-12))
    assert upper < 1e-7  # the limit the docstring promises out there
    assert lower < 1e-13  # the lower tail keeps full precision
    assert upper > lower  # the asymmetry is real, not a rounding accident


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_norm_ppf_rejects_probabilities_outside_the_open_unit_interval(p: float) -> None:
    with pytest.raises(ValueError):
        norm_ppf(p)


# --- sharpe / skewness / excess_kurtosis --------------------------------------


def test_sharpe_is_mean_over_sample_std() -> None:
    returns = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = 3.0 / math.sqrt(2.5)  # mean 3, ddof=1 variance 10/4
    assert sharpe(returns) == pytest.approx(expected, rel=1e-12)


def test_sharpe_is_not_annualised() -> None:
    # Mean 0.1, population variance 1.0 over 101 points centred on 0.1: the per-observation
    # ratio is ~0.0995, while any annualisation would multiply it by sqrt(periods per year).
    returns = [0.1 + offset for offset in np.linspace(-1.0, 1.0, 3)]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    per_observation = mean / math.sqrt(variance)
    assert sharpe(returns) == pytest.approx(per_observation, rel=1e-12)
    assert sharpe(returns) != pytest.approx(per_observation * math.sqrt(252.0), rel=1e-3)


def test_sharpe_of_a_constant_series_is_zero() -> None:
    assert sharpe([0.7] * 10) == 0.0


def test_sharpe_of_a_degenerate_series_is_zero() -> None:
    assert sharpe([]) == 0.0
    assert sharpe([1.0]) == 0.0


def test_skewness_uses_population_moments() -> None:
    values = [0.0, 0.0, 0.0, 1.0]
    mean = 0.25
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    m3 = sum((v - mean) ** 3 for v in values) / len(values)
    assert skewness(values) == pytest.approx(m3 / m2**1.5, rel=1e-12)
    assert skewness(values) == pytest.approx(2.0 / math.sqrt(3.0), rel=1e-12)


def test_skewness_of_a_symmetric_series_is_zero() -> None:
    assert skewness([-2.0, -1.0, 0.0, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


def test_excess_kurtosis_uses_population_moments() -> None:
    values = [0.0, 0.0, 0.0, 1.0]
    mean = 0.25
    m2 = sum((v - mean) ** 2 for v in values) / len(values)
    m4 = sum((v - mean) ** 4 for v in values) / len(values)
    assert excess_kurtosis(values) == pytest.approx(m4 / m2**2 - 3.0, rel=1e-12)
    assert excess_kurtosis(values) == pytest.approx(-2.0 / 3.0, rel=1e-12)


def test_excess_kurtosis_of_a_normal_sample_is_near_zero() -> None:
    sample = np.random.default_rng(11).normal(0.0, 1.0, 20_000)
    assert excess_kurtosis(sample) == pytest.approx(0.0, abs=0.15)


def test_moments_of_a_constant_series_are_zero() -> None:
    assert skewness([2.0] * 8) == 0.0
    assert excess_kurtosis([2.0] * 8) == 0.0


# --- deflated Sharpe ----------------------------------------------------------

EULER_MASCHERONI = 0.5772156649
"""Gamma, the constant in the expected-maximum term of the deflated Sharpe threshold."""


def expected_sr_star(n_trials: int, variance: float) -> float:
    """`sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`, written out longhand."""
    if n_trials <= 1:
        return 0.0
    g = EULER_MASCHERONI
    return math.sqrt(variance) * (
        (1.0 - g) * norm_ppf(1.0 - 1.0 / n_trials) + g * norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def expected_prob(returns: np.ndarray, sr_star_value: float) -> float:
    """The PSR/DSR probability, recomputed here from raw numpy moments."""
    array = np.asarray(returns, dtype=float)
    n = array.size
    sr = float(array.mean()) / float(array.std(ddof=1))
    centred = array - array.mean()
    m2 = float((centred**2).mean())
    g3 = float((centred**3).mean()) / m2**1.5
    g4 = float((centred**4).mean()) / m2**2  # non-excess kurtosis
    denominator = math.sqrt(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr)
    return norm_cdf((sr - sr_star_value) * math.sqrt(n - 1) / denominator)


def planted(mean: float, n: int = 200, seed: int = 0) -> np.ndarray:
    """A seeded normal R series with a planted per-trade edge of `mean` R."""
    return np.random.default_rng(seed).normal(mean, 1.0, n)


def test_a_single_trial_is_not_deflated_and_reduces_to_the_psr() -> None:
    returns = planted(0.3)
    result = deflated_sharpe(returns, n_trials=1)
    assert result.sr_star == 0.0
    assert result.prob == pytest.approx(expected_prob(returns, 0.0), rel=1e-12)


def test_sr_star_increases_strictly_with_the_number_of_trials() -> None:
    returns = planted(0.3)
    thresholds = [deflated_sharpe(returns, n_trials=n).sr_star for n in (1, 2, 10, 100, 1000, 2016, 10_000)]
    assert all(a < b for a, b in zip(thresholds[:-1], thresholds[1:], strict=True))


def test_sr_star_increases_strictly_with_the_trial_sharpe_variance() -> None:
    returns = planted(0.3)
    thresholds = [
        deflated_sharpe(returns, n_trials=2016, trial_sr_variance=v).sr_star
        for v in (0.001, 0.005, 0.02, 0.1, 1.0)
    ]
    assert all(a < b for a, b in zip(thresholds[:-1], thresholds[1:], strict=True))


def test_sr_star_matches_the_formula_by_hand_at_a_thousand_trials() -> None:
    result = deflated_sharpe(planted(0.3), n_trials=1000, trial_sr_variance=1.0)
    assert result.sr_star == pytest.approx(expected_sr_star(1000, 1.0), abs=1e-9)


def test_a_strong_edge_clears_the_deflation_threshold_at_tournament_scale() -> None:
    result = deflated_sharpe(planted(0.5), n_trials=2016)
    assert result.prob > 0.95
    assert result.n == 200
    assert result.n_trials == 2016


def test_zero_mean_noise_does_not_clear_the_deflation_threshold() -> None:
    for seed in (0, 1, 2, 3, 4):
        assert deflated_sharpe(planted(0.0, seed=seed), n_trials=2016).prob < 0.95


def test_a_moderate_edge_is_deflated_away_at_two_thousand_trials() -> None:
    # A +0.3R mean over 200 trades is a per-observation SR of about 0.33 with a standard
    # error of ~0.07; the expected maximum SR over 2,016 null trials is ~0.244, so the
    # edge sits barely one standard error above the threshold and cannot reach 95%.
    returns = planted(0.3)
    result = deflated_sharpe(returns, n_trials=2016)
    assert result.sr_star == pytest.approx(expected_sr_star(2016, 1.0 / 199.0), abs=1e-12)
    assert result.prob == pytest.approx(expected_prob(returns, result.sr_star), rel=1e-12)
    assert result.prob < 0.95


def test_the_reference_sharpe_reproduces_the_formula_end_to_end() -> None:
    # Self-consistency check on the plan's reference point: an annualised SR of 2.5 on
    # daily observations is a per-observation SR of 2.5/sqrt(252) ~= 0.1575. The expected
    # value below is the formula written out by hand, not a published DSR figure.
    per_observation_sr = 2.5 / math.sqrt(252.0)
    sample = np.random.default_rng(3).normal(0.0, 1.0, 1000)
    returns = (sample - sample.mean()) / sample.std(ddof=1) * 1.0 + per_observation_sr
    result = deflated_sharpe(returns, n_trials=1000)
    assert result.sr == pytest.approx(per_observation_sr, rel=1e-12)
    assert result.sr_star == pytest.approx(expected_sr_star(1000, 1.0 / 999.0), abs=1e-12)
    assert result.prob == pytest.approx(expected_prob(returns, result.sr_star), rel=1e-12)


def test_dsr_reports_the_moments_that_enter_the_formula() -> None:
    returns = planted(0.3)
    centred = returns - returns.mean()
    m2 = float((centred**2).mean())
    assert deflated_sharpe(returns, n_trials=10).skew == pytest.approx(
        float((centred**3).mean()) / m2**1.5, rel=1e-12
    )
    assert deflated_sharpe(returns, n_trials=10).kurt == pytest.approx(
        float((centred**4).mean()) / m2**2, rel=1e-12
    )


@pytest.mark.parametrize("returns", [[], [0.5], [0.5, 0.5], [1.0] * 50])
def test_degenerate_series_get_zero_probability(returns: list[float]) -> None:
    result = deflated_sharpe(returns, n_trials=2016)
    assert result.sr == 0.0
    assert result.prob == 0.0
    assert result.n == len(returns)


def test_dsr_result_is_frozen() -> None:
    result = deflated_sharpe(planted(0.5), n_trials=2016)
    with pytest.raises(ValidationError):
        result.prob = 0.1


# --- stationary bootstrap -----------------------------------------------------


def test_bootstrap_indices_resample_to_the_input_length() -> None:
    indices = stationary_bootstrap_indices(17, 25)
    assert indices.shape == (25, 17)
    assert indices.min() >= 0
    assert indices.max() < 17


def test_bootstrap_blocks_wrap_around_the_end_of_the_series() -> None:
    # p very small: a resample is one long block, so it must wrap past the last index and
    # continue from 0 rather than stopping or clipping.
    indices = stationary_bootstrap_indices(10, 1, p=1e-9, rng=np.random.default_rng(4))
    row = indices[0]
    steps = (row[1:] - row[:-1]) % 10
    assert set(steps.tolist()) == {1}
    assert row.max() - row.min() == 9  # it visited every index, so it wrapped


def test_bootstrap_with_unit_restart_probability_is_an_iid_bootstrap() -> None:
    # p == 1: every position restarts, so consecutive draws carry no block structure.
    indices = stationary_bootstrap_indices(50, 200, p=1.0, rng=np.random.default_rng(5))
    continued = ((indices[:, 1:] - indices[:, :-1]) % 50 == 1).mean()
    assert continued == pytest.approx(1.0 / 50.0, abs=0.02)


def test_bootstrap_mean_block_length_is_one_over_p() -> None:
    indices = stationary_bootstrap_indices(400, 400, p=0.1, rng=np.random.default_rng(6))
    restarts = ((indices[:, 1:] - indices[:, :-1]) % 400 != 1).mean()
    assert restarts == pytest.approx(0.1, abs=0.02)


def test_bootstrap_p5_of_a_constant_series_is_the_constant() -> None:
    assert stationary_bootstrap_p5([1.25] * 40) == pytest.approx(1.25, rel=1e-12)


def test_bootstrap_p5_is_reproducible_for_a_seed_and_moves_with_it() -> None:
    series = planted(0.2, n=120, seed=9)
    assert stationary_bootstrap_p5(series, seed=3) == stationary_bootstrap_p5(series, seed=3)
    assert stationary_bootstrap_p5(series, seed=3) != stationary_bootstrap_p5(series, seed=4)


def test_bootstrap_p5_is_positive_for_a_strong_edge_and_negative_for_noise() -> None:
    assert stationary_bootstrap_p5(planted(0.5, n=200, seed=0)) > 0.0
    noise = planted(0.0, n=200, seed=0)
    assert stationary_bootstrap_p5(noise - noise.mean()) < 0.0


def test_bootstrap_p5_accepts_an_alternative_statistic() -> None:
    series = planted(0.4, n=200, seed=2)
    assert stationary_bootstrap_p5(series, statistic=np.median) != stationary_bootstrap_p5(series)


@pytest.mark.parametrize("seed", [0, 5, 17])
def test_the_vectorised_mean_matches_the_generic_statistic_loop(seed: int) -> None:
    # `np.mean` takes the vectorised path; an equivalent callable takes the row-by-row one.
    series = np.random.default_rng(21).normal(0.05, 1.0, 250)
    fast = stationary_bootstrap_p5(series, seed=seed)
    generic = stationary_bootstrap_p5(series, seed=seed, statistic=lambda row: float(row.mean()))
    assert fast == pytest.approx(generic, abs=1e-12)


def test_bootstrap_p5_rejects_an_empty_series() -> None:
    with pytest.raises(ValueError):
        stationary_bootstrap_p5([])


def test_difference_p5_is_negative_when_both_arms_are_the_same_series() -> None:
    series = planted(0.3, n=200, seed=1)
    assert bootstrap_difference_p5(series, series) < 0.0


def test_difference_p5_is_positive_when_one_arm_dominates() -> None:
    assert bootstrap_difference_p5(planted(0.8, n=200, seed=1), planted(-0.2, n=200, seed=2)) > 0.0


def test_difference_p5_is_reproducible_for_a_seed() -> None:
    a, b = planted(0.4, n=80, seed=1), planted(0.1, n=90, seed=2)
    assert bootstrap_difference_p5(a, b, seed=7) == bootstrap_difference_p5(a, b, seed=7)


# --- drawdown, CAGR, MAR, equity curve ----------------------------------------

HAND_CURVE = [100.0, 110.0, 99.0, 120.0]
"""Peak 110 then a trough of 99 (a 10% drawdown), ending 20% above the 100 it started at."""


def test_max_drawdown_of_the_hand_curve() -> None:
    assert max_drawdown(HAND_CURVE) == pytest.approx((110.0 - 99.0) / 110.0, rel=1e-12)
    assert max_drawdown(HAND_CURVE) == pytest.approx(0.1, rel=1e-12)


def test_max_drawdown_of_a_monotone_curve_is_zero() -> None:
    assert max_drawdown([100.0, 101.0, 140.0]) == 0.0
    assert max_drawdown([100.0]) == 0.0


def test_max_drawdown_is_measured_against_the_running_peak_not_the_start() -> None:
    # The 50 -> 25 leg is a 50% drawdown even though 25 is only 75% below the 100 start.
    assert max_drawdown([100.0, 20.0, 50.0, 25.0]) == pytest.approx(0.8, rel=1e-12)


def test_cagr_of_the_hand_curve_over_one_year() -> None:
    assert cagr(HAND_CURVE, 1.0) == pytest.approx(0.2, rel=1e-12)


def test_cagr_compounds_over_multiple_years() -> None:
    assert cagr([100.0, 121.0], 2.0) == pytest.approx(0.1, rel=1e-12)


def test_mar_of_the_hand_curve() -> None:
    assert mar(HAND_CURVE, 1.0) == pytest.approx(0.2 / 0.1, rel=1e-12)
    assert mar(HAND_CURVE, 1.0) == pytest.approx(2.0, rel=1e-12)


def test_mar_without_a_drawdown_is_infinite_when_growing_and_zero_otherwise() -> None:
    assert mar([100.0, 150.0], 1.0) == math.inf
    assert mar([100.0, 100.0], 1.0) == 0.0


@pytest.mark.parametrize("years", [0.0, -1.0])
def test_metrics_reject_a_non_positive_horizon(years: float) -> None:
    with pytest.raises(ValueError):
        cagr(HAND_CURVE, years)
    with pytest.raises(ValueError):
        mar(HAND_CURVE, years)


@pytest.mark.parametrize("bad", [[], [100.0, 0.0, 50.0], [-1.0, 2.0]])
def test_metrics_reject_a_non_positive_equity_curve(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        max_drawdown(bad)


def test_equity_curve_from_r_compounds_one_point_per_trade_plus_the_start() -> None:
    curve = equity_curve_from_r([1.0, -1.0, 2.0], risk_pct=0.01, start=1.0)
    assert len(curve) == 4
    expected = [1.0, 1.01, 1.01 * 0.99, 1.01 * 0.99 * 1.02]
    assert curve == pytest.approx(expected, rel=1e-12)


def test_equity_curve_from_r_honours_start_and_risk_pct() -> None:
    curve = equity_curve_from_r([2.0], risk_pct=0.02, start=500.0)
    assert curve == pytest.approx([500.0, 520.0], rel=1e-12)


def test_equity_curve_from_no_trades_is_just_the_starting_equity() -> None:
    assert equity_curve_from_r([]) == [1.0]


# --- the six-rule gate --------------------------------------------------------

FLAT_BH = [1.0, 1.0]
"""A buy-and-hold curve that neither grew nor drew down, so its MAR is 0.0."""

RISING_BH = [1.0, 2.0]
"""A buy-and-hold curve that only ever rose, so its MAR is `inf` - the rule 5 tie case."""


def all_positive(n: int = 200, seed: int = 8) -> np.ndarray:
    """Winners only: the compounded curve never draws down, so its MAR is `inf` too."""
    return np.abs(np.random.default_rng(seed).normal(0.0, 1.0, n)) + 0.1


def zero_mean(n: int = 200, seed: int = 0) -> np.ndarray:
    """Noise with an exactly zero sample mean - the baseline a real edge must beat."""
    sample = planted(0.0, n=n, seed=seed)
    return sample - sample.mean()


def test_too_few_trades_fails_rule_one_and_leaves_every_other_rule_unevaluated() -> None:
    result = evaluate(planted(0.5, n=59), zero_mean(), FLAT_BH, years=1.0)
    assert result.n == 59
    assert result.rule1 is False
    assert result.rule2 is None
    assert result.rule3 is None
    assert result.rule4 is None
    assert result.rule5 is None
    assert result.rule6 is None
    assert result.passed is False
    assert result.dsr is None
    assert result.boot_p5 is None
    assert result.diff_p5 is None
    assert result.mar_config is None
    assert result.mar_bh is None
    assert result.stressed is None


def test_the_minimum_trade_count_is_inclusive_and_configurable() -> None:
    assert evaluate(planted(0.5, n=60), zero_mean(), FLAT_BH, years=1.0).rule1 is True
    assert evaluate(planted(0.5, n=30), zero_mean(), FLAT_BH, years=1.0, min_trades=30).rule1 is True


def test_a_planted_edge_beats_noise_on_bootstrap_and_difference() -> None:
    result = evaluate(planted(0.3, n=200), zero_mean(), FLAT_BH, years=1.0)
    assert result.rule1 is True
    assert result.rule3 is True
    assert result.rule4 is True
    assert result.rule5 is True
    # +0.3R over 200 trades is a per-observation SR of ~0.33 against an expected maximum
    # null SR of ~0.244 over 2,016 trials: real, but not 95% real. See the DSR tests.
    assert result.rule2 is False
    assert result.dsr is not None
    assert result.boot_p5 is not None and result.boot_p5 > 0.0
    assert result.diff_p5 is not None and result.diff_p5 > 0.0


def test_a_strong_planted_edge_clears_rules_two_three_and_four() -> None:
    result = evaluate(planted(0.5, n=200), zero_mean(), FLAT_BH, years=1.0)
    assert result.rule2 is True
    assert result.rule3 is True
    assert result.rule4 is True


def test_a_config_identical_to_its_baseline_fails_rule_four() -> None:
    series = planted(0.5, n=200)
    result = evaluate(series, series, FLAT_BH, years=1.0)
    assert result.rule4 is False
    assert result.passed is False


def test_an_empty_baseline_fails_rule_four_rather_than_skipping_it() -> None:
    result = evaluate(planted(0.5, n=200), [], FLAT_BH, years=1.0)
    assert result.rule4 is False
    assert result.diff_p5 is None
    assert result.rule3 is True  # the other rules were still evaluated


def test_losing_to_buy_and_hold_fails_rule_five() -> None:
    strong_bh = [1.0, 10.0]  # 900% CAGR, no drawdown at all
    result = evaluate(planted(0.5, n=200), zero_mean(), strong_bh, years=1.0)
    assert result.mar_bh == math.inf
    assert result.rule5 is False


def test_rule_five_compares_mar_of_the_compounded_r_curve_to_buy_and_hold() -> None:
    returns = planted(0.5, n=200)
    result = evaluate(returns, zero_mean(), FLAT_BH, years=2.0)
    assert result.mar_config == pytest.approx(mar(equity_curve_from_r(returns), 2.0), rel=1e-12)
    assert result.mar_bh == pytest.approx(mar(FLAT_BH, 2.0), rel=1e-12)


def test_no_stressed_series_leaves_rule_six_unevaluated() -> None:
    result = evaluate(planted(0.8, n=200), zero_mean(), FLAT_BH, years=1.0)
    assert result.rule6 is None
    assert result.stressed is None
    assert result.passed is False  # rule 6 was never shown to hold


def test_a_stressed_series_that_loses_the_edge_fails_rule_six_alone() -> None:
    result = evaluate(planted(0.8, n=200), zero_mean(), FLAT_BH, years=1.0, stressed_r=zero_mean(seed=5))
    assert (result.rule1, result.rule2, result.rule3, result.rule4, result.rule5) == (True,) * 5
    assert result.rule6 is False
    assert result.passed is False
    assert result.stressed is not None
    assert result.stressed.n == 200
    assert result.stressed.rule6 is None  # the nested run is rules 1-5 only


def test_a_config_that_survives_cost_stress_passes_the_whole_gate() -> None:
    result = evaluate(
        planted(0.8, n=200), zero_mean(), FLAT_BH, years=1.0, stressed_r=planted(0.6, n=200, seed=4)
    )
    assert result.rules() == {f"rule{i}": True for i in range(1, 7)}
    assert result.passed is True


def test_a_stressed_series_with_too_few_trades_fails_rule_six() -> None:
    result = evaluate(
        planted(0.8, n=200), zero_mean(), FLAT_BH, years=1.0, stressed_r=planted(0.8, n=10, seed=4)
    )
    assert result.rule6 is False
    assert result.stressed is not None
    assert result.stressed.rule1 is False


def test_rules_reports_every_rule_including_the_unevaluated_ones() -> None:
    rules = evaluate(planted(0.5, n=59), zero_mean(), FLAT_BH, years=1.0).rules()
    assert list(rules) == [f"rule{i}" for i in range(1, 7)]
    assert rules == {"rule1": False, **{f"rule{i}": None for i in range(2, 7)}}


def test_gate_result_is_frozen() -> None:
    result = evaluate(planted(0.5, n=200), zero_mean(), FLAT_BH, years=1.0)
    with pytest.raises(ValidationError):
        result.rule1 = False


def test_evaluate_is_reproducible_for_a_seed() -> None:
    args = (planted(0.5, n=200), zero_mean(), FLAT_BH)
    first = evaluate(*args, years=1.0, seed=11)
    second = evaluate(*args, years=1.0, seed=11)
    assert first == second
    assert evaluate(*args, years=1.0, seed=12).boot_p5 != first.boot_p5


def test_evaluate_passes_the_trial_variance_through_to_the_deflated_sharpe() -> None:
    returns = planted(0.5, n=200)
    result = evaluate(returns, zero_mean(), FLAT_BH, years=1.0, trial_sr_variance=1.0)
    assert result.dsr is not None
    assert result.dsr.sr_star == pytest.approx(expected_sr_star(2016, 1.0), abs=1e-12)
    assert result.rule2 is False  # a variance that large deflates any 200-trade record away


# --- input validation ---------------------------------------------------------


@pytest.mark.parametrize("bad", [[1.0, float("nan")], [1.0, float("inf")], [-float("inf")]])
def test_non_finite_r_multiples_are_rejected(bad: list[float]) -> None:
    # A NaN R multiple means a trade whose risk was never established; silently averaging
    # it away would corrupt every statistic downstream.
    with pytest.raises(ValueError):
        sharpe(bad)
    with pytest.raises(ValueError):
        deflated_sharpe(bad, n_trials=10)


def test_moments_of_a_series_too_short_to_have_them_are_zero() -> None:
    for short in ([], [1.5]):
        assert skewness(short) == 0.0
        assert excess_kurtosis(short) == 0.0


def test_a_negative_trial_variance_is_rejected() -> None:
    with pytest.raises(ValueError):
        deflated_sharpe(planted(0.5), n_trials=100, trial_sr_variance=-1.0)


@pytest.mark.parametrize("variance", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_trial_variance_is_rejected(variance: float) -> None:
    # A NaN V would sail through `sqrt` and poison sr_star, prob and rule 2 silently.
    with pytest.raises(ValueError):
        deflated_sharpe(planted(0.5), n_trials=2016, trial_sr_variance=variance)


@pytest.mark.parametrize(
    ("n", "n_resamples", "p"),
    [(0, 10, 0.1), (-1, 10, 0.1), (10, 0, 0.1), (10, 10, 0.0), (10, 10, 1.5), (10, 10, -0.2)],
)
def test_bootstrap_indices_reject_degenerate_arguments(n: int, n_resamples: int, p: float) -> None:
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(n, n_resamples, p=p)


def test_difference_p5_rejects_an_empty_arm() -> None:
    with pytest.raises(ValueError):
        bootstrap_difference_p5([], planted(0.1, n=10))
    with pytest.raises(ValueError):
        bootstrap_difference_p5(planted(0.1, n=10), [])


def test_a_non_finite_equity_curve_is_rejected() -> None:
    with pytest.raises(ValueError):
        max_drawdown([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError):
        max_drawdown([1.0, float("inf")])


@pytest.mark.parametrize(("risk_pct", "start"), [(0.0, 1.0), (-0.01, 1.0), (0.01, 0.0), (0.01, -5.0)])
def test_equity_curve_rejects_a_degenerate_risk_or_starting_equity(risk_pct: float, start: float) -> None:
    with pytest.raises(ValueError):
        equity_curve_from_r([1.0], risk_pct=risk_pct, start=start)


def test_equity_curve_rejects_an_r_multiple_that_would_wipe_the_account_out() -> None:
    # -100R at 1% risk lands exactly on zero equity: ruin, not a curve with a deep drawdown.
    with pytest.raises(ValueError, match=r"index 1"):
        equity_curve_from_r([1.0, -100.0, 1.0], risk_pct=0.01)
    with pytest.raises(ValueError, match=r"-150"):
        equity_curve_from_r([-150.0], risk_pct=0.01)
    assert equity_curve_from_r([-99.0], risk_pct=0.01) == pytest.approx([1.0, 0.01], rel=1e-12)


def test_rule_three_and_rule_four_draw_the_config_resamples_once() -> None:
    # The p5 the gate reports is exactly the standalone one: rules 3 and 4 share the draw.
    config_r = planted(0.5, n=200)
    result = evaluate(config_r, zero_mean(), FLAT_BH, years=1.0, seed=13)
    assert result.boot_p5 == stationary_bootstrap_p5(config_r, seed=13)


def test_the_baseline_trade_count_is_reported_so_an_empty_baseline_is_legible() -> None:
    assert evaluate(planted(0.5, n=200), zero_mean(n=140), FLAT_BH, years=1.0).baseline_n == 140
    empty = evaluate(planted(0.5, n=200), [], FLAT_BH, years=1.0)
    assert empty.baseline_n == 0
    assert empty.rule4 is False  # the report can now say "no baseline trades", not just "failed"


def test_rules_one_to_five_passed_summarises_the_nested_stressed_verdict() -> None:
    result = evaluate(
        planted(0.8, n=200), zero_mean(), FLAT_BH, years=1.0, stressed_r=planted(0.6, n=200, seed=4)
    )
    assert result.stressed is not None
    assert result.stressed.rules_1_to_5_passed is True
    assert result.stressed.passed is False  # the nested result's own rule 6 is never evaluated
    assert result.rule6 is True


def test_rules_one_to_five_passed_is_false_when_the_stressed_series_fails_a_rule() -> None:
    result = evaluate(planted(0.8, n=200), zero_mean(), FLAT_BH, years=1.0, stressed_r=zero_mean(seed=5))
    assert result.stressed is not None
    assert result.stressed.rules_1_to_5_passed is False
    assert result.rule6 is False


def test_a_zero_drawdown_tie_goes_to_buy_and_hold() -> None:
    result = evaluate(all_positive(), zero_mean(), RISING_BH, years=1.0)
    assert result.mar_config == math.inf
    assert result.mar_bh == math.inf
    assert result.rule5 is False  # the comparison is strict, so a tie is a failure


def test_an_infinite_mar_survives_json_serialisation_as_a_constant() -> None:
    result = evaluate(all_positive(), zero_mean(), RISING_BH, years=1.0)
    dumped = result.model_dump_json()
    assert "Infinity" in dumped
    assert math.isinf(json.loads(dumped)["mar_config"])
    assert math.isinf(json.loads(dumped)["mar_bh"])
