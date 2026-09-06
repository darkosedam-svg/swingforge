"""Statistical gate for the tournament: does an out-of-sample track record survive the search?

2,016 configs are swept, so the best in-sample Sharpe is mostly the maximum of 2,016 draws
from a null distribution. Everything here exists to price that in. The six rules of spec
section 6 run over pooled OOS trades expressed as plain R multiples (P&L per unit of risk),
so the gate is independent of the engine, the broker and the exit rules - hand it a
sequence of floats.

No scipy: the normal CDF comes from `math.erfc` and the quantile from Acklam's rational
approximation plus one Halley refinement, which lands within a few ulps of the exact
inverse everywhere the tournament looks - see `norm_ppf` for the far upper tail, the one
place it does not.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field

__all__ = [
    "DSRResult",
    "EULER_MASCHERONI",
    "GateResult",
    "bootstrap_difference_p5",
    "cagr",
    "deflated_sharpe",
    "equity_curve_from_r",
    "evaluate",
    "excess_kurtosis",
    "mar",
    "max_drawdown",
    "norm_cdf",
    "norm_ppf",
    "sharpe",
    "skewness",
    "stationary_bootstrap_indices",
    "stationary_bootstrap_p5",
]

EULER_MASCHERONI = 0.5772156649
"""Gamma. Weights the two order-statistic terms of the expected maximum null Sharpe."""

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Acklam's coefficients for the inverse normal CDF (relative error < 1.15e-9 before
# refinement); `_P_LOW` splits the central rational branch from the two tail branches.
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def norm_cdf(x: float) -> float:
    """Standard normal CDF, exact to within the error of `math.erfc`.

    `erfc` rather than `1 + erf`: forming the 1 cancels the leading digits of a small left
    tail away (at `x = -7` the erf route is wrong in the sixth significant figure), and the
    deflated Sharpe reads exactly that tail when a track record is far from significant.
    """
    return 0.5 * math.erfc(-float(x) / _SQRT2)


def norm_ppf(p: float) -> float:
    """Standard normal quantile for `0 < p < 1`.

    Acklam's rational approximation refined once by Halley's method against `math.erfc`.
    Through the centre and the whole lower tail that lands within a few ulps of the exact
    inverse; the two quantiles the tournament actually asks for, `1 - 1/2016` and
    `1 - 1/(2016 e)`, are good to ~1e-14 and round-trip through `norm_cdf` exactly.

    The far *upper* tail is weaker, because the Halley residual `Phi(x) - p` cancels once
    both terms sit within an ulp of 1: the error grows to roughly 1e-8 absolute at
    `p = 1 - 1e-12`. Past that no implementation can do better - the double nearest such a
    `p` is itself worth ~1.5e-5 in `x`. Mirror through `-norm_ppf(1 - p)` if a far upper
    quantile ever has to be exact.

    Raises `ValueError` outside the open unit interval, where the quantile is infinite or
    undefined.
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must lie strictly between 0 and 1, got {p!r}")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = _acklam_tail(q)
    elif p > 1.0 - _P_LOW:
        q = math.sqrt(-2.0 * math.log1p(-p))
        x = -_acklam_tail(q)
    else:
        q = p - 0.5
        r = q * q
        num = (
            (((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r + _ACKLAM_A[4]
        ) * r + _ACKLAM_A[5]  # noqa: E501
        den = (
            (((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r + _ACKLAM_B[4]
        ) * r + 1.0  # noqa: E501
        x = num * q / den

    # One Halley step on `Phi(x) - p = 0`, using erfc for a tail-accurate residual.
    error = 0.5 * math.erfc(-x / _SQRT2) - p
    step = error * _SQRT_2PI * math.exp(x * x / 2.0)
    return x - step / (1.0 + x * step / 2.0)


def _acklam_tail(q: float) -> float:
    """Acklam's lower-tail branch evaluated at `q = sqrt(-2 ln p)`."""
    num = (
        (((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q + _ACKLAM_C[4]
    ) * q + _ACKLAM_C[5]  # noqa: E501
    den = (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0
    return num / den


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """A 1-D float64 view of a sequence of R multiples."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size and not np.all(np.isfinite(array)):
        raise ValueError("returns must be finite; NaN and infinity are rejected")
    return array


def sharpe(returns: Sequence[float] | np.ndarray) -> float:
    """Per-observation Sharpe ratio: mean over the sample standard deviation (ddof=1).

    Deliberately *not* annualised. Every trial in the tournament is measured on the same
    per-trade frequency, so a common sqrt(periods) factor would cancel out of the deflated
    Sharpe anyway while inviting a wrong periods-per-year constant. Returns 0.0 for fewer
    than two observations or a zero standard deviation, where the ratio is undefined.
    """
    array = _as_array(returns)
    if array.size < 2:
        return 0.0
    std = float(array.std(ddof=1))
    if std <= 0.0:
        return 0.0
    return float(array.mean()) / std


def _central_moment(array: np.ndarray, order: int) -> float:
    return float(((array - array.mean()) ** order).mean())


def skewness(returns: Sequence[float] | np.ndarray) -> float:
    """Population (biased) skewness `m3 / m2**1.5`, as used by Bailey & Lopez de Prado.

    No sample-size correction: the deflated Sharpe formula is stated in terms of the
    population moments of the return distribution. Zero for a degenerate series.
    """
    array = _as_array(returns)
    if array.size < 2:
        return 0.0
    m2 = _central_moment(array, 2)
    if m2 <= 0.0:
        return 0.0
    return _central_moment(array, 3) / m2**1.5


def excess_kurtosis(returns: Sequence[float] | np.ndarray) -> float:
    """Population excess kurtosis `m4 / m2**2 - 3`; 0.0 for a normal sample.

    Population moments, matching `skewness`. Add 3 to recover the non-excess kurtosis the
    deflated Sharpe formula calls for.
    """
    array = _as_array(returns)
    if array.size < 2:
        return 0.0
    m2 = _central_moment(array, 2)
    if m2 <= 0.0:
        return 0.0
    return _central_moment(array, 4) / m2**2 - 3.0


class DSRResult(BaseModel):
    """The deflated Sharpe ratio of one track record, with everything that produced it.

    `sr` is the per-observation Sharpe of the returns, `sr_star` the expected maximum
    Sharpe of `n_trials` independent null strategies, and `prob` the probability that the
    true Sharpe exceeds that threshold - the number gate rule 2 thresholds at 0.95.
    `skew` and `kurt` are the population moments that enter the formula, `kurt`
    **non-excess** (a normal sample reports 3.0, not 0.0).

    `ser_json_inf_nan="constants"`: an infinity here would otherwise serialise to `null`
    and read back as a missing measurement rather than an extreme one.
    """

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    sr: float
    sr_star: float
    prob: float
    n: int
    skew: float
    kurt: float
    n_trials: int


def _expected_max_sharpe(n_trials: int, variance: float) -> float:
    """`sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))`, Bailey & Lopez de Prado.

    The expected maximum of `N` independent Sharpe estimates drawn under the null, from the
    Gumbel approximation to the maximum of a normal sample. A single trial is not a search,
    so `N <= 1` deflates by nothing and returns 0.0 - the formula itself would ask for
    `Phi^-1(0)`, which is minus infinity.

    `variance` is checked before anything else: a NaN or infinite `V` would pass straight
    through `sqrt` into `sr_star`, and rule 2 would then report a silent `False` for every
    config instead of a loud failure.
    """
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError(f"trial_sr_variance must be finite and >= 0, got {variance!r}")
    if n_trials <= 1:
        return 0.0
    return math.sqrt(variance) * (
        (1.0 - EULER_MASCHERONI) * norm_ppf(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(
    returns: Sequence[float] | np.ndarray,
    n_trials: int,
    *,
    trial_sr_variance: float | None = None,
) -> DSRResult:
    """Deflated Sharpe ratio of `returns` against a search over `n_trials` configurations.

    Bailey & Lopez de Prado (2014). The Sharpe of the best of many trials is inflated by
    the selection itself, so the null is not "Sharpe > 0" but "Sharpe > the expected
    maximum of `n_trials` null trials". `prob` is that probability, adjusted for the skew
    and fat tails of the return distribution: negative skew and high kurtosis make a given
    Sharpe less trustworthy and push `prob` down.

    `trial_sr_variance` is `V`, the variance of the Sharpe estimates across the trials.
    Pass it when the tournament has actually observed the spread of Sharpe ratios over its
    configs. The default is the null-hypothesis variance of the Sharpe *estimator*,
    `1 / (T - 1)`, which is what to use when the per-trial Sharpes are not to hand. A
    negative, NaN or infinite `V` raises rather than propagating into `sr_star`.

    Fewer than three observations, or a zero standard deviation, leave the ratio undefined:
    the result then reports `sr = 0.0` and `prob = 0.0` - a track record that cannot be
    measured never passes the gate.
    """
    array = _as_array(returns)
    n = int(array.size)
    if n < 3 or float(array.std(ddof=1)) <= 0.0:  # short-circuits before ddof=1 can divide by 0
        return DSRResult(sr=0.0, sr_star=0.0, prob=0.0, n=n, skew=0.0, kurt=0.0, n_trials=n_trials)

    sr = sharpe(array)
    variance = 1.0 / (n - 1) if trial_sr_variance is None else float(trial_sr_variance)
    threshold = _expected_max_sharpe(n_trials, variance)
    g3 = skewness(array)
    g4 = excess_kurtosis(array) + 3.0

    # Non-negative for any real distribution: the moment inequality g4 >= g3**2 + 1 makes
    # this quadratic in SR have a non-positive discriminant. Guarded anyway.
    spread = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    prob = 0.0 if spread <= 0.0 else norm_cdf((sr - threshold) * math.sqrt(n - 1) / math.sqrt(spread))
    return DSRResult(sr=sr, sr_star=threshold, prob=prob, n=n, skew=g3, kurt=g4, n_trials=n_trials)


# --- stationary bootstrap -----------------------------------------------------


_N_RESAMPLES = 1000
"""Resamples per bootstrap, shared by rules 3 and 4 so their p5s are on the same footing."""

_BLOCK_P = 0.1
"""Restart probability: mean block length 10 trades. A recorded, conservative choice."""


def stationary_bootstrap_indices(
    n: int,
    n_resamples: int,
    *,
    p: float = _BLOCK_P,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Row `i` of the returned `(n_resamples, n)` array indexes one stationary resample.

    Politis & Romano (1994). Each position either continues the previous block (with
    probability `1 - p`, stepping one index on and wrapping circularly past the end) or
    starts a fresh block at a uniformly drawn index. Block lengths are therefore geometric
    with mean `1 / p`, which is what preserves the short-range serial dependence an i.i.d.
    bootstrap would destroy - trades cluster, and a resample that ignored that would
    understate the spread of expectancy.

    Every resample is exactly `n` long, so a bootstrapped track record is the same size as
    the one it came from.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p!r}")
    generator = np.random.default_rng(0) if rng is None else rng

    fresh = generator.integers(0, n, size=(n_resamples, n))
    restart = generator.random((n_resamples, n)) < p
    indices = np.empty((n_resamples, n), dtype=np.int64)
    indices[:, 0] = fresh[:, 0]
    for column in range(1, n):
        continued = (indices[:, column - 1] + 1) % n
        indices[:, column] = np.where(restart[:, column], fresh[:, column], continued)
    return indices


def _bootstrap_statistics(
    array: np.ndarray,
    n_resamples: int,
    p: float,
    rng: np.random.Generator,
    statistic: Callable[[np.ndarray], float],
) -> np.ndarray:
    """`statistic` evaluated on each of `n_resamples` stationary resamples of `array`.

    Every gate rule uses the mean, and this is essentially the whole cost of `evaluate`, so
    `np.mean` takes a vectorised row-mean over the `(n_resamples, n)` block instead of a
    Python loop. Any other callable falls back to the loop; the two agree to within
    floating-point summation noise, which the tests pin at 1e-12.
    """
    indices = stationary_bootstrap_indices(array.size, n_resamples, p=p, rng=rng)
    resamples = array[indices]
    if statistic is np.mean:
        return np.asarray(resamples.mean(axis=1), dtype=np.float64)
    return np.array([float(statistic(row)) for row in resamples], dtype=np.float64)


def stationary_bootstrap_p5(
    x: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = _N_RESAMPLES,
    p: float = _BLOCK_P,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.mean,
) -> float:
    """5th percentile of `statistic` over stationary resamples of `x` - gate rule 3.

    The pessimistic end of the sampling distribution of expectancy. Requiring it to stay
    above 0 R asks for an edge that survives a plausible reordering of the same trades,
    not merely one that happened to average positive in the order it was dealt.
    """
    array = _as_array(x)
    if array.size == 0:
        raise ValueError("cannot bootstrap an empty series")
    stats = _bootstrap_statistics(array, n_resamples, p, np.random.default_rng(seed), statistic)
    return float(np.percentile(stats, 5))


def _difference_p5(
    left_means: np.ndarray,
    right: np.ndarray,
    n_resamples: int,
    p: float,
    rng: np.random.Generator,
) -> float:
    """Rule 4's p5 from an already-drawn left arm, so `evaluate` can draw that arm once."""
    right_means = _bootstrap_statistics(right, n_resamples, p, rng, np.mean)
    return float(np.percentile(left_means - right_means, 5))


def bootstrap_difference_p5(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = _N_RESAMPLES,
    p: float = _BLOCK_P,
    seed: int = 0,
) -> float:
    """5th percentile of `mean(a*) - mean(b*)` over stationary resamples - gate rule 4.

    The two arms are resampled **independently**, from one seeded generator so the whole
    comparison is reproducible. Independent is the right construction: the records are
    frequency-matched, not aligned - trade `i` of the config is not the same event as trade
    `i` of the baseline, and the two need not even be the same length, so there is nothing
    to pair. Above 0 means the config out-earns the baseline by a margin that survives
    resampling.
    """
    left = _as_array(a)
    right = _as_array(b)
    if left.size == 0 or right.size == 0:
        raise ValueError("cannot bootstrap an empty series")
    rng = np.random.default_rng(seed)
    left_means = _bootstrap_statistics(left, n_resamples, p, rng, np.mean)
    return _difference_p5(left_means, right, n_resamples, p, rng)


# --- equity curve metrics -----------------------------------------------------


def _curve(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """A validated equity curve: non-empty, finite and strictly positive throughout.

    A curve that touches zero has no meaningful percentage drawdown or growth rate, and
    silently absorbing one would report a nonsense MAR for a blown-up account.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("equity curve must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("equity curve must be finite")
    if not np.all(array > 0.0):
        raise ValueError("equity curve must be strictly positive")
    return array


def max_drawdown(curve: Sequence[float] | np.ndarray) -> float:
    """Deepest peak-to-trough fall as a fraction of the running peak; always >= 0."""
    array = _curve(curve)
    peak = np.maximum.accumulate(array)
    return float(max(((peak - array) / peak).max(), 0.0))


def cagr(curve: Sequence[float] | np.ndarray, years: float) -> float:
    """Compound annual growth rate of an equity curve over `years` calendar years."""
    array = _curve(curve)
    if years <= 0.0:
        raise ValueError("years must be > 0")
    return float((array[-1] / array[0]) ** (1.0 / years) - 1.0)


def mar(curve: Sequence[float] | np.ndarray, years: float) -> float:
    """MAR (Calmar) ratio: CAGR divided by maximum drawdown - gate rule 5's yardstick.

    Return per unit of the worst loss actually endured, which is what makes a config
    comparable to buy-and-hold on the same instrument. With no drawdown at all the ratio
    is undefined: a curve that only ever rose scores `inf`, and a flat one scores 0.0. Two
    such curves therefore tie at `inf`, and rule 5's comparison is strict, so a tie goes to
    buy-and-hold.
    """
    drawdown = max_drawdown(curve)
    growth = cagr(curve, years)
    if drawdown == 0.0:
        return math.inf if growth > 0.0 else 0.0
    return growth / drawdown


def equity_curve_from_r(
    rs: Sequence[float] | np.ndarray,
    *,
    risk_pct: float = 0.01,
    start: float = 1.0,
) -> list[float]:
    """Compound a sequence of R multiples into an equity curve at fixed fractional risk.

    Each trade risks `risk_pct` of current equity, so equity multiplies by
    `1 + risk_pct * r`. The curve is one point longer than `rs`: it opens at `start`
    before the first trade.

    A trade that would take equity to zero or below - `-100R` at 1% risk, and worse beyond
    that - is ruin, not a drawdown, and raises `ValueError` naming the R multiple and its
    index rather than returning a curve `max_drawdown` and `cagr` cannot describe. Callers
    sweeping many configs should catch it per config so one blown-up config is a failed
    row, not a lost run.
    """
    if risk_pct <= 0.0:
        raise ValueError("risk_pct must be > 0")
    if start <= 0.0:
        raise ValueError("start must be > 0")
    equity = float(start)
    curve = [equity]
    for index, value in enumerate(_as_array(rs)):
        r = float(value)
        factor = 1.0 + risk_pct * r
        if factor <= 0.0:
            raise ValueError(
                f"R multiple {r} at index {index} wipes the account out at risk_pct={risk_pct}: "
                f"equity would go to {equity * factor}"
            )
        equity *= factor
        curve.append(equity)
    return curve


# --- the six-rule gate --------------------------------------------------------

_RULES = ("rule1", "rule2", "rule3", "rule4", "rule5", "rule6")


class GateResult(BaseModel):
    """The verdict of spec section 6's six rules on one config's pooled OOS trades.

    Each rule is `True`, `False`, or `None` for "not evaluated". Rule 1 is the trade-count
    floor and it short-circuits: below it, none of the statistics mean anything, so rules
    2-6 stay `None` rather than being reported as failures. Rule 6 is `None` when no
    cost-stressed series was supplied - the gate was never asked the question.

    `stressed` holds the nested rules 1-5 verdict on the cost-stressed series; its own
    `rule6` is always `None`, so its `passed` is always `False`. Read `rule6` on the outer
    result, or `stressed.rules_1_to_5_passed`, for the stress verdict - never
    `stressed.passed`.

    `baseline_n` is how many baseline trades rule 4 had to compare against. 0 means there
    were none, so the report can say "no baseline trades" instead of reporting a bare
    failure against a baseline that never existed.

    `ser_json_inf_nan="constants"`: `mar_config` and `mar_bh` are `inf` for a curve that
    never drew down, and the default serialisation would flatten that to `null` - reading
    back as "not measured" rather than "never drew down".
    """

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    n: int
    baseline_n: int = 0
    rule1: bool | None = None
    rule2: bool | None = None
    rule3: bool | None = None
    rule4: bool | None = None
    rule5: bool | None = None
    rule6: bool | None = None
    dsr: DSRResult | None = None
    boot_p5: float | None = None
    diff_p5: float | None = None
    mar_config: float | None = None
    mar_bh: float | None = None
    stressed: GateResult | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """True only when all six rules are `True`; derived, so it cannot drift from them."""
        return all(getattr(self, rule) is True for rule in _RULES)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rules_1_to_5_passed(self) -> bool:
        """True when rules 1-5 all hold, rule 6 ignored - a nested result's stress verdict.

        `passed` is useless on a nested `stressed` result, whose `rule6` is always `None`;
        this is the honest reading of one, and what `evaluate` derives rule 6 from.
        """
        return all(getattr(self, rule) is True for rule in _RULES[:5])

    def rules(self) -> dict[str, bool | None]:
        """The six verdicts keyed `rule1`..`rule6`, for the run report's gate table."""
        return {rule: getattr(self, rule) for rule in _RULES}


def evaluate(
    config_r: Sequence[float] | np.ndarray,
    baseline_r: Sequence[float] | np.ndarray,
    bh_curve: Sequence[float] | np.ndarray,
    *,
    years: float,
    n_trials: int = 2016,
    min_trades: int = 60,
    stressed_r: Sequence[float] | np.ndarray | None = None,
    seed: int = 0,
    trial_sr_variance: float | None = None,
) -> GateResult:
    """Run the six gate rules of spec section 6 over one config's pooled OOS trades.

    1. at least `min_trades` trades;
    2. deflated Sharpe probability >= 0.95 against `n_trials` searched configs;
    3. stationary-bootstrap 5th-percentile expectancy above 0 R;
    4. bootstrap difference against the frequency-matched baseline, 5th percentile above 0;
    5. MAR of the compounded R curve above buy-and-hold's MAR on the same instrument;
    6. rules 1-5 hold again on `stressed_r`, the same trades re-priced at doubled
       spread/slippage and 1.5x funding.

    `config_r`, `baseline_r` and `stressed_r` are plain R multiples, one per closed OOS
    trade, in chronological order - the bootstrap's block structure assumes that order.
    `bh_curve` is a buy-and-hold equity curve for the same instrument and window, and
    `years` its length in calendar years (shared by both MAR figures, so it must cover the
    same span).

    An empty `baseline_r` fails rule 4 rather than skipping it: with nothing to beat, the
    config has not been shown to beat anything - `baseline_n` records that it was empty, so
    the report can say so rather than implying the config lost a comparison.

    Rules 3 and 4 **share** the config's resamples: the left arm is drawn once and feeds
    both `boot_p5` and the difference distribution (`boot_p5` is therefore exactly
    `stationary_bootstrap_p5(config_r, seed=seed)`). The two rules are consequently not
    independent tests of the same data - benign here, since the gate demands both rather
    than combining their p-values, and it halves the work.

    Rule 5 compares strictly, so two curves that both never drew down tie at `inf` and the
    tie goes to buy-and-hold. `equity_curve_from_r` raises `ValueError` on an R multiple
    that would wipe the account out, so a tournament sweeping configs should wrap each
    `evaluate` call and record the failure as a row rather than losing the run.
    """
    array = _as_array(config_r)
    n = int(array.size)
    if n < min_trades:
        return GateResult(n=n, rule1=False)

    dsr = deflated_sharpe(array, n_trials, trial_sr_variance=trial_sr_variance)

    # One draw of the config's resamples, feeding rule 3's p5 and rule 4's left arm. The
    # generator is then handed on to the right arm, so both match the standalone helpers.
    rng = np.random.default_rng(seed)
    left_means = _bootstrap_statistics(array, _N_RESAMPLES, _BLOCK_P, rng, np.mean)
    boot_p5 = float(np.percentile(left_means, 5))

    baseline = _as_array(baseline_r)
    diff_p5 = (
        None if baseline.size == 0 else _difference_p5(left_means, baseline, _N_RESAMPLES, _BLOCK_P, rng)
    )

    mar_config = mar(equity_curve_from_r(array), years)
    mar_bh = mar(bh_curve, years)

    stressed = None
    rule6 = None
    if stressed_r is not None:
        stressed = evaluate(
            stressed_r,
            baseline_r,
            bh_curve,
            years=years,
            n_trials=n_trials,
            min_trades=min_trades,
            stressed_r=None,
            seed=seed,
            trial_sr_variance=trial_sr_variance,
        )
        rule6 = stressed.rules_1_to_5_passed

    return GateResult(
        n=n,
        baseline_n=int(baseline.size),
        rule1=True,
        rule2=dsr.prob >= 0.95,
        rule3=boot_p5 > 0.0,
        rule4=False if diff_p5 is None else diff_p5 > 0.0,
        rule5=mar_config > mar_bh,
        rule6=rule6,
        dsr=dsr,
        boot_p5=boot_p5,
        diff_p5=diff_p5,
        mar_config=mar_config,
        mar_bh=mar_bh,
        stressed=stressed,
    )
