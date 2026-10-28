"""Steps 3-4: industry regressions on the market and the vol shock, full sample and by regime."""

import numpy as np
import pandas as pd

from factors import vol_shock
from ols import add_intercept, ols

COEF_NAMES = ["alpha", "beta", "gamma"]


def fit_industries(returns: np.ndarray, mkt: np.ndarray, dvix: np.ndarray) -> np.ndarray:
    """Step 3: regress every industry's excess return on [1, Mkt-RF, vol shock].

    returns: (days, industries). mkt, dvix: (days,).
    Returns b: (3, industries). Row 0 = alpha, row 1 = beta (market), row 2 = gamma (vol shock).
    """
    # Why the vol shock is built here, from the same days as the regression: when this is
    # called on one regime's days, the shock is that regime's own Gram-Schmidt residual,
    # so beta is that regime's plain market beta (DEV_GUIDE decision).
    shock = vol_shock(dvix, mkt)
    X = add_intercept(np.column_stack([mkt, shock]))
    # Why one call for all industries: X is the same for every industry, so OLS handles
    # all 49 columns of `returns` at once (identical to fitting them one by one).
    return ols(X, returns)


def fit_by_regime(returns: np.ndarray, mkt: np.ndarray, dvix: np.ndarray, high: np.ndarray):
    """Step 4: run Step 3 separately on low-VIX days and high-VIX days.

    high: (days,) booleans from regime_flag. Returns (b_low, b_high), each (3, industries).
    """
    low = ~high
    b_low = fit_industries(returns[low], mkt[low], dvix[low])
    b_high = fit_industries(returns[high], mkt[high], dvix[high])
    return b_low, b_high


def coef_table(b: np.ndarray, industries) -> pd.DataFrame:
    """Turn a (3, industries) coefficient array into a table: one row per industry."""
    return pd.DataFrame(b.T, index=industries, columns=COEF_NAMES)


# Step 6: moving-block bootstrap

BLOCK_LENGTH = 20    # trading days, about one month; ~ n^(1/3) for n = 9,228 days
N_RESAMPLES = 1000
Z_95 = 1.96          # a normal distribution has 95% of its mass within ±1.96 SDs


def block_indices(n_days: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap resample of day positions: random runs of consecutive days, glued together.

    Returns an array of n_days positions into the original timeline (repeats allowed).
    """
    # Why blocks of consecutive days: crisis days cluster, so a day's noise is related to its
    # neighbours'. Drawing whole runs keeps that clustering inside each resample; drawing
    # single days would scatter it and make the intervals too narrow.
    n_blocks = int(np.ceil(n_days / block_length))
    starts = rng.integers(0, n_days - block_length + 1, size=n_blocks)
    # Each row is one block: its start plus 0, 1, ..., block_length-1. Flatten, trim to n_days.
    offsets = np.arange(block_length)
    positions = (starts[:, None] + offsets[None, :]).ravel()
    return positions[:n_days]


def bootstrap_gammas(returns, mkt, dvix, high, block_length=BLOCK_LENGTH,
                     n_resamples=N_RESAMPLES, seed=0):
    """Step 6: re-estimate gamma_low and gamma_high on many block-bootstrap resamples.

    Returns (draws_low, draws_high), each (n_resamples, industries).
    """
    rng = np.random.default_rng(seed)  # Why a fixed seed: rerunning gives the same intervals
    n_days, n_industries = returns.shape
    draws_low = np.zeros((n_resamples, n_industries))
    draws_high = np.zeros((n_resamples, n_industries))
    for i in range(n_resamples):
        # Why resample the full timeline, not each regime on its own: blocks keep crisis runs
        # intact even where they cross the regime line, and each day carries its own regime
        # flag, so the low and high estimates in one resample come from the same fake history.
        # That pairing is what makes the high-minus-low interval valid.
        idx = block_indices(n_days, block_length, rng)
        b_low, b_high = fit_by_regime(returns[idx], mkt[idx], dvix[idx], high[idx])
        draws_low[i] = b_low[2]
        draws_high[i] = b_high[2]
    return draws_low, draws_high


def se_interval(estimate: np.ndarray, draws: np.ndarray):
    """95% interval: estimate ± 1.96 × bootstrap standard error.

    estimate: (industries,). draws: (n_resamples, industries), the same statistic per resample.
    Returns (lower, upper, se), each (industries,).
    """
    # Why a standard-error interval and not percentiles: the bootstrap only has to estimate a
    # spread (variance), which is the job the n^(1/3) block-length rule is optimal for
    # (Hall, Horowitz & Jing 1995). Assumes the draws are roughly bell-shaped; checked by histogram.
    se = draws.std(axis=0, ddof=1)
    return estimate - Z_95 * se, estimate + Z_95 * se, se
