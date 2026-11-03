"""Step 5: Monte Carlo check. Build fake returns with known answers, run Steps 2-4 and 6, and see
whether the estimates land on the truth and the 95% intervals contain it about 95% of the time."""

import numpy as np

from estimation import bootstrap_vol_betas, fit_by_regime, se_interval
from factors import vol_shock
from ols import add_intercept, residuals

VOL_BETA_RATIO = 1.5    # true vol_beta_high = 1.5 x vol_beta_low, a known regime difference to recover
N_SIMS = 100


def regime_shocks(mkt, dvix, high):
    """The vol shock built inside each regime (Step 2 per regime), on the original day order."""
    shock = np.zeros(len(mkt))
    shock[~high] = vol_shock(dvix[~high], mkt[~high])
    shock[high] = vol_shock(dvix[high], mkt[high])
    return shock


def noise_sd_by_regime(returns, mkt, dvix, high):
    """SD of the real regression misses per industry, in each regime. Returns (sd_low, sd_high)."""
    shock = regime_shocks(mkt, dvix, high)
    b_low, b_high = fit_by_regime(returns, mkt, dvix, high)
    sds = []
    for mask, b in [(~high, b_low), (high, b_high)]:
        X = add_intercept(np.column_stack([mkt[mask], shock[mask]]))
        sds.append(residuals(X, returns[mask], b).std(axis=0))
    return sds[0], sds[1]


def simulate_returns(mkt, dvix, high, beta, vol_beta_low, vol_beta_high, sd_low, sd_high, rng):
    """Fake industry returns with known betas: beta_mkt*mkt + beta_vol*shock + noise.

    mkt, dvix, high are the REAL series. beta, vol_beta_*, sd_*: (industries,).
    Returns (days, industries).
    """
    # Why use the real market, VIX and regime days: the check then runs on exactly the factor
    # behaviour (fat tails, crisis spikes, regime sizes) the real estimates face. Only the
    # industry returns are fake, so we know their true betas.
    shock = regime_shocks(mkt, dvix, high)
    vol_beta_by_day = np.where(high[:, None], vol_beta_high, vol_beta_low)  # (days, industries)
    # Why bigger noise on high-VIX days: real industry misses are bigger in stressed markets.
    sd_by_day = np.where(high[:, None], sd_high, sd_low)
    noise = rng.normal(size=vol_beta_by_day.shape) * sd_by_day
    # Why no alpha: the average return doesn't affect either beta, so it adds nothing to test.
    return mkt[:, None] * beta + shock[:, None] * vol_beta_by_day + noise


def monte_carlo_coverage(mkt, dvix, high, beta, vol_beta_low, vol_beta_high, sd_low, sd_high,
                         n_sims=N_SIMS, seed=0):
    """Run the full pipeline on n_sims fake datasets.

    Returns a dict with, for each statistic, the estimates (n_sims, industries) and whether its
    95% interval covered the truth (n_sims, industries, booleans).
    Statistics: vol_beta_low, vol_beta_high, diff = vol_beta_high - vol_beta_low, absdiff = |high| - |low|.
    """
    rng = np.random.default_rng(seed)
    truth = {
        "vol_beta_low": vol_beta_low,
        "vol_beta_high": vol_beta_high,
        "diff": vol_beta_high - vol_beta_low,
        "absdiff": np.abs(vol_beta_high) - np.abs(vol_beta_low),
    }
    estimates = {}
    covered = {}
    for name in truth:
        estimates[name] = np.zeros((n_sims, len(beta)))
        covered[name] = np.zeros((n_sims, len(beta)), dtype=bool)

    for s in range(n_sims):
        fake = simulate_returns(mkt, dvix, high, beta, vol_beta_low, vol_beta_high, sd_low, sd_high, rng)
        b_low, b_high = fit_by_regime(fake, mkt, dvix, high)
        # Why a new bootstrap seed per simulation: each fake dataset gets its own resamples.
        draws_low, draws_high = bootstrap_vol_betas(fake, mkt, dvix, high, seed=s + 1)
        stats = {
            "vol_beta_low": (b_low[2], draws_low),
            "vol_beta_high": (b_high[2], draws_high),
            "diff": (b_high[2] - b_low[2], draws_high - draws_low),
            "absdiff": (np.abs(b_high[2]) - np.abs(b_low[2]), np.abs(draws_high) - np.abs(draws_low)),
        }
        for name in stats:
            est, draws = stats[name]
            lower, upper, _ = se_interval(est, draws)
            estimates[name][s] = est
            covered[name][s] = (lower <= truth[name]) & (truth[name] <= upper)

    return {"truth": truth, "estimates": estimates, "covered": covered}
