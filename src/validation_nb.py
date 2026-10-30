import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import BLUE, GRAY, ORANGE, new_figure
    from data import INDUSTRY_NAMES, build_data
    from factors import vol_shock
    from ols import add_intercept, ols
    from estimation import BLOCK_LENGTH, N_RESAMPLES, bootstrap_gammas, coef_table, fit_by_regime, fit_industries, regime_interval_table, se_interval
    from validation import GAMMA_RATIO, N_SIMS, monte_carlo_coverage, noise_sd_by_regime
    from hypotheses import evaluate_h1, evaluate_h2, significance_counts

    return (
        BLOCK_LENGTH,
        BLUE,
        GAMMA_RATIO,
        GRAY,
        INDUSTRY_NAMES,
        N_RESAMPLES,
        N_SIMS,
        ORANGE,
        add_intercept,
        bootstrap_gammas,
        build_data,
        coef_table,
        evaluate_h1,
        evaluate_h2,
        fit_by_regime,
        fit_industries,
        mo,
        monte_carlo_coverage,
        new_figure,
        noise_sd_by_regime,
        np,
        ols,
        pd,
        regime_interval_table,
        se_interval,
        significance_counts,
        vol_shock,
    )



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 4: Monte Carlo check of the confidence intervals 🤖

    *Companion to `main.py`. Code: `src/validation.py`.*

    **Bottom line**

    - On 100 simulated datasets with **known** answers, the pipeline's estimates land on the truth.
    - The 95% intervals contain the true value **94.5–94.9%** of the time for all four statistics (4,900 intervals each), right at the 95% target.
    - Limit: the simulated noise doesn't cluster like real crisis noise, so this checks the machinery (code, interval formula, regime handling), not the 20-day block length.

    *Runs the full pipeline 100 times: takes about a minute to load.*

    **Units used throughout:** returns are in **percent per day** (0.5 = 0.5%); VIX changes are in **VIX points**. So **γ (gamma)** is the % move in an industry for a 1-point vol shock: γ = −0.10 means a 1-point surprise jump in VIX goes with the industry being 0.10% lower that day.
    """)
    return


@app.cell
def _(build_data):
    returns_df, factors_df = build_data()
    industries = returns_df.columns

    # NumPy arrays for the math; the DataFrames keep dates and names for display
    R = returns_df.to_numpy()                 # (days, 49) industry excess returns, decimals
    mkt = factors_df["Mkt-RF"].to_numpy()     # (days,) market excess return, decimals
    dvix_arr = factors_df["dVIX"].to_numpy()  # (days,) VIX change, points
    high = factors_df["high"].to_numpy()      # (days,) True = high-VIX regime
    return R, dvix_arr, factors_df, high, industries, mkt


@app.cell(hide_code=True)
def _(R, dvix_arr, fit_industries, mkt):
    b_full = fit_industries(R, mkt, dvix_arr)
    return (b_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Monte Carlo check (before trusting any interval)

    **Question.** If we *knew* the true answer, would Steps 2–4 and 6 find it? Do the 95% intervals really contain the truth about 95% of the time?

    **How it's built.**

    1. Keep the **real** market, VIX and regime days. Invent 49 fake industries with a known β and γ:
       - β = each real industry's full-sample β;
       - γ_low = each real industry's full-sample γ;
       - γ_high = 1.5 × γ_low (a known regime difference to recover);
       - noise: random normal, sized like each real industry's misses in each regime (about 1.5× bigger on high-VIX days).
    2. Run the exact pipeline (regime fits + 1,000-resample bootstrap) on the fake data.
    3. Repeat 100 times → 100 × 49 = 4,900 intervals per statistic. Count how often they contain the truth.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Real factors, fake industries.** | The check then faces the real fat tails, crisis spikes and regime sizes. Only the industry returns are made up, so we know their true γ. | Fully fake factors: easier, but tests the pipeline on data that looks nothing like ours. |
    | **Realistic sizes for β, γ and noise** (taken from the real fits). | Interval widths and coverage then match what the real data will produce. | Round numbers like γ = 0.1: could make the problem much easier or harder than the real one. |
    | **γ_high = 1.5 × γ_low.** | Gives a known, moderate regime difference, so we can check the difference interval centers on it. | No difference: wouldn't test whether a real difference is recovered. |
    | **Exactly the same settings as the real run** (20-day blocks, 1,000 resamples). | We test the procedure we actually use. | Cheaper settings: faster, but then we'd be testing something else. |
    | **100 simulations.** | 4,900 intervals pin overall coverage to about ±0.6%. Runs in about a minute. | More: slower for little gain ("keep it light"). |

    **What this can't check.** The fake noise is independent day to day; real crisis noise clusters. So this checks the machinery (code, SE formula, regime handling), not whether 20 days is the right block length.
    """)
    return


@app.cell
def _(GAMMA_RATIO, N_SIMS, R, b_full, dvix_arr, high, mkt, monte_carlo_coverage, noise_sd_by_regime, np, pd):
    sd_low, sd_high = noise_sd_by_regime(R, mkt, dvix_arr, high)
    print(f"noise SD, high-VIX ÷ low-VIX days (median industry): {np.median(sd_high / sd_low):.2f}")

    mc = monte_carlo_coverage(
        mkt, dvix_arr, high,
        beta=b_full[1], gamma_low=b_full[2], gamma_high=GAMMA_RATIO * b_full[2],
        sd_low=sd_low, sd_high=sd_high, n_sims=N_SIMS,
    )

    _labels = {
        "gamma_low": "γ low",
        "gamma_high": "γ high",
        "diff": "γ high − γ low",
        "absdiff": "|γ high| − |γ low|",
    }
    _rows = {}
    for _name in _labels:
        _covered = mc["covered"][_name]
        _est = mc["estimates"][_name]
        # Why bias in SE units: compares the average miss to the typical interval half-width,
        # so it's on the same scale for every industry. Near 0 = centered on the truth.
        _bias_in_se = (_est.mean(axis=0) - mc["truth"][_name]) / _est.std(axis=0, ddof=1)
        _per_industry = _covered.mean(axis=0)
        _rows[_labels[_name]] = {
            "coverage, all 4,900 intervals": _covered.mean(),
            "lowest industry": _per_industry.min(),
            "highest industry": _per_industry.max(),
            "avg bias (in SEs)": np.abs(_bias_in_se).mean(),
        }
    mc_table = pd.DataFrame(_rows).T.round(3)
    print("target: coverage ≈ 0.95; avg bias ≈ 0 (a few tenths at most is noise from 100 simulations)")
    mc_table
    return (mc,)


@app.cell
def _(BLUE, GRAY, ORANGE, mc, new_figure, np):
    _fig, _axes = new_figure(n_panels=2, width=11, height=4.2)

    # Left: average estimate vs truth for gamma high, one dot per industry
    _ax = _axes[0]
    _truth = mc["truth"]["gamma_high"] * 100
    _avg = mc["estimates"]["gamma_high"].mean(axis=0) * 100
    _lim = np.array([_truth.min(), _truth.max()]) * 1.15
    _ax.plot(_lim, _lim, color=GRAY, linewidth=1, label="perfect recovery")
    _ax.scatter(_truth, _avg, color=ORANGE, s=18, zorder=3, label="one fake industry")
    _ax.set_xlabel("true γ high (%/pt)")
    _ax.set_ylabel("average estimate over 100 simulations")
    _ax.set_title("Estimates land on the truth", fontsize=10)
    _ax.legend(frameon=False)

    # Right: coverage per industry for the difference interval
    _ax = _axes[1]
    _cov = mc["covered"]["diff"].mean(axis=0) * 100
    _ax.scatter(np.arange(len(_cov)), np.sort(_cov), color=BLUE, s=18, zorder=3)
    _ax.axhline(95, color=GRAY, linewidth=1)
    # Why this band: with 100 simulations, a correct 95% interval still scores
    # anywhere in about 91–99% per industry just from luck (binomial ± 2 SDs).
    _ax.axhspan(95 - 2 * np.sqrt(95 * 5 / 100), 95 + 2 * np.sqrt(95 * 5 / 100), color=GRAY, alpha=0.15)
    _ax.set_ylim(80, 101)
    _ax.set_xlabel("fake industries, sorted by coverage")
    _ax.set_ylabel("% of 100 intervals containing the truth")
    _ax.set_title("γ high − γ low interval coverage (gray band = expected luck)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
