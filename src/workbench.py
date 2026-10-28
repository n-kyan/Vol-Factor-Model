import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from data import build_data
    from factors import vol_shock
    from ols import add_intercept, ols
    from estimation import fit_industries, fit_by_regime, coef_table
    from estimation import BLOCK_LENGTH, N_RESAMPLES, bootstrap_gammas, se_interval
    from validation import GAMMA_RATIO, N_SIMS, monte_carlo_coverage, noise_sd_by_regime
    from hypotheses import evaluate_h1, evaluate_h2, significance_counts

    return (
        BLOCK_LENGTH,
        GAMMA_RATIO,
        N_RESAMPLES,
        N_SIMS,
        add_intercept,
        bootstrap_gammas,
        build_data,
        coef_table,
        evaluate_h1,
        evaluate_h2,
        fit_by_regime,
        fit_industries,
        monte_carlo_coverage,
        noise_sd_by_regime,
        se_interval,
        mo,
        np,
        ols,
        pd,
        plt,
        vol_shock,
    )


@app.cell
def _(plt):
    BLUE = "#2a78d6"     # low-VIX regime / main series
    ORANGE = "#eb6834"   # high-VIX regime
    GRAY = "#8a8985"     # reference lines and points
    INK = "#52514e"      # axis text


    def new_figure(n_panels=1, width=6.0, height=3.6):
        """Create a figure with n side-by-side panels and quiet axes."""
        fig, axes = plt.subplots(1, n_panels, figsize=(width, height), squeeze=False)
        axes = axes[0]
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(GRAY)
            ax.spines["bottom"].set_color(GRAY)
            ax.tick_params(colors=INK)
            ax.grid(alpha=0.2)
        return fig, axes

    return BLUE, GRAY, ORANGE, new_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Workbench 🤖

    Claude's build notebook. Each step runs on the real data and lays out three things:

    1. **Question:** what the step answers.
    2. **Decisions and why:** what was chosen and why, and the obvious alternative and why it loses.
    3. **Result:** what came out.

    This is scratch space for checking the work. The final presentation goes in `main.py`, in your words.

    **Units used throughout:**

    - Returns are in **percent per day**: 0.5 means 0.5%.
    - VIX changes are in **VIX points**.
    - So **γ (gamma)** is the % move in an industry for a 1-point vol shock. γ = −0.10 means "a 1-point surprise jump in VIX goes with the industry being 0.10% lower that day."
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
def _(mo):
    mo.md(r"""
    ## Step 2: the vol shock (Gram-Schmidt)

    **Question.** On a typical day, how much of VIX's move is just the market falling or rising, and what's left over? The leftover is the **vol shock**: news about volatility that the market move doesn't explain.

    **How it's built.** Regress each day's VIX change on that day's market return, and keep the miss (the residual). If the market fell 2% and that usually goes with VIX rising 3 points, but today VIX rose 5, the vol shock is about +2.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Market goes first.** VIX is cleaned of the market, not the other way round. | The question is about vol moves *beyond* market moves. The market keeps everything it can explain, and only the leftover counts as a vol shock. | Clean the market of VIX instead. Then "market" would mean "market moves VIX can't explain", β would stop being the ordinary market beta, and the question would change. |
    | **Include an intercept** in this regression. | It makes the vol shock average exactly 0, so it only measures surprises, not VIX's average drift. | Leave it out: any average drift in VIX would leak into the shock. |
    | **VIX changes in points** (already decided). | It matches how options desks measure vol moves (vega), and it's simple to read. | Percentage changes: in the post-project analysis. |

    **Why cleaning makes the regressors uncorrelated.** The vol shock is a regression miss, and regression misses are uncorrelated with the regressor by construction (Lesson 1, "no drift"). So the market and the vol shock don't overlap. Each industry's response can then be split cleanly into "market part" and "vol-shock part".
    """)
    return


@app.cell
def _(dvix_arr, high, mkt, np, pd, vol_shock):
    def corr(a, b):
        """Correlation between two arrays."""
        return np.corrcoef(a, b)[0, 1]


    shock = vol_shock(dvix_arr, mkt)

    _rows = {
        "all days": corr(mkt, dvix_arr),
        "low-VIX days": corr(mkt[~high], dvix_arr[~high]),
        "high-VIX days": corr(mkt[high], dvix_arr[high]),
    }
    _shares = {}
    for _name in _rows:
        _shares[_name] = _rows[_name] ** 2  # R²: share of ΔVIX variance the market explains
    market_vix_corr = pd.DataFrame({
        "corr(Mkt-RF, ΔVIX)": _rows,
        "share of ΔVIX variance explained by market": _shares,
        "days": {"all days": len(mkt), "low-VIX days": int((~high).sum()), "high-VIX days": int(high.sum())},
    }).round(3)

    print("check: corr(vol shock, Mkt-RF) =", round(corr(shock, mkt), 12), "← should be 0")
    print("check: mean vol shock =", round(shock.mean(), 12), "← should be 0")
    market_vix_corr
    return (shock,)


@app.cell
def _(BLUE, GRAY, ORANGE, add_intercept, dvix_arr, high, mkt, new_figure, np, ols):
    _fig, _axes = new_figure(n_panels=2, width=11, height=4.2)
    _cases = [("Low-VIX days", ~high, BLUE), ("High-VIX days", high, ORANGE)]
    for _ax, (_name, _mask, _color) in zip(_axes, _cases):
        _m = mkt[_mask] * 100
        _d = dvix_arr[_mask]
        _b = ols(add_intercept(_m), _d)
        _ax.scatter(_m, _d, s=3, alpha=0.25, color=_color)
        _grid = np.array([-10.0, 10.0])
        _ax.plot(_grid, _b[0] + _b[1] * _grid, color=GRAY, linewidth=2)
        _ax.set_xlim(-10, 10)
        _ax.set_ylim(-20, 25)
        _ax.set_title(f"{_name}: 1% market drop ≈ VIX {-_b[1]:+.2f} points", fontsize=10)
        _ax.set_xlabel("market excess return, % that day")
    _axes[0].set_ylabel("VIX change, points")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each dot is one day. The gray line is what the market alone predicts for VIX. A day's **vol shock** is how far its dot sits above or below that line.

    ## Step 3: full-sample sensitivities

    **Question.** For each industry: how much does it move with the market (**β**, beta), and how much with vol shocks (**γ**, gamma)?

    **How it's built.** For each industry, one regression across all 9,228 days:

    $$
    \text{industry excess return}_t = \alpha + \beta \cdot \text{Mkt-RF}_t + \gamma \cdot \text{vol shock}_t + \text{miss}_t
    $$

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Time-series regression**: one regression per industry, across days. | We want each industry's *own* sensitivity. | Cross-sectional regression (across industries, day by day) answers a different question: whether exposure relates to average returns. That's in the back pocket, not the core. |
    | **Excess returns** (minus the risk-free rate) on both sides. | Matches how Mkt-RF is defined, so β is the standard market beta. | Raw returns: the risk-free rate would leak into α. |
    | **Intercept α** included. | Each industry has its own average daily return. Without α, β and γ would bend to absorb it (Lesson 1). | — |
    | **All 49 industries fit in one call.** | They share the same regressors, so it's identical to 49 separate fits, just faster. | — (a check below confirms it's identical) |

    **Why the vol shock and not raw ΔVIX?** γ comes out **exactly the same** either way, because the regression already includes the market. The check below confirms it. What changes is β. With raw ΔVIX, β is "market effect holding VIX fixed", which isn't the plain beta people mean. With the vol shock, β is the plain market beta, and γ is purely the extra effect of vol news.
    """)
    return


@app.cell
def _(R, add_intercept, coef_table, dvix_arr, fit_industries, industries, mkt, np, ols):
    b_full = fit_industries(R, mkt, dvix_arr)
    full_table = coef_table(b_full, industries)

    # Check 1: gamma is identical with raw dVIX in place of the vol shock
    _b_raw = ols(add_intercept(np.column_stack([mkt, dvix_arr])), R)
    print("check: gamma identical with raw ΔVIX:", np.allclose(_b_raw[2], b_full[2]))
    print("       beta identical with raw ΔVIX: ", np.allclose(_b_raw[1], b_full[1]), "← expected False")

    # Check 2: our OLS matches NumPy's answer key on the real data
    _X = add_intercept(np.column_stack([mkt, dvix_arr]))
    _b_key = np.linalg.lstsq(_X, R, rcond=None)[0]
    print("check: matches np.linalg.lstsq answer key:", np.allclose(_b_raw, _b_key))

    # Check 3: one industry fit alone matches its column in the 49-at-once fit
    _b_one = ols(_X, R[:, 0])
    print(f"check: {industries[0]} fit alone matches:", np.allclose(_b_one, _b_raw[:, 0]))

    # Display: alpha and gamma in % (per day / per VIX point); beta is unitless
    full_display = full_table.copy()
    full_display["alpha"] = full_display["alpha"] * 100
    full_display["gamma"] = full_display["gamma"] * 100
    full_display.columns = ["alpha (% per day)", "beta", "gamma (% per VIX point)"]
    full_display = full_display.sort_values("gamma (% per VIX point)").round(3)
    full_display
    return (b_full,)


@app.cell
def _(BLUE, GRAY, b_full, industries, new_figure, np):
    _gamma = b_full[2] * 100
    _order = np.argsort(_gamma)
    _fig, _axes = new_figure(width=7, height=10)
    _ax = _axes[0]
    _ax.barh(np.arange(len(_order)), _gamma[_order], color=BLUE, height=0.7)
    _ax.set_yticks(np.arange(len(_order)))
    _ax.set_yticklabels(np.array(industries)[_order], fontsize=7)
    _ax.axvline(0, color=GRAY, linewidth=1)
    _ax.set_xlabel("gamma: % return per 1-point vol shock (full sample)")
    _ax.set_title("Full-sample vol-shock sensitivity by industry", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: low-VIX vs. high-VIX regimes

    **Question.** Does each industry react to vol shocks differently on calm days than on stressed days?

    **How it's built.** Split the days into two groups by the regime flag, then rerun Steps 2 and 3 inside each group.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Regime set by the *prior* day's VIX close.** | Today's close is partly caused by today's shock. Flagging a day by its own close would put big up-shock days into "high" automatically and bias the comparison. | Same-day close: mixes cause and effect. |
    | **Split at the 1990–2026 median** (17.56), fixed whatever the sample start. | VIX is right-skewed (spikes pull the mean up), so the median gives an even split without an arbitrary cutoff. Fixing it keeps tests over different time spans comparable. | A fixed level like 25: arbitrary, and leaves far fewer high days. |
    | **Vol shock rebuilt inside each regime** (Gram-Schmidt within regime). | γ is identical either way (check below), so the hypothesis tests don't depend on this. Within-regime cleaning keeps the two regressors uncorrelated inside each regime, so β is that regime's plain market beta. It also gives each regime's own market–VIX relationship (Step 2 table). | Reuse the full-sample shock: same γ, but β would no longer be a plain beta within each regime. |
    | **Separate regressions per regime**, not one regression with an interaction term. | It's simpler to read, and each regime gets its own β too, not just its own γ. | One pooled regression with "γ × high" terms. Equivalent for γ only if β is also allowed to differ, and harder to explain. |

    > ⚠️ **No uncertainty yet.** The numbers below are point estimates. Before reading anything into them we need the Monte Carlo check (Step 5) and the bootstrap intervals (Step 6). The counts at the bottom are first impressions only.
    """)
    return


@app.cell
def _(R, add_intercept, dvix_arr, fit_by_regime, high, industries, mkt, np, ols, pd, shock):
    b_low, b_high = fit_by_regime(R, mkt, dvix_arr, high)

    # Check: gamma within a regime is the same whether the shock is rebuilt in-regime or reused from the full sample
    _X_reuse = add_intercept(np.column_stack([mkt[high], shock[high]]))
    _b_reuse = ols(_X_reuse, R[high])
    print("check: high-regime gamma identical with full-sample shock:", np.allclose(_b_reuse[2], b_high[2]))
    print("       high-regime beta identical:                         ", np.allclose(_b_reuse[1], b_high[1]), "← expected False")

    regime_table = pd.DataFrame({
        "beta low": b_low[1],
        "beta high": b_high[1],
        "gamma low (%/pt)": b_low[2] * 100,
        "gamma high (%/pt)": b_high[2] * 100,
    }, index=industries)
    regime_table["same sign"] = np.sign(b_low[2]) == np.sign(b_high[2])
    regime_table["|high| > |low|"] = np.abs(b_high[2]) > np.abs(b_low[2])

    print()
    print("first impressions (no uncertainty yet):")
    print(f"  same gamma sign in both regimes : {regime_table['same sign'].sum()} of {len(industries)}")
    print(f"  |gamma high| > |gamma low|      : {regime_table['|high| > |low|'].sum()} of {len(industries)}")
    regime_table.sort_values("gamma high (%/pt)").round(3)
    return b_high, b_low


@app.cell
def _(BLUE, GRAY, ORANGE, b_high, b_low, industries, new_figure, np):
    _g_low = b_low[2] * 100
    _g_high = b_high[2] * 100
    _order = np.argsort(_g_high)
    _y = np.arange(len(_order))

    _fig, _axes = new_figure(width=7, height=10)
    _ax = _axes[0]
    _ax.hlines(_y, _g_low[_order], _g_high[_order], color=GRAY, linewidth=1)
    _ax.scatter(_g_low[_order], _y, color=BLUE, s=18, zorder=3, label="low-VIX days")
    _ax.scatter(_g_high[_order], _y, color=ORANGE, s=18, zorder=3, label="high-VIX days")
    _ax.axvline(0, color=GRAY, linewidth=1)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(np.array(industries)[_order], fontsize=7)
    _ax.set_xlabel("gamma: % return per 1-point vol shock")
    _ax.set_title("Vol-shock sensitivity: low- vs high-VIX regime (point estimates)", fontsize=10)
    _ax.legend(frameon=False, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: block bootstrap (the machinery)

    **Question.** How much would each γ move around if history had played out a bit differently? That spread gives each γ a 95% confidence interval. If the interval excludes 0, the sign is trustworthy.

    **How it's built.**

    1. Cut the 9,228-day timeline into random runs of 20 consecutive days and glue enough of them together to make a fake 9,228-day history. Each day keeps its own regime flag.
    2. Rerun Step 4 on that fake history to get γ_low and γ_high for all 49 industries.
    3. Repeat 1,000 times. The spread (standard deviation) of the 1,000 γ's is the **bootstrap standard error (SE)**.
    4. Interval = estimate ± 1.96 × SE. The same goes for the difference γ_high − γ_low and for |γ_high| − |γ_low| (for H2): those are computed inside each resample, then spread.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Bootstrap**, not the textbook OLS formula for SEs. | The formula assumes every day's miss is independent and equally noisy. Crisis days are neither: they're noisier and come in runs. | Textbook SEs: too narrow here, so too many industries would look significant. |
    | **Blocks of consecutive days**, not single days. | A 3-week crisis has to stay together in the resample, or the fake histories look calmer than the real one and the intervals come out too narrow. | Single-day (iid) bootstrap: breaks up crisis runs, same problem as the textbook formula. |
    | **Block length 20 days** (about a month). | Too short a block cuts crisis runs apart (bias); too long leaves too few distinct blocks (noisy SEs). Balancing the two puts the best length proportional to n^(1/3) (Hall, Horowitz & Jing 1995), and 9,228^(1/3) ≈ 21. | Other lengths: the theory gives the growth rate, not the exact constant. Rerunning with 10 and 60 days is in the post-project list. |
    | **SE intervals** (± 1.96 × SE), not percentile intervals. | The n^(1/3) rule is optimal for estimating a *spread*, which is exactly what an SE interval uses. For percentile intervals the theory gives a different rate (n^(1/5)). | Percentile intervals (2.5th–97.5th of the draws): would call for ~6-day blocks, which may cut crisis runs apart. |
    | **Resample the full timeline**, not each regime on its own. | Crisis runs stay intact even where they cross the regime line. And γ_low and γ_high in one resample come from the *same* fake history, so their difference has the right spread. | Resample each regime separately: breaks that pairing, so the high-minus-low interval would be wrong. |
    | **1,000 resamples, fixed seed.** | Plenty to pin down an SE; the fixed seed makes reruns give identical numbers. | — |

    **Caveat about the SE interval.** It assumes the 1,000 draws are roughly bell-shaped. That's checked with histograms further down. It can fail for |γ_high| − |γ_low| when a γ is near 0: there the absolute value folds the draws over and they get lopsided. Those industries are flagged.

    ## Step 5: Monte Carlo check (before trusting any interval)

    **Question.** If we *knew* the true answer, would Steps 2–4 and 6 find it? Do the 95% intervals really contain the truth about 95% of the time?

    **How it's built.**

    1. Keep the **real** market, VIX and regime days. Invent 49 fake industries with a known β and γ:
       - β = each real industry's full-sample β;
       - γ_low = each real industry's full-sample γ;
       - γ_high = 1.5 × γ_low (a known regime difference to recover);
       - noise: random normal, sized like each real industry's misses in each regime (about 1.5× bigger on high-VIX days).
    2. Run the exact pipeline (Step 4 fits + 1,000-resample bootstrap) on the fake data.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 6: block-bootstrap results on the real data

    Each table row is an industry: the estimate and its 95% interval for γ in each regime and for the two differences. γ's are in % per VIX point.

    - **"sig"** = the interval excludes 0.
    - **"fold risk"** = at least one regime's γ interval includes 0, so the |γ_high| − |γ_low| interval may be lopsided (the caveat above). This flag is cautious: it fires for most industries, but the measured lopsidedness (skewness) of those draws stays small, between −0.5 and +0.5, and the Monte Carlo coverage for this statistic was still about 95%.

    > Still no hypothesis verdicts here. Those, with the chance baseline, are Step 7.
    """)
    return


@app.cell
def _(BLOCK_LENGTH, N_RESAMPLES, R, b_high, b_low, bootstrap_gammas, dvix_arr, high, industries, mkt, np, pd, se_interval):
    draws_low, draws_high = bootstrap_gammas(R, mkt, dvix_arr, high)
    print(f"{N_RESAMPLES} resamples, {BLOCK_LENGTH}-day blocks")

    g_low = b_low[2]
    g_high = b_high[2]
    boot = {
        "gamma low": (g_low, draws_low),
        "gamma high": (g_high, draws_high),
        "high - low": (g_high - g_low, draws_high - draws_low),
        "|high| - |low|": (np.abs(g_high) - np.abs(g_low), np.abs(draws_high) - np.abs(draws_low)),
    }
    intervals = {}
    _table = pd.DataFrame(index=industries)
    for _name in boot:
        _est, _draws = boot[_name]
        _lo, _hi, _se = se_interval(_est, _draws)
        intervals[_name] = (_lo, _hi)
        _table[f"{_name}"] = _est * 100
        _table[f"{_name} lo"] = _lo * 100
        _table[f"{_name} hi"] = _hi * 100
        _table[f"{_name} sig"] = (_lo > 0) | (_hi < 0)

    _table["fold risk"] = ~(_table["gamma low sig"] & _table["gamma high sig"])

    print()
    print("intervals excluding 0 (by chance alone, expect about 2–3 of 49 each):")
    for _name in boot:
        print(f"  {_name:15s}: {_table[f'{_name} sig'].sum()} of {len(industries)}")
    print(f"  fold risk on |high| - |low|: {_table['fold risk'].sum()} of {len(industries)}")
    boot_table = _table
    boot_table.sort_values("gamma high").round(3)
    return boot, boot_table, draws_high, draws_low


@app.cell
def _(BLUE, GRAY, ORANGE, b_high, b_low, boot, industries, new_figure, np):
    # Bell-shape check for the SE interval: histograms of the bootstrap draws with the matching
    # normal curve. Shown: the most extreme gamma-high industry, and the |high| - |low| statistic
    # for the industry whose gamma-low is closest to 0 (where folding should be worst).
    _names = list(industries)
    _i_extreme = int(np.argmax(np.abs(b_high[2])))
    _i_fold = int(np.argmin(np.abs(b_low[2])))
    _panels = [
        ("gamma high", _i_extreme, ORANGE),
        ("high - low", _i_extreme, BLUE),
        ("|high| - |low|", _i_fold, GRAY),
    ]
    _fig, _axes = new_figure(n_panels=3, width=12, height=3.6)
    for _ax, (_stat, _i, _color) in zip(_axes, _panels):
        _draws = boot[_stat][1][:, _i] * 100
        _ax.hist(_draws, bins=40, density=True, color=_color, alpha=0.6)
        _grid = np.linspace(_draws.min(), _draws.max(), 200)
        _m, _s = _draws.mean(), _draws.std(ddof=1)
        _ax.plot(_grid, np.exp(-0.5 * ((_grid - _m) / _s) ** 2) / (_s * np.sqrt(2 * np.pi)), color="black", linewidth=1.5)
        _ax.set_title(f"{_names[_i]}: {_stat}", fontsize=10)
        _ax.set_xlabel("bootstrap draws, %/pt")
        _ax.locator_params(axis="x", nbins=5)  # fewer ticks so the labels don't overlap
    _axes[0].set_ylabel("density (black = normal curve)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(BLUE, GRAY, ORANGE, boot_table, new_figure, np):
    _t = boot_table.sort_values("gamma high")
    _y = np.arange(len(_t))
    _fig, _axes = new_figure(n_panels=2, width=12, height=10)

    _ax = _axes[0]
    _ax.errorbar(_t["gamma low"], _y - 0.15, xerr=[_t["gamma low"] - _t["gamma low lo"], _t["gamma low hi"] - _t["gamma low"]],
                 fmt="o", color=BLUE, markersize=4, linewidth=1, label="low-VIX days")
    _ax.errorbar(_t["gamma high"], _y + 0.15, xerr=[_t["gamma high"] - _t["gamma high lo"], _t["gamma high hi"] - _t["gamma high"]],
                 fmt="o", color=ORANGE, markersize=4, linewidth=1, label="high-VIX days")
    _ax.axvline(0, color=GRAY, linewidth=1)
    _ax.set_yticks(_y)
    _ax.set_yticklabels(_t.index, fontsize=7)
    _ax.set_xlabel("γ, % per VIX point, with 95% interval")
    _ax.set_title("γ by regime", fontsize=10)
    _ax.legend(frameon=False, loc="lower right")

    _ax = _axes[1]
    _ax.errorbar(_t["high - low"], _y, xerr=[_t["high - low"] - _t["high - low lo"], _t["high - low hi"] - _t["high - low"]],
                 fmt="o", color=GRAY, markersize=4, linewidth=1)
    _ax.axvline(0, color=GRAY, linewidth=1)
    _ax.set_yticks(_y)
    _ax.set_yticklabels([])
    _ax.set_xlabel("γ high − γ low, with 95% interval")
    _ax.set_title("Regime difference", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 7: H1 and H2 against the chance baseline

    **The hypotheses, as written before seeing any results:**

    - **H1, direction:** each industry's vol-shock sensitivity has the same sign in low- and high-VIX periods. Judged only for industries whose confidence interval excludes zero (in both regimes).
    - **H2, magnitude:** vol-shock sensitivities are larger in absolute value in high-VIX periods.
    - Judge both by the overall pattern across the 49 industries, not individual industries. At 95% confidence, about 2–3 industries will look significant by chance.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **H1 kept exactly as written**, even though few industries qualify. | The rule was set before seeing the data. Loosening it after seeing the results would be choosing the test to fit the answer (data snooping), and a reviewer would rightly distrust it. | Loosen it (e.g. significant in *either* regime): more industries to judge, but no longer an honest test. |
    | **Chance baseline for every count.** | With 49 industries some will look significant by pure luck (5% each, about 2.5 in total). A count only means something compared with what luck alone would give. | Read counts on their own: 6 significant industries sounds like a finding, but luck alone gives about 2.5. |
    | **H2 measured two ways, both across all 49:** (1) how many have \|γ_high\| > \|γ_low\|; (2) how many \|γ_high\| − \|γ_low\| intervals exclude 0. | (1) uses every industry's point estimate; (2) uses only differences big enough to trust. Agreement between them is stronger than either alone. | Only (1): it counts tiny, noisy differences the same as big ones. Only (2): ignores most industries. |
    | **Binomial "chance of this many or more"** for each count. | It turns "33 of 49" into one number: how likely a count this high would be if there were no real effect. | — |
    | **Significant-industry count per regime shown as supporting context only.** | It wasn't a pre-set test, so it's labeled descriptive. It helps read H2: it shows whether γ's are not just bigger but clearly non-zero more often at high VIX. | Present it as a third hypothesis: would be adding a test after seeing the data. |

    **Caveat that applies to every binomial number below.** The binomial assumes the 49 industries are independent coin flips. They aren't: industries move together, and one crisis hits them all at once. So the real evidence is weaker than the binomial numbers suggest. Treat them as "how far from luck", not as exact probabilities.
    """)
    return


@app.cell
def _(b_high, b_low, boot_table, evaluate_h1, industries):
    sig_low = boot_table["gamma low sig"].to_numpy()
    sig_high = boot_table["gamma high sig"].to_numpy()
    h1_table, h1_qualifying = evaluate_h1(b_low[2], b_high[2], sig_low, sig_high, industries)

    print("H1: which industries have a gamma interval excluding 0 in each regime?")
    print(h1_table)
    print()
    print(f"qualify for H1 (significant in both): {len(h1_qualifying)} of {len(industries)}")
    print(h1_qualifying)
    return sig_high, sig_low


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **H1 verdict: untestable as written.** Only the top-left cell counts, and it holds 2 industries (Softw, Chips). Both keep the same (positive) sign. But 2 of 49 is too few to judge an overall pattern, which is how the hypothesis says to judge it. The reason is visible in the table: only 6 industries have a clear γ on low-VIX days at all, so very few can be clear in both regimes.
    """)
    return


@app.cell
def _(b_high, b_low, boot_table, evaluate_h2, pd, sig_high, sig_low, significance_counts):
    h2 = evaluate_h2(
        b_low[2], b_high[2],
        boot_table["|high| - |low| lo"].to_numpy(), boot_table["|high| - |low| hi"].to_numpy(),
    )
    print("H2 measure 1: count of |gamma high| > |gamma low|")
    print(f"  {h2['|high| > |low|']} of {h2['industries']}  (chance: {h2['expected by chance']}),"
          f"  chance of this many or more: {h2['chance of this many or more']:.3f}")
    print()
    print("H2 measure 2: |high| - |low| intervals excluding 0")
    print(f"  bigger at high VIX : {h2['interval says bigger at high VIX']}"
          f"  (chance: {h2['expected by chance (each direction)']:.1f}),"
          f"  chance of this many or more: {h2['chance of this many bigger or more']:.1e}")
    print(f"  smaller at high VIX: {h2['interval says smaller at high VIX']}")
    print()
    print("Supporting context (descriptive, not a pre-set test): significant gammas per regime")
    _counts = significance_counts(sig_low, sig_high)
    _counts["chance of this many or more"] = _counts["chance of this many or more"].map("{:.1e}".format)
    _counts
    return


@app.cell
def _(BLUE, GRAY, ORANGE, boot_table, new_figure, np):
    _abs_low = boot_table["gamma low"].abs()
    _abs_high = boot_table["gamma high"].abs()
    _sig = boot_table["|high| - |low| sig"]
    _fig, _axes = new_figure(width=6.5, height=6)
    _ax = _axes[0]
    _lim = np.array([0.0, max(_abs_low.max(), _abs_high.max()) * 1.08])
    _ax.plot(_lim, _lim, color=GRAY, linewidth=1, label="equal in both regimes")
    _ax.scatter(_abs_low[~_sig], _abs_high[~_sig], color=BLUE, s=22, zorder=3, label="difference not significant")
    _ax.scatter(_abs_low[_sig], _abs_high[_sig], color=ORANGE, s=30, zorder=3, label="|high| − |low| interval excludes 0")
    for _name in boot_table.index[_sig]:
        _ax.annotate(_name, (_abs_low[_name], _abs_high[_name]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    _ax.set_xlim(_lim)
    _ax.set_ylim(_lim)
    _ax.set_xlabel("|γ| on low-VIX days, % per point")
    _ax.set_ylabel("|γ| on high-VIX days, % per point")
    _ax.set_title("H2: dots above the line = bigger sensitivity at high VIX", fontsize=10)
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **H2 verdict: supported.**

    - **Measure 1:** 33 of 49 industries have a bigger |γ| at high VIX, against 24.5 expected by chance (binomial chance of 33 or more ≈ 1%).
    - **Measure 2:** 8 industries have a |γ_high| − |γ_low| interval entirely above 0, against about 1.2 expected by chance. **None** is significantly *smaller* at high VIX.
    - **Supporting context:** 20 industries have a clearly non-zero γ at high VIX vs. 6 at low VIX (about 2.5 expected by chance in each). This isn't from narrower intervals (the typical interval is slightly *wider* at high VIX), so it comes from bigger γ's.

    **Limits to state with it.**

    - The binomial numbers overstate the evidence, because industries move together (caveat above).
    - High-VIX days come from a handful of crisis episodes, so the effective sample is smaller than the day count.
    - Measuring VIX in points biases *against* H2: if industries really respond to percentage VIX moves, per-point γ shrinks at high VIX. H2 holds despite that. Percentage/log units are in the post-project analysis.
    """)
    return


if __name__ == "__main__":
    app.run()
