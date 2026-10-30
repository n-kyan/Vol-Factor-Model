import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import BLUE, GRAY, ORANGE, altair_scatter, new_figure
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
        altair_scatter,
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
    # Phases 3–4: Industry sensitivities and their confidence intervals 🤖

    *Companion to `main.py`. Code: `src/estimation.py`. How each regression is solved: `ols_nb.py`.*

    **Bottom line**

    - **Full sample:** market betas range from 0.38 (Precious Metals) to 1.33 (Electronic Equipment). Vol-shock sensitivities γ range from −0.11 (Petroleum and Natural Gas) to +0.10 (Computers) % per point; 32 of 49 are negative.
    - **By regime:** on low-VIX days most γ's are near zero; on high-VIX days they spread out much more.
    - **Confidence intervals** come from a 20-day block bootstrap (1,000 resamples). 6 industries have a clearly non-zero γ on low-VIX days, 20 on high-VIX days, and 9 have a clearly non-zero high-minus-low difference.

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
def _(dvix_arr, mkt, vol_shock):
    shock = vol_shock(dvix_arr, mkt)
    return (shock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each dot is one day. The gray line is what the market alone predicts for VIX. A day's **vol shock** is how far its dot sits above or below that line.

    ## 1. Full-sample sensitivities

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
    | **Intercept α** included. | Each industry has its own average daily return. Without α, β and γ would bend to absorb it (see `ols_nb.py`). | — |
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


@app.cell(hide_code=True)
def _(BLUE, INDUSTRY_NAMES, ORANGE, altair_scatter, b_full, industries, mo, pd):
    full_points = pd.DataFrame({
        "industry": industries.map(INDUSTRY_NAMES),
        "code": industries,
        "market beta β": b_full[1].round(3),
        "vol-shock γ (%/pt)": (b_full[2] * 100).round(3),
    })
    full_points["γ sign"] = (full_points["vol-shock γ (%/pt)"] < 0).map({True: "falls on vol shocks", False: "rises on vol shocks"})
    full_chart = mo.ui.altair_chart(
        altair_scatter(
            full_points, x="market beta β", y="vol-shock γ (%/pt)",
            x_title="market beta β (1 = moves one-for-one with the market)", y_title="vol-shock γ, % per VIX point",
            color_field="γ sign", color_domain=["falls on vol shocks", "rises on vol shocks"], color_range=[ORANGE, BLUE],
            tooltip=["industry", "market beta β", "vol-shock γ (%/pt)"],
        )
    )
    return full_chart, full_points


@app.cell(hide_code=True)
def _(full_chart, full_points, mo):
    mo.vstack([
        mo.md(r"""
    **Market exposure vs. vol-shock exposure, full sample.** Each dot is an industry. Right = moves more with the market; down = falls more when VIX jumps beyond what the market explains. Hover for names; **drag a box over dots** to list them below.
    """),
        full_chart,
        mo.ui.table(full_chart.apply_selection(full_points), selection=None, page_size=10),
        mo.md(r"""
    The two exposures are different things: high-beta industries are not automatically the most vol-sensitive. Computers (β 1.25) and electronic equipment (β 1.33) are high-β and *rise* on vol shocks, while mining (β 1.14) is also high-β but *falls*. Oil (β 0.89) and precious metals (β 0.38) are below-average β and fall on them. These are full-sample point estimates; the intervals come in section 4.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(BLUE, GRAY, INDUSTRY_NAMES, ORANGE, b_full, industries, mo, new_figure, np):
    _gamma = b_full[2] * 100
    _order = np.argsort(_gamma)
    _names = np.array(industries.map(INDUSTRY_NAMES))[_order]
    _colors = np.where(_gamma[_order] < 0, ORANGE, BLUE)
    _fig, _axes = new_figure(width=8, height=10)
    _ax = _axes[0]
    _ax.barh(np.arange(len(_order)), _gamma[_order], color=_colors, height=0.7)
    _ax.set_yticks(np.arange(len(_order)))
    _ax.set_yticklabels(_names, fontsize=7)
    _ax.axvline(0, color=GRAY, linewidth=1)
    _ax.set_xlabel("γ: % return per 1-point vol shock (full sample)")
    _ax.set_title("All 49 industries ranked by full-sample vol-shock sensitivity", fontsize=10)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(rf"""
    {int((_gamma < 0).sum())} of 49 industries fall on vol shocks (orange). Commodities, consumer staples and healthcare sit at the bottom; computers, electronic equipment and software at the top.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Low-VIX vs. high-VIX regimes

    **Question.** Does each industry react to vol shocks differently on calm days than on stressed days?

    **How it's built.** Split the days into two groups by the regime flag, then rerun Steps 2 and 3 inside each group.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Regime set by the *prior* day's VIX close.** | Today's close is partly caused by today's shock. Flagging a day by its own close would put big up-shock days into "high" automatically and bias the comparison. | Same-day close: mixes cause and effect. |
    | **Split at the 1990–2026 median** (17.56), fixed whatever the sample start. | VIX is right-skewed (spikes pull the mean up), so the median gives an even split without an arbitrary cutoff. Fixing it keeps tests over different time spans comparable. | A fixed level like 25: arbitrary, and leaves far fewer high days. |
    | **Vol shock rebuilt inside each regime** (Gram-Schmidt within regime). | γ is identical either way (check below), so the hypothesis tests don't depend on this. Within-regime cleaning keeps the two regressors uncorrelated inside each regime, so β is that regime's plain market beta. It also gives each regime's own market–VIX relationship (Step 2 table). | Reuse the full-sample shock: same γ, but β would no longer be a plain beta within each regime. |
    | **Separate regressions per regime**, not one regression with an interaction term. | It's simpler to read, and each regime gets its own β too, not just its own γ. | One pooled regression with "γ × high" terms. Equivalent for γ only if β is also allowed to differ, and harder to explain. |

    > ⚠️ **No uncertainty yet.** The numbers below are point estimates. Before reading anything into them we need the bootstrap intervals (section 3) and the Monte Carlo check of those intervals (`validation_nb.py`). The counts at the bottom are first impressions only.
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
    ## 3. Block bootstrap (the machinery)

    **Question.** How much would each γ move around if history had played out a bit differently? That spread gives each γ a 95% confidence interval. If the interval excludes 0, the sign is trustworthy.

    **How it's built.**

    1. Cut the 9,228-day timeline into random runs of 20 consecutive days and glue enough of them together to make a fake 9,228-day history. Each day keeps its own regime flag.
    2. Rerun the regime fits (section 2) on that fake history to get γ_low and γ_high for all 49 industries.
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
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Block-bootstrap results on the real data

    Each table row is an industry: the estimate and its 95% interval for γ in each regime and for the two differences. γ's are in % per VIX point.

    - **"t"** = the estimate divided by its bootstrap standard error: how many standard errors it sits from zero.
    - **"sig"** = the interval excludes 0, which is the same test as |t| > 1.96.
    - **"fold risk"** = at least one regime's γ interval includes 0, so the |γ_high| − |γ_low| interval may be lopsided (the caveat above). This flag is cautious: it fires for most industries, but the measured lopsidedness (skewness) of those draws stays small, between −0.5 and +0.5, and the Monte Carlo coverage for this statistic was still about 95%.

    > Still no hypothesis verdicts here. Those, with the chance baseline, are in `hypotheses_nb.py`.
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
        # Why: t = estimate / bootstrap SE; |t| > 1.96 is the same test as the interval excluding 0
        _table[f"{_name} t"] = _est / _se
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


@app.cell(hide_code=True)
def _(INDUSTRY_NAMES, b_high, industries, mo, np):
    _options = {}
    for _code in industries:
        _options[f"{INDUSTRY_NAMES[_code]} ({_code})"] = _code
    _default = industries[int(np.argmax(np.abs(b_high[2])))]
    industry_pick = mo.ui.dropdown(
        options=_options,
        value=f"{INDUSTRY_NAMES[_default]} ({_default})",
        label="Industry",
    )
    mo.vstack([
        mo.md(r"""
    ### Bell-shape check: pick any industry

    The SE interval assumes each statistic's 1,000 bootstrap draws are roughly bell-shaped. Pick an industry to see its draws against a normal curve with the same mean and SD. Starts on the industry with the largest high-VIX γ.
    """),
        industry_pick,
    ])
    return (industry_pick,)


@app.cell(hide_code=True)
def _(BLUE, GRAY, ORANGE, boot, boot_table, industries, industry_pick, mo, new_figure, np):
    _i = list(industries).index(industry_pick.value)
    _panels = [("gamma low", BLUE), ("gamma high", ORANGE), ("high - low", GRAY), ("|high| - |low|", GRAY)]
    _fig, _axes = new_figure(n_panels=4, width=14, height=3.4)
    for _ax, (_stat, _color) in zip(_axes, _panels):
        _draws = boot[_stat][1][:, _i] * 100
        _ax.hist(_draws, bins=40, density=True, color=_color, alpha=0.6)
        _grid = np.linspace(_draws.min(), _draws.max(), 200)
        _m, _s = _draws.mean(), _draws.std(ddof=1)
        _ax.plot(_grid, np.exp(-0.5 * ((_grid - _m) / _s) ** 2) / (_s * np.sqrt(2 * np.pi)), color="black", linewidth=1.5)
        _ax.axvline(0, color=GRAY, linewidth=1, linestyle="--")
        _ax.set_title(_stat, fontsize=10)
        _ax.set_xlabel("bootstrap draws, %/pt")
        _ax.locator_params(axis="x", nbins=5)  # fewer ticks so the labels don't overlap
    _axes[0].set_ylabel("density (black = normal curve)")
    _fig.tight_layout()
    _row = boot_table.loc[industry_pick.value]
    mo.vstack([
        _fig,
        mo.md(rf"""
    Estimates (% per VIX point): γ low **{_row['gamma low']:+.3f}**, γ high **{_row['gamma high']:+.3f}**, difference **{_row['high - low']:+.3f}**. Dashed line = 0: an interval excludes 0 when nearly all the draws sit on one side of it. The |high| − |low| panel is where lopsidedness would show up first, when a γ is near 0.
    """),
    ])
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


if __name__ == "__main__":
    app.run()
