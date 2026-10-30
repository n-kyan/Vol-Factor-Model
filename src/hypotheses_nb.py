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
    # Phase 5: Hypotheses 🤖

    *Companion to `main.py`. Code: `src/hypotheses.py`.*

    **Bottom line**

    - **H2 (larger sensitivity at high VIX): supported.** 33 of 49 industries have a larger |γ| at high VIX (24.5 expected from luck); 8 are significantly larger and none significantly smaller (about 1.2 expected from luck in each direction).
    - **H1 (same sign in both regimes): can't be judged as written.** Only 2 industries are clearly non-zero in both regimes; both keep their sign.
    - The industries aren't independent, so the "chance from luck" numbers overstate the evidence.

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
def _(R, dvix_arr, fit_by_regime, high, industries, mkt, regime_interval_table):
    b_low, b_high = fit_by_regime(R, mkt, dvix_arr, high)
    # Why one table: the same estimates, 95% intervals and significance flags used everywhere
    boot_table = regime_interval_table(R, mkt, dvix_arr, high, industries)
    return b_high, b_low, boot_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## H1 and H2 against the chance baseline

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


if __name__ == "__main__":
    app.run()
