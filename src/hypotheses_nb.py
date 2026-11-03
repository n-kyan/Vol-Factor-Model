import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import BLUE, GRAY, ORANGE, new_figure
    from data import build_data
    from estimation import fit_by_regime, regime_interval_table
    from hypotheses import evaluate_h1, evaluate_h2, significance_counts

    return (
        BLUE,
        GRAY,
        ORANGE,
        build_data,
        evaluate_h1,
        evaluate_h2,
        fit_by_regime,
        mo,
        new_figure,
        np,
        regime_interval_table,
        significance_counts,
    )


@app.cell
def _(build_data):
    returns_df, factors_df = build_data()
    industries = returns_df.columns

    # NumPy arrays for the math; the DataFrames keep dates and names for display
    R = returns_df.to_numpy()                 # (days, 49) industry excess returns, decimals
    mkt = factors_df["Mkt-RF"].to_numpy()     # (days,) market excess return, decimals
    dvix_arr = factors_df["dVIX"].to_numpy()  # (days,) VIX change, points
    high = factors_df["high"].to_numpy()      # (days,) True = high-VIX regime
    return R, dvix_arr, high, industries, mkt


@app.cell
def _(R, dvix_arr, fit_by_regime, high, industries, mkt, regime_interval_table):
    b_low, b_high = fit_by_regime(R, mkt, dvix_arr, high)
    # Why one table: the same estimates, 95% intervals and significance flags used everywhere
    boot_table = regime_interval_table(R, mkt, dvix_arr, high, industries)
    return b_high, b_low, boot_table


@app.cell
def _(b_high, b_low, boot_table, evaluate_h1, evaluate_h2, industries, significance_counts):
    sig_low = boot_table["vol beta low sig"].to_numpy()
    sig_high = boot_table["vol beta high sig"].to_numpy()
    h1 = evaluate_h1(
        b_low[2], b_high[2],
        boot_table["|high| - |low| lo"].to_numpy(), boot_table["|high| - |low| hi"].to_numpy(),
    )
    h2_table, h2_qualifying = evaluate_h2(b_low[2], b_high[2], sig_low, sig_high, industries)
    sig_counts = significance_counts(sig_low, sig_high)
    return h1, h2_qualifying, h2_table, sig_counts


@app.cell(hide_code=True)
def _(h1, h2_qualifying, mo, sig_counts):
    mo.md(rf"""
    # Phase 5: Hypotheses 🤖

    *Companion to `main.py`. Code: `src/hypotheses.py`.*

    **Bottom line**

    - **H1 (larger sensitivity at high VIX): supported.** {h1['|high| > |low|']} of {h1['industries']} industries have a larger |β_vol| at high VIX ({h1['expected by chance']} expected from luck); {h1['interval says bigger at high VIX']} are significantly larger and {h1['interval says smaller at high VIX']} significantly smaller (about {h1['expected by chance (each direction)']:.1f} expected from luck in each direction).
    - **H2 (same sign in both regimes): can't be judged as written.** Only {len(h2_qualifying)} industries are clearly non-zero in both regimes; {int(h2_qualifying['same sign'].sum())} keep their sign.
    - Supporting context for H1: {int(sig_counts['significant vol betas'].iloc[1])} industries have a clearly non-zero β_vol at high VIX vs. {int(sig_counts['significant vol betas'].iloc[0])} at low VIX.
    - The industries aren't independent, so the "chance from luck" numbers overstate the evidence.

    **Units used throughout:** returns are in **percent per day** (0.5 = 0.5%); VIX changes are in **VIX points**. So **β_vol (vol beta)** is the % move in an industry for a 1-point vol shock: β_vol = −0.10 means a 1-point surprise jump in VIX goes with the industry being 0.10% lower that day.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## H1 and H2 against the chance baseline

    **The hypotheses, as written before seeing any results:**

    - **H1, magnitude:** vol-shock sensitivities are larger in absolute value in high-VIX periods.
    - **H2, direction:** each industry's vol-shock sensitivity has the same sign in low- and high-VIX periods. Judged only for industries whose confidence interval excludes zero (in both regimes).
    - Judge both by the overall pattern across the 49 industries, not individual industries. At 95% confidence, about 2–3 industries will look significant by chance.

    (Numbered direction-first while building; renumbered later to match `main.py`. Labels only, the tests are unchanged.)

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **H2 kept exactly as written**, even though few industries qualify. | The rule was set before seeing the data. Loosening it after seeing the results would be choosing the test to fit the answer (data snooping), and a reviewer would rightly distrust it. | Loosen it (e.g. significant in *either* regime): more industries to judge, but no longer an honest test. |
    | **Chance baseline for every count.** | With 49 industries some will look significant by pure luck (5% each, about 2.5 in total). A count only means something compared with what luck alone would give. | Read counts on their own: a handful of significant industries sounds like a finding, but luck alone gives about 2.5. |
    | **H1 measured two ways, both across all 49:** (1) how many have \|β_vol high\| > \|β_vol low\|; (2) how many \|β_vol high\| − \|β_vol low\| intervals exclude 0. | (1) uses every industry's point estimate; (2) uses only differences big enough to trust. Agreement between them is stronger than either alone. | Only (1): it counts tiny, noisy differences the same as big ones. Only (2): ignores most industries. |
    | **Binomial "chance of this many or more"** for each count. | It turns a count like "34 of 49" into one number: how likely a count this high would be if there were no real effect. | — |
    | **Significant-industry count per regime shown as supporting context only.** | It wasn't a pre-set test, so it's labeled descriptive. It helps read H1: it shows whether vol betas are not just bigger but clearly non-zero more often at high VIX. | Present it as a third hypothesis: would be adding a test after seeing the data. |

    **Caveat that applies to every binomial number below.** The binomial assumes the 49 industries are independent coin flips. They aren't: industries move together, and one crisis hits them all at once. So the real evidence is weaker than the binomial numbers suggest. Treat them as "how far from luck", not as exact probabilities.
    """)
    return


@app.cell
def _(h1, sig_counts):
    print("H1 measure 1: count of |vol beta high| > |vol beta low|")
    print(f"  {h1['|high| > |low|']} of {h1['industries']}  (chance: {h1['expected by chance']}),"
          f"  chance of this many or more: {h1['chance of this many or more']:.3f}")
    print()
    print("H1 measure 2: |high| - |low| intervals excluding 0")
    print(f"  bigger at high VIX : {h1['interval says bigger at high VIX']}"
          f"  (chance: {h1['expected by chance (each direction)']:.1f}),"
          f"  chance of this many or more: {h1['chance of this many bigger or more']:.1e}")
    print(f"  smaller at high VIX: {h1['interval says smaller at high VIX']}")
    print()
    print("Supporting context (descriptive, not a pre-set test): significant vol betas per regime")
    _counts = sig_counts.copy()
    _counts["chance of this many or more"] = _counts["chance of this many or more"].map("{:.1e}".format)
    _counts
    return


@app.cell
def _(BLUE, GRAY, ORANGE, boot_table, new_figure, np):
    _abs_low = boot_table["vol beta low"].abs()
    _abs_high = boot_table["vol beta high"].abs()
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
    _ax.set_xlabel("|β_vol| on low-VIX days, % per point")
    _ax.set_ylabel("|β_vol| on high-VIX days, % per point")
    _ax.set_title("H1: dots above the line = bigger sensitivity at high VIX", fontsize=10)
    _ax.legend(frameon=False, fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(h1, mo, sig_counts):
    mo.md(rf"""
    **H1 verdict: supported.**

    - **Measure 1:** {h1['|high| > |low|']} of {h1['industries']} industries have a bigger |β_vol| at high VIX, against {h1['expected by chance']} expected by chance (binomial chance of {h1['|high| > |low|']} or more ≈ {h1['chance of this many or more']:.1%}).
    - **Measure 2:** {h1['interval says bigger at high VIX']} industries have a |β_vol high| − |β_vol low| interval entirely above 0, against about {h1['expected by chance (each direction)']:.1f} expected by chance. {h1['interval says smaller at high VIX']} are significantly *smaller* at high VIX.
    - **Supporting context:** {int(sig_counts['significant vol betas'].iloc[1])} industries have a clearly non-zero β_vol at high VIX vs. {int(sig_counts['significant vol betas'].iloc[0])} at low VIX (about {sig_counts['expected by chance'].iloc[0]:.1f} expected by chance in each). This isn't from narrower intervals (the typical interval is slightly *wider* at high VIX), so it comes from bigger vol betas.

    **Limits to state with it.**

    - The binomial numbers overstate the evidence, because industries move together (caveat above).
    - High-VIX days come from a handful of crisis episodes, so the effective sample is smaller than the day count.
    - Measuring VIX in points biases *against* H1: if industries really respond to percentage VIX moves, per-point β_vol shrinks at high VIX. H1 holds despite that. Percentage/log units are in the post-project analysis.
    """)
    return


@app.cell
def _(h2_qualifying, h2_table, industries):
    print("H2: which industries have a vol beta interval excluding 0 in each regime?")
    print(h2_table)
    print()
    print(f"qualify for H2 (significant in both): {len(h2_qualifying)} of {len(industries)}")
    print(h2_qualifying)
    return


@app.cell(hide_code=True)
def _(h2_qualifying, h2_table, mo):
    mo.md(rf"""
    **H2 verdict: untestable as written.** Only the top-left cell counts, and it holds {len(h2_qualifying)} industries ({", ".join(h2_qualifying.index)}). {int(h2_qualifying['same sign'].sum())} of them keep the same sign. But {len(h2_qualifying)} of 49 is too few to judge an overall pattern, which is how the hypothesis says to judge it. The reason is visible in the table: only {int(h2_table.iloc[0].sum())} industries have a clear β_vol on low-VIX days at all, so very few can be clear in both regimes.
    """)
    return


if __name__ == "__main__":
    app.run()
