import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import (
        BLUE,
        ORANGE,
        altair_scatter,
        vol_beta_intervals,
        gt_table,
        vix_timeline,
    )
    from great_tables import loc
    from stats import vol_beta_summary, tail_counts
    from data import INDUSTRY_NAMES, build_data, load_vix_table
    from estimation import fit_industries, regime_interval_table
    from factors import regime_threshold
    from hypotheses import evaluate_h1, evaluate_h2, significance_counts
    from validation import VOL_BETA_RATIO, monte_carlo_coverage, noise_sd_by_regime

    return (
        BLUE,
        INDUSTRY_NAMES,
        ORANGE,
        VOL_BETA_RATIO,
        altair_scatter,
        build_data,
        evaluate_h1,
        evaluate_h2,
        fit_industries,
        gt_table,
        load_vix_table,
        loc,
        mo,
        monte_carlo_coverage,
        noise_sd_by_regime,
        np,
        pd,
        regime_interval_table,
        regime_threshold,
        significance_counts,
        tail_counts,
        vix_timeline,
        vol_beta_intervals,
        vol_beta_summary,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    # Factor Modeling with Volatility
    Kyan Nelson n.kyan@outlook.com 10/15/26
    """),
            mo.accordion(
                {
                    "Running the project and AI disclaimer": mo.md(r"""
    This project is managed using uv. To run the project use `uv run marimo run
    src/main.py`.

    AI Disclaimer: I will be using Claude throughout this process to:
    1. Explain concepts and math
    2. Research available datasets
    3. Correcting and explaning syntax
    4. Write boilerplate and skeleton code

    All explanations and documentation will be written myself. Any exceptions will be
    labled with this symbol 🤖.
    """),
                }
            ),
        ]
    )
    return


@app.cell
def _(
    evaluate_h1,
    evaluate_h2,
    gt_table,
    loc,
    pd,
    returns,
    significance_counts,
    vol_beta_table,
):
    _vb_low = vol_beta_table["vol beta low"].to_numpy()
    _vb_high = vol_beta_table["vol beta high"].to_numpy()
    _sig_low = vol_beta_table["vol beta low sig"].to_numpy()
    _sig_high = vol_beta_table["vol beta high sig"].to_numpy()
    h1 = evaluate_h1(
        _vb_low,
        _vb_high,
        vol_beta_table["|high| - |low| lo"].to_numpy(),
        vol_beta_table["|high| - |low| hi"].to_numpy(),
    )
    h2_table, h2_qualifying = evaluate_h2(
        _vb_low, _vb_high, _sig_low, _sig_high, returns.columns
    )
    sig_counts = significance_counts(_sig_low, _sig_high)
    # Why built here: the top of the notebook and section 5 show this same table
    h1_evidence = pd.DataFrame(
        {
            "Industries": [
                f"{h1['|high| > |low|']} of {h1['industries']}",
                str(h1["interval says bigger at high VIX"]),
                str(h1["interval says smaller at high VIX"]),
            ],
            "Expected by luck": [
                f"{h1['expected by chance']:.1f}",
                f"{h1['expected by chance (each direction)']:.1f}",
                f"{h1['expected by chance (each direction)']:.1f}",
            ],
            "Chance by luck": [
                f"{h1['chance of this many or more']:.1%}",
                f"{h1['chance of this many bigger or more']:.4%}",
                "—",
            ],
        },
        index=[
            "Larger |β_vol| at high VIX",
            "Significantly larger",
            "Significantly smaller",
        ],
    )
    _sample = f"{returns.shape[1]} industries from {returns.index.min().year}–{returns.index.max().year}"
    h1_gt = gt_table(
        h1_evidence,
        title="H1: larger sensitivity at high VIX",
        subtitle=_sample,
    ).tab_footnote(
        "Calculations assume independent industries.",
        locations=loc.column_labels(columns="Chance by luck"),
    )
    # Why relabel: a 2×2 reads best as "low-VIX rows × high-VIX columns" under one spanner
    _h2 = h2_table.copy()
    _h2.columns = ["Clear", "Not clear"]
    _h2.index = ["Clear at low VIX", "Not clear at low VIX"]
    h2_gt = (
        gt_table(
            _h2,
            title="H2: industries with a clear β_vol",
            subtitle="Clear = 95% interval excludes 0",
        )
        .tab_spanner(label="At high VIX", columns=["Clear", "Not clear"])
        .tab_footnote(
            "Only these can test H2.",
            locations=loc.body(columns="Clear", rows=[0]),
        )
    )
    return h1, h1_gt, h2_gt, h2_qualifying, sig_counts


@app.cell(hide_code=True)
def _(h1_gt, h2_gt, mo):
    mo.vstack(
        [
            mo.md(r"""

    **How do US industries react to changes in the VIX beyond what market moves explain
    (Vol-shocks), and is the relationship dependent on the level of the VIX?**

    ## Hypothesis
    1. US industries are more sensitive to changes in Vol-shocks when the VIX is high
       compared to when it is low.
    2. The direction of the relationship persists across VIX regimes.

    ## Conclusion
    1. Suppported. Industries tend to have a larger sensitivity to vol shocks at high VIX.
    """),
            h1_gt,
            mo.md(r"""
    2. Inconslusive. There was not enough industries with statistically significant directions to their sensitivities in both high and low VIX.
    """),
            h2_gt
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methodology Overview

    ### 1. Data
    Daily returns for 49 industries, the market, and the VIX. Total sample ranges from 1990–2026. Each day is categorized into either a high or low VIX regime.

    ### 2. Factors
    Two factors are used: market excess returns and daily vol shocks. Vol shocks are daily changes in the vix that are not explained by daily marker excess returns.

    ### 3. Estimation
    Regresses each industry's daily return on the market and the vol shock, separately for low- and high-VIX days; **β_vol (vol beta)** is the industry's % move per 1-point vol shock. Confidence intervals come from a block bootstrap.

    ### 4. Validation
    A Monte Carlo test that the confidence intervals contain the truth 95% of the time.

    ### 5. Hypotheses
    Judges H1 and H2 across all 49 industries against what random chance alone would produce.

    Each phase has a dedicated section below and a separate
    that explains it in detail. Open one with `uv run marimo run src/<notebook>`.
    """)
    return


@app.cell
def _(build_data, load_vix_table, regime_threshold):
    returns, factors = build_data()
    vix_full = load_vix_table()
    threshold = regime_threshold(vix_full)

    R = returns.to_numpy()
    mkt = factors["Mkt-RF"].to_numpy()
    dvix = factors["dVIX"].to_numpy()
    high = factors["high"].to_numpy()
    return R, dvix, factors, high, mkt, returns, threshold, vix_full


@app.cell(hide_code=True)
def _(factors, mo, tail_counts, threshold, vix_full, vix_timeline):
    _tails = tail_counts(factors["Mkt-RF"], [4]).iloc[0]
    mo.vstack(
        [
            mo.md(f"""
    ## 1. Data

    <table>

    Each day is categorized into either a high or low VIX regime. The threshold delimitting the two VIX regimes is **{threshold:.2f}**, the 1990–2026 median VIX level. The resulting split is  {(~factors["high"]).mean():.2%} days in the low regime  and {factors["high"].mean():.2%} in the high regime.

    The categoization is dependent on the previous day's close, ensuring regime categorization is not influenced by the day's own shock.

    - Daily moves have **fat tails**: market days beyond 4 standard deviations happened
      {int(_tails["actual days"])} times; a bell curve expects
      {_tails["normal distribution expects"]:.1f}. So confidence intervals come from a
      bootstrap, not textbook formulas.
    """),
            vix_timeline(vix_full.loc["1990":], threshold),
            mo.md(r"""
    High-VIX days come in a few long episodes (orange), which is why the effective
    sample is smaller than the day count.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(dvix, gt_table, high, mkt, mo, np, pd):
    _rows = {}
    for _name, _mask in [
        ("all days", np.ones(len(mkt), dtype=bool)),
        ("low-VIX days", ~high),
        ("high-VIX days", high),
    ]:
        _corr = np.corrcoef(mkt[_mask], dvix[_mask])[0, 1]
        _rows[_name] = {
            "Correlation": f"{_corr:.2f}",
            "Market explains": f"{_corr**2:.0%}",
        }
    _share = np.corrcoef(mkt, dvix)[0, 1] ** 2
    mo.vstack(
        [
            mo.md(r"""
    ## 2. The vol shock 🤖

    VIX usually jumps when the market falls, so a raw VIX change is mostly a market move
    in disguise. To isolate news about volatility itself, each day's VIX change is
    regressed on the market's return; the leftover (the residual) is the **vol
    shock**. By construction it is uncorrelated with the market, so each industry's
    market effect and vol-shock effect can be measured separately (a Gram-Schmidt step,
    market first).
    """),
            gt_table(
                pd.DataFrame(_rows).T,
                title="Market vs. VIX change",
                source="Sources: Ken French data library, Cboe.",
            ),
            mo.md(rf"""
    The market explains about {_share:.0%} of daily VIX changes. The vol shock is the
    other {1 - _share:.0%}.
    """),
        ]
    )
    return


@app.cell
def _(INDUSTRY_NAMES, R, dvix, high, mkt, regime_interval_table, returns):
    # Why computed once here: every later section reads from this one table (1,000 bootstrap resamples, ~1 s)
    vol_beta_table = regime_interval_table(R, mkt, dvix, high, returns.columns)
    industry_labels = returns.columns.map(INDUSTRY_NAMES)
    return industry_labels, vol_beta_table


@app.cell(hide_code=True)
def _(industry_labels, mo, vol_beta_intervals, vol_beta_table):
    mo.vstack(
        [
            mo.md(r"""
    ## 3. Estimation: sensitivities by regime 🤖

    For each industry and each regime: daily excess return =

    needs to be latex and on its own line:
    α + β_mkt × market + **β_vol × vol
    shock** + noise. 

    **β_vol** is the industry's % move for a 1-point vol shock (e.g. β_vol =
    −0.10 means a 1-point surprise VIX jump goes with the industry 0.10% lower that
    day). Bars are 95% confidence intervals.

    **How the intervals are built.** A **block bootstrap**: rebuild 1,000 alternative
    histories from random 20-day stretches of the real timeline, re-estimate every β_vol on
    each, and use the spread of those 1,000 estimates (interval = estimate ± 1.96 × that
    spread). Keeping 20-day stretches intact preserves the way crisis days cluster.
    """),
            vol_beta_intervals(vol_beta_table, industry_labels),
            mo.md(rf"""
    - **Low-VIX days (blue):** most vol betas sit near zero with intervals crossing it. On
      calm days, vol shocks barely move industries.
    - **High-VIX days (orange):** a clear pattern. Oil, agriculture, mining, consumer
      staples and healthcare fall on vol shocks; computers, software, electronic
      equipment and measuring equipment rise.
    - **Right panel:** the high-minus-low difference is clearly non-zero for
      {int(vol_beta_table["high - low sig"].sum())} industries.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Validation

    To ensure the confidence intervals are accurate, a monte carlo test is used to check the calculation pipeline against a fake industry with pre-determined vol betas with a ratio between the two of 1.5x.

    <need an explanation of how a monte carlo test works with an implied statement of why its the approach we used>
    """)
    return


@app.cell(hide_code=True)
def _(
    R,
    VOL_BETA_RATIO,
    dvix,
    fit_industries,
    high,
    mkt,
    mo,
    monte_carlo_coverage,
    noise_sd_by_regime,
):
    # Why cached: 100 simulations x 1,000 resamples takes about a minute; marimo saves the result to
    # __marimo__/cache and reloads it instantly on later runs. Delete that folder to force a rerun.
    with mo.persistent_cache(name="monte_carlo"):
        _b_full = fit_industries(R, mkt, dvix)
        _sd_low, _sd_high = noise_sd_by_regime(R, mkt, dvix, high)
        mc = monte_carlo_coverage(
            mkt,
            dvix,
            high,
            beta=_b_full[1],
            vol_beta_low=_b_full[2],
            vol_beta_high=VOL_BETA_RATIO * _b_full[2],
            sd_low=_sd_low,
            sd_high=_sd_high,
        )
    return (mc,)


@app.cell(hide_code=True)
def _(gt_table, mc, mo, pd):
    _labels = {
        "vol_beta_low": "β_vol, low-VIX days",
        "vol_beta_high": "β_vol, high-VIX days",
        "diff": "β_vol high − β_vol low",
        "absdiff": "|β_vol high| − |β_vol low|",
    }
    _rows = {}
    for _key in _labels:
        _rows[_labels[_key]] = {
            "Coverage": f"{mc['covered'][_key].mean():.1%}"
        }
    mo.vstack(
        [
            gt_table(
                pd.DataFrame(_rows).T,
                title="Monte Carlo Coverage",
                subtitle=f"Target 95% · {mc['covered']['vol_beta_low'].size:,} intervals each",
            ),
            mo.md(rf""" All four are within a fraction of a percent of the 95% target
    ({mc["covered"]["vol_beta_low"].size:,} intervals each). Limit: the simulated noise
    doesn't cluster like real crisis noise, so this checks the machinery, not the 20-day
    block length.  """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(BLUE, ORANGE, altair_scatter, industry_labels, mo, pd, vol_beta_table):
    h1_points = pd.DataFrame(
        {
            "industry": industry_labels,
            "code": vol_beta_table.index,
            "|β_vol| low-VIX days": vol_beta_table["vol beta low"].abs().round(3).to_numpy(),
            "|β_vol| high-VIX days": vol_beta_table["vol beta high"].abs().round(3).to_numpy(),
            "difference": vol_beta_table["|high| - |low| sig"]
            .map({True: "significantly larger at high VIX", False: "not significant"})
            .to_numpy(),
        }
    )
    h1_points["label"] = h1_points["industry"].where(
        vol_beta_table["|high| - |low| sig"].to_numpy(), ""
    )
    h1_chart = mo.ui.altair_chart(
        altair_scatter(
            h1_points,
            x="|β_vol| low-VIX days",
            y="|β_vol| high-VIX days",
            x_title="|β_vol| on low-VIX days (% per point)",
            y_title="|β_vol| on high-VIX days (% per point)",
            color_field="difference",
            color_domain=["not significant", "significantly larger at high VIX"],
            color_range=[BLUE, ORANGE],
            tooltip=["industry", "|β_vol| low-VIX days", "|β_vol| high-VIX days", "difference"],
            diagonal=True,
            label_field="label",
        )
    )
    return h1_chart, h1_points


@app.cell(hide_code=True)
def _(
    INDUSTRY_NAMES,
    gt_table,
    h1,
    h1_chart,
    h1_gt,
    h1_points,
    h2_gt,
    h2_qualifying,
    industry_labels,
    mo,
    pd,
    sig_counts,
    vol_beta_summary,
    vol_beta_table,
):
    _context = pd.DataFrame(
        {
            "Clear β_vol": sig_counts["significant vol betas"],
            "Expected by luck": sig_counts["expected by chance"].map(
                "{:.1f}".format
            ),
        }
    )
    # Why the table follows the chart: dragging a box over points narrows the table to those industries
    _selected_codes = h1_chart.apply_selection(h1_points)["code"]
    _summary = vol_beta_summary(vol_beta_table, industry_labels)
    _summary = _summary[_summary["code"].isin(_selected_codes)]
    _qualifying_names = []
    for _code in h2_qualifying.index:
        _qualifying_names.append(INDUSTRY_NAMES[_code])
    mo.vstack(
        [
            mo.md("""
    ## 5. Hypotheses

    <insert H1 and H2 again to remind the reader and makes it so they dont have to scroll back to the top to refresh themselves>

    Both were set before seeing any results and are judged across all 49 industries,
    against what luck alone would produce (at 95% confidence, about 2–3 of 49 industries
    look significant by chance).

    ### H1, magnitude: sensitivities are larger in high-VIX periods → **supported**
    """),
            h1_gt,
            mo.md(rf"""
    **Each dot is an industry.** Dots above the dashed line are more sensitive on
    high-VIX days; the {h1["interval says bigger at high VIX"]} significantly larger
    ones are named. Hover any dot for its name; **drag a box over dots** to narrow the
    table below to those industries.

    In the table, **t** is the estimate divided by its bootstrap standard error: how
    many standard errors it sits from zero. |t| > 1.96 is the same test as "the 95%
    interval excludes 0". The standard error comes from the block bootstrap, not the
    textbook formula, which assumes independent, equally noisy days and would overstate
    significance here.
    """),
            h1_chart,
            mo.ui.table(
                _summary,
                selection=None,
                page_size=10,
                label="β_vol estimate [95% interval], % per VIX point; * = interval excludes 0",
            ),
            mo.md(r"""
    **Supporting context** (descriptive, not a pre-set test): far more industries have a
    clearly non-zero β_vol at high VIX.
    """),
            gt_table(
                _context,
                title="Industries with a clear β_vol",
                subtitle="Descriptive only",
            ),
            mo.md(r"""
    ### H2, direction: sensitivities keep the same sign in both regimes → **can't be judged as written**

    H2 is judged only on industries whose β_vol is clearly non-zero in *both* regimes:
    """),
            h2_gt,
            mo.md(rf"""
    Only {len(h2_qualifying)} industries qualify ({", ".join(_qualifying_names)}), and
    {int(h2_qualifying["same sign"].sum())} of them keep their sign. That is too few to
    judge a pattern across 49 industries. The test was not loosened after seeing this,
    since choosing a test to fit the data would undermine the result.
    """),
            mo.md(r"""
    **Caveats**

    - The "chance from luck" numbers treat industries as independent. They aren't
      (industries move together), so the real evidence is weaker than those numbers
      suggest.
    - High-VIX days come from a handful of crisis episodes, so the effective sample is
      smaller than the day count.
    - Measuring VIX in points biases *against* H1: if industries respond to percentage
      VIX moves, per-point β_vol shrinks at high VIX. H1 holds anyway.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next

    - **VIX units:** rerun with percentage or log VIX changes, to see whether the
      high-VIX effect is partly a units effect.
    - **Block length:** rerun the bootstrap with 10- and 60-day blocks and check whether
      the conclusions change. Test
    """)

    return


if __name__ == "__main__":
    app.run()
