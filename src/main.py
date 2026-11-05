import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import (
        gt_table,
        h1_bar_chart,
        h1_dumbbell,
        vix_timeline,
        vol_beta_intervals,
    )
    from great_tables import loc
    from stats import vol_beta_summary, tail_counts
    from data import INDUSTRY_NAMES, build_data, load_vix_table
    from estimation import fit_industries, regime_interval_table
    from factors import regime_threshold, vol_shock
    from hypotheses import evaluate_h1, evaluate_h2, significance_counts
    from validation import VOL_BETA_RATIO, monte_carlo_coverage, noise_sd_by_regime

    return (
        INDUSTRY_NAMES,
        VOL_BETA_RATIO,
        build_data,
        evaluate_h1,
        evaluate_h2,
        fit_industries,
        gt_table,
        h1_bar_chart,
        h1_dumbbell,
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
        vol_shock,
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


@app.cell(hide_code=True)
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

    def p_value(p):
        # Why this format: the usual way to report p-values; very small ones as a bound
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    # Why built here: the top of the notebook and section 6 show this same table
    h1_evidence = pd.DataFrame(
        {
            "Industries": [
                f"{h1['|high| > |low|']} of {h1['industries']}",
                str(h1["interval says bigger at high VIX"]),
                str(h1["interval says smaller at high VIX"]),
            ],
            "Expected under the null": [
                f"{h1['expected by chance']:.1f}",
                f"{h1['expected by chance (each direction)']:.1f}",
                f"{h1['expected by chance (each direction)']:.1f}",
            ],
            "p-value": [
                p_value(h1["chance of this many or more"]),
                p_value(h1["chance of this many bigger or more"]),
                "—",
            ],
        },
        index=[
            "|β_vol| larger at high VIX",
            "Larger at high VIX (5% level)",
            "Smaller at high VIX (5% level)",
        ],
    )
    _sample = f"{returns.shape[1]} industries from {returns.index.min().year}–{returns.index.max().year}"
    h1_gt = (
        gt_table(
            h1_evidence,
            title="H1: larger sensitivity at high VIX",
            subtitle=_sample,
        )
        .tab_footnote(
            "Null hypothesis: vol-beta size doesn't depend on the regime. Each industry is then 50/50 to be larger at high VIX, and 2.5% of industries are significant in each direction by chance.",
            locations=loc.column_labels(columns="Expected under the null"),
        )
        .tab_footnote(
            "Calculations assume independent industries. One-sided binomial test; industries move together, so the true evidence is weaker than these p-values suggest.",
            locations=loc.column_labels(columns="p-value"),
        )
        .tab_footnote(
            "5% level: the 95% bootstrap confidence interval for |β_vol high| − |β_vol low| excludes 0.",
            locations=loc.stub(rows=[1, 2]),
        )
    )
    # Why relabel: a 2×2 reads best as "low-VIX rows × high-VIX columns" under one spanner
    _h2 = h2_table.copy()
    _h2.columns = ["Significant", "Not significant"]
    _h2.index = ["Significant at low VIX", "Not significant at low VIX"]
    h2_gt = (
        gt_table(
            _h2,
            title="H2: industries with a statistically significant β_vol",
            subtitle="Significant = 95% bootstrap confidence interval excludes 0 (5% level)",
        )
        .tab_spanner(label="At high VIX", columns=["Significant", "Not significant"])
        .tab_footnote(
            "Only these can test H2.",
            locations=loc.body(columns="Significant", rows=[0]),
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
    1. Supported. Industries tend to have a larger sensitivity to vol shocks at high
       VIX.
    """),
            h1_gt,
            mo.md(r"""
    2. Inconclusive. There was not enough industries with statistically significant
       directions to their sensitivities in both high and low VIX.
    """),
            h2_gt,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methodology Overview

    ### 1. Data
    Daily returns for 49 industries, the market, and the VIX. Total sample ranges from
    1990–2026. Each day is categorized into either a high or low VIX regime.

    [Data notebook](?file=data_nb.py)

    ### 2. Factors
    Two factors are used: market excess returns and daily vol shocks. Vol shocks are the
    part of each day's VIX change that the same day's market excess return can't account
    for: the residual from regressing the VIX change on the market return. 🤖

    [Factors notebook](?file=factors_nb.py)

    ### 3. Regression
    Each industry's daily excess return is regressed on the two factors. Its **β_vol
    (vol beta)** is the industry's % move per 1-point vol shock, holding the market
    fixed. 🤖

    [Regression notebook](?file=ols_nb.py)

    ### 4. Estimation
    The regression is run separately on low- and high-VIX days. 🤖 Confidence intervals
    come from a block bootstrap.

    [Estimation notebook](?file=estimation_nb.py)

    ### 5. Validation
    A Monte Carlo test that the confidence intervals contain the truth 95% of the time.

    [Validation notebook](?file=validation_nb.py)

    ### 6. Hypotheses
    Judges H1 and H2 across all 49 industries against what random chance alone would
    produce.

    [Hypotheses notebook](?file=hypotheses_nb.py)

    Each phase has a dedicated section below and a separate notebook, linked above and
    at the end of each section, that explains it in detail. 🤖
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
def _(
    factors,
    gt_table,
    mo,
    np,
    pd,
    returns,
    tail_counts,
    threshold,
    vix_full,
    vix_timeline,
    vol_shock,
):
    _tails = tail_counts(factors["Mkt-RF"], [4]).iloc[0]
    _sources = pd.DataFrame(
        {
            "Provider": ["Ken French Data Library", "Ken French Data Library", "Cboe"],
            "Series": [
                "Value-weighted daily returns",
                "Mkt-RF and RF (Fama-French factors)",
                "Daily close",
            ],
            "Units": ["% per day", "% per day", "Index points"],
            "Used for": [
                "Industry excess returns (return − RF)",
                "Market factor; RF for excess returns",
                "VIX changes, vol shock and regimes",
            ],
        },
        index=["49 industry portfolios", "Market and risk-free rate", "VIX"],
    )
    _sources_gt = gt_table(
        _sources,
        title="Data sources",
        subtitle=f"Daily · {returns.index.min():%b %Y} to {returns.index.max():%b %Y} · {len(returns):,} trading days",
    )
    # Why compare labels: shows how much same-day labels would sort days by their own shock
    _shock = vol_shock(factors["dVIX"].to_numpy(), factors["Mkt-RF"].to_numpy())
    _biggest = _shock >= np.quantile(_shock, 0.99)
    _same_day_high = (factors["VIX"] >= threshold).to_numpy()
    _share_prior = factors["high"].to_numpy()[_biggest].mean()
    _share_same = _same_day_high[_biggest].mean()
    # Why runs: how concentrated high-VIX days are in long episodes
    _runs = []
    _length = 0
    for _is_high in factors["high"].to_numpy():
        if _is_high:
            _length += 1
        elif _length > 0:
            _runs.append(_length)
            _length = 0
    if _length > 0:
        _runs.append(_length)
    _runs = np.array(_runs)
    _long = _runs >= 60
    _long_share = _runs[_long].sum() / _runs.sum()
    mo.vstack(
        [
            mo.md(r"""
    ## 1. Data: sources, sample and VIX regimes
    """),
            _sources_gt,
            mo.md(rf"""
    Each day is categorized into either a high or low VIX regime. The threshold
    delimiting the two VIX regimes is **{threshold:.2f}**, the 1990–2026 median VIX
    level. The resulting split is {(~factors["high"]).mean():.2%} days in the low regime
    and {factors["high"].mean():.2%} in the high regime.

    The categorization is dependent on the previous day's close, ensuring regime
    categorization is not influenced by the day's own shock.

    **Why the prior day's close.** 🤖 The regime is meant to describe the market
    *before* a shock arrives, so the question becomes: does the same vol shock land
    differently on a calm market than on a stressed one? A day's own close already
    contains that day's shock (today's VIX = yesterday's VIX + today's change), so
    labeling a day by its own close would sort days partly by the very shock being
    measured. With same-day labels, {_share_same:.0%} of the largest 1% of vol shocks
    would be labeled high-VIX, against {_share_prior:.0%} with prior-day labels. The
    {_share_prior:.0%} is real: large shocks are more common when the market is already
    stressed. The extra {(_share_same - _share_prior) * 100:.0f} percentage points would
    be created by the labeling itself. If industries react more than proportionally to
    large shocks, that size effect would then show up as a "high-VIX" effect and inflate
    H1 even if the VIX level didn't matter. Prior-day labels keep the size of the shock
    and the state of the market separate. So a day that starts at a VIX of 15 and closes
    at 50 counts as low-VIX: a calm market hit by a very large shock, which is exactly
    the case the low regime should measure. This follows the usual practice in
    conditional factor models of conditioning only on information known before the
    return.

    - Daily moves have **fat tails**: market days beyond 4 standard deviations happened
      {int(_tails["actual days"])} times; a bell curve expects
      {_tails["normal distribution expects"]:.1f}. So confidence intervals come from a
      bootstrap, not textbook formulas.

    ### VIX since 1990
    """),
            vix_timeline(vix_full.loc["1990":], threshold, title=False),
            mo.md(rf"""
    Orange marks high-VIX days. 🤖 Both regimes hold about half the days by
    construction, but high-VIX days bunch together: {_long_share:.0%} of them fall in
    just {int(_long.sum())} stretches of 60 or more consecutive trading days. Days
    inside one stretch share the same conditions, so they carry less independent
    information than their count suggests (the effective sample is smaller than the day
    count). That is why the confidence intervals in section 4 use a block bootstrap,
    which keeps these stretches together.

    [Data notebook](data_nb.html)
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(dvix, gt_table, high, loc, mkt, mo, np, pd):
    _rows = {}
    for _name, _mask in [
        ("all days", np.ones(len(mkt), dtype=bool)),
        ("low-VIX days", ~high),
        ("high-VIX days", high),
    ]:
        _corr = np.corrcoef(mkt[_mask], dvix[_mask])[0, 1]
        _rows[_name] = {
            "Correlation": f"{_corr:.2f}",
            "R²": f"{_corr**2:.0%}",
        }
    _share = np.corrcoef(mkt, dvix)[0, 1] ** 2
    mo.vstack(
        [
            mo.md(r"""
    ## 2. Factors: isolating the vol shock 🤖

    **In plain terms:** the VIX usually jumps when stocks fall, so much of a day's VIX
    move is the market move seen from another angle. The vol shock is what's left after
    removing that part: volatility news that the market return doesn't capture.

    **What was done.** Each day's VIX change (in index points, the units of vega) is
    regressed on that day's market excess return, and the residual is the vol shock:

    $$\Delta\text{VIX}_t = a + c \cdot \text{MKT}_t + \text{VolShock}_t$$

    **Why.** The question is about volatility *beyond* the market. Raw VIX changes are
    strongly negatively correlated with the market (table below), so an industry's
    response to them would partly be its market beta in disguise. The residual is
    uncorrelated with the market by construction (a Gram-Schmidt step, with the market
    first), so each industry's exposure to the market and to volatility news can be
    measured separately in section 3.
    """),
            gt_table(
                pd.DataFrame(_rows).T,
                title="Market return vs. VIX change",
                source="Sources: Ken French data library, Cboe.",
            ).tab_footnote(
                "Share of the variance of daily VIX changes accounted for by the same day's market return.",
                locations=loc.column_labels(columns="R²"),
            ),
            mo.md(rf"""
    **So what.** The market accounts for about {_share:.0%} of daily VIX moves. The
    other {1 - _share:.0%} is volatility news that a market-only model misses, and that
    is the part the vol beta measures.

    [Factors notebook](factors_nb.html)
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Regression: the model and how it's solved 🤖

    **In plain terms:** for each industry, how much does it move with the market, and
    how much with volatility news on top of that?

    $$r_{i,t} = \alpha_i + \beta_{\text{mkt},i} \cdot \text{MKT}_t +
    \beta_{\text{vol},i} \cdot \text{VolShock}_t + \varepsilon_{i,t}$$

    - $r_{i,t}$: industry $i$'s excess return on day $t$ (its return minus the risk-free
      rate). $\text{MKT}_t$: the market's excess return. $\text{VolShock}_t$: from
      section 2. $\varepsilon_{i,t}$: the part neither factor explains.
    - **β_mkt (market beta):** how much the industry moves per 1% market move.
    - **β_vol (vol beta):** the industry's % move per 1-point vol shock, holding the
      market fixed. β_vol = −0.10 means a 1-point surprise jump in VIX goes with the
      industry being 0.10% lower that day than its market beta alone implies.

    The coefficients are estimated by ordinary least squares (OLS): the values of α,
    β_mkt and β_vol that make the squared misses ε as small as possible. OLS is written
    by hand, by solving its normal equations, rather than called from a library, so
    every step can be inspected.

    [Regression notebook](ols_nb.html)
    """)
    return


@app.cell
def _(INDUSTRY_NAMES, R, dvix, high, mkt, regime_interval_table, returns):
    # Why computed once here: every later section reads from this one table (1,000 bootstrap resamples, ~1 s)
    vol_beta_table = regime_interval_table(R, mkt, dvix, high, returns.columns)
    industry_labels = returns.columns.map(INDUSTRY_NAMES)
    return industry_labels, vol_beta_table


@app.cell(hide_code=True)
def _(industry_labels, mo, returns, vol_beta_intervals, vol_beta_table):
    _n = len(returns)
    mo.vstack(
        [
            mo.md(rf"""
    ## 4. Estimation: vol betas by regime, with bootstrap confidence intervals 🤖

    **In plain terms:** the section 3 regression is run twice for every industry, once
    on low-VIX days and once on high-VIX days. Each vol beta comes with a 95% confidence
    interval: the range of values consistent with the data.

    ### How the intervals are built: the block bootstrap

    **The problem.** We observe one history of the market. The question is how different
    each β_vol would be in other histories the same market could have produced. That
    spread is the estimate's uncertainty, its standard error.

    **The idea.** History can't be rerun, so the observed days are treated as the best
    available picture of how this market behaves, and new histories are built by drawing
    from them at random, with replacement. If the sample is representative, estimates
    vary across the rebuilt histories about as much as they would across real
    alternative histories. This is the bootstrap (Efron, 1979). It doesn't assume
    returns follow a bell curve, which matters because daily returns have fat tails
    (section 1).

    **Why blocks of 20 days.** Days aren't independent: volatility clusters, so a crisis
    day tends to be followed by more crisis days. Drawing single days would scatter
    crises across the rebuilt history and understate the uncertainty. Drawing runs of 20
    consecutive days keeps that clustering. Longer blocks keep more of it but leave
    fewer distinct blocks to draw from. The standard rule for this balance sets the
    block length proportional to $n^{{1/3}}$ (Hall, Horowitz & Jing, 1995), and with $n$
    = {_n:,} days, $n^{{1/3}} \approx$ {_n ** (1 / 3):.0f}.

    **From rebuilt histories to an interval.** Every β_vol is re-estimated on 1,000
    rebuilt histories. The standard deviation of those 1,000 estimates is the bootstrap
    standard error, $\text{{SE}}_{{\text{{boot}}}}$. The 1,000 estimates are close to
    bell-shaped (checked in the notebook), and 95% of a bell curve lies within 1.96
    standard deviations of its center, so:

    $$\text{{95\% CI}} = \hat{{\beta}}_{{\text{{vol}}}} \pm 1.96 \times
    \text{{SE}}_{{\text{{boot}}}}$$

    An interval that excludes 0 means the vol beta is **statistically significant at the
    5% level**.

    **Can these intervals be trusted?** Section 5 tests the whole recipe on simulated
    data where the true answer is known.

    ### Vol betas by regime, with 95% confidence intervals
    """),
            vol_beta_intervals(vol_beta_table, industry_labels),
            mo.md(rf"""
    - **Low-VIX days (blue):** most vol betas sit near zero with intervals crossing
      it. On calm days, vol shocks barely move industries beyond what the market
      explains.
    - **High-VIX days (orange):** a clear pattern. Oil, agriculture, mining, consumer
      staples and healthcare fall on vol shocks; computers, software, electronic
      equipment and measuring equipment rise, relative to what their market beta
      implies.
    - **Right panel:** the high-minus-low difference is statistically significant for
      {int(vol_beta_table["high - low sig"].sum())} industries.

    [Estimation notebook](estimation_nb.html)
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Validation: testing the confidence intervals on simulated data

    To ensure the confidence intervals are accurate, a Monte Carlo test is used to check
    the calculation pipeline against 49 fake industries with pre-determined vol betas
    with a ratio between the two of 1.5x.

    **In plain terms:** 🤖 before trusting the intervals on real data, check that the
    method finds the right answer on fake data where the right answer is known.

    **How a Monte Carlo test works.** On real data the true vol betas are unknown, so
    there is no way to check whether an interval contains them. A simulation fixes that
    by building data where the truth is set in advance:

    1. Keep the real market, VIX and regime days, and invent 49 fake industries. Each
       gets a true β_mkt and low-VIX β_vol equal to a real industry's full-sample
       estimates, a high-VIX β_vol 1.5× larger, and random noise sized like that real
       industry's noise in each regime.
    2. Run the exact pipeline from sections 2–4 on the fake data: vol shock, regime
       regressions, a 1,000-resample block bootstrap and 95% intervals.
    3. Repeat 100 times with fresh noise, and count how often each interval contains the
       true value. That share is the interval's **coverage**.

    A correct 95% interval should contain the truth about 95% of the time. Coverage well
    below 95% would mean the intervals are too narrow, and the significance counts in
    section 6 would be overstated.

    **Why this approach.** It is the most direct way to grade an interval method: run it
    against a known answer, using the real data's fat tails, crisis spikes and regime
    sizes rather than idealized inputs.
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
    _worst = 0.0
    for _key in _labels:
        _coverage = mc["covered"][_key].mean()
        _worst = max(_worst, abs(_coverage - 0.95))
        _rows[_labels[_key]] = {
            "Intervals containing the true value": f"{_coverage:.1%}"
        }
    mo.vstack(
        [
            gt_table(
                pd.DataFrame(_rows).T,
                title="Monte Carlo Coverage",
                subtitle=f"Target 95% · {mc['covered']['vol_beta_low'].size:,} intervals each",
            ),
            mo.md(rf"""
    **So what.** 🤖 Coverage is within {_worst * 100:.1f} percentage points of 95% for
    all four statistics ({mc["covered"]["vol_beta_low"].size:,} intervals each), so the
    intervals that section 6 counts can be trusted to mean what they say.

    **Limit.** The fake noise is drawn independently day by day, so it doesn't cluster
    the way real crisis noise does. This test therefore checks the code and the interval
    formula, but not whether 20-day blocks are long enough to capture real
    clustering. Rerunning with shorter and longer blocks is listed under next steps.

    [Validation notebook](validation_nb.html)
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    INDUSTRY_NAMES,
    gt_table,
    h1,
    h1_bar_chart,
    h1_dumbbell,
    h1_gt,
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
            "Industries": sig_counts["significant vol betas"],
            "Expected under the null": sig_counts["expected by chance"].map(
                "{:.1f}".format
            ),
        }
    )
    _summary = vol_beta_summary(vol_beta_table, industry_labels).drop(columns="code")
    _qualifying_names = []
    for _code in h2_qualifying.index:
        _qualifying_names.append(INDUSTRY_NAMES[_code])
    mo.vstack(
        [
            mo.md(r"""
    ## 6. Hypotheses: testing H1 and H2

    1. US industries are more sensitive to changes in Vol-shocks when the VIX is high
       compared to when it is low.
    2. The direction of the relationship persists across VIX regimes.

    Both were set before seeing any results and are judged across all 49 industries,
    against what random chance alone would produce under the null hypothesis of no
    regime difference (at the 5% level, about 2–3 of 49 industries look significant by
    chance).

    ### H1, magnitude: sensitivities are larger in high-VIX periods → **supported**
    """),
            h1_gt,
            mo.md(rf"""
    **In plain terms:** 🤖 {h1["|high| > |low|"]} of {h1["industries"]} industries react
    more strongly to vol shocks when the VIX is high. Where the difference is
    statistically significant, {h1["interval says bigger at high VIX"]} industries are
    larger at high VIX and {h1["interval says smaller at high VIX"]} are smaller.

    ### Change in vol-beta size, high minus low VIX
    """),
            h1_bar_chart(vol_beta_table, industry_labels),
            mo.md(rf"""
    Each bar is one industry's |β_vol high| − |β_vol low|, with its 95% confidence
    interval. 🤖 Bars right of 0 are more sensitive at high VIX ({h1["|high| > |low|"]}
    of {h1["industries"]}); colored bars have an interval that excludes 0. Hover a bar
    for its numbers.

    ### Vol-beta size on low- vs. high-VIX days
    """),
            h1_dumbbell(vol_beta_table, industry_labels),
            mo.md(r"""
    Each row is one industry: blue is its |β_vol| on low-VIX days, orange on high-VIX
    days. 🤖 Orange to the right of blue means more sensitive at high VIX.
    """),
            mo.accordion(
                {
                    "All 49 industries: vol betas, 95% intervals and t-stats": mo.vstack(
                        [
                            mo.md(r"""
    **t** is the estimate divided by its bootstrap standard error: how many standard
    errors it sits from zero. |t| > 1.96 is the same test as "the 95% interval excludes
    0". The standard error comes from the block bootstrap, not the textbook formula,
    which assumes independent, equally noisy days and would overstate significance
    here. 🤖
    """),
                            mo.ui.table(
                                _summary,
                                selection=None,
                                page_size=10,
                                label="β_vol estimate [95% interval], % per VIX point; * = significant at the 5% level",
                            ),
                        ]
                    )
                }
            ),
            mo.md(r"""
    **Supporting context** (descriptive, not a pre-set test): far more industries have a
    statistically significant β_vol at high VIX.
    """),
            gt_table(
                _context,
                title="Industries with a statistically significant β_vol",
                subtitle="Descriptive only · 5% level",
            ),
            mo.md(r"""
    ### H2, direction: sensitivities keep the same sign in both regimes → **can't be judged as written**

    H2 is judged only on industries whose β_vol is statistically significant in *both*
    regimes:
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

    - The p-values treat industries as independent. They aren't (industries move
      together), so the real evidence is weaker than those numbers suggest.
    - High-VIX days bunch into a few long crisis episodes (section 1), so they carry
      less independent information than their count suggests. The block bootstrap
      accounts for this inside each interval, but the result still rests on a limited
      number of episodes.
    - Measuring VIX in points biases *against* H1: if industries respond to percentage
      VIX moves, per-point β_vol shrinks at high VIX. H1 holds anyway.

    [Hypotheses notebook](hypotheses_nb.html)
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(h1, mc, mo, sig_counts):
    _coverage = []
    for _key in mc["covered"]:
        _coverage.append(mc["covered"][_key].mean())
    mo.md(rf"""
    ## Conclusion 🤖

    **Claim.** An industry's sensitivity to volatility news depends on the regime: vol
    betas are larger when the VIX is already high (H1 supported). Whether the
    *direction* of that sensitivity persists across regimes can't be judged from this
    data (H2), because too few industries have a statistically significant vol beta in
    calm markets.

    **Grounds.**

    - {h1["|high| > |low|"]} of {h1["industries"]} industries have a larger |β_vol| at
      high VIX (p = {h1["chance of this many or more"]:.3f} against a 50/50 null).
    - {h1["interval says bigger at high VIX"]} are larger at the 5% level, against
      {h1["expected by chance (each direction)"]:.1f} expected by chance, and
      {h1["interval says smaller at high VIX"]} are smaller.
    - {int(sig_counts["significant vol betas"].iloc[1])} industries have a statistically
      significant β_vol at high VIX, versus
      {int(sig_counts["significant vol betas"].iloc[0])} at low VIX.
    - In simulation, the 95% intervals contained the true value
      {min(_coverage):.1%}–{max(_coverage):.1%} of the time.

    **Why it matters.**

    - **For risk models.** A factor model with one vol beta per industry averages calm
      and stressed periods. Calm-period vol betas are near zero and stressed-period ones
      are not, so the average understates exposure to vol shocks exactly when the VIX is
      high, which is when that exposure matters most. A regime-dependent loading would
      capture it.
    - **For portfolio managers.** In calm markets most industries look vol-neutral, so
      calm-period data can't show which holdings will react to a vol shock once markets
      are stressed. At high VIX, oil, agriculture, mining, consumer staples and
      healthcare do worse on vol shocks than their market beta implies, while computers,
      software and electronic equipment do better. That exposure doesn't show up in a
      market beta.
    - **What it doesn't show.** These are historical sensitivities, not forecasts, and
      the high-VIX evidence comes from a limited number of crisis episodes. The
      implications above follow from the results but weren't tested here: no risk-model
      or hedging backtest was run.

    **Alternatives considered.**

    - VIX changes in points vs. percentage or log changes: points match vega and are
      simpler to read; percentage units are a next step.
    - Regime threshold at the long-run median vs. a fixed level of 25: a fixed level
      felt arbitrary, and the median splits the sample evenly without choosing a number.
    - Prior-day vs. same-day regime labels: see section 1.
    - Standard-error intervals with 20-day blocks vs. percentile intervals, which would
      need much shorter blocks (about 6 days) to be optimal.
    - Vol shock rebuilt within each regime vs. reusing the full-sample shock: β_vol is
      identical either way; rebuilding keeps β_mkt a plain market beta within each
      regime.
    - Separate regressions per regime vs. one pooled regression with interaction terms:
      simpler to read, and each regime gets its own β_mkt.
    - Keeping H2 as written vs. loosening it after seeing the data: loosening would be
      choosing the test to fit the result.

    **Next steps.**

    - **VIX units:** rerun with percentage or log VIX changes, to see whether the
      high-VIX effect is partly a units effect.
    - **Block length:** rerun the bootstrap with 10- and 60-day blocks and check whether
      the conclusions change.
    """)
    return


if __name__ == "__main__":
    app.run()
