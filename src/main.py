import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import BLUE, ORANGE, altair_scatter, gamma_intervals, vix_timeline
    from stats import gamma_summary
    from data import INDUSTRY_NAMES, build_data, load_vix_table
    from estimation import fit_industries, regime_interval_table
    from factors import regime_threshold
    from hypotheses import evaluate_h1, evaluate_h2, significance_counts
    from validation import GAMMA_RATIO, monte_carlo_coverage, noise_sd_by_regime

    return (
        BLUE,
        GAMMA_RATIO,
        INDUSTRY_NAMES,
        ORANGE,
        altair_scatter,
        build_data,
        evaluate_h1,
        evaluate_h2,
        fit_industries,
        gamma_intervals,
        gamma_summary,
        load_vix_table,
        mo,
        monte_carlo_coverage,
        noise_sd_by_regime,
        np,
        pd,
        regime_interval_table,
        regime_threshold,
        significance_counts,
        vix_timeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Factor Modeling with Volatility
    Kyan Nelson n.kyan@outlook.com 10/15/26

    This project is managed using uv. To run the project use 'uv run marimo run
    src/main.py'.

    AI Disclaimer: I will be using Claude throughout this process for educational
    purposes and for understanding what data is readily available to me and how to get
    it. I will be writing all of the code and documentation myself. I will do my best to
    specify where AI was used with this symbol 🤖.
    """)
    return


@app.cell(hide_code=True)
def _(h1_table, h2, mc, mo, sig_counts):
    _coverage = []
    for _key in mc["covered"]:
        _coverage.append(mc["covered"][_key].mean())
    mo.vstack(
        [
            mo.md(r"""
    ## Driving Question

    #### How do US industries react to changes in the VIX beyond what market moves explain, and is the relationship dependent on the level of the VIX?
    """),
            mo.callout(
                mo.md(r"""
            **Hypothesis**
            1. US industries are more sensitive to changes in the VIX when the VIX is
               high compared to when it is low.
            2. US industies' VIX sensitivity preserves the same sign regardless of VIX
               regime.  **Vol-shock sensitivity is much stronger in stressed markets.**
               On calm days a surprise VIX jump barely moves most industries. On
               high-VIX days a clear pattern appears: oil, agriculture, mining, consumer
               staples and healthcare fall, while computers, software and electronic
               equipment rise.
    """),
                kind="success",
            ),
            mo.hstack(
                [
                    mo.stat(
                        value=f"{h2['|high| > |low|']} of {h2['industries']}",
                        label="industries more sensitive at high VIX",
                        caption=f"luck alone: ~{h2['expected by chance']:.1f}",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{h2['interval says bigger at high VIX']} vs {h2['interval says smaller at high VIX']}",
                        label="significantly larger vs. smaller",
                        caption=f"luck alone: ~{h2['expected by chance (each direction)']:.1f} each",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{int(sig_counts['significant gammas'].iloc[1])} vs {int(sig_counts['significant gammas'].iloc[0])}",
                        label="clearly sensitive industries, high vs. low VIX",
                        caption=f"luck alone: ~{sig_counts['expected by chance'].iloc[0]:.1f} each",
                        bordered=True,
                    ),
                    mo.stat(
                        value=f"{min(_coverage):.1%}–{max(_coverage):.1%}",
                        label="95% intervals containing the truth",
                        caption="Monte Carlo check, target 95%",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            mo.md(rf"""
    - **H2, larger sensitivity at high VIX: supported.**
    - **H1, same sign in both regimes: can't be judged as written.** Only
      {int(h1_table.iloc[0, 0])} industries (software and electronic equipment) are
      clearly non-zero in both regimes. Both keep their sign, but that is too few to
      call a pattern.
    - **Main caveats:** industries move together, so they are not 49 independent pieces
      of evidence. High-VIX days come from a handful of crisis episodes. Measuring VIX
      in points, not percentages, if anything works *against* the high-VIX finding.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md(r"""
    ## How the analysis works 🤖

    1. **Data:** daily returns for 49 industries, the market and VIX, 1990–2026.
    2. **Vol shock:** the part of each day's VIX change that the market's move doesn't
       explain.
    3. **Sensitivities:** for each industry, regress its daily return on the market and
       the vol shock, separately for low- and high-VIX days. **γ (gamma)** is the
       industry's % move per 1-point vol shock.
    4. **Confidence intervals:** a block bootstrap, checked by Monte Carlo simulation.
    5. **Hypotheses:** judged across all 49 industries against what luck alone would
       produce.

    Each section below gives one phase's takeaway. Every phase has a companion notebook
    with the full explanation.
    """),
            mo.accordion(
                {
                    "Companion notebooks: every decision, its reasoning, and more tables and charts": mo.md(r"""
    | Phase | Open with | Code |
    |---|---|---|
    | 1. Data | `uv run marimo run src/data_nb.py` | `data.py` |
    | 2. Vol shock | `uv run marimo run src/factors_nb.py` | `factors.py` |
    | How the regressions are solved | `uv run marimo run src/ols_nb.py` | `ols.py` |
    | 3. Sensitivities and 4. confidence intervals | `uv run marimo run src/estimation_nb.py` | `estimation.py` |
    | 4. Monte Carlo check | `uv run marimo run src/validation_nb.py` | `validation.py` |
    | 5. Hypotheses | `uv run marimo run src/hypotheses_nb.py` | `hypotheses.py` |
    """),
                }
            ),
        ]
    )
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
def _(factors, mo, returns, threshold, vix_full, vix_timeline):
    mo.vstack(
        [
            mo.md(rf"""
    ## 1. Data 🤖

    - **{len(returns):,} trading days**, {returns.index.min().date()} to
      {returns.index.max().date()}, for all **{returns.shape[1]} Ken French industries**
      (value-weighted, excess of the risk-free rate), the market (Mkt-RF) and the Cboe
      VIX close.
    - A day is **high-VIX** if the *previous* day's VIX close is at or above
      **{threshold:.2f}**, the 1990–2026 median. That splits the sample
      {(~factors["high"]).mean():.0%} low / {factors["high"].mean():.0%} high. Using
      yesterday's close keeps the flag from being caused by the day's own shock.
    - Daily moves have **fat tails**: market days beyond 4 standard deviations happened
      55 times; a bell curve expects fewer than 1. So confidence intervals come from a
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
def _(dvix, high, mkt, mo, np, pd):
    _rows = {}
    for _name, _mask in [
        ("all days", np.ones(len(mkt), dtype=bool)),
        ("low-VIX days", ~high),
        ("high-VIX days", high),
    ]:
        _corr = np.corrcoef(mkt[_mask], dvix[_mask])[0, 1]
        _rows[_name] = {
            "correlation: market vs. VIX change": f"{_corr:.2f}",
            "share of VIX change the market explains": f"{_corr**2:.0%}",
        }
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
            pd.DataFrame(_rows).T,
            mo.md(r"""
    The market explains about 60% of daily VIX changes. The vol shock is the other ~40%.
    """),
        ]
    )
    return


@app.cell
def _(INDUSTRY_NAMES, R, dvix, high, mkt, regime_interval_table, returns):
    # Why computed once here: every later section reads from this one table (1,000 bootstrap resamples, ~1 s)
    gamma_table = regime_interval_table(R, mkt, dvix, high, returns.columns)
    industry_labels = returns.columns.map(INDUSTRY_NAMES)
    return gamma_table, industry_labels


@app.cell(hide_code=True)
def _(gamma_intervals, gamma_table, industry_labels, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## 3. Sensitivities by regime 🤖

    For each industry and each regime: daily excess return = α + β × market + **γ × vol
    shock** + noise. **γ** is the industry's % move for a 1-point vol shock (e.g. γ =
    −0.10 means a 1-point surprise VIX jump goes with the industry 0.10% lower that
    day). Bars are 95% confidence intervals.
    """),
            gamma_intervals(gamma_table, industry_labels),
            mo.md(r"""
    - **Low-VIX days (blue):** most γ's sit near zero with intervals crossing it. On
      calm days, vol shocks barely move industries.
    - **High-VIX days (orange):** a clear pattern. Oil, agriculture, mining, consumer
      staples and healthcare fall on vol shocks; computers, software, electronic
      equipment and measuring equipment rise.
    - **Right panel:** the high-minus-low difference is clearly non-zero for 9
      industries.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Are the confidence intervals trustworthy? 🤖

    **How they are built.** A **block bootstrap**: rebuild 1,000 alternative histories
    from random 20-day stretches of the real timeline, re-estimate every γ on each, and
    use the spread of those 1,000 estimates (interval = estimate ± 1.96 × that
    spread). Keeping 20-day stretches intact preserves the way crisis days cluster.

    **How they were checked.** A **Monte Carlo test**: keep the real market, VIX and
    regime days, invent 49 industries with *known* γ's (and a known 1.5× high-VIX
    effect), run the whole pipeline, and repeat 100 times. A trustworthy 95% interval
    should contain the true value about 95% of the time.
    """)
    return


@app.cell
def _(
    GAMMA_RATIO,
    R,
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
            gamma_low=_b_full[2],
            gamma_high=GAMMA_RATIO * _b_full[2],
            sd_low=_sd_low,
            sd_high=_sd_high,
        )
    return (mc,)


@app.cell(hide_code=True)
def _(mc, mo, pd):
    _labels = {
        "gamma_low": "γ, low-VIX days",
        "gamma_high": "γ, high-VIX days",
        "diff": "γ high − γ low",
        "absdiff": "|γ high| − |γ low|",
    }
    _rows = {}
    for _key in _labels:
        _rows[_labels[_key]] = {
            "intervals containing the true value": f"{mc['covered'][_key].mean():.1%}"
        }
    mo.vstack(
        [
            pd.DataFrame(_rows).T,
            mo.md(r""" All four are within a fraction of a percent of the 95% target
    (4,900 intervals each). Limit: the simulated noise doesn't cluster like real crisis
    noise, so this checks the machinery, not the 20-day block length.  """),
        ]
    )
    return


@app.cell
def _(evaluate_h1, evaluate_h2, gamma_table, returns, significance_counts):
    _g_low = gamma_table["gamma low"].to_numpy()
    _g_high = gamma_table["gamma high"].to_numpy()
    _sig_low = gamma_table["gamma low sig"].to_numpy()
    _sig_high = gamma_table["gamma high sig"].to_numpy()
    h1_table, h1_qualifying = evaluate_h1(
        _g_low, _g_high, _sig_low, _sig_high, returns.columns
    )
    h2 = evaluate_h2(
        _g_low,
        _g_high,
        gamma_table["|high| - |low| lo"].to_numpy(),
        gamma_table["|high| - |low| hi"].to_numpy(),
    )
    sig_counts = significance_counts(_sig_low, _sig_high)
    return h1_table, h2, sig_counts


@app.cell(hide_code=True)
def _(BLUE, ORANGE, altair_scatter, gamma_table, industry_labels, mo, pd):
    h2_points = pd.DataFrame(
        {
            "industry": industry_labels,
            "code": gamma_table.index,
            "|γ| low-VIX days": gamma_table["gamma low"].abs().round(3).to_numpy(),
            "|γ| high-VIX days": gamma_table["gamma high"].abs().round(3).to_numpy(),
            "difference": gamma_table["|high| - |low| sig"]
            .map({True: "significantly larger at high VIX", False: "not significant"})
            .to_numpy(),
        }
    )
    h2_points["label"] = h2_points["industry"].where(
        gamma_table["|high| - |low| sig"].to_numpy(), ""
    )
    h2_chart = mo.ui.altair_chart(
        altair_scatter(
            h2_points,
            x="|γ| low-VIX days",
            y="|γ| high-VIX days",
            x_title="|γ| on low-VIX days (% per point)",
            y_title="|γ| on high-VIX days (% per point)",
            color_field="difference",
            color_domain=["not significant", "significantly larger at high VIX"],
            color_range=[BLUE, ORANGE],
            tooltip=["industry", "|γ| low-VIX days", "|γ| high-VIX days", "difference"],
            diagonal=True,
            label_field="label",
        )
    )
    return h2_chart, h2_points


@app.cell(hide_code=True)
def _(
    gamma_summary,
    gamma_table,
    h1_table,
    h2,
    h2_chart,
    h2_points,
    industry_labels,
    mo,
    pd,
    sig_counts,
):
    _h2_table = pd.DataFrame(
        {
            "result": [
                f"{h2['|high| > |low|']} of {h2['industries']}",
                str(h2["interval says bigger at high VIX"]),
                str(h2["interval says smaller at high VIX"]),
            ],
            "expected from luck alone": [
                f"{h2['expected by chance']:.1f}",
                f"{h2['expected by chance (each direction)']:.1f}",
                f"{h2['expected by chance (each direction)']:.1f}",
            ],
            "chance of a result this strong from luck": [
                f"{h2['chance of this many or more']:.1%}",
                f"{h2['chance of this many bigger or more']:.4%}",
                "—",
            ],
        },
        index=[
            "industries with larger |γ| at high VIX",
            "…significantly larger",
            "…significantly smaller",
        ],
    )
    _context = pd.DataFrame(
        {
            "industries with γ clearly non-zero": sig_counts["significant gammas"],
            "expected from luck alone": sig_counts["expected by chance"].map(
                "{:.1f}".format
            ),
        }
    )
    # Why the table follows the chart: dragging a box over points narrows the table to those industries
    _selected_codes = h2_chart.apply_selection(h2_points)["code"]
    _summary = gamma_summary(gamma_table, industry_labels)
    _summary = _summary[_summary["code"].isin(_selected_codes)]
    mo.vstack(
        [
            mo.md(r"""
    ## 5. Hypotheses 🤖

    Both were set before seeing any results and are judged across all 49 industries,
    against what luck alone would produce (at 95% confidence, about 2–3 of 49 industries
    look significant by chance).

    ### H2, magnitude: sensitivities are larger in high-VIX periods → **supported**
    """),
            _h2_table,
            mo.md(r"""
    **Each dot is an industry.** Dots above the dashed line are more sensitive on
    high-VIX days; the 8 significantly larger ones are named. Hover any dot for its
    name; **drag a box over dots** to narrow the table below to those industries.

    In the table, **t** is the estimate divided by its bootstrap standard error: how
    many standard errors it sits from zero. |t| > 1.96 is the same test as "the 95%
    interval excludes 0". The standard error comes from the block bootstrap, not the
    textbook formula, which assumes independent, equally noisy days and would overstate
    significance here.
    """),
            h2_chart,
            mo.ui.table(
                _summary,
                selection=None,
                page_size=10,
                label="γ estimate [95% interval], % per VIX point; * = interval excludes 0",
            ),
            mo.md(r"""
    **Supporting context** (descriptive, not a pre-set test): far more industries have a
    clearly non-zero γ at high VIX.
    """),
            _context,
            mo.md(r"""
    ### H1, direction: sensitivities keep the same sign in both regimes → **can't be judged as written**

    H1 is judged only on industries whose γ is clearly non-zero in *both* regimes:
    """),
            h1_table,
            mo.md(r"""
    Only 2 industries qualify (software and electronic equipment), and both keep a
    positive sign. Two is too few to judge a pattern across 49 industries. The test was
    not loosened after seeing this, since choosing a test to fit the data would
    undermine the result.
    """),
            mo.md(r"""
    **Caveats**

    - The "chance from luck" numbers treat industries as independent. They aren't
      (industries move together), so the real evidence is weaker than those numbers
      suggest.
    - High-VIX days come from a handful of crisis episodes, so the effective sample is
      smaller than the day count.
    - Measuring VIX in points biases *against* H2: if industries respond to percentage
      VIX moves, per-point γ shrinks at high VIX. H2 holds anyway.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What's next 🤖

    - **VIX units:** rerun with percentage or log VIX changes, to see whether the
      high-VIX effect is partly a units effect.
    - **Block length:** rerun the bootstrap with 10- and 60-day blocks and check whether
      the conclusions change. Test
    """)

    mo.md(r"""
    awefiawioefjaiwoehf awefi aweiofha wioefio wefaj fiaje fioajfioje ij iejij ej ejf
    iefoi ejfoiwejf owjefoi wjiofjw iofwijfio wfjweij f iwjfoiwjef oiwfiojw efwjf wijf
    wef wefjw fwiej f
    """)
    return


if __name__ == "__main__":
    app.run()
