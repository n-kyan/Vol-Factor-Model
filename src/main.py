import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import (
        BLUE,
        ORANGE,
        altair_scatter,
        gamma_intervals,
        result_table,
        vix_timeline,
    )
    from stats import gamma_summary, tail_counts
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
        result_table,
        significance_counts,
        tail_counts,
        vix_timeline,
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
def _(h1_evidence, h2_table, mo, result_table):
    mo.vstack(
        [
            mo.md(r"""
    How do US industries react to changes in the VIX beyond what market moves explain
    (Vol-shocks), and is the relationship dependent on the level of the VIX?

    **Hypothesis**
    1. US industries are more sensitive to changes in Vol-shocks when the VIX is high
       compared to when it is low.
    2. The direction of the relationship persists across VIX regimes.

    **Conclusion**

    H1, larger sensitivity at high VIX: supported. 🤖
    """),
            result_table(h1_evidence),
            mo.md(r"""
    H2, same sign in both regimes: can't be judged as written.
    """),
            result_table(h2_table),
            mo.md(r"""
    - **Main caveats:** Each of the 49 industries cannot be assumed to be independent
      pieces of evidence. I chose to measure the VIX in points for simplicity and to
      follow the precedent set by the units of the Vega Greek in options pricing.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methodology Overview

    | Phase | What it does | Notebook | Code |
    |---|---|---|---|
    | 1. Data | Daily returns for 49 industries, the market and VIX, 1990–2026; splits days into low- and high-VIX regimes. | `data_nb.py` | `data.py` |
    | 2. Vol shock | The part of each day's VIX change that the market's move doesn't explain. | `factors_nb.py` | `factors.py` |
    | 3. Estimation | Regresses each industry's daily return on the market and the vol shock, separately for low- and high-VIX days; **γ (gamma)** is the industry's % move per 1-point vol shock. Confidence intervals come from a block bootstrap. | `estimation_nb.py` (the regression math: `ols_nb.py`) | `estimation.py`, `ols.py` |
    | 4. Validation | A Monte Carlo test that the confidence intervals contain the truth 95% of the time. | `validation_nb.py` | `validation.py` |
    | 5. Hypotheses | Judges H1 and H2 across all 49 industries against what luck alone would produce. | `hypotheses_nb.py` | `hypotheses.py` |

    Each phase has its own section below with its main takeaway, and its own notebook
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
def _(factors, mo, returns, tail_counts, threshold, vix_full, vix_timeline):
    _tails = tail_counts(factors["Mkt-RF"], [4]).iloc[0]
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
            pd.DataFrame(_rows).T,
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
    gamma_table = regime_interval_table(R, mkt, dvix, high, returns.columns)
    industry_labels = returns.columns.map(INDUSTRY_NAMES)
    return gamma_table, industry_labels


@app.cell(hide_code=True)
def _(gamma_intervals, gamma_table, industry_labels, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## 3. Estimation: sensitivities by regime 🤖

    For each industry and each regime: daily excess return = α + β × market + **γ × vol
    shock** + noise. **γ** is the industry's % move for a 1-point vol shock (e.g. γ =
    −0.10 means a 1-point surprise VIX jump goes with the industry 0.10% lower that
    day). Bars are 95% confidence intervals.

    **How the intervals are built.** A **block bootstrap**: rebuild 1,000 alternative
    histories from random 20-day stretches of the real timeline, re-estimate every γ on
    each, and use the spread of those 1,000 estimates (interval = estimate ± 1.96 × that
    spread). Keeping 20-day stretches intact preserves the way crisis days cluster.
    """),
            gamma_intervals(gamma_table, industry_labels),
            mo.md(rf"""
    - **Low-VIX days (blue):** most γ's sit near zero with intervals crossing it. On
      calm days, vol shocks barely move industries.
    - **High-VIX days (orange):** a clear pattern. Oil, agriculture, mining, consumer
      staples and healthcare fall on vol shocks; computers, software, electronic
      equipment and measuring equipment rise.
    - **Right panel:** the high-minus-low difference is clearly non-zero for
      {int(gamma_table["high - low sig"].sum())} industries.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Validation: are the confidence intervals trustworthy? 🤖

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
def _(evaluate_h1, evaluate_h2, gamma_table, pd, returns, significance_counts):
    _g_low = gamma_table["gamma low"].to_numpy()
    _g_high = gamma_table["gamma high"].to_numpy()
    _sig_low = gamma_table["gamma low sig"].to_numpy()
    _sig_high = gamma_table["gamma high sig"].to_numpy()
    h1 = evaluate_h1(
        _g_low,
        _g_high,
        gamma_table["|high| - |low| lo"].to_numpy(),
        gamma_table["|high| - |low| hi"].to_numpy(),
    )
    h2_table, h2_qualifying = evaluate_h2(
        _g_low, _g_high, _sig_low, _sig_high, returns.columns
    )
    sig_counts = significance_counts(_sig_low, _sig_high)
    # Why built here: the top of the notebook and section 5 show this same table
    h1_evidence = pd.DataFrame(
        {
            "result": [
                f"{h1['|high| > |low|']} of {h1['industries']}",
                str(h1["interval says bigger at high VIX"]),
                str(h1["interval says smaller at high VIX"]),
            ],
            "expected from luck alone": [
                f"{h1['expected by chance']:.1f}",
                f"{h1['expected by chance (each direction)']:.1f}",
                f"{h1['expected by chance (each direction)']:.1f}",
            ],
            "chance of a result this strong from luck": [
                f"{h1['chance of this many or more']:.1%}",
                f"{h1['chance of this many bigger or more']:.4%}",
                "—",
            ],
        },
        index=[
            "industries with larger |γ| at high VIX",
            "…significantly larger",
            "…significantly smaller",
        ],
    )
    return h1, h1_evidence, h2_qualifying, h2_table, sig_counts


@app.cell(hide_code=True)
def _(BLUE, ORANGE, altair_scatter, gamma_table, industry_labels, mo, pd):
    h1_points = pd.DataFrame(
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
    h1_points["label"] = h1_points["industry"].where(
        gamma_table["|high| - |low| sig"].to_numpy(), ""
    )
    h1_chart = mo.ui.altair_chart(
        altair_scatter(
            h1_points,
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
    return h1_chart, h1_points


@app.cell(hide_code=True)
def _(
    INDUSTRY_NAMES,
    gamma_summary,
    gamma_table,
    h1,
    h1_chart,
    h1_evidence,
    h1_points,
    h2_qualifying,
    h2_table,
    industry_labels,
    mo,
    pd,
    result_table,
    sig_counts,
):
    _context = pd.DataFrame(
        {
            "industries with γ clearly non-zero": sig_counts["significant gammas"],
            "expected from luck alone": sig_counts["expected by chance"].map(
                "{:.1f}".format
            ),
        }
    )
    # Why the table follows the chart: dragging a box over points narrows the table to those industries
    _selected_codes = h1_chart.apply_selection(h1_points)["code"]
    _summary = gamma_summary(gamma_table, industry_labels)
    _summary = _summary[_summary["code"].isin(_selected_codes)]
    _qualifying_names = []
    for _code in h2_qualifying.index:
        _qualifying_names.append(INDUSTRY_NAMES[_code])
    mo.vstack(
        [
            mo.md(r"""
    ## 5. Hypotheses 🤖

    Both were set before seeing any results and are judged across all 49 industries,
    against what luck alone would produce (at 95% confidence, about 2–3 of 49 industries
    look significant by chance).

    ### H1, magnitude: sensitivities are larger in high-VIX periods → **supported**
    """),
            result_table(h1_evidence),
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
                label="γ estimate [95% interval], % per VIX point; * = interval excludes 0",
            ),
            mo.md(r"""
    **Supporting context** (descriptive, not a pre-set test): far more industries have a
    clearly non-zero γ at high VIX.
    """),
            result_table(_context),
            mo.md(r"""
    ### H2, direction: sensitivities keep the same sign in both regimes → **can't be judged as written**

    H2 is judged only on industries whose γ is clearly non-zero in *both* regimes:
    """),
            result_table(h2_table),
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
      VIX moves, per-point γ shrinks at high VIX. H1 holds anyway.
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
