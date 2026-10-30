import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd

    from charts import BLUE, GRAY, INK, ORANGE, compare_to_normal, new_figure, vix_timeline
    from data import INDUSTRY_NAMES, build_data, load_vix_table
    from factors import MEDIAN_WINDOW, regime_threshold
    from stats import describe_moves, tail_counts

    return (
        BLUE,
        GRAY,
        INDUSTRY_NAMES,
        INK,
        MEDIAN_WINDOW,
        ORANGE,
        build_data,
        compare_to_normal,
        describe_moves,
        load_vix_table,
        mo,
        new_figure,
        np,
        pd,
        regime_threshold,
        tail_counts,
        vix_timeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 1: Data 🤖

    *Companion to `main.py`. Code: `src/data.py` (loading) and `src/factors.py` (VIX change, regime flag).*

    **Bottom line**

    - **9,228 trading days** (1990-01-03 to 2026-08-31) for **all 49 industries**, with no gaps and no industry dropped.
    - Days are split into **low- and high-VIX regimes** at the 1990–2026 VIX median of **17.56**, using the **prior day's** close. That gives a near 50/50 split.
    - Daily moves have **fat tails**: market days beyond 4 SDs happen ~95× more often than a bell curve predicts. This is why confidence intervals later come from a block bootstrap, not textbook formulas.
    """)
    return


@app.cell
def _(MEDIAN_WINDOW, build_data, load_vix_table, regime_threshold):
    returns, factors = build_data()
    vix_full = load_vix_table()
    threshold = regime_threshold(vix_full)
    vix_window = vix_full.loc[MEDIAN_WINDOW[0]:MEDIAN_WINDOW[1]]
    return factors, returns, threshold, vix_full, vix_window


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Sources

    | Source | What it provides | Why the project needs it | Units in the file |
    |---|---|---|---|
    | **Ken French: 49 Industry Portfolios** (daily, value-weighted section) | Each industry's daily return | The 49 things whose sensitivities we measure | % per day |
    | **Ken French: Fama-French factors** (daily) | **Mkt-RF**: the market's return minus the risk-free rate. **RF**: the risk-free (1-month T-bill) rate | Mkt-RF is the market factor in every regression. RF turns industry returns into excess returns, so both sides of the regression are measured the same way | % per day |
    | **Cboe VIX history** | VIX daily close | The source of the vol shock and of the regime flag | VIX points (20 = 20% expected annualized volatility) |

    **What the VIX is.** The market's expected volatility of the S&P 500 over the next 30 days, read from S&P 500 option prices and annualized. It is quoted in percentage points: a VIX of 20 means option prices imply about 20% annualized volatility.

    **How Ken French builds the industry portfolios.** In his words: *"We assign each NYSE, AMEX, and NASDAQ stock to an industry portfolio at the end of June of year t based on its four-digit SIC code at that time."* So each portfolio holds every listed stock in that industry, not a top-N subset. The project uses the **value-weighted** version, where bigger companies count more (by market value).

    **Why value-weighted.** Mkt-RF is itself value-weighted, so industries and the market are built the same way. An equal-weighted industry would be dominated by its many small stocks, which behave differently from the market factor we regress on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. From raw files to the panel

    Each box is one step in `build_data()`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        flowchart LR
            A["Download once,<br/>cache in data/raw/"] --> B["Parse Ken French CSVs<br/>% → decimals,<br/>-99.99 → missing"]
            B --> C["Join industries, factors<br/>and VIX on dates<br/>(keep shared days only)"]
            C --> D["Industry return − RF<br/>= excess return"]
            C --> E["VIX change in points<br/>ΔVIX = today − yesterday"]
            C --> F["Regime flag<br/>yesterday's VIX ≥ 17.56?"]
            D --> G["Drop any industry<br/>with missing days<br/>(none dropped)"]
            E --> G
            F --> G
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Step | Why |
    |---|---|
    | **Cache downloads, never re-download** | Reruns give identical results and don't depend on the websites being up. Delete a file in `data/raw/` to refresh it. |
    | **Keep only days present in all three sources** | A regression needs every variable on the same day. A day without a VIX close can't have a vol shock. |
    | **Excess returns** (industry return − RF) | Matches how Mkt-RF is defined, so β is the standard market beta and the risk-free rate doesn't leak into α. |
    | **ΔVIX in points** | Matches vega, the industry standard in options market making: sensitivities can be read as "per 1 vol point". Also the simplest to interpret. Percentage/log changes are in the post-project analysis. |
    | **Drop industries with missing days** | Every industry then uses the exact same days, so results are comparable across industries. From 1990 none are missing. |
    """)
    return


@app.cell(hide_code=True)
def _(mo, pd, returns, vix_full):
    _sample = pd.DataFrame(
        {"value": [
            str(returns.index.min().date()),
            str(returns.index.max().date()),
            f"{len(returns):,}",
            str(returns.shape[1]),
        ]},
        index=["first day", "last day", "trading days", "industries"],
    )
    mo.vstack([
        mo.md(r"""
    ## 3. The sample
    """),
        _sample,
        mo.md(rf"""
    - **Start, 1990-01-03.** Cboe's VIX history starts {vix_full.index.min().date()}, and a VIX *change* needs the day before, so the first usable day is the next trading day. VIX is the latest-starting source (Ken French goes back to 1926), so it sets the start.
    - **End, 2026-08-31.** The last day in the cached Ken French files. The VIX file runs later, so Ken French sets the end. (Ken French's site now has data through September 2026; deleting the cached files would extend the sample.)
    - **The start date is a parameter** (`build_data(start=...)`), so the analysis can be rerun on any sub-period. The regime threshold doesn't move with it (section 5).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(INDUSTRY_NAMES, mo, pd, returns):
    _industries = pd.DataFrame({
        "industry": returns.columns.map(INDUSTRY_NAMES),
        "code": returns.columns,
    }).sort_values("industry").reset_index(drop=True)
    _industries.index = _industries.index + 1
    mo.vstack([
        mo.md(r"""
    ## 4. The 49 industries

    Full names from Ken French's industry definitions file. **Code** is the column name used in the data and in every later table. A few names are not obvious: **Chips** = Electronic Equipment, **Hardw** = Computers, **Fin** = Trading, **Other** = Almost Nothing.
    """),
        _industries,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The regime split

    **The question it serves:** do industries react to vol shocks differently on calm days than on stressed days? That needs a rule for which days are "stressed".

    **The rule:** a day is **high-VIX** if the *previous* day's VIX close is at or above **17.56**, the median VIX close over 1990–2026.
    """)
    return


@app.cell(hide_code=True)
def _(threshold, vix_full, vix_timeline):
    vix_timeline(vix_full.loc["1990":], threshold)
    return


@app.cell(hide_code=True)
def _(BLUE, GRAY, INK, ORANGE, mo, new_figure, np, vix_window):
    _median = vix_window.median()
    _mean = vix_window.mean()
    _fig, _axes = new_figure(width=8, height=3.6)
    _ax = _axes[0]
    _ax.hist(vix_window, bins=np.arange(8, 84, 1), color=GRAY, alpha=0.6)
    _ax.axvline(_median, color=BLUE, linewidth=2, label=f"median {_median:.2f}: {(vix_window >= _median).mean():.0%} of days above")
    _ax.axvline(_mean, color=ORANGE, linewidth=2, label=f"mean {_mean:.2f}: {(vix_window >= _mean).mean():.0%} of days above")
    _ax.set_xlabel("VIX close (points)")
    _ax.set_ylabel("number of days")
    _ax.set_title("Distribution of VIX closes, 1990–2026", fontsize=10)
    _ax.legend(frameon=False, fontsize=8)
    _ax.tick_params(colors=INK)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(r"""
    **Why the median, not the mean.** VIX is **right-skewed**: 4 in 5 days fall between about 12 and 29, and a long tail of crisis days stretches out to 80. The median is the middle day, so the tail doesn't move it. The mean adds up every day, so a few days at 50–80 drag it up to 19.4. Splitting at the mean would put only 40% of days in the high regime. The median splits the days evenly without choosing an arbitrary level like 25.

    **Why a fixed 1990–2026 window.** The threshold is always computed on the full 1990–2026 VIX history, whatever sample period is being analyzed. A 2020-only run still calls a day "high" at 17.56, so results from different periods are measured against the same yardstick.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(factors, mo, pd, threshold):
    _days = factors.loc["2020-02-19":"2020-03-02", ["VIX", "dVIX", "high"]].copy()
    _example = pd.DataFrame({
        "VIX close": _days["VIX"],
        "ΔVIX (points)": _days["dVIX"],
        "regime by yesterday's close (used)": _days["high"].map({True: "high", False: "low"}),
        "regime by today's close (not used)": (_days["VIX"] >= threshold).map({True: "high", False: "low"}),
    })
    _example.index = _example.index.date

    _same_day = factors["VIX"] >= threshold
    _bias = pd.DataFrame(
        {
            "flag by yesterday's close (used)": [factors.loc[factors["high"], "dVIX"].mean(), factors.loc[~factors["high"], "dVIX"].mean()],
            "flag by today's close": [factors.loc[_same_day, "dVIX"].mean(), factors.loc[~_same_day, "dVIX"].mean()],
        },
        index=["average ΔVIX on high days", "average ΔVIX on low days"],
    ).round(3)
    mo.vstack([
        mo.md(r"""
    ### Why yesterday's close

    Today's VIX close is partly *caused* by today's vol shock. Flag a day by its own close and a big jump in VIX pushes that same day into "high" automatically. The start of the COVID crash shows it:
    """),
        _example,
        mo.md(r"""
    On **2020-02-24** VIX jumped **7.95 points**, from 17.08 to 25.03. By today's close that day is "high", but only *because of* the shock we are trying to measure. By yesterday's close (17.08, below 17.56) it is a low-VIX day that received a big shock, which is what actually happened.

    Across the whole sample, the same-day rule would sort VIX increases into the high regime by construction:
    """),
        _bias,
        mo.md(r"""
    With today's close, high days average a VIX *rise*, which is the flag reacting to the shock. With yesterday's close, the flag is set before the day's shock happens, so it can't be caused by it.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(BLUE, GRAY, ORANGE, factors, mo, new_figure, np, pd):
    _counts = pd.DataFrame(
        {
            "days": [int((~factors["high"]).sum()), int(factors["high"].sum())],
            "share of sample": [f"{(~factors['high']).mean():.1%}", f"{factors['high'].mean():.1%}"],
        },
        index=["low-VIX days", "high-VIX days"],
    )
    _by_year = factors["high"].groupby(factors.index.year).mean()
    _fig, _axes = new_figure(width=11, height=3.2)
    _ax = _axes[0]
    _colors = np.where(_by_year.values >= 0.5, ORANGE, BLUE)
    _ax.bar(_by_year.index, _by_year.values * 100, color=_colors, width=0.8)
    _ax.axhline(50, color=GRAY, linewidth=1)
    _ax.set_ylabel("% of days high-VIX")
    _ax.set_title("High-VIX days come in long episodes (orange = most of that year was high-VIX)", fontsize=10)
    _fig.tight_layout()
    mo.vstack([
        mo.md(r"""
    ### How the days split
    """),
        _counts,
        _fig,
        mo.md(r"""
    The split is close to 50/50, as a median split should be: the median *defines* the halfway point of the 1990–2026 history. (It isn't exactly 50% because the sample starts one day later and flags use the prior day.)

    The chart shows the catch. High-VIX days come in long stretches: 1997–2003, 2008–2011, 2020–2022. Calm years like 1993–1995, 2005 and 2017 have almost none. So the ~4,600 high days come from a **handful of episodes**, and the effective amount of independent information is much smaller than the day count. This is a stated caveat of the project, and one reason the bootstrap resamples blocks of consecutive days.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(describe_moves, factors, mo, pd):
    _mkt = describe_moves(factors["Mkt-RF"] * 100)
    _dvix = describe_moves(factors["dVIX"])
    _rows = {}
    for _name in _mkt:
        if "days moving" in _name:
            _rows[_name] = {"market excess return (%)": f"{_mkt[_name]:.0%}", "VIX change (points)": f"{_dvix[_name]:.0%}"}
        else:
            _rows[_name] = {"market excess return (%)": f"{_mkt[_name]:.2f}", "VIX change (points)": f"{_dvix[_name]:.2f}"}
    mo.vstack([
        mo.md(r"""
    ## 6. How big are daily moves?

    The **standard deviation (SD)** measures how spread out the daily moves are. The table turns it into plain counts of days.
    """),
        pd.DataFrame(_rows).T,
        mo.md(r"""
    - On about **4 in 5 days** the market moves less than 1.14% and VIX less than 1.66 points.
    - On **half of all days** the market moves less than 0.51% and VIX less than 0.62 points.
    - The **largest days** are about 10× and 15× the SD.

    The charts below compare these moves with a normal (bell-curve) distribution with the same mean and SD. A normal distribution is what textbook confidence-interval formulas assume.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(BLUE, compare_to_normal, factors):
    compare_to_normal(factors["Mkt-RF"] * 100, "Daily market excess return", BLUE)
    return


@app.cell(hide_code=True)
def _(GRAY, compare_to_normal, factors):
    compare_to_normal(factors["dVIX"], "Daily VIX change", GRAY)
    return


@app.cell(hide_code=True)
def _(factors, mo, pd, tail_counts):
    _cutoffs = [3, 4, 5, 6]
    _mkt = tail_counts(factors["Mkt-RF"], _cutoffs)
    _dvix = tail_counts(factors["dVIX"], _cutoffs)
    _table = pd.DataFrame({
        "normal distribution expects": _mkt["normal distribution expects"].map("{:.2f}".format),
        "market: actual days": _mkt["actual days"].astype(int),
        "VIX change: actual days": _dvix["actual days"].astype(int),
    })
    mo.vstack([
        mo.md(rf"""
    ### Extreme days: actual vs. a normal distribution ({len(factors):,} days)
    """),
        _table,
        mo.md(r"""
    **Reading the charts and the table.**

    - **The middle (left charts):** real days are *more* bunched near zero than the bell curve. Most days are quieter than a normal distribution with the same SD would suggest.
    - **The tails (right charts and table):** real days also have far more extreme moves. Beyond 4 SDs, a normal distribution expects about **0.6 days** in 36 years. The market had **55** and VIX had **73**. Beyond 6 SDs the normal expects essentially none; there were 15 and 30.
    - Together these are **fat tails**: quiet most of the time, with occasional moves far larger than a bell curve allows.

    **Why it matters for the project.** Textbook confidence intervals for regression coefficients assume well-behaved, independent noise. Fat tails, plus crisis days that cluster together (section 5), break those assumptions. So the project gets its confidence intervals from a **block bootstrap** (Phase 5), which uses the data's own behavior instead of assuming a bell curve.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Known data caveats

    - **Mkt-RF vs. VIX coverage.** Mkt-RF covers all CRSP US stocks; VIX comes from S&P 500 options. Close, but not identical.
    - **Few independent crisis episodes.** High-VIX days come in long stretches (section 5), so the effective sample is smaller than 4,636 days.
    - **Points vs. percentage VIX changes.** A 5-point jump means more at VIX 12 than at VIX 60. If industries actually respond to *percentage* moves, per-point sensitivities will look smaller at high VIX, which works *against* the high-VIX hypothesis. Checked in the post-project analysis.
    """)
    return


if __name__ == "__main__":
    app.run()
