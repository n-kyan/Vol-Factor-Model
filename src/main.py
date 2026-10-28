import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from data import load_kfrench_table, load_vix_table, build_data, summarize_data
    from factors import dvix, regime_flag

    return build_data, load_vix_table, mo, summarize_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Factor Modeling with Volatility
    Kyan Nelson
    n.kyan@outlook.com
    10/15/26

    This project is managed using uv. To run the project use 'uv run marimo run src/main.py'.

    AI Disclaimer: I will be using Claude throughout this process for educational purposes and for understanding what data is readily  available to me and how to get it. I will be writing all of the code and documentation myself. I will do my best to specify where AI was used with this symbol 🤖.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project Outline 🤖

    **Question:** How do industries react to volatility shocks beyond what market moves explain, and does that change at different VIX levels?

    **Hypotheses** (judged by the overall pattern across the 49 industries; at 95% confidence about 2–3 will look significant by chance)

    - **H1, direction:** each industry's vol-shock sensitivity has the same sign in low- and high-VIX periods. Judged only for industries whose confidence interval excludes zero in both regimes.
    - **H2, magnitude:** vol-shock sensitivities are larger in absolute value in high-VIX periods.

    **Data** (daily, 1990 to latest available)

    - Ken French 49 industry portfolios, value-weighted, daily returns
    - Ken French daily Mkt-RF and RF (industry excess return = industry return − RF)
    - Cboe VIX daily close

    **Steps**

    1. **Data:** download, handle missing values (-99.99), align dates, compute industry excess returns and daily VIX changes in points.
    2. **Gram-Schmidt:** regress VIX changes on Mkt-RF; the residual is the vol shock. Market goes first. Report the market–VIX correlation.
    3. **Time-series regressions:** regress each industry's excess return on Mkt-RF and the vol shock to get its market sensitivity (β) and vol-shock sensitivity (γ). Full sample.
    4. **Regimes:** split days by prior-day VIX close, above vs. below the long-run median. Re-run step 3 within each regime.
    5. **Monte Carlo validation:** simulate data with known β, γ and a known regime difference; confirm steps 2–4 recover them and that confidence intervals contain the truth about 95% of the time.
    6. **Block bootstrap:** confidence intervals for γ in each regime and for the high-minus-low difference.
    7. **Evaluate H1 and H2** against the chance baseline.

    **Known caveats**

    - If industries respond to percentage VIX moves, per-point sensitivity will mechanically shrink at high VIX. This biases against H2.
    - Mkt-RF covers all CRSP US firms, while VIX is based on S&P 500 options. Close, but not identical.
    - High-VIX days come from a few episodes, so the effective sample is smaller than the day count.
    """)
    return


@app.cell
def _(load_vix_table):
    vix = load_vix_table()
    vix
    return


@app.cell
def _():
                                                 
    return


@app.cell
def _(build_data, summarize_data):
    returns, factors = build_data()

    factors

    s = summarize_data(returns, factors)

    s
    return


if __name__ == "__main__":
    app.run()
