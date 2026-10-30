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
    # Phase 2: The vol shock (Gram-Schmidt) 🤖

    *Companion to `main.py`. Code: `vol_shock` in `src/factors.py`.*

    **Bottom line**

    - The market explains about **60%** of daily VIX changes (correlation −0.79): when the market falls, VIX jumps.
    - The **vol shock** is the other ~40%: each day's VIX change minus what that day's market move predicts. It is uncorrelated with the market by construction.
    - The relationship is slightly stronger on high-VIX days (a 1% market drop goes with VIX +1.16 points, vs. +1.10 on low-VIX days), and VIX moves are about twice as large.

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
def _(mo):
    mo.md(r"""
    ## The vol shock (Gram-Schmidt)

    **Question.** On a typical day, how much of VIX's move is just the market falling or rising, and what's left over? The leftover is the **vol shock**: news about volatility that the market move doesn't explain.

    **How it's built.** Regress each day's VIX change on that day's market return, and keep the miss (the residual). If the market fell 2% and that usually goes with VIX rising 3 points, but today VIX rose 5, the vol shock is about +2.

    **Decisions and why**

    | Decision | Why | Obvious alternative, and why not |
    |---|---|---|
    | **Market goes first.** VIX is cleaned of the market, not the other way round. | The question is about vol moves *beyond* market moves. The market keeps everything it can explain, and only the leftover counts as a vol shock. | Clean the market of VIX instead. Then "market" would mean "market moves VIX can't explain", β would stop being the ordinary market beta, and the question would change. |
    | **Include an intercept** in this regression. | It makes the vol shock average exactly 0, so it only measures surprises, not VIX's average drift. | Leave it out: any average drift in VIX would leak into the shock. |
    | **VIX changes in points** (already decided). | It matches how options desks measure vol moves (vega), and it's simple to read. | Percentage changes: in the post-project analysis. |

    **Why cleaning makes the regressors uncorrelated.** The vol shock is a regression miss, and regression misses are uncorrelated with the regressor by construction (`ols_nb.py`, "no drift"). So the market and the vol shock don't overlap. Each industry's response can then be split cleanly into "market part" and "vol-shock part".
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


if __name__ == "__main__":
    app.run()
