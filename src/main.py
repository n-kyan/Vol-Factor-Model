import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from data import load_kfrench_table, load_vix_table, build_data, summarize_data
    from factors import vix_log_returns, regime_flag

    return build_data, mo, summarize_data


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Factor Modeling with Volatility
    Kyan Nelson
    n.kyan@outlook.com
    10/15/26

    This project is managed using uv. To run the project use 'uv run marimo run main.py'.

    AI Disclaimer: I will be using Claude throughout this process for educational purposes and for understanding what data is readily  available to me and how to get it. I will be writing all of the code and documentation myself. I will do my best to specify where AI was used with this symbol 🤖.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline of Project with Personal Notes:

    **Data:**
    Stock data from yfinance and download daily/monthly data from the Ken French Data Library. Look into factor normalization.

    **Statistical Estimator:** Code a Ordinary Least Squares (OLS) Estimator using Lin Alg via NumPy. Write it from scratch so you can understand the math. Decide if its as cross-sectional or time series regression and why.

    **Factor Model:** Look into factor normalization, multicolinearity, linear independence, and Gram-Schmidt Projection. Minimize residual variance; consider the concept of inverting matricies and minimizing rather than trying to maximize.

    **Portfolio Covariance:** Construct the covariance matrix of the assets (more Lin Alg). Use it to calculate things like portfolio vol and VaR. Generate confidence intervals with Bootstrapping via Monte Carlo Sim. Covariance shifts over time.

    **Optimization:** Find the weights that maximize sharpe ratio. Maybe use the invert and minimize tech here. Look into Markowitz Optimization and alternatives. How does this all work as portfolio evolves over time?

    **Final Write-Up:** Clean up code and make sure documentation doesn't have any spelling or grammar errors. Write a conclusion a-la scientific method.

    ## General Ideas About Direction
    Rather than picking arbitrary factors to model I want to look into using volatiltiy as my factors. Initial thoughts are that this could look like using vix data, vix futures data, implied vol on vix options, implied vol of snp500, realized vol of snp500, sector implied vol, realized vol of sector. If not then something else that could be interesting is looking at interest rates. This piques my curiosity more than looking at things like size, momentum, or P/E ratio as of right now. After some chatting with Claude, I have settled on using SnP500 20-day rolling realized vol and the vol risk premium (difference between implied vol and realized vol).

    For the price data, I am planning on pulling daily data for sector ETFs, SnP500, and the VIX. This should be fun for modeling because prices, RV and VIX are in different units, and realized vol and the VRP will be highly correlated. I need to decide if I am using prices/VIX level or returns/change in VIX.
    """)
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
