import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First order of business is fetching all of the raw data that we need.

    1. Ken French Data from their website
    2. Stock data from yfinance
    """)
    return


@app.cell
def _():
    import requests
    import zipfile
    from pathlib import Path

    return Path, requests, zipfile


@app.cell(hide_code=True)
def _(Path, mo, requests, zipfile):


    def fetch_french():
        # Urls
        base = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

        file_names = ["F-F_Research_Data_Factors_daily_CSV.zip", "49_Industry_Portfolios_daily_CSV.zip"]

        # Setup
        destination_dir = Path("raw")
        destination_dir.mkdir(exist_ok=True)

        # Download
        for file_name in file_names:
            zip_path = destination_dir / file_name
            response = requests.get(base + file_name, timeout=10)
            response.raise_for_status()
            zip_path.write_bytes(response.content)

        # Unzip
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(destination_dir)

        # Delete the zip file
            zip_path.unlink()

    mo.md("## FF3 Fetch")
    return


@app.cell
def _():
    # print(fetch_french())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ok time for yfinance
    """)
    return


@app.cell
def _():
    import yfinance as yf
    import pandas as pd

    return pd, yf


@app.cell(hide_code=True)
def _(Path, mo, yf):
    def fetch_yfin():

        # Setup
        tickers = ["SPY", "^VIX"]

        destination_dir = Path("raw")
        destination_dir.mkdir(exist_ok=True)

        for ticker in tickers:

            file_path = destination_dir / f"{ticker.lstrip('^')}.csv"
            if file_path.exists():
                continue

            df = yf.download(
                ticker,
                period="max",
                auto_adjust=True, # splits and dividends
                multi_level_index=False,
            )
            df.to_csv(file_path)


    mo.md("## SPY and VIX Fetch")
    return (fetch_yfin,)


@app.cell
def _(fetch_yfin):
    fetch_yfin()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    All the data is collected and it is time for cleaning, formatting, and merging.

    Tasks:
    1. Fix formatting differences across files. (ex: date, NaN, units)
    2.

    VRP = the difference between today's fear about the short term future vs observed recent turbulence.
    """)
    return


@app.cell
def _():
    import numpy as np

    return (np,)


@app.cell
def _():
    industries_path = "raw/49_Industry_Portfolios_Daily.csv"
    famaFrench_path = "raw/F-F_Research_Data_Factors_daily.csv"
    vix_path = "raw/VIX.csv"
    spy_path = "raw/SPY.csv"
    return famaFrench_path, industries_path, spy_path, vix_path


@app.cell(hide_code=True)
def _(famaFrench_path, mo, pd):
    # FF3
    _raw = pd.read_csv(famaFrench_path, header=None, dtype=str, skip_blank_lines=False, names=range(5))
    _is_date = _raw[0].str.fullmatch(r"\d{8}")
    ff = _raw[_is_date].copy()

    _column_names = ["date"] + _raw.loc[4, 1:4].str.strip().to_list()
    ff.columns = _column_names

    ff = ff.set_index("date")[["Mkt-RF", "RF"]].astype(float)
    ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
    ff = ff/100


    _title = mo.md(
        '''
        # (FF3) Market Returns DataFrame

        Just was with the Industry data, this data set uses returns as the unit so we divide by 100. The VIX is in units of annualized vol in "points", where a VIX of 20 == 20% annualized vol. 

        Why the numbers look strange: A fraction can only be represented exactly only when the number base modulo the fraction in lowest terms' denominator's prime factors equals zero. For example, in base 10, 1/3 repeats forever because 10%3 != 0, resulting in 0.33 repeating. In binary the base is 2, and for many numbers x/100, the set of the prime factors of the denominator is {2, 5}}. 2%5 != 0 so in binary x/100 repeats forever. A float64 uses 64 bits with 52 allocated for representing the significant figures. The computer stores the repeating number with accuracy of 52 bits. The absolute error is <= x * 1.1e-16. This is because the neighbors to x are spaced by 2e-. I believe that this is below the precision of the data.

        ''')
    mo.vstack([_title, ff])
    return (ff,)


@app.cell(hide_code=True)
def _(ff, industries_path, mo, np, pd):
    # Industries

    # Isolating the right table
    _raw = pd.read_csv(industries_path, header=None, dtype=str, skip_blank_lines=False, names=range(50))

    first_col = _raw[0]
    start = first_col.str.contains("Average Value Weighted").idxmax()

    _is_date = first_col.str.fullmatch(r"\d{8}")
    end = (~_is_date.loc[start + 2:]).idxmax()

    ind = _raw.loc[start + 2 : end - 1].copy()

    # Data processing
    ind.columns = ["date"] + _raw.loc[start + 1, 1:].str.strip().tolist()

    ind = ind.set_index("date")
    ind.index = pd.to_datetime(ind.index, format="%Y%m%d")

    ind = ind.astype(float)
    ind = ind.replace([-99.99, -999], np.nan) # data source specifies these as their NaN values
    ind = ind/100

    ind = ind.sub(ff["RF"], axis=0) # remove risk free rate

    # Remove NaN columns
    _bad = ind.columns[ind.isna().any()].tolist()
    print(_bad)
    ind = ind.drop(columns=_bad)

    _title = mo.md(
        '''
        # Industires DataFrame

        The units of this table is returns, or percent change in price per day, where 0.56 == 0.56%. So, we divide each cell of data by 100 to convert the units into decimal.



        ''')
    mo.vstack([_title, ind])
    return (ind,)


@app.cell
def _():
    # Check that the ind dataset was total returns not excess returns
    return


@app.cell
def _(mo, pd, vix_path):
    vix_raw = pd.read_csv(vix_path, index_col=0, parse_dates=True)

    vix = (vix_raw["Close"] / 100).rename("VIX")

    _title = mo.md(
        '''
        # VIX DataFrame

        The VIX is in units of annualized vol in "points", where a VIX of 20 == 20% annualized vol. Each cell is divided to convert the units to decimals. 

        A key decision must be made here. Should the model use daily VIX Level or the daily change in the VIX Level. I decided to use change in VIX becuase is an apprimation for the new information from the VIX each day. This is motivated by Robert Mertons concept of isolating "innovations" (ICPAM )

        "Low autocorrelation is a symptom of being an innovation, not the goal." - 🤖

        "Better heuristic: to measure reaction to news or a priced risk, use an innovation. To measure state-dependence, use a conditioning variable with SEs that account for persistence." - 🤖

        ''')
    mo.vstack([_title, vix])
    return (vix,)


@app.cell
def _():
    return


@app.cell
def _(mo, np, pd, spy_path):
    spy_raw = pd.read_csv(spy_path, index_col=0, parse_dates=True)
    spy = spy_raw["Close"]

    log_returns = np.log(spy).diff()
    rv = (log_returns.rolling(20).std() * np.sqrt(252)).rename("RV")


    _title = mo.md(
        '''# Realized Vol DataFrame

        test''')
    mo.vstack([_title, rv])
    return (rv,)


@app.cell
def _(ff, ind, pd, rv, vix):
    final = pd.concat([ff["Mkt-RF"], vix, rv, ind], axis=1, join="inner")

    final
    return


@app.cell
def _(np):
    rng = np.random.default_rng(0)                  # seeded random generator, so results repeat
    n = 100_000
    mkt = rng.standard_normal(n)                    # n draws from N(0, 1)
    dvix = -0.7 * mkt + np.sqrt(0.51) * rng.standard_normal(n)
    y = 1.0 * mkt + 0.5 * rng.standard_normal(n)    # market beta 1, no ΔVIX effect, plus noise
    X = np.column_stack([mkt, dvix])
    np.linalg.solve(X.T @ X, X.T @ y), (dvix @ y) / (dvix @ dvix)
    return


@app.cell
def _(np):
    def resid(y, X):
        return y - X @ np.linalg.solve(X.T @ X, X.T @ y)

    return (resid,)


@app.cell
def _(ff, np, pd, resid, vix):
    d = pd.concat([ff["Mkt-RF"], vix.shift(1), vix.diff()], axis=1, join="inner").dropna()
    d.columns = ["mkt", "L", "dvix"]
    d = d - d.mean()                              # demean every column

    M = d[["mkt"]].to_numpy()
    ML = d[["mkt", "L"]].to_numpy()
    MD = d[["mkt", "dvix"]].to_numpy()

    dvix_o1 = resid(d["dvix"].to_numpy(), ML)     # Mkt → L → ΔVIX
    dvix_o2 = resid(d["dvix"].to_numpy(), M)      # Mkt → ΔVIX → L
    L_o1 = resid(d["L"].to_numpy(), M)            # L after Mkt only
    L_o2 = resid(d["L"].to_numpy(), MD)           # L after Mkt and ΔVIX

    np.corrcoef(dvix_o1, dvix_o2)[0, 1], np.corrcoef(L_o1, L_o2)[0, 1]
    return


@app.cell
def _(np, pd, vix):
    v = vix.dropna()
    bucket = pd.qcut(vix.shift(1), 5)
    stds = pd.concat(
        [vix.diff().groupby(bucket, observed=True).std(),
         np.log(vix).diff().groupby(bucket, observed=True).std()],
        axis=1, keys=["dVIX", "dlogVIX"])
    v.skew(), np.log(v).skew(), stds
    return


@app.cell
def _(ff, pd, vix):
    ff["Mkt-RF"].groupby(pd.qcut(vix.shift(1), 5), observed=True).std()
    return


if __name__ == "__main__":
    app.run()
