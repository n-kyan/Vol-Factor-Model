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
    from pathlib import Path

    import pandas as pd
    import numpy as np
    import requests

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    KEN_FRENCH_URL_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

    return KEN_FRENCH_URL_BASE, Path, RAW_DIR, np, pd, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Utility Functions
    """)
    return


@app.cell
def _(Path, np, pd, requests):
    def download_raw_data(url: str, dest: Path) -> Path:
        # Download url to dest unles dest already exists
        # return dest

        if dest.exists():
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest

    def load_kfrench_table(zip_path: Path, section_title: str | None = None) -> pd.DataFrame:
        # Parse Ken French CSV zip for individual data tables by title
        # return table as a df with DatetimeIndex, decimal returns, NaN for missing

        # Read csv
        raw = pd.read_csv(zip_path, header=None, names=range(100), dtype=str, skip_blank_lines=False)

        # Remove leading and trailing whitespace from cells in col 0
        raw[0] = raw[0].str.strip()
        # Bool series matching the date format
        is_date = raw[0].str.fullmatch(r"\d{8}", na=False)

        if section_title is None:
            search_from = 0
        else:
            # Is the title in the cell?
            is_title = raw[0].str.contains(section_title, regex=False, na=False)
        
            if not is_title.any():
                raise ValueError(f"Section not found: {section_title}")

            # Row of the section title
            search_from = is_title.idxmax()

        data_start = is_date.iloc[search_from:].idxmax()

        # Find first non-date row after the table
        not_date = ~is_date.iloc[data_start:]
        data_end = not_date.idxmax() if not_date.any() else len(raw)

        header = raw.iloc[data_start - 1, 1:].dropna().str.strip().tolist()
        # +1 is needed because of the date column
        table = raw.iloc[data_start:data_end, : len(header) + 1].copy()
        table.columns = ["date"] + header

        # Formatting
        table = table.set_index("date")
        table.index = pd.to_datetime(table.index, format="%Y%m%d")
        table = table.astype(float)
        table = table.replace([-99.99, -999.0], np.nan)
        table = table / 100
        return table

    return download_raw_data, load_kfrench_table


@app.cell
def _(KEN_FRENCH_URL_BASE, RAW_DIR, download_raw_data, load_kfrench_table):
    ind_zip = download_raw_data(KEN_FRENCH_URL_BASE + "49_Industry_Portfolios_daily_CSV.zip",
                           RAW_DIR / "49_Industry_Portfolios_daily_CSV.zip")
    ff_zip = download_raw_data(KEN_FRENCH_URL_BASE + "F-F_Research_Data_Factors_daily_CSV.zip",
                          RAW_DIR / "F-F_Research_Data_Factors_daily_CSV.zip")

    ind = load_kfrench_table(ind_zip, "Average Value Weighted")
    ff = load_kfrench_table(ff_zip)
    return ff, ind


@app.cell
def _(Path, pd):
    CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

    def load_vix_table(csv_path: Path) -> pd.Series:
        """return VIX close in index points with a DatetimeIndex, sorted by date."""
    
        raw = pd.read_csv(csv_path)
        vix = raw.set_index("Date")["Close"]
        vix.index = pd.to_datetime(vix.index, format="%Y-%m-%d")
        vix = vix.astype(float).sort_index().rename("VIX")
        vix.index.name = "date"
    
        return vix

    return CBOE_VIX_URL, load_vix_table


@app.cell
def _(CBOE_VIX_URL, RAW_DIR, download_raw_data, load_vix_table):
    vix_csv = download_raw_data(CBOE_VIX_URL, RAW_DIR/"VIX.csv")

    vix = load_vix_table(vix_csv)
    vix
    return (vix,)


@app.cell(hide_code=True)
def _(ff, mo):

    _title = mo.md(
        '''
        # (FF3) Market Returns DataFrame

        Just was with the Industry data, this data set uses returns as the unit so we divide by 100. The VIX is in units of annualized vol in "points", where a VIX of 20 == 20% annualized vol. 

        Why the numbers look strange: A fraction can only be represented exactly only when the number base modulo the fraction in lowest terms' denominator's prime factors equals zero. For example, in base 10, 1/3 repeats forever because 10%3 != 0, resulting in 0.33 repeating. In binary the base is 2, and for many numbers x/100, the set of the prime factors of the denominator is {2, 5}}. 2%5 != 0 so in binary x/100 repeats forever. A float64 uses 64 bits with 52 allocated for representing the significant figures. The computer stores the repeating number with accuracy of 52 bits. The absolute error is <= x * 1.1e-16. This is because the neighbors to x are spaced by 2e-. I believe that this is below the precision of the data.

        ''')
    mo.vstack([_title, ff])
    return


@app.cell(hide_code=True)
def _(ind, mo):
    # Industries


    _title = mo.md(
        '''
        # Industires DataFrame

        The units of this table is returns, or percent change in price per day, where 0.56 == 0.56%. So, we divide each cell of data by 100 to convert the units into decimal.



        ''')
    mo.vstack([_title, ind])
    return


@app.cell
def _(mo, vix):


    _title = mo.md(
        '''
        # VIX DataFrame

        The VIX is in units of annualized vol in "points", where a VIX of 20 == 20% annualized vol. Each cell is divided to convert the units to decimals. 

        A key decision must be made here. Should the model use daily VIX Level or the daily change in the VIX Level. I decided to use change in VIX becuase is an apprimation for the new information from the VIX each day. This is motivated by Robert Mertons concept of isolating "innovations" (ICPAM )

        "Low autocorrelation is a symptom of being an innovation, not the goal." - 🤖

        "Better heuristic: to measure reaction to news or a priced risk, use an innovation. To measure state-dependence, use a conditioning variable with SEs that account for persistence." - 🤖

        ''')
    mo.vstack([_title, vix])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
