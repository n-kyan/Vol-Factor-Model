from factors import dvix, regime_flag, regime_threshold

# First order of business is fetching all of the raw data that we need.
#
# 1. Ken French Data from their website
# 2. VIX close data from CBOE

from pathlib import Path

import pandas as pd
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
KEN_FRENCH_URL_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"

# Full industry names, from Ken French's Siccodes49.txt (industry definitions file)
INDUSTRY_NAMES = {
    "Agric": "Agriculture",
    "Food": "Food Products",
    "Soda": "Candy & Soda",
    "Beer": "Beer & Liquor",
    "Smoke": "Tobacco Products",
    "Toys": "Recreation",
    "Fun": "Entertainment",
    "Books": "Printing and Publishing",
    "Hshld": "Consumer Goods",
    "Clths": "Apparel",
    "Hlth": "Healthcare",
    "MedEq": "Medical Equipment",
    "Drugs": "Pharmaceutical Products",
    "Chems": "Chemicals",
    "Rubbr": "Rubber and Plastic Products",
    "Txtls": "Textiles",
    "BldMt": "Construction Materials",
    "Cnstr": "Construction",
    "Steel": "Steel Works Etc",
    "FabPr": "Fabricated Products",
    "Mach": "Machinery",
    "ElcEq": "Electrical Equipment",
    "Autos": "Automobiles and Trucks",
    "Aero": "Aircraft",
    "Ships": "Shipbuilding, Railroad Equipment",
    "Guns": "Defense",
    "Gold": "Precious Metals",
    "Mines": "Non-Metallic and Industrial Metal Mining",
    "Coal": "Coal",
    "Oil": "Petroleum and Natural Gas",
    "Util": "Utilities",
    "Telcm": "Communication",
    "PerSv": "Personal Services",
    "BusSv": "Business Services",
    "Hardw": "Computers",
    "Softw": "Computer Software",
    "Chips": "Electronic Equipment",
    "LabEq": "Measuring and Control Equipment",
    "Paper": "Business Supplies",
    "Boxes": "Shipping Containers",
    "Trans": "Transportation",
    "Whlsl": "Wholesale",
    "Rtail": "Retail",
    "Meals": "Restaurants, Hotels, Motels",
    "Banks": "Banking",
    "Insur": "Insurance",
    "RlEst": "Real Estate",
    "Fin": "Trading",
    "Other": "Almost Nothing",
}

# Utility Functions
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

def load_kfrench_table(file_name: str, section_title: str | None = None) -> pd.DataFrame:
    """Download (if needed) and parse one table from a Ken French CSV zip: DatetimeIndex, decimal returns, NaN for missing."""
    zip_path = download_raw_data(KEN_FRENCH_URL_BASE + file_name, RAW_DIR / file_name)
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

def load_vix_table() -> pd.Series:
    """return VIX close in index points with a DatetimeIndex, sorted by date."""

    csv_path = download_raw_data(CBOE_VIX_URL, RAW_DIR / "VIX_History.csv")

    raw = pd.read_csv(csv_path)
    vix = raw.set_index("DATE")["CLOSE"]
    vix.index = pd.to_datetime(vix.index, format="%m/%d/%Y")
    vix = vix.astype(float).sort_index().rename("VIX")
    vix.index.name = "date"

    return vix


# (FF3) Market Returns DataFrame
#
# Just was with the Industry data, this data set uses returns as the unit so we divide by 100. The VIX is in units of annualized vol in "points", where a VIX of 20 == 20% annualized vol.
#
# Why the numbers look strange: A fraction can only be represented exactly only when the number base modulo the fraction in lowest terms' denominator's prime factors equals zero. For example, in base 10, 1/3 repeats forever because 10%3 != 0, resulting in 0.33 repeating. In binary the base is 2, and for many numbers x/100, the set of the prime factors of the denominator is {2, 5}}. 2%5 != 0 so in binary x/100 repeats forever. A float64 uses 64 bits with 52 allocated for representing the significant figures. The computer stores the repeating number with accuracy of 52 bits. The absolute error is <= x * 1.1e-16. This is because the neighbors to x are spaced by 2e-. I believe that this is below the precision of the data.


# Industries
#
# Industires DataFrame
#
# The units of this table is returns, or percent change in price per day, where 0.56 == 0.56%. So, we divide each cell of data by 100 to convert the units into decimal.


# VIX DataFrame
#
# The VIX is in units of annualized vol in "points", where a VIX of 20 == 20% annualized vol. .
#
# A key decision must be made here. Should the model use daily VIX Level or the daily change in the VIX Level. I decided to use change in VIX becuase is an apprimation for the new information from the VIX each day. This is motivated by Robert Mertons concept of isolating "innovations" (ICPAM )
#
# "Low autocorrelation is a symptom of being an innovation, not the goal." - 🤖
#
# "Better heuristic: to measure reaction to news or a priced risk, use an innovation. To measure state-dependence, use a conditioning variable with SEs that account for persistence." - 🤖


def build_data(start: str = "1990-01-03") -> pd.DataFrame:
    """Returns panel of industry excess returns and a tables of the Mkt-RF, VIX, dVIX, and regime flag time series; inner-joined."""
    ind = load_kfrench_table("49_Industry_Portfolios_daily_CSV.zip", "Average Value Weighted")
    ff = load_kfrench_table("F-F_Research_Data_Factors_daily_CSV.zip")
    vix = load_vix_table()

    panel = pd.concat(
        [ind, ff[["Mkt-RF", "RF"]], vix, dvix(vix), regime_flag(vix)],
        axis=1, join="inner",
    ).loc[start:]

    industries = ind.columns
    panel[industries] = panel[industries].sub(panel["RF"], axis=0)
    panel = panel.drop(columns="RF")

    missing = panel[industries].isna().any()
    dropped = missing[missing].index.tolist()
    if dropped:
        print(f"Dropped industries with missing data since {start}: {dropped}")
    panel = panel.drop(columns=dropped)

    if panel.isna().any().any():
        raise ValueError("Panel still contains NaN")

    industries_kept = [c for c in industries if c not in dropped]
    return panel[industries_kept], panel.drop(columns=industries_kept)

def summarize_data(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.Series:
    """Sample size, regime split, and factor moments."""
    return pd.Series({
        "start": returns.index.min().date(),
        "end": returns.index.max().date(),
        "days": len(returns),
        "industries": returns.shape[1],
        "regime_threshold": regime_threshold(load_vix_table()),
        "days_high": int(factors["high"].sum()),
        "days_low": int((~factors["high"]).sum()),
        "share_high": factors["high"].mean(),
        "VIX_mean": factors["VIX"].mean(),
        "VIX_median": factors["VIX"].median(),
        "Mkt-RF_mean": factors["Mkt-RF"].mean(),
        "Mkt-RF_sd": factors["Mkt-RF"].std(),
        "dVIX_mean": factors["dVIX"].mean(),
        "dVIX_sd": factors["dVIX"].std(),
    })

# Ayo what's good. - Written in emacs.
