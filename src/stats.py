"""Descriptive statistics shown in the notebooks: size of daily moves and extreme-day counts."""

from math import erfc, sqrt

import pandas as pd


def describe_moves(x: pd.Series) -> dict:
    """Size of daily moves: SD, share of days within 1 and 2 SDs, median size, largest."""
    sd = x.std()
    return {
        "standard deviation (SD)": sd,
        "days moving less than 1 SD": (x.abs() <= sd).mean(),
        "days moving less than 2 SD": (x.abs() <= 2 * sd).mean(),
        "median day: size of move": x.abs().median(),
        "largest day: size of move": x.abs().max(),
    }


def tail_counts(x: pd.Series, cutoffs) -> pd.DataFrame:
    """Days more than k SDs from the mean: actual count vs. what a normal distribution expects."""
    z = (x - x.mean()) / x.std()
    rows = {}
    for k in cutoffs:
        rows[f"beyond ±{k} SD"] = {
            # Why erfc: for a normal distribution, the share of days beyond ±k SDs is erfc(k / √2)
            "normal distribution expects": len(x) * erfc(k / sqrt(2)),
            "actual days": int((z.abs() > k).sum()),
        }
    return pd.DataFrame(rows).T


def gamma_summary(table: pd.DataFrame, names) -> pd.DataFrame:
    """Readable per-industry table from estimation.regime_interval_table.

    Per statistic: estimate [95% interval] with * when significant, then its t-stat (bootstrap SE).
    """
    out = pd.DataFrame({"industry": names, "code": table.index})
    for stat in ["gamma low", "gamma high", "high - low"]:
        cells = []
        for code in table.index:
            row = table.loc[code]
            star = " *" if row[f"{stat} sig"] else ""
            cells.append(f"{row[stat]:+.3f} [{row[f'{stat} lo']:+.3f}, {row[f'{stat} hi']:+.3f}]{star}")
        label = stat.replace("gamma", "γ")
        out[f"{label} (%/pt)"] = cells
        out[f"{label} t"] = table[f"{stat} t"].round(2).to_numpy()
    return out.reset_index(drop=True)
