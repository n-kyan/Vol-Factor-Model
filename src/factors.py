import numpy as np
import pandas as pd

from ols import add_intercept, ols, residuals



# Fixed window so the threshold is the same no matter what sample start is chosen.
MEDIAN_WINDOW = ("1990-01-01", "2026-12-31")


def regime_threshold(vix: pd.Series) -> float:
    """Long-run median VIX close over MEDIAN_WINDOW."""
    return vix.loc[MEDIAN_WINDOW[0]:MEDIAN_WINDOW[1]].median()


def regime_flag(vix: pd.Series) -> pd.Series:
    """True if the prior trading day's VIX close is at or above the long-run median. Prior day because today's close is partly caused by today's shock."""
    prior = vix.shift(1).dropna()
    return (prior >= regime_threshold(vix)).rename("high")



def dvix(vix: pd.Series) -> pd.Series:
    """Daily change of VIX. Follows vega convention."""

    return vix.diff().dropna().rename("dVIX")


def vol_shock(dvix: np.ndarray, mkt: np.ndarray) -> np.ndarray:
    """Step 2 (Gram-Schmidt): the part of each day's VIX change that the market doesn't explain.

    Regress dVIX on [1, Mkt-RF] and keep the residual. Inputs and output are NumPy arrays (days,).
    """
    # Why market first: the question is about vol moves *beyond* what market moves explain,
    # so the market keeps everything it can explain and the vol shock gets only the leftover.
    X = add_intercept(mkt)
    b = ols(X, dvix)
    return residuals(X, dvix, b)
