import numpy as np
import pandas as pd



def regime_flag(vix: pd.Series, threshold: float = 25.0) -> pd.Series:
    """True if the prior trading day's VIX close is at or above threshold."""
    prior = vix.shift(1).dropna()
    return (prior >= threshold).rename("high")



def vix_log_returns(vix: pd.Series) -> pd.Series:
    """Daily log returns of VIX. ln(VIX_t/)"""

    return np.log(vix).diff().dropna().rename("VIX_log_returns")