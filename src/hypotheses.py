"""Step 7: evaluate H1 and H2 against the chance baseline."""

from math import comb

import numpy as np
import pandas as pd

CHANCE_RATE = 0.05   # a 95% interval excludes 0 by luck 5% of the time when the truth is 0


def binomial_tail(k: int, n: int, p: float) -> float:
    """Chance of k or more 'hits' out of n independent tries, each a hit with probability p."""
    total = 0.0
    for j in range(k, n + 1):
        total += comb(n, j) * p**j * (1 - p) ** (n - j)
    return total


def significant(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """True where the interval excludes 0."""
    return (lower > 0) | (upper < 0)


def evaluate_h1(g_low, g_high, sig_low, sig_high, industries):
    """H1: same gamma sign in both regimes, judged only where gamma is significant in BOTH.

    Returns (table_2x2, qualifying), where qualifying lists the industries significant in both
    regimes with their signs.
    """
    table = pd.DataFrame(
        {
            "significant at high VIX": [int((sig_low & sig_high).sum()), int((~sig_low & sig_high).sum())],
            "not significant at high VIX": [int((sig_low & ~sig_high).sum()), int((~sig_low & ~sig_high).sum())],
        },
        index=["significant at low VIX", "not significant at low VIX"],
    )
    both = sig_low & sig_high
    qualifying = pd.DataFrame(
        {"sign low": np.sign(g_low[both]), "sign high": np.sign(g_high[both])},
        index=np.array(industries)[both],
    )
    qualifying["same sign"] = qualifying["sign low"] == qualifying["sign high"]
    return table, qualifying


def evaluate_h2(g_low, g_high, absdiff_lower, absdiff_upper):
    """H2: |gamma| bigger at high VIX. Two pre-set measures across all industries.

    1. Count of |gamma_high| > |gamma_low|, vs. half expected if regimes didn't differ.
    2. Count of |high| - |low| intervals excluding 0, split by direction.
    Returns a dict of counts and chance probabilities.
    """
    n = len(g_low)
    bigger = int((np.abs(g_high) > np.abs(g_low)).sum())
    sig = significant(absdiff_lower, absdiff_upper)
    sig_bigger = int((sig & (absdiff_lower > 0)).sum())
    sig_smaller = int((sig & (absdiff_upper < 0)).sum())
    # Why p = 0.5 for the count: if regimes didn't differ, each industry is a coin flip
    # for which regime has the bigger |gamma|.
    # Why p = 0.025 for the significant-bigger count: with no real difference, a 95% interval
    # lands entirely above 0 by luck 2.5% of the time (the other 2.5% is entirely below).
    return {
        "industries": n,
        "|high| > |low|": bigger,
        "expected by chance": n / 2,
        "chance of this many or more": binomial_tail(bigger, n, 0.5),
        "interval says bigger at high VIX": sig_bigger,
        "interval says smaller at high VIX": sig_smaller,
        "expected by chance (each direction)": n * CHANCE_RATE / 2,
        "chance of this many bigger or more": binomial_tail(sig_bigger, n, CHANCE_RATE / 2),
    }


def significance_counts(sig_low, sig_high):
    """Supporting context for H2 (descriptive, not a pre-set test): significant gammas per regime."""
    n = len(sig_low)
    return pd.DataFrame(
        {
            "significant gammas": [int(sig_low.sum()), int(sig_high.sum())],
            "expected by chance": [n * CHANCE_RATE, n * CHANCE_RATE],
            "chance of this many or more": [
                binomial_tail(int(sig_low.sum()), n, CHANCE_RATE),
                binomial_tail(int(sig_high.sum()), n, CHANCE_RATE),
            ],
        },
        index=["low-VIX days", "high-VIX days"],
    )
