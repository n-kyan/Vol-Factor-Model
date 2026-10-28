# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

@DEV_GUIDE.md

`DEV_GUIDE.md` is the source of truth for this project: context, hard rules for Claude, the final project plan, decisions made, and a corrections log. Follow every rule in it. When the user flags a new Claude mistake, add a row to its corrections log. `README.md` is a to-do list for final deliverables, not a dev doc.

Follow the **Working method** in `DEV_GUIDE.md`. In the build phase Claude implements Steps 2–7 and explains the question, the results and the why behind every design decision in `src/workbench.py`, pausing at three checkpoints. In the understanding phase the user rebuilds `src/main.py` in their own words. `src/regression.py` and `src/math_nb.py` are the user's own learning code: don't edit them. Anything AI-sourced in the user's notebooks is marked with 🤖.

## What this is

A solo risk-modeling research project (not trading): how do the Ken French 49 value-weighted industry portfolios react to VIX shocks orthogonal to the market (Gram-Schmidt residual of ΔVIX on Mkt-RF), and does that differ between high- and low-VIX regimes? It is its own git repo, nested inside the Nexus C++ app, and unrelated to that build.

## Commands

Python 3.11, managed with uv.

```bash
uv sync                          # install deps
uv run marimo edit src/main.py   # develop the notebook
uv run marimo run src/main.py    # view it as an app
```

There are no tests, lint config, or CI.

## Architecture

- `src/main.py` is the **only** marimo notebook and the final deliverable. It imports plain functions from the sibling modules (`from data import ...`), so those modules must stay plain Python files, not marimo notebooks.
- `src/data.py`: data loading and panel assembly.
  - `download_raw_data` caches downloads in `data/raw/` (gitignored) and never re-downloads if the file exists. Delete the file to refresh it.
  - `load_kfrench_table` parses Ken French's multi-section CSV zips by locating the `section_title` row and then the next run of `YYYYMMDD` rows. It converts percent to decimal and -99.99/-999 to NaN.
  - `build_data` inner-joins industries, FF factors, and Cboe VIX, subtracts RF from the industries, and drops any industry with NaNs. It returns a tuple `(industry_excess_returns, factors)`, even though its annotation says `DataFrame`.
- `src/ols.py`: hand-written `solve` (Gaussian elimination) and `ols`. `y` can be (days, 49) to fit all industries in one call.
- `src/estimation.py`: `fit_industries` (Step 3) and `fit_by_regime` (Step 4). The vol shock is rebuilt inside each fit, so a regime fit uses that regime's own Gram-Schmidt residual. Step 6: `bootstrap_gammas` (moving-block bootstrap of the full timeline, 20-day blocks, 1,000 resamples) and `se_interval` (estimate ± 1.96 × bootstrap SE).
- `src/validation.py`: Step 5 Monte Carlo. Real factors and regime days, fake industries with known β/γ (γ_high = 1.5 × γ_low) and regime-sized noise; `monte_carlo_coverage` runs the full pipeline 100 times (~1 min).
- `src/hypotheses.py`: Step 7. `evaluate_h1` (2×2 significance table), `evaluate_h2` (two pre-set counts with binomial chance baselines), `significance_counts` (descriptive context for H2).
- `src/workbench.py`: Claude's build notebook. `src/lessons.py`: Lesson 1 (OLS), kept as reference.
- `src/factors.py`: factor and regime construction.
  - `dvix`: daily VIX change in points.
  - `vol_shock`: Step 2, the residual of ΔVIX regressed on `[1, Mkt-RF]`.
  - `regime_flag`: True when the prior day's VIX close is at or above `regime_threshold`, the median over the fixed `MEDIAN_WINDOW` (1990–2026). The window doesn't depend on `build_data(start=...)`, so tests over different spans stay comparable. Compute it on the full Cboe series, never on the sliced panel.
