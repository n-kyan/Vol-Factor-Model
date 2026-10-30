# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

@DEV_GUIDE.md

`DEV_GUIDE.md` is the source of truth for this project: context, hard rules for Claude, the final project plan, decisions made, and a corrections log. Follow every rule in it. When the user flags a new Claude mistake, add a row to its corrections log. `README.md` is a to-do list for final deliverables, not a dev doc.

Follow the **Working method** in `DEV_GUIDE.md`. `src/main.py` is the short, bottom-line-first takeaway notebook for readers; each phase has a companion notebook `<module>_nb.py` with the full explanation. Claude writes both; the user asks questions in `notes.txt` (one section per phase) and rewrites the markdown in his own voice. Text not yet rewritten stays marked 🤖. Every `mo.md` is one `r"""` / `rf"""` string. `src/regression.py` and `src/math_nb.py` are the user's own learning code: don't edit them.

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

- **Notebooks** (marimo): `src/main.py` (the deliverable, takeaways only) and the companions `data_nb.py`, `factors_nb.py`, `ols_nb.py`, `estimation_nb.py`, `validation_nb.py`, `hypotheses_nb.py`. They import plain functions from the sibling modules (`from data import ...`), so the modules must stay plain Python files. `main.py` caches the Monte Carlo run with `mo.persistent_cache` in `src/__marimo__/cache/`; delete that folder after changing `validation.py` or `estimation.py`.
- `src/charts.py`: shared palette and chart helpers. `src/stats.py`: describe_moves, tail_counts.
- `src/data.py`: data loading and panel assembly.
  - `download_raw_data` caches downloads in `data/raw/` (gitignored) and never re-downloads if the file exists. Delete the file to refresh it.
  - `load_kfrench_table` parses Ken French's multi-section CSV zips by locating the `section_title` row and then the next run of `YYYYMMDD` rows. It converts percent to decimal and -99.99/-999 to NaN.
  - `build_data` inner-joins industries, FF factors, and Cboe VIX, subtracts RF from the industries, and drops any industry with NaNs. It returns a tuple `(industry_excess_returns, factors)`, even though its annotation says `DataFrame`.
- `src/ols.py`: hand-written `solve` (Gaussian elimination) and `ols`. `y` can be (days, 49) to fit all industries in one call.
- `src/estimation.py`: `fit_industries` (Step 3) and `fit_by_regime` (Step 4). The vol shock is rebuilt inside each fit, so a regime fit uses that regime's own Gram-Schmidt residual. Step 6: `bootstrap_gammas` (moving-block bootstrap of the full timeline, 20-day blocks, 1,000 resamples), `se_interval` (estimate ± 1.96 × bootstrap SE), and `regime_interval_table` (estimates, intervals and significance for every statistic in one table).
- `src/validation.py`: Step 5 Monte Carlo. Real factors and regime days, fake industries with known β/γ (γ_high = 1.5 × γ_low) and regime-sized noise; `monte_carlo_coverage` runs the full pipeline 100 times (~1 min).
- `src/hypotheses.py`: Step 7. `evaluate_h1` (2×2 significance table), `evaluate_h2` (two pre-set counts with binomial chance baselines), `significance_counts` (descriptive context for H2).
- `src/workbench.py`: Claude's build-phase notebook, superseded by the companion notebooks.
- `src/factors.py`: factor and regime construction.
  - `dvix`: daily VIX change in points.
  - `vol_shock`: Step 2, the residual of ΔVIX regressed on `[1, Mkt-RF]`.
  - `regime_flag`: True when the prior day's VIX close is at or above `regime_threshold`, the median over the fixed `MEDIAN_WINDOW` (1990–2026). The window doesn't depend on `build_data(start=...)`, so tests over different spans stay comparable. Compute it on the full Cboe series, never on the sliced panel.
