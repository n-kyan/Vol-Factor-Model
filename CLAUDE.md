# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

@DEV_GUIDE.md

`DEV_GUIDE.md` is the source of truth for this project: context, hard rules for Claude, the final project plan, decisions made, and a corrections log. Follow every rule in it. When the user flags a new Claude mistake, add a row to its corrections log. `README.md` is a to-do list for final deliverables, not a dev doc.

The user writes all code and documentation themselves; Claude's role is mainly to teach just-in-time and review. Don't write project code unless asked. Anything AI-sourced that ends up in the notebook is marked with 🤖.

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
- `src/factors.py`: factor and regime construction.
  - `dvix`: daily VIX change in points.
  - `regime_flag`: True when the prior day's VIX close is at or above `regime_threshold`, the median over the fixed `MEDIAN_WINDOW` (1990–2026). The window doesn't depend on `build_data(start=...)`, so tests over different spans stay comparable. Compute it on the full Cboe series, never on the sliced panel.
