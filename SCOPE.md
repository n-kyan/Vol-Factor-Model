---
title: Scope
marimo-version: 0.25.1
---

```python {.marimo}
import marimo as mo
```

# Volatility Factor Model — Project Scope 🤖

**Status:** Locked v1.0  **Date:** 2026-10-15  **Time budget:** 10 hours (hard cap)

> **The contract:** This document defines the whole project. It changes only through the Amendment Log (§12), and amendments may only **cut** scope or **fix a correctness error**, never add. New ideas go to the Parking Lot (§9), not into code.
>
> Exception: the "Actual" time column (§7) and checkboxes (§8) are meant to be filled in as I work.

---

## 1. Why this project exists

**Context.** I interviewed for *Quantitative Analyst – Equities* (US equity team) at Janus Henderson, after an earlier networking call with the interviewer. The interview exposed gaps in linear algebra, factor models, and statistical estimators.

**Goals**
1. Learn those topics by building something real.
2. Make the project and the directory presentable and post on GitHub showing I closed the gap on my own.
3. Have a portfolio piece for quant roles. The volatility focus also suits derivatives- and prop-oriented roles.

**Non-goal:** a tradable strategy. This is a model and risk study, not alpha research.

### Interviewer's technical topics → where they are covered

| Topic from interview notes | Phase |
|---|---|
| Cross-sectional regression | P5 |
| Normalization of a factor | P5 (cross-sectional z-score); P1 (units) |
| Covariance matrix | P6 |
| Covariance can shift | P6 (regimes), P7 (H4) |
| Optimization | P7 |
| Linear independence | P3 |
| Invert to minimize | P2 (OLS normal equations), P7 (min-variance) |
| Monte Carlo / bootstrapping | P6 |
| Confidence intervals & distributions, not just point estimates | P4 (SEs), P5 (FM t-stats), P6 (bootstrap CIs) |
| Gram-Schmidt projection | P3 |

### What he said makes a strong junior → how the project shows it

| Trait | Evidence in project |
|---|---|
| Quality execution, operations focus | Unit tests, one-command reproducible run, clean modular code |
| Self-driven learning | The project itself, plus the decision log (§3) |
| Makes people's lives easier, owns things | Reusable functions, clear README, documented limitations |

---

## 2. Research question & hypotheses

**Question:** How do US industries respond to market *fear* (volatility shocks) beyond what market direction explains, and how does the risk structure of an industry portfolio change when volatility is high?

| # | Hypothesis | Test statistic |
|---|---|---|
| H1 | Industries' exposures to fear shocks (`β_VIX⊥`) differ from zero, and from each other, in both directions | Full-sample OLS t-stats and 95% CIs per industry (P4) |
| H2 | The average daily cross-sectional return to fear exposure (`γ_VIX`) differs between high- and low-VIX regimes | Fama-MacBeth means and t-stats by regime, plus a difference test (P5) |
| H3 | In high-VIX regimes, average pairwise industry correlation is higher and diversification is weaker (higher portfolio vol and VaR) | Bootstrap 95% CIs for high − low differences (P6) |
| H4 | Min-variance weights estimated in calm periods underestimate realized risk in high-VIX periods | Predicted vs. realized portfolio vol (P7) |

Results are reported as found. A null or "not supported" result is a valid result and is **not** a reason to add scope.

---

## 3. Key decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Model type | **Two-stage Fama-MacBeth.** Stage 1: time-series regression per industry gives betas. Stage 2: cross-sectional regression per day gives factor returns. | Answers both "time-series or cross-sectional?" questions; Stage 2 is the cross-sectional regression the interviewer described. |
| ~~Superseded~~ | ~~One cross-sectional regression of sector returns on RV and VRP~~ | Impossible: market-wide factors take the same value for every asset on a given date, so the column is collinear with the intercept and `XᵀX` is singular. |
| Assets | Ken French **49 Industry Portfolios, value-weighted, daily** | A 49-asset cross-section (vs. 11 sector ETFs), long history, free. Some sector ETFs (XLRE, XLC) are too recent. |
| Sample | 2000-01-03 → last date in the French file | Covers the dot-com crash, 2008, 2020, and 2022. |
| Levels vs. changes | **Daily excess returns and daily changes in vol** | Levels are persistent and trending, which produces spurious regressions; prices and VIX are also in different units. |
| Units | Everything in decimals. French data: % ÷ 100. VIX ÷ 100. RV annualized. | So that factors are comparable and VRP is a valid subtraction. |
| Realized vol | `RV_t = std(last 20 daily SPY log returns) × √252` | Matches VIX's annualized units. VIX is on the S&P 500, so SPY is the right underlying. |
| Factors | `Mkt-RF` (French), `ΔVIX`, `ΔRV` | VRP is captured implicitly (next row). |
| VRP | **Not a separate regressor** | `VRP = VIX − RV`, so `ΔVRP = ΔVIX − ΔRV`, a perfect linear combination; including all three makes `XᵀX` singular. The VRP information is fully contained in ΔVIX and ΔRV. Demonstrated in P3. |
| Orthogonalization | Gram-Schmidt on demeaned series, in the order **Mkt → ΔVIX → ΔRV** | `ΔVIX⊥` = fear change not explained by the market move. `ΔRV⊥` = realized turbulence not explained by the market or implied vol. Demeaning makes orthogonal mean uncorrelated. |
| Beta estimation | Trailing **252 trading days**, re-estimated at each **month-end**, held through the next month | No look-ahead; standard practice; cheap to compute. |
| Stage 2 normalization | Each date: winsorize betas at ±3 SD, then **z-score across industries** | This is "normalization of a factor." Coefficients then read as return per 1-SD of exposure. |
| Regimes | **High VIX if the prior day's VIX close ≥ 25**, else low | Using the prior day avoids look-ahead; a fixed threshold means no tuning. |
| Estimator | OLS hand-coded in NumPy via normal equations | Learning goal. statsmodels is used **only** in tests, for verification. |
| Inference | Classic OLS SEs; FM t = `mean(γ) / (sd(γ)/√T)`; bootstrap with 1,000 iid resamples within regime and 95% percentile CIs | Simple and correct for this scope. Refinements go to the Parking Lot. |
| Optimization | **Global min-variance (primary)**, unconstrained: `w = Σ⁻¹1 / (1ᵀΣ⁻¹1)`. **Max-Sharpe (tangency)** `w ∝ Σ⁻¹μ` computed once, as a comparison only. | Expected returns are very noisy and covariance is more reliable. The tangency comparison shows estimation-error sensitivity. |
| Risk measures | Annualized portfolio vol; 1-day 95% VaR, parametric (`1.645σ`) and historical (5th percentile) | Covers both model-based and empirical risk. |
| Tools | Python, NumPy, pandas, matplotlib, yfinance, requests, pytest, statsmodels (tests only) | No scikit-learn, no scipy.optimize, so the math stays visible. |

---

## 4. Data specification

| Source | Series | Notes |
|---|---|---|
| Ken French Data Library: *49 Industry Portfolios [Daily]* | Value-weighted section | In %; missing values coded −99.99 / −999 |
| Ken French Data Library: *Fama/French 3 Factors [Daily]* | `Mkt-RF`, `RF` | In % |
| yfinance `^VIX` | Close | Annualized %, ÷ 100 |
| yfinance `SPY` | Adjusted close | Used only to compute RV |

**Processing pipeline**
1. Download and cache raw files in `data/raw/` (gitignored).
2. Parse the French CSVs: % → decimals; missing codes → NaN.
3. Industry excess return = industry return − RF.
4. Inner-join all series on date.
5. Drop any industry with NaNs in the sample, and log which ones.
6. Compute RV, ΔVIX, ΔRV (and ΔVRP, for P3 only); drop warm-up rows.
7. Save `data/processed/panel.csv`.

**Known caveats (for the write-up):** VIX closes at 4:15 pm ET, stocks at 4:00 pm. The French data lags a few months. SPY ≠ the total-market factor (fine for this use). VRP here uses *backward-looking* RV as a real-time proxy, while textbook VRP compares against *forward* realized vol.

---

## 5. Phase-by-phase execution plan

Each phase lists: tasks → deliverable → done when → fallback if over budget.

### Phase 1 — Setup, data & factors (2.0h)
- Repo skeleton, `requirements.txt`, `.gitignore`.
- `src/data.py`: download, parse, align (§4).
- `src/factors.py`: RV, ΔVIX, ΔRV, ΔVRP.
- **Deliverable:** clean panel, plus a summary table (date range, N days, N industries, factor means and SDs).
- **Done when:** one function returns an aligned DataFrame with no NaNs. Sanity ranges: VIX and RV roughly 0.09–0.85; daily industry returns mostly within ±10%.
- **Fallback:** if French parsing takes more than 45 minutes, download the CSVs by hand into `data/raw/`.

### Phase 2 — OLS estimator from scratch (1.0h)
- `src/ols.py`: `ols(X, Y)` returns `β`, SEs, t-stats, R², residuals.
  - `B = (XᵀX)⁻¹XᵀY`, which works for one asset or a matrix of assets at once
  - `σ² = eᵀe / (n − k)`; `SE = sqrt(diag(σ²(XᵀX)⁻¹))`
- `tests/test_ols.py`: (a) recover known β from simulated data; (b) match statsmodels to 1e-8.
- **Learning notes (short markdown):** derive the normal equations by minimizing `‖y − Xβ‖²` (this is "invert to minimize"); explain why `inv` is fine here but `solve`/`lstsq` is numerically preferred.
- **Done when:** tests pass.
- **Fallback:** drop multi-column Y support.

### Phase 3 — Linear independence & Gram-Schmidt (1.0h)
- Correlation matrix of `[Mkt, ΔVIX, ΔRV, ΔVRP]`.
- Rank and condition number of `XᵀX` for three cases: all four factors (rank 3, singular), three raw factors, three orthogonalized factors.
- `src/linalg.py`: `gram_schmidt(columns, normalize=False)`, modified Gram-Schmidt, which by default returns residual vectors in their original units.
- `tests/test_linalg.py`: output columns are mutually orthogonal; they match `np.linalg.qr` up to sign and scale; **`ΔVIX⊥` equals the residual of `ols(ΔVIX ~ 1 + Mkt)`**.
- **Deliverable:** Figure 1, correlation heatmaps before and after orthogonalization; a short note: *projection = regression*.
- **Done when:** tests pass and the condition-number drop is reported.

### Phase 4 — Stage 1: time-series betas (1.0h)
- Full-sample regression for each industry:
  `r_i − rf = α + β_M·Mkt + β_V·ΔVIX⊥ + β_R·ΔRV⊥ + ε`
- Output a table and **Figure 2:** `β_V` by industry with 95% CIs, sorted. → **Tests H1.**
- Rolling: at each month-end, fit on the trailing 252 days → a beta panel (months × industries × 3).
- **Figure 3:** rolling `β_V` for the 2 most positive and 2 most negative industries from Figure 2.
- **Done when:** the beta panel is saved, and an assertion checks that each estimation window ends before the month its betas are applied to.
- **Fallback:** re-estimate quarterly instead of monthly.

### Phase 5 — Stage 2: cross-sectional regressions & Fama-MacBeth (1.25h)
- For each day t:
  - `X_t = [1, z(β_M), z(β_V), z(β_R)]`, using the betas from the latest month-end before t, winsorized then z-scored.
  - `Y_t` = the 49 industry excess returns.
  - `γ_t = ols(X_t, Y_t)`, reusing P2.
- Fama-MacBeth mean, SE, and t-stat for each γ: over the full sample, the high-VIX regime, and the low-VIX regime. Difference test: `(mean_H − mean_L) / sqrt(SE_H² + SE_L²)`. → **Tests H2.**
- **Sanity check:** `corr(γ_V,t, ΔVIX⊥_t)` should be clearly positive (on days fear rises, high-`β_V` industries should outperform). If it isn't, find out why before moving on.
- **Figure 4:** distribution of `γ_V` by regime, plus cumulative `γ_V`.
- **Done when:** the FM table is produced and the sanity check passes (or its failure is explained).

### Phase 6 — Covariance, regimes, risk & bootstrap (1.25h)
- `Σ_high`, `Σ_low`: sample covariance of industry excess returns within each regime.
- Per regime: average pairwise correlation; for the equal-weight portfolio, annualized vol and 1-day 95% VaR (parametric and historical).
- `src/bootstrap.py`: 1,000 resamples of days, with replacement, within each regime, fixed seed. Compute 95% CIs for each statistic and for each high − low difference. → **Tests H3.**
- **Figure 5:** correlation heatmaps, high vs. low, with industries in the same order.
- **Figure 6:** bootstrap distributions of the high − low difference in average correlation and in VaR.
- **Done when:** a table shows the point estimate and CI for each statistic in each regime.
- **Fallback:** 500 resamples.

### Phase 7 — Optimization (0.75h)
- Derive the min-variance weights with a Lagrangian (short markdown). Implement with `np.linalg.solve`.
- Compute `w_low`, `w_high`, `w_full`.
- **H4:** compare the predicted vol of `w_low` (`wᵀΣ_low w`) with its realized vol in the high regime (`wᵀΣ_high w`); include equal-weight for reference.
- Compute tangency weights on the full sample; report gross leverage `Σ|w|` and the largest weights, plus one paragraph on why they are unstable.
- **Figure 7:** min-variance weights, low vs. high regime.
- **Done when:** the predicted vs. realized vol table exists.
- **Documented limitations:** in-sample, unconstrained (shorts allowed), no transaction costs.

### Phase 8 — Write-up & ship (1.75h)
- `scripts/run_all.py` regenerates every figure and table in one command.
- README:
  1. Question
  2. Hypotheses
  3. Data
  4. Method
  5. Results (Figures 1–7)
  6. Conclusion per hypothesis: supported or not, with the numbers
  7. Limitations
  8. Next steps (taken from the Parking Lot)
- Code pass: docstrings, remove dead code, `pytest` green.
- Spelling and grammar check on all docs.
- Push to GitHub.
- Email the interviewer, about 5 sentences: thanks; the interview showed me a gap; what I built; 2–3 findings; ask for his feedback and mention I'm still interested in the team.

---

## 6. Repo structure

```
vol-factor-model/
├── SCOPE.md            # this file
├── README.md           # final write-up
├── requirements.txt
├── .gitignore          # data/raw/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data.py         # P1
│   ├── factors.py      # P1
│   ├── ols.py          # P2
│   ├── linalg.py       # P3
│   ├── stage1.py       # P4
│   ├── stage2.py       # P5
│   ├── risk.py         # P6
│   ├── bootstrap.py    # P6
│   └── optimize.py     # P7
├── tests/
│   ├── test_ols.py
│   └── test_linalg.py
├── scripts/
│   └── run_all.py
└── reports/figures/    # fig1–fig7
```

---

## 7. Time budget

| Phase | Budget | Actual |
|---|---|---|
| P1 Setup, data & factors | 2.0h | |
| P2 OLS from scratch | 1.0h | |
| P3 Linear independence & Gram-Schmidt | 1.0h | |
| P4 Stage 1 betas | 1.0h | |
| P5 Stage 2 & Fama-MacBeth | 1.25h | |
| P6 Covariance, risk, bootstrap | 1.25h | |
| P7 Optimization | 0.75h | |
| P8 Write-up & ship | 1.75h | |
| **Total** | **10.0h** | |

**Overrun rules**
- A phase that reaches 150% of its budget stops. Apply its fallback, or cut its lowest-priority item, and log the change in §12.
- If the total reaches 8.25h before P8 has started, go straight to P8 with whatever exists.

---

## 8. Definition of done

- [ ] `python scripts/run_all.py` regenerates all 7 figures and all tables from raw data
- [ ] `pytest` passes
- [ ] H1–H4 each have a stated conclusion backed by numbers
- [ ] README includes limitations and next steps
- [ ] No spelling or grammar errors in the docs
- [ ] Pushed to GitHub
- [ ] Email sent to the interviewer

---

## 9. Out of scope — Parking Lot

Not in this project. Add new ideas here instead of building them.

**Data / factors**
- VIX futures, VIX options implied vol, VVIX
- S&P 500 implied vol surface; sector or single-stock implied vol
- Interest rates, credit spreads, cross-market spillovers (planned as project #2)

**Statistics**
- Newey-West SEs; block or stationary bootstrap (both handle volatility clustering)
- Regime-threshold sensitivity analysis
- Forward-looking RV for "true" VRP

**Risk / covariance**
- Factor-implied covariance `BFBᵀ + D`
- EWMA covariance; Ledoit-Wolf shrinkage

**Optimization**
- Constrained / long-only optimization; risk parity
- Rolling rebalanced out-of-sample backtest; transaction costs

**Presentation**
- Dashboards, web app, ML of any kind

---

## 10. Scope rules (the contract)

1. No data sources, factors, models, figures, or statistics beyond those listed in §3–§5.
2. Ideas go to §9. Nothing else happens to them during the project.
3. Amendments (§12) may only cut scope or fix a correctness error, and each is dated and gives a reason.
4. Unremarkable results are reported honestly, not "fixed" by adding scope.
5. Done beats perfect: the 10-hour cap outranks polish.

---

## 11. Glossary

- **Factor:** an explanatory variable for returns. In time-series models it's a market-wide series (ΔVIX); in cross-sectional models it's an asset characteristic (an industry's `β_V`).
- **Beta (exposure):** the coefficient on a factor in a time-series regression.
- **Factor return (γ):** the coefficient in a cross-sectional regression, i.e. that day's payoff per unit of exposure.
- **Fama-MacBeth:** run a cross-sectional regression every period, then average the coefficients; t-stats come from the time series of those coefficients.
- **Normalization:** rescaling exposures across assets on each date (winsorize, then z-score) so they're comparable.
- **Linear independence:** no column is a combination of the others. Without it, `XᵀX` is singular.
- **Condition number:** a measure of how close a matrix is to singular. High values mean unstable estimates.
- **Gram-Schmidt:** sequentially remove each vector's projection onto the earlier vectors. Equivalent to taking OLS residuals.
- **VRP (volatility risk premium):** implied vol − realized vol, the "price of insurance."
- **Bootstrap:** resample the data with replacement to approximate the sampling distribution of a statistic.
- **Min-variance portfolio:** the weights minimizing `wᵀΣw` subject to `Σw = 1`; solved by inverting Σ.

---

## 12. Amendment log

| Date | Change | Reason (cut scope / fix correctness) | Hours affected |
|---|---|---|---|
| | | | |

---

**Committed:** I commit to no scope creep.