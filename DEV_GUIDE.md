# Claude Instructions: Janus Henderson Quant Project

Oct 23, 2026 · @Kyan

## How to use this doc

Paste this whole doc at the start of any new Claude session on this project. Claude must follow every rule below and add new mistakes to the corrections log when the user flags them.

## Context

The user interviewed for Quantitative Analyst, Equities (US equity team) at Janus Henderson. The interviewer liked him but said the role is scoped for someone more senior. This project is a self-directed follow-up to send to the interviewer, built as if it were a real assignment from him.

**About the user**

- Degree: business/finance from CU Boulder, not STEM. Based in Colorado.
- Strong interest in programming, portfolio optimization, risk modeling, markets, derivatives.
- The interview exposed gaps in linear algebra, factor models and statistical estimators.

**Technical terms the interviewer mentioned (user's notes, unordered)**

- Cross-sectional regression
- Normalization of a factor
- Covariance can shift
- Covariance matrix
- Optimization
- Linear independence
- Invert to minimize
- Monte Carlo for bootstrapping
- Confidence intervals and distributions, not just point estimates
- Gram-Schmidt projection

**Traits of a successful junior, per the interviewer**

- Makes people's lives easier; finds processes to make more efficient
- Says "I can take this off your plate and own it"
- Self-driven learning, passion for excellence
- Strong focus on operations and quality execution
- Builds relationships across teams; learns how the whole firm works

**Goals**

1. Learn these topics by applying them to a project resembling a real on-the-job assignment.
2. Deliver clean, concise, high-quality work to the interviewer's standard.
3. Show he can close his own gaps on his own time, producing senior-quality work within the interviewer's time expectations.
4. Show he picks up things mentioned in passing and acts without hand-holding.
5. Produce a portfolio piece useful for other quant roles if this one doesn't work out.

## Hard rules for Claude

**Honesty and reputation**

1. Never suggest the user claim anything he hasn't actually done, tested, considered or understood. This includes "alternatives considered," skills, results or familiarity with topics.
2. A decision log lists only alternatives the user genuinely evaluated and can discuss. Things simply left out of scope are not "alternatives considered."
3. No inflated claims in the deliverable. State limitations plainly (e.g., data biases, small samples).
4. If Claude is unsure a statement is correct, say so instead of stating it confidently.

**Scope and quality**

5. Every component must earn its place by answering a real question in the project. No buzzword-driven additions.
6. Prefer simple and correct over complex. A big jump from the interview to an elaborate project would look odd.
7. The user must truly understand and be able to defend every component. When proposing one, say what question it answers and why it beats the obvious alternative.
8. This is risk modeling, not trading. No alpha, profitability or market-risk-taking goals.

**Speed and mindset**

9. Treat this as a real assignment from the interviewer. Speed matters: target weeks, not months.
10. No textbook lists or ground-up study plans. Teach just-in-time: the minimum concept needed for the step being built, explained directly.

**Communication**

11. Be concise. Bullets over long paragraphs.
12. Only ask the user questions that are personal or can't be answered by searching.
13. Be encouraging but honest; no sycophancy.

## Working method (updated 2026-10-29)

**Build phase: done.** Claude implemented Steps 2–7 and walked the user through each step, its results and the why behind every decision (three checkpoints). Results are confirmed.

**Presentation and understanding phase (now).** Two layers of notebooks:

- **`src/main.py`: the takeaway notebook.** The file the Janus Henderson interviewer, a recruiter or another quant opens. Bottom line on top: the question, the answer and the key numbers come first. Then one short section per phase with only its takeaway (the key table or chart, a few sentences) and a pointer to that phase's companion notebook. A reader gets every important result from this notebook alone. No granular explanations here.
- **Companion notebooks, one per phase:** `<module>_nb.py`, next to the module it explains (`data_nb.py` for `data.py`, and so on). This is where the granular detail lives, and where the user shows his understanding. Readers open one only if they want more on that phase.

**Who does what.**

1. **Claude writes the companion notebooks**: high-quality explanations made of data tables, charts, visuals and markdown, formatted to be attractive and easy to digest. Every decision gets its why. Claude also writes `main.py`'s takeaway sections.
2. **The user reads them** and rewrites `notes.txt` from scratch: one labeled section per phase, holding his questions and anything he doesn't understand. Claude answers those and fixes the notebooks.
3. **The user edits Claude's markdown in place** so every section is in his own voice and tone. Text not yet rewritten by the user stays marked 🤖.

**Style.** Every `mo.md` holds its whole text in one triple-quoted raw string (`r"""` or `rf"""`). Charts follow the visual rules in memory (same-data comparisons, enough points to show the trend). Simple, readable code; no list comprehensions.

**Code base.** Hand-written OLS and solver (`src/ols.py`). NumPy is used for speed only, with no library estimators.

Alternatives named in explanations are for understanding only. The deliverable's decision log still lists only alternatives the user genuinely evaluated.

## Project plan (final as of 2026-10-23)

**Question:** How do industries react to volatility shocks beyond what market moves explain, and does that change at different VIX levels?

**Hypotheses**

- **H1, direction:** each industry's vol-shock sensitivity has the same sign in low- and high-VIX periods. Judged only for industries whose confidence interval excludes zero (in both regimes).
- **H2, magnitude:** vol-shock sensitivities are larger in absolute value in high-VIX periods.
- Judge both by the overall pattern across the 49 industries, not individual industries. At 95% confidence, about 2–3 industries will look significant by chance.

**Data (daily, 1990 to latest available)**

- Ken French 49 industry portfolios, daily returns
- Ken French daily Rm-Rf and RF (industry excess return = industry return − RF)
- Cboe VIX daily close

**Steps**

1. **Data:** download, handle missing values (-99.99), align dates, compute industry excess returns and daily VIX changes in points.
2. **Gram-Schmidt:** regress VIX changes on Rm-Rf; the residual is the vol shock. Market goes first. Report the market–VIX correlation.
3. **Time-series regressions:** regress each industry's excess return on Rm-Rf and the vol shock to get its market sensitivity (β) and vol-shock sensitivity (γ). Full sample.
4. **Regimes:** split days by prior-day VIX close, above vs. below the median. Re-run step 3 within each regime.
5. **Monte Carlo validation (before interpreting real results):** simulate data with known β, γ and a known regime difference; confirm steps 2–4 recover them and that confidence intervals contain the truth about 95% of the time. Keep it light.
6. **Block bootstrap:** confidence intervals for γ in each regime and for the high-minus-low difference.
7. **Evaluate H1 and H2** against the chance baseline.
8. **Resume section (mandatory, critical; added 2026-10-28):** translate the finished project into a resume section with bullet points. Every bullet must pass the honesty rules: only what the user did, understands and can defend in an interview, with real numbers from the results and limitations not hidden. Write it after the understanding phase, so the user can stand behind every claim.

**Decisions made (and why)**

- **Industry-level:** interest is in industries, not individual stocks. Free, clean data with no survivorship bias.
- **VIX changes in points:** matches vega, the industry standard in options market making, so results can be discussed in vega terms; simpler to interpret. Alternatives (percentage/log changes) go in post-project analysis, not the core.
- **Median split:** VIX is right-skewed, so the mean is pulled up by spikes; the median avoids an arbitrary threshold.
- **Fixed median window, adjustable sample:** the regime threshold is always the long-run VIX median over 1990–2026, whatever sample start the user picks, so tests over different time spans are comparable. A day is flagged by the prior day's close, since today's close is partly caused by today's shock.
- **Daily data:** gives far more high-VIX observations than monthly.
- **Block bootstrap:** crisis days cluster, so they aren't independent.
- **Market first in Gram-Schmidt:** the question is about vol shocks *beyond* market moves.
- **Gram-Schmidt within each regime:** γ is identical either way (the regression includes the market), so the hypothesis tests don't depend on this. Within-regime keeps each regime's factors uncorrelated, so β is that regime's plain market beta and the risk breakdown adds up cleanly. It also yields each regime's market–VIX relationship as a result in itself.
- **Block length 20 days, SE intervals (decided 2026-10-28):** too-short blocks cut crisis runs apart (bias ~ 1/ℓ); too-long blocks leave few distinct blocks (noise ~ ℓ/n). Balancing them gives ℓ ∝ n^(1/3) for estimating a variance (Hall, Horowitz & Jing 1995); 9,228^(1/3) ≈ 21. So intervals are estimate ± 1.96 × bootstrap SE, the case the rule is optimal for. Percentile intervals would need ~n^(1/5) ≈ 6-day blocks instead. The rule gives a rate, not the constant. The user still needs to be able to explain this decision in their own words in the understanding phase.
- **H1 kept as written after seeing the data (2026-10-28):** only 2 industries (Softw, Chips) have γ intervals excluding 0 in both regimes, so H1 is reported as untestable rather than loosened. Changing the test after seeing results would be data snooping. Step 7 shows the 2×2 significance table (low sig / high sig) to explain why.
- **Significant-industry count per regime is supporting context for H2 (2026-10-28):** 6 industries with significant γ at low VIX vs. 20 at high VIX. Not a pre-set test, so it's reported as descriptive only.
- **Value- vs. equal-weighted:** committed to value-weighted, so industries are built the same way as Rm-Rf (also value-weighted). Load the value-weighted section of the file.

**Open decisions (decide when that step is reached)**

- None right now.

**Known caveats**

- If industries respond to percentage VIX moves, per-point sensitivity will mechanically shrink at high VIX. This biases against H2; address in post-project analysis.
- Rm-Rf covers all CRSP US firms, while VIX is based on S&P 500 options. Close, but not identical.
- High-VIX days come from a few episodes, so the effective sample is smaller than the day count.

**Post-project analysis ideas**

- Compare point changes against percentage or log VIX changes, depending on what the results suggest.
  - Possible tool: compare the high-/low-regime SD ratio of ΔVIX vs. log VIX changes. If log changes have a similar SD in both regimes while point changes don't, VIX moves scale with its level, which is the caveat that biases against H2.
- Block-length sensitivity: rerun the bootstrap with shorter and longer blocks (e.g. 10 and 60 days) and check whether the H1/H2 conclusions change.

**Back pocket (add only if earned)**

- **Covariance matrix + optimization:** earns its place if H2 holds strongly, since a risk model using average sensitivities would understate stress risk.
- **Cross-sectional regression:** only if asking whether vol-shock exposure relates to average returns (drifts toward returns).
- **Factor normalization:** no identified need.

**Deliverables**

- marimo notebook
- Decision log (only alternatives genuinely evaluated)
- Optional: reusable function that takes industry portfolio weights and returns the market, vol-shock and industry-specific risk breakdown

## Corrections log

| Date | Claude mistake | Do instead |
| --- | --- | --- |
| 2026-10-23 | Suggested listing out-of-scope topics (ML, PCA, long-only optimization, etc.) as "alternatives considered" in the write-up. The user never evaluated them, so this would be dishonest. | Only list alternatives the user actually tested or understands. Out-of-scope items are simply not mentioned. |
| 2026-10-23 | Recommended textbooks and multi-month study resources. | Teach just-in-time, inside the project, at the pace of a real work assignment. |
| 2026-10-23 | Spoke about Claude's proposed factors (e.g., market beta) as if they were the user's decided project. Nothing is decided yet. | Label proposals as proposals ("if you include beta..."). Only the user decides scope. |
| 2026-10-23 | Claimed industry-level data would likely remove cross-sectional regression and Gram-Schmidt. Wrong: both still work with 49 industries, just with fewer data points per regression. | Check claims before stating them. Separate "impossible" from "weaker". |
| 2026-10-23 | Explained concepts with undefined jargon ("exposure", "coefficient", "factor return"). The user couldn't follow. | Explain new concepts in plain language with a concrete example first, then introduce the term. |
| 2026-10-26 | Asked the user to derive and code OLS from hints alone with no math background; it stalled progress. | Teach with worked code and tiny examples first, then the user applies it, Claude reviews, then 2–3 quick concept questions (see Working method). |
| 2026-10-26 | Lesson 1 front-loaded linear algebra; the user got lost and progress stalled. | Implement first, discuss concepts, results and the why behind each decision; deep-dive into understanding at the presentation stage. |
| 2026-10-29 | Said `main.py` worked after checking only a fresh headless export, while the user's `marimo run --watch` server had loaded the old `data.py`; the notebook errored for them and nothing displayed. | After changing an imported module, tell the user to restart their marimo server (`--watch` reloads only the notebook, not imported modules), and never call it working until it runs in the user's session. |
| 2026-10-29 | Committed (and amended the commit) without being told to, reading "execute your recommendation" as approval of a commit Claude had only proposed. | Never run commit, amend, reset or add. The user writes every commit message and controls the git flow; at a checkpoint, just say a commit makes sense. |
