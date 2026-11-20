# SEM Bidding Simulator

**Search-ads optimisation lab in Python, SQL and statistics.** It generates a realistic synthetic Google-Ads-style account, analyses it with portable analytical SQL, runs frequentist and Bayesian A/B tests, compares multi-armed bandits against a static split, and benchmarks an expected-value bidding algorithm with Bayesian shrinkage against classic rule-based bidding in a walk-forward auction simulation.

> **All data in this repository are synthetic.** They come from a simulated world with known ground truth, which is what makes the methods *measurable* (we can compare every estimate against the truth). The numbers below show what the methods do on that world. They are not results from a real client account.

```
sem-bidding-simulator/
├── app.py                     Streamlit dashboard (4 tabs)
├── src/
│   ├── data_generator.py      SEM world model, GSP auction, DuckDB + SQLite writer
│   ├── sql_runner.py          runs sql/*.sql on DuckDB or SQLite
│   ├── stats_engine.py        z-test, power, sample size, SRM, Beta-Binomial, bandits
│   ├── bidding_optimizer.py   EV bidding + empirical Bayes, rules, simulation, bid landscape
│   ├── pipeline.py            end-to-end sample run -> outputs/
│   ├── report.py              self-contained HTML report (no Streamlit needed)
│   └── charts.py              Plotly figures shared by app and report
├── sql/                       7 commented analytical queries (CTEs + window functions)
├── tests/                     56 pytest tests (statistics, SQL parity, simulator, app)
├── outputs/                   summary.json, report.html, query results (sample run)
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_generator --sqlite data/sem.sqlite    # ~3 s: 406k hourly rows -> DuckDB (+ SQLite copy)
python -m src.pipeline --report outputs/report.html      # ~8 s: every module, writes outputs/
python -m pytest                                         # 56 tests, ~10 s
streamlit run app.py                                     # interactive dashboard
```

`make all` does the same with the Makefile. No Streamlit? `outputs/report.html` contains the same charts as a single offline HTML file.

**Docker:** `docker build -t sem-bidding-simulator . && docker run --rm -p 127.0.0.1:8501:8501 sem-bidding-simulator` (the DuckDB data is generated at build time; healthcheck on `/_stcore/health`). An interactive live demo of this image is available on request.

---

## 1. Synthetic data: the world model (`src/data_generator.py`)

5 campaigns (Brand, Generic Laptops, Generic Phones, Accessories, Competitor), 47 keywords across exact / phrase / broad match, 179 search terms, 120 days × 24 hours × 3 devices. The output is a star schema in **DuckDB** with an identical **SQLite** copy:

| table | grain | rows |
|---|---|---|
| `fact_keyword_hourly` | date × hour × device × keyword | 406,080 |
| `fact_search_term_daily` | date × search term | 21,480 |
| `dim_campaign`, `dim_keyword` | dimensions | 5 / 47 |
| `truth_keyword`, `truth_search_term` | latent ground truth (used **only** for evaluation) | 47 / 179 |
| `meta` | seed and parameters to rebuild the world | 1 |

**Latent keyword parameters.** Each keyword has a true CTR (depends on Quality Score: `ctr ∝ (QS/8)^0.8`), a true CVR (campaign base × lognormal), an AOV, a heavy-tailed search volume (many low-volume keywords, a few big ones) and a competitor ad-rank level. Phrase and broad keywords expand into several search terms, including irrelevant "junk" queries (`… repair`, `free …`, `… jobs`, `… manual pdf`) that click but almost never convert. That is the raw material for wasted-spend detection.

**GSP auction.** The strongest competing ad rank in each auction is $R \sim \text{Weibull}(C, a)$. With our ad rank $r = \text{bid} \times QS$ and $x = r/C$:

$$
\text{impression share } s = P(R < r) = 1 - e^{-x^{a}}, \qquad
\text{CPC} = \frac{\mathbb{E}[R \mid R < r]}{QS} + 0.01 = \frac{C\,\Gamma(1+\tfrac1a)\,P(1+\tfrac1a,\,x^a)}{s\cdot QS} + 0.01 \le \text{bid}
$$

where $P$ is the regularised lower incomplete gamma function. This gives the real mechanics of a generalised second-price auction for free:

* you never pay more than your bid, and a higher Quality Score directly lowers CPC;
* **diminishing returns:** impression share saturates, while the *marginal* CPC of the extra clicks keeps rising, because you start winning auctions against stronger rivals;
* the CTR gets a position multiplier `0.55 + 0.45·s`.

**Seasonality and noise.** Hour-of-day (two peaks, with conversions concentrated in the evening), day-of-week, a slow trend, a yearly wave and a Black-Friday/Cyber-Monday window (volume ×1.75, CVR ×1.25, competition ×1.25). Devices differ in CTR, CVR and AOV. Demand follows a Gamma-Poisson process (overdispersed), clicks and conversions are Binomial, and order values are Gamma. Historical bids follow a mean-reverting AR(1) process in log space (weekly manager changes), which provides the bid variation that landscape forecasting needs.

The exact same `SemWorld.simulate()` call generates the history *and* evaluates the bidding strategies, so the evaluation uses the same ground truth that produced the training data.

## 2. SQL analytics layer (`sql/`)

Every query is commented, uses CTEs and window functions, and **runs unchanged on DuckDB and SQLite**. `tests/test_sql.py` checks that both engines return identical results. Portability tricks: calendar columns (`week_start`, `day_of_week`) are precomputed, `* 1.0` forces float division (SQLite truncates int/int), `CASE` stands in for `GREATEST/LEAST`, and every ratio is a ratio of sums.

| file | what it answers | techniques |
|---|---|---|
| `01_campaign_kpis.sql` | CTR, CPC, CVR, CPA, ROAS, share of spend per campaign | CTE, `SUM() OVER ()` |
| `02_daily_rolling_trends.sql` | 7-day rolling ROAS/CPA, same-weekday WoW change | named `WINDOW`, `ROWS BETWEEN 6 PRECEDING`, `LAG(…, 7)` |
| `03_keyword_roas_cohorts.sql` | Pareto view: spend vs value by keyword ROAS quartile | `NTILE(4)`, running share |
| `04_weekly_cpa_cohorts.sql` | weekly CPA/ROAS cohorts, WoW deltas, best/worst weeks | `LAG`, `RANK`, 4-week rolling ROAS |
| `05_wasted_spend_search_terms.sql` | negative-keyword candidates, ranked by money lost | params CTE, CASE flags, cumulative waste share |
| `06_hour_device_bid_modifiers.sql` | value-per-click index → suggested bid modifiers | index vs account, clamping |
| `07_keyword_bid_landscape.sql` | weekly (bid → clicks, cost) points, bid elasticity | `LAG`, elasticity = %Δclicks / %Δbid |

**Wasted spend definition.** A term with at least €25 of spend is flagged `ZERO_CONV` (the whole cost counts as waste) or `HIGH_CPA` when its CPA is more than 3× the campaign CPA (waste = cost − conversions × campaign CPA). In the sample run this flags **€18,082 (23.8 % of spend) in 53 search terms**. The top three are "how to fix laptop for programming" (€2,073, 0 conversions), "dual sim phone driver download" (€1,260, 0 conversions) and "how to fix compact phone" (€1,079, 1 conversion).

## 3. Statistical engine (`src/stats_engine.py`)

### Frequentist A/B test (CTR / CVR)

$$
z = \frac{\hat p_B - \hat p_A}{\sqrt{\bar p(1-\bar p)\left(\frac1{n_A}+\frac1{n_B}\right)}}, \qquad
\text{CI} = (\hat p_B-\hat p_A) \pm z_{1-\alpha/2}\sqrt{\tfrac{\hat p_A(1-\hat p_A)}{n_A}+\tfrac{\hat p_B(1-\hat p_B)}{n_B}}
$$

The test statistic uses the pooled standard error (under H0), and the confidence interval uses the unpooled (Wald) standard error.

**Minimum sample size per arm** for baseline $p_1$ and target $p_2 = p_1(1+\text{MDE})$:

$$
n = \frac{\left(z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)} + z_{1-\beta}\sqrt{p_1(1-p_1)+p_2(1-p_2)}\right)^2}{(p_2-p_1)^2}
$$

**Power** is $\Phi\big((|p_2-p_1| - z_{1-\alpha/2}SE_0)/SE_1\big)$. The **SRM check** (a χ² test on the traffic split) catches broken randomisation before anyone trusts a lift.

### Bayesian A/B test (Beta-Binomial)

With prior $\text{Beta}(\alpha_0,\beta_0)$, the posterior is $\text{Beta}(\alpha_0+s,\ \beta_0+n-s)$. The probability that B beats A is computed **exactly**, in log space:

$$
P(p_B>p_A)=\sum_{i=0}^{\alpha_B-1}\frac{B(\alpha_A+i,\ \beta_A+\beta_B)}{(\beta_B+i)\,B(1+i,\ \beta_B)\,B(\alpha_A,\ \beta_A)}
$$

There is a Monte-Carlo fallback for non-integer parameters. The test also returns credible intervals, the posterior of the relative lift and the **expected loss** $\mathbb{E}[\max(p_A-p_B,0)]$, which answers the business question "how much do we risk if we ship B now?".

### Multi-armed bandits

The engine runs fully vectorised Monte-Carlo replications of **Thompson Sampling** (sample $\tilde p_k \sim \text{Beta}(1+s_k, 1+f_k)$ and play the arg-max), **UCB1** ($\hat p_k + \sqrt{2\ln t / n_k}$) and a **static equal split** (50/50 for two arms). It measures pseudo-regret $\sum_t (p^* - p_{a_t})$ (expected conversions lost against always showing the best ad) and revenue. All policies use common random numbers.

## 4. Bid optimisation and forecasting (`src/bidding_optimizer.py`)

### Expected-value bidding with Bayesian shrinkage

$$
\text{Bid}_k = \text{pCVR}_k \times \text{Target CPA}_c, \qquad \text{Target CPA}_c = \frac{\text{AOV}_c}{\text{Target ROAS}}
$$

**pCVR with empirical-Bayes shrinkage (two-level partial pooling).** The prior mean of each keyword is its campaign's CVR $m_c$. One prior strength $\kappa$ (measured in "prior clicks") is shared across the account and fitted by maximising the Beta-Binomial marginal likelihood:

$$
\hat\kappa=\arg\max_\kappa \sum_k \Big[\ln B\big(c_k+m_c\kappa,\ n_k-c_k+(1-m_c)\kappa\big)-\ln B\big(m_c\kappa,(1-m_c)\kappa\big)\Big],
\qquad
\text{pCVR}_k=\frac{c_k + m_c\hat\kappa}{n_k+\hat\kappa}
$$

A keyword with 40 clicks and 0 conversions is no longer bid down to the floor, and a keyword with 3 lucky conversions out of 30 clicks is no longer bid to the moon. Each one is pulled toward its campaign mean in proportion to how little evidence it has: the weight on the prior is $\hat\kappa/(n_k+\hat\kappa)$. Why not fit a separate prior per campaign by the method of moments? It collapses to "no heterogeneity" (full pooling) exactly when data are sparse, and `fit_beta_prior(method="moments")` is kept in the code to show that.

**Why target ROAS = 1 / margin by default?** In a second-price auction, bidding your true value per click is optimal: bid = pCVR × AOV × margin. That is the same as a target ROAS at the break-even point, so this target maximises profit = value × margin − cost.

### Baselines

* **Rule-based** (typical automated rules, weekly, trailing 14 days): ≥ 20 clicks and 0 conversions → bid × 0.70; CPA > 1.2 × target → × 0.85; CPA < 0.8 × target → × 1.15.
* **Static bids**: keep the last bids (control).
* **EV, raw CVR**: the same formula without shrinkage (an ablation).
* **Oracle**: the true value per click from the simulator. This is an upper bound and is not achievable in practice.

### Walk-forward simulation

All strategies start from the same 120-day history and the same last bids. They re-bid weekly for 4 weeks against the same simulated world and see only their own new data. Each strategy runs on 5 seeds with common random numbers, so per-seed differences are paired comparisons.

### Bid landscape forecasting

For each keyword the forecaster fits $\text{clicks}(b) = a(1-e^{-cb})$ and $\text{CPC}(b) = u(1-e^{-vb})$ on its weekly history (from `sql/07`). It then forecasts clicks, cost, conversions, value and profit for any bid, and returns the profit-maximising bid. The search stops at 1.5× the highest bid ever observed, because beyond that the curve is pure extrapolation.

## 5. Sample run (synthetic, seed 42)

Output of `python -m src.pipeline` (full numbers in `outputs/summary.json`, charts in `outputs/report.html`):

**Bidding, 4 weeks forward, target ROAS 2.86 (35 % margin), mean of 5 seeds:**

| strategy | spend € | conversions | ROAS | profit € |
|---|---:|---:|---:|---:|
| Static bids (control) | 17,976 | 757.6 | 8.91 | 38,056 |
| Rule-based | 22,838 | 895.6 | 7.74 | 39,028 |
| EV, raw CVR | 18,522 | 971.4 | 10.08 | 46,861 |
| **EV + Bayesian shrinkage** | 18,656 | 977.0 | **10.04** | **46,904** |
| Oracle (true value/click) | 18,535 | 980.2 | 10.24 | 47,914 |

* EV bidding beats the rules in **5 of 5 paired seeds**: ROAS +2.3 on average (at least +1.6), profit +€7.9k on average (at least +€1.7k). It gets **9 % more conversions from 18 % less spend**, and lands within about 2 % of the oracle's profit.
* **Where shrinkage matters.** With 120 days of history most keywords have plenty of clicks, so raw and shrunk CVR bid almost the same (the difference is inside seed noise). With only the **last 28 days** of history (a cold start), the shrunk version earns €47,981 against €46,379 for raw CVR, with half the seed-to-seed standard deviation (€2.2k vs €5.0k). The CVR estimation error (MAE vs. truth, 28 days) falls by **25 % overall and 41 % for keywords under 300 clicks**. It rises by 21 % in the 300–1,000-click bucket, because shrinkage trades a little bias for a lot less variance. That trade only pays off where variance dominates.

**Bandits** (3 variants with CVR 3.0 / 3.4 / 4.0 %, 20,000 visitors, 200 replications):

| policy | cumulative regret | revenue € | lift vs static | traffic to best arm |
|---|---:|---:|---:|---:|
| Static equal split | 106.7 | 41,554 | – | 33 % |
| **Thompson Sampling** | **38.6** | **45,878** | **+10.4 %** | 74 % |
| UCB1 | 94.5 | 42,255 | +1.7 % | 40 % |

UCB1's exploration bonus $\sqrt{2\ln t/n}$ is calibrated for rewards in [0, 1]. Next to conversion rates of a few percent the bonus is huge, so UCB1 keeps exploring for a long time. Thompson Sampling adapts to the actual posterior uncertainty. This is a known practical result, and it is why most ad-tech systems use Thompson-style allocation.

**A/B test** (true CTR 4.0 % vs 4.4 %, 40,000 impressions per arm): the observed lift is +16.1 % with p < 0.0001, the 95 % CI of the difference is [+0.35 pp, +0.91 pp], and P(B > A) = 1.000. The minimum sample size for a 10 % relative MDE is **39,475 per arm**. Note that the observed lift (16 %) overshoots the true lift (10 %). This is a real-world reminder that point estimates from a "winning" test are noisy, and the posterior of the lift shows the full range.

## 6. Business impact: how this maps to a real account

| module | decision it supports | typical lever |
|---|---|---|
| Wasted-spend SQL | negative keyword list, match-type tightening | recover spend that buys no conversions (here 24 %) |
| Hour × device index | bid modifiers / dayparting | move budget to hours and devices with a higher value per click |
| Sample-size planner + SRM | stop underpowered tests and broken splits | fewer false winners, fewer wasted test weeks |
| Bayesian A/B | ship / don't ship with an explicit risk number | decisions in business language ("98 % chance B is better, expected loss 0.002 pp") |
| Thompson Sampling | ad-copy rotation | ship the winner *while* testing; less regret than A/B-then-ship |
| EV bidding + shrinkage | keyword bids from value, not gut feel | higher ROAS at the same or lower spend; stable bids for long-tail keywords |
| Bid landscape | "what if we bid X?" | estimate headroom before you spend it |

## 7. Tests (`tests/`, 56 tests)

* **Statistics:** z² equals χ² (scipy), the textbook sample size (10 % → 12 % ⇒ 3,841/arm), power at the computed n is 0.80, simulated **CI coverage ≈ 95 %**, **type-I error ≈ 5 %** and **empirical power ≈ 80 %**, the exact P(B > A) matches Monte Carlo, symmetry, and expected loss.
* **Bandits:** static-split regret equals the exact linear formula, Thompson beats static and UCB1, UCB1 regret is sublinear, and the regret bands are ordered.
* **Simulator:** CPC ≤ bid, QS lowers CPC, diminishing returns (marginal clicks fall, marginal CPC rises), funnel consistency, search terms roll up exactly to keywords, seasonality, promo window, junk terms convert worse.
* **SQL:** every query returns identical results on DuckDB and SQLite, matches pandas, and computes rolling ratios of sums and wasted-spend rules correctly.
* **Bidding:** the prior fit recovers the simulated population, shrinkage pulls low-volume keywords harder, the EV formula and rule actions are exact, the landscape fit recovers the curve, and EV beats rules in the forward simulation.
* **App:** the end-to-end pipeline, the HTML report, and the Streamlit app rendered headless via `streamlit.testing`.

## 8. Limitations and next steps

* The auction is a single-slot GSP abstraction (no reserve prices, Smart Bidding competitors or budget pacing). Budget-constrained bidding (a Lagrangian on the daily budget) is the natural next step.
* The pCVR model is keyword-level only. A hierarchical GLM (keyword × device × hour) would also learn the bid modifiers that `sql/06` currently suggests.
* Conversions are attributed immediately. Delayed conversions (conversion lag) would penalise rules that react to recent CPA even more.
* Bandits use stationary arms. Discounted Thompson Sampling handles ad fatigue.

## Requirements

Python ≥ 3.10. `numpy`, `scipy`, `pandas`, `duckdb`, `plotly`, `streamlit` (optional: the HTML report works without it), `pytest`.
