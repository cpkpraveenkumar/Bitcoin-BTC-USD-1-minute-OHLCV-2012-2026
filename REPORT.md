# Advanced Analytics Project — Bitcoin BTC/USD 1-minute OHLCV (2012–2026)

**Dataset:** `btcusd_1-min_data.csv` · **Scope of this report:** complete data-understanding, quality audit, cleaning, EDA, statistics, machine learning, segmentation, time-series, anomaly detection, KPI design, insights and executive summary.
**Analytical stance:** descriptive findings (what happened) vs statistical findings (what is significant) vs predictive findings (what can be forecast) vs business interpretations are kept rigorously separate throughout.

---

## 1. Dataset Overview

| Attribute | Value |
|---|---|
| Format | CSV, 392.6 MB (≈217 MB as parquet, float32) |
| Rows | **7,743,013** minute bars |
| Columns | 6 (Timestamp, Open, High, Low, Close, Volume) |
| Frequency | 1-minute, **perfectly contiguous** (0 gaps) |
| Period | 2012-01-01 00:01 UTC → 2026-09-21 02:13 UTC (~14.7 years) |
| Source type | Bitcoin to US-Dollar OHLCV, single exchange (typical Kaggle/Bitstamp-origin feed) |

### Variable inventory

| Variable | Type | Meaning | Notes |
|---|---|---|---|
| `Timestamp` | Datetime (UTC) | Unix seconds of bar open | Unique, no duplicates |
| `Open` | Continuous | Price at minute open ($) | |
| `High` / `Low` | Continuous | Intraminute range ($) | |
| `Close` | Continuous | Price at minute close ($) | Used for returns |
| `Volume` | Continuous (count) | BTC traded in that minute | 0 is valid (illiquid minutes) |

**Domain context:** cryptocurrency market microstructure. Typical objectives (used here): price evolution, return/risk behaviour, volatility dynamics, day-of-week/intraday seasonality, predictability (market efficiency), regime clustering, anomaly surveillance, dashboard KPIs. **Target variable:** no explicit target — daily & 1-min returns are engineered, and "next-day direction" is the ML target.

**Identifier check:** `Timestamp` is a natural key. **0 duplicate timestamps, 0 duplicate rows.** No suspicious identifier artefacts.

**Specimen values:** first bar Open=4.58 (2012), final bar Close≈81,319 (2026-09-21). Price trajectory $4.58 → **$126,272** (all-time high, 2025-10-06) → $81,319 at sample end.

---

## 2. Data Quality Report

### 2.1 Completeness — excellent
- **Missing values: 0 everywhere** (Timestamp 0, OHLC 0, Volume 0).
- **Contiguity:** the minute grid from first to last timestamp has exactly 7,743,013 expected minutes and 7,743,013 actual — **not one missing minute** across 14.7 years.
- **Duplicates:** 0 exact rows; 0 duplicate timestamps.

### 2.2 Plausibility / validity checks

| Check | Result | Verdict |
|---|---|---|
| O and C inside [Low, High] | 0 violations | Clean |
| Negative prices | 0 | Clean |
| Negative volume | 0 | Clean |
| Zero prices | 0 | Clean |
| High==Low (flat minutes) | 2,288,043 (29.6% of bars) | Expected in illiquid era |
| Volume = 0 | 1,312,415 minutes (16.95%) | **Concentrated in early years — see below** |
| Unique close prices | 1,874,921 | High-resolution, valid |
| Min/max prices | $3.80 / $126,272 | Plausible for BTC |

### 2.3 The dominant data-quality issue: early-era illiquidity

Zero-volume share by year collapses over time:

| Year | % zero-volume minutes |
|---|---|
| 2012 | **94.9%** |
| 2013 | **39.2%** |
| 2014–2016 | 24–33% |
| 2017 | 8.0% |
| 2018–2026 | 0.4–4.0% |

**Analytical impact:** in 2012 almost every minute traded 0 BTC yet carries a price — quotes with no trades. Summary stats dominated by 2012–2013 (e.g., the ±62% single-minute log returns at the extremes) come from this ultra-illiquid era. Lucas-Cavalluzzo style volume-price inference must use ≥2017 data.

**Treatment:** keep bars (they are valid observations of quoted prices), but (a) filter volatility/volume statistics to ≥2017 where microstructure is meaningful, (b) winsorize the 1-minute return extremes for distribution work, (c) prefer daily aggregation for most analytics.

### 2.4 Other issues and treatments

| Issue | Detail | Implication | Treatment |
|---|---|---|---|
| Extreme 1-min returns | max +0.617, min −0.617 log-returns (2012–13) | Obesity of tails → naive std is unstable | Robust stats (IQR/MAD), log returns, winsorization for plots |
| Massive scale drift | close $4.58 → $126k | Correlations on raw levels are spurious | Log-transform; returns, not levels |
| Zero-volume semantics | valid (no trades) but = noise | Volume-change features explode (div by 0) | Guarded division; log1p(Volume) |
| Day-1 bars | 2012-01-01 00:00 is the first full minute | No warm-up for lag features | Rolling features have NaN warm-up — dropped/backfilled only via minimum history |
| Panel/reporting bias | single feed, 2012 start | Missing pre-2012 (negligible volume) and exchange-specific | Flagged as limitation (see §13) |

**High-cardinality / constant columns:** none constant. `Timestamp` and prices are effectively 1,874,921-cardinality (fine). Near-zero-variance columns: none (volume std = 21.5 BTC/min, all prices vary). **No data leakage identified** in the raw file; engineered features were built strictly point-in-time.

---

## 3. Data Cleaning Summary

Cleaned/enriched assets written to `out/`:

1. **`btc_1min_clean.parquet`** — raw data, `Timestamp` converted to datetime (UTC), sorted, deduplicated (0 found), float32.
2. **`btc_daily.parquet`** — daily OHLCV aggregation (5,378 days) with engineered features.
3. **`btc_monthly.parquet`**, **`btc_yearly.parquet`**.
4. **`daily_regimes.parquet`** — daily rows with regime labels.

Treatment decisions and justifications:

| Step | Decision | Justification |
|---|---|---|
| Missing data | None existed — nothing imputed | Audit showed 0% missing |
| Duplicates | Removed — none existed | Not an active step |
| Dtypes | float32 for OHLC+vol, datetime for index | 45% memory reduction; precision more than adequate (price resolution ≤ $0.01) |
| Timestamp | Unix-epoch → datetime UTC | Enables resampling/seasonality |
| Categorical | None raw; engineered `dow` (0–6), `month` (1–12) | Needed for calendar effects |
| Outliers | Not deleted — winsorized only for distribution plots; robust stats used | Deleting the 2012–13 extremes would erase real (if extreme) market events |
| Zero volume | Retained; guarded division; `log1p` volume | Valid observations |
| Liquidity regime | Analysis split "all data" vs "2017+" for microstructure | Early-era illiquidity otherwise dominates metrics |

### Engineered variables (all strictly point-in-time — only info known at close of bar t)

| Family | Variables | Purpose |
|---|---|---|
| Returns | `ret`, `logret`, `ret_2…ret_30` | Return profiles, momentum |
| Volatility | `vol_7/30/90` (rolling daily-ret std × √365, annualized) | Risk regimes |
| Trend | `sma7/30/200_ratio` (close vs moving average) | Trend position |
| Volume | `log_vol`, `vol_change`, `vol_smaN` | Liquidity/activity |
| Structure | `range_pct`, `body`, `gap`, `streak`, `rsi14` | Market microstructure + momentum |
| Calendar | `dow`, `month`, `year` | Seasonality |
| Target (ML only) | `ret_next`, `y=(ret_next>0)` (1-day-ahead) | Classification |

**Variables removed:** none at daily level. Within minute level, `Timestamp` is folded into the index (kept, not dropped). Raw `Close`/`Open` kept as original; all engineering clearly separated from originals.

---

## 4. Exploratory Data Analysis

### 4.1 The headline picture
- **16,263× total price growth**, CAGR ≈ **93%/yr** over 14.7 years — but via violent cycles, not smooth compounding.
- 5,378 daily bars: **52.5% up-days**, mean daily return **+0.263%**, daily std **4.02%**, annualized vol ≈ **77%**.
- **Best day +40.1% (2013-11-18)**, **worst day −48.5% (2013-04-11)** (Mt. Gox-era crashes), max **drawdown −84.9%** (Dec-2013 → Feb-2017), longest losing streak 8 days, longest winning streak 10 days.

### 4.2 Annual performance (daily-based)

| Year | Return % | Win rate % | Ann. vol % | Max DD % | Year-end close $ |
|---|---|---|---|---|---|
| 2012 | +133 | 53.8 | 84 | −41 | 13.2 |
| 2013 | +512 | 61.6 | 145 | −71 | 732 |
| 2014 | −54 | 45.2 | 75 | −66 | 321 |
| 2015 | +54 | 51.5 | 70 | −46 | 431 |
| 2016 | +92 | 57.4 | 48 | −30 | 966 |
| 2017 | +311 | 59.5 | 94 | −35 | 13,880 |
| 2018 | −97 | 49.0 | 84 | −82 | 3,693 |
| 2019 | +91 | 51.5 | 71 | −49 | 7,168 |
| 2020 | +171 | 57.9 | 76 | −53 | 28,993 |
| 2021 | +80 | 51.0 | 81 | −53 | 46,214 |
| 2022 | −82 | 46.0 | 64 | −67 | 16,528 |
| 2023 | +104 | 49.6 | 44 | −20 | 42,258 |
| 2024 | +94 | 52.5 | 54 | −26 | 93,381 |
| 2025 | +2.3 | 50.1 | 42 | −32 | 87,496 |
| 2026 (to 9/21) | +0.4 | 49.2 | 46 | **−40** | 81,319 |

**Reading:** decade-long volatility has *halved* (145% → ~42–46% annualized) — the market matured. Only 4 down-years in 15; down-years followed up-years at cycle peaks (2014 after 2013, 2018 after 2017, 2022 after 2021). **2025–2026 is the weakest macro period in the sample** in terms of forward returns (−53% drawdown into Sept-2026 from the Oct-2025 ATH of $126.3k).

### 4.3 Univariate distributions
- Daily returns: **not normal**. Kurtosis 17.9 (simple) / 28.3 (log); JB test p≈0 (reject normality). Fat tails confirmed — the empirical frequency of |return| >3σ is ≈ **0–0.3%**... rather: 1-day ±3σ bands (±12.1%) are breached ~0.8% of days vs 0.27% under normal — a ~3× excess of tail events. QQ plot (c9) shows both tails heavy; left tail heavier (log-return skew −1.31) — **crashes are more violent than rallies**.
- Volume: heavy right skew (min-mean 4.9 BTC, max 5,854 BTC); log scale used throughout.

### 4.4 Calendar effects
- **Weekday (c5):** Monday mean +0.66% (win 54.1%) is the strongest; Thursday weakest (+0.06%, win 49.9%). Pattern in *returns* is borderline significant (K-W p=0.06) but the **volatility-by-weekday effect is very strong** (K-W p<0.0001).
- **Month:** no robust monthly seasonality (K-W p=0.42).
- **Intraday (c6, ≥2017, n=5.11M minutes):** lowest volatility 04–05 UTC (mean |1-min move| ≈ 5.2–5.3 bps), highest 14–15 UTC (≈7.3–7.6 bps) — the US cash-session overlap. Hour-of-day return differences are statistically real (K-W p≈4.5e-08) but economically tiny (−1 to +7 bps/hr).

### 4.5 Bivariate / multivariate
- **Spearman correlations (c16):**
  - `ret` vs `Volume`: ρ=0.040 (p=0.0036) — *daily return level is essentially unrelated to volume*.
  - `|ret|` vs `Volume`: ρ=**0.40** (p<0.0001) — volume is a **volatility/activity** flag, not a direction signal.
  - `log(Close)` vs `log(Volume)`: ρ=−0.40 — volume *shares* fell as price rose (structural liquidity concentration, not a causal negative).
- **Volatility clustering (c10, c3):** ACF(returns)₁ ≈ −0.046 ≈ white noise; ACF(returns²)₁ ≈ **0.412**, Ljung-Box on squared returns p≈0 → **strong persistence in volatility**. GARCH(1,1) fit: α=0.157, β=0.817, persistence 0.974 → **shocks decay with a ~26-day half-life**.
- **Vol-of-vol regime bands (c3):** realized 90-day vol swung from ~25% (2016, 2023) to >160% (2013), and every bull cycle opens with rising vol.

*Charts:* `c1_log_close` (cycle map), `c2_returns_dist`, `c3_volatility`, `c4_drawdown_vol`, `c5_dow`, `c6_hour`, `c7_vol_close`, `c8_calendar`, `c9_qq`, `c10_acf`, `c16_corr_heat`, `c17_dashboard`.

---

## 5. Statistical Analysis

Assumptions stated: returns are non-normal (tests that assume normality — t, ANOVA, chi-square — are used either on large samples where CLT helps, or justified as robustness checks against non-parametric alternatives). Where data are non-normal and samples small, non-parametric tests are used (Wilcoxon, Mann-Whitney, Kruskal-Wallis, Spearman).

| Question | Method | Result | Interpretation |
|---|---|---|---|
| Are daily returns positive on average? | one-sample t-test (n=5,377) | t=4.79, **p=1.75e-06**; 95% CI of daily mean **[+0.155%, +0.370%]** | Expected daily drift ≈ +0.26% (≈ +97%/yr compounded) — strongly positive over the full window |
| Is the median positive? | Wilcoxon | p=2.8e-08 | Robust to outliers; confirms drift |
| Are prices / log-prices stationary? | ADF | levels p=0.77; log-price p=0.17 — **non-stationary** | Prices are cumulative; modelling must use returns |
| Are daily log returns stationary? | ADF p≈0, KPSS p=0.06 | **stationary** | Returns are the correct modelling unit |
| Are daily returns white noise? | Ljung-Box lag 30 | stat 121, **p=6e-13** | Technically not pure white noise, but ACF₁=−0.046 → economically negligible (≈ n×ρ² = tiny predictability) |
| Is volatility persistent? | Ljung-Box on squared returns; GARCH(1,1) | p≈0; persistence 0.974, ~26-day half-life | **Vol clustering is the strongest statistical regularity** |
| Weekday effect on return / on vol | Kruskal-Wallis | return p=0.06; velocity p<0.0001 | Day-of-week matters for risk, not for expected return |
| Hour effect (2017+) | Kruskal-Wallis (n=5.11M) | returns p=4.5e-08; |ret| p≈0 | Real but small; largest at US session |
| Return ↔ volume | Spearman | ρ=0.040 (p=0.004) | Significant only through sample size; **no practical value** |
| |ret| ↔ volume | Spearman | ρ=0.40 (p≈0) | **Practical:** volume spikes flag risk |
| Momentum (ret→ret_next) | Spearman | ρ=−0.023 (p=0.09) | **No day-to-day momentum/reversal edge** |
| Trend fit, log price ~ t | OLS | slope 0.0017/day ≈83%/yr, R²=0.88 | Descriptive, not a forecast (violates iid assumptions) |
| Halving: 18-mo post vs pre | Mann-Whitney + Cohen's d | daily means 0.54% vs 0.24%; p=0.016; **d=0.07** | Significant by n, **trivial effect size** — the halving trade is not a reliably exploitable edge |

**Statistical significance vs practical significance:** the dataset is so large (n≈7.7M minutes; 5.4k days) that almost everything becomes "statistically significant" (e.g., ρ=0.04 return-volume). Practical significance is assessed via effect sizes (d), annualized economics, and out-of-sample performance — the recurring conclusion is that **direction predictability is economically nil while volatility/risk structure is real and exploitable**.

---

## 6. Advanced Analytics / Machine Learning

**Problem formulation:** predict **next-day return direction** (`y = sign(ret_next)`) from only point-in-time daily features. Justified: the target is the central business question and daily contiguity supports walk-forward validation. A 2-class problem with 52.5/47.5 base rates.

**Leakage controls:** all features built from data ≤ day t; rolling windows include only history; 1-day embargo on every fold; expanding-window (recursive) validation — no shuffling, no future info.

**Features (24):** `ret_1..ret_30`, `vol_7/30/90`, `sma7/30/200_ratio`, `rsi14`, `log_vol`, `vol_change`, `range_pct`, `gap`, `streak`, `dow`, `month` (training from 2013-01-01, n≈4,900).

**Models compared:** Logistic Regression (interpretable), Random Forest (400–600 trees), HistGradientBoost. Walk-forward folds: each calendar year 2021–2026.

| Model | Pooled OOS AUC | OOS Accuracy | Balanced Acc | PR-AUC |
|---|---|---|---|---|
| LogisticRegression | 0.509 | 49.5% | 49.5% | 0.510 |
| RandomForest | 0.516 | 51.1% | 51.1% | 0.508 |
| HistGradientBoost | 0.515 | 50.3% | 50.3% | 0.509 |
| Baseline "always up" | 0.50 | 49.7% | 49.7% | — |

Best single fold-performance peaks at AUC 0.54 (2022/2024) and dips to 0.46 (2026). **Final model (RF, train ≤2024, test 2025–2026, n=628): AUC 0.497, accuracy 47.9%, PR-AUC 0.502** — *indistinguishable from a coin flip*. Confidence-decile test shows **zero monotonicity** between predicted probability and realized outcome (rank correlation −0.07): the model's "high-conviction" bins perform no better than its low ones.

**Interpretability (drivers of the *attempt*, not of future returns):**
- RF permutation importance (test period): `ret_1` (0.0074) and `ret_14` (0.0072) lead, but all imports are tiny — no feature carries real signal.
- LR coefficients (trained ≤2024): strongest "up" coefficients on `rsi14` (+0.16), `ret_21` (+0.15), `log_vol` (+0.14), `sma200_ratio` (+0.14); strongest "down" on `sma30_ratio` (−0.22), `vol_90` (−0.11), `streak` (−0.09). These encode a *mean-reverting, high-risk-premium-after-extended-rally* flavour — economically sensible but **not out-of-sample profitable**.

**Conclusion (predictive finding):** daily BTC direction is **not predictable** from price/volume history in this sample. No selected model beats the naive base rate. This is an *informative null*: it quantifies market efficiency at daily horizon and says *time-series alpha from OHLCV alone is absent* — consistent with Ljung-Box and momentum-tests in §5. (Methods rejected as inappropriate: Poisson regression, naive Bayes, KNN, time-series CV without embargo, random CV — all would inflate apparent skill.)

---

## 7. Segmentation Analysis

**Method:** KMeans (k=4, k chosen for interpretability; annualized-profile separability checked) on standardized daily `[log_vol, |ret|, range_pct, ret]` over 5,374 days.

| Segment (regime) | % of days | Mean ret | Median ret | Ann. vol | Mean |ret| | Next-day mean ret | Next-day vol |
|---|---|---|---|---|---|---|---|---|
| **Bull (steady up)** | 46.9% | +0.40% | +0.39% | 50% | 2.2% | +0.29% | 4.0% |
| **Base (calm/mixed)** | 42.5% | +0.011% | 0.00% | 30% | 1.2% | +0.08% | 2.7% |
| **Euphoria (extreme up)** | 5.4% | +9.9% | +8.8% | 90% | 9.9% | +0.64% | 6.6% |
| **Stress (extreme down)** | 5.2% | −8.8% | −7.4% | 98% | 8.8% | **+1.05%** | 8.0% |

**Findings**
- Extreme days are rare (~5% each) and symmetric (5.4% euphoria vs 5.2% stress) — fat tails in BOTH directions.
- **Mean-reversion after extremes:** after a Stress day, next-day average +1.05% (though with 8.0% vol); after Euphoria, +0.64%. Risk-taking in Fear is *statistically* rewarded, but the volatility dwarfs the edge.
- **Regime history (c12):** Yearly mix changed drastically — 2013–2014 dominated by Euphoria/Stress (bubble), 2018–2019 heavy Stress/Bear, 2023–2025 shifted toward Base/Bull (mature market). **2026 shows elevated Stress share (recent −53% drawdown).**

---

## 8. Time-Series Analysis (application to BTC price)

### 8.1 Trend & cycles
- OLS on log price: ~83%/yr long-run drift proxy (R²=0.88) — descriptive only.
- Monthly log-close decomposition (2017+, period-12): residual std 0.275 log vs seasonal amplitude 0.133 log → **trend+cycle dominates; calendar seasonality is second-order** and mostly reflects US-session/liquidity microstructure, not a trading edge.
- **Cycle peaks/troughs (month-end closes):** 2017-12 ≈$13.9k → 2018-12 ≈$3.7k; 2021-10 ≈$61.4k → 2022-12 ≈$16.5k; 2025-07 ≈$115.8k → 2026-09 ≈$81.3k (still trending down). Real-time ATH **$126.3k on 2025-10-06**; at sample end (2026-09-21) price is **−36% off that high**.

### 8.2 Drawdown episodes ≥ 30% (peak-to-trough, close-based)

| Period | Depth |
|---|---|
| 2013-12 → 2017-02 | **−84.9%** |
| 2017-12 → 2020-11 | **−83.4%** |
| 2021-11 → 2024-03 | **−76.7%** |
| 2013-04 → 2013-11 | −71.0% |
| 2022-01 → 2022-11 (sub-period) | −67.0% |
| 2014-01 → 2014-10 | −66.2% |
| **2025-10 → 2026-09** | **−53.1%** (ongoing) |

### 8.3 Forecasting (walk-forward, 2021–2026, out-of-sample on log price)
| Model | RMSE h=1 | h=5 | h=21 |
|---|---|---|---|
| Naive (last level) | 47.1% | 47.1% | 47.9% |
| ARIMA(1,1,0) | 47.1% | 47.2% | 47.9% |
| AR(1)/AR(5) | 65–66% | 65–66% | 66–67% |

Interpretation: the 47% RMSE is dominated by each test year's *unpredictable drift* (e.g., 2021 moved ~40% within one year). **No linear/ARIMA model beats "predict the last price"** — mean reversion models actively hurt. Return-level benchmark: 30-day-average as next-day forecast has RMSE 4.00% vs the unconditional 4.02% daily std — **zero predictable skill**. This corroborates §6.

---

## 9. Anomaly Detection

Two independent lenses on daily data:
1. **Robust modified z-score (MAD-based, Iglewicz-Banerjee)** on daily log returns: threshold |MZ|>5 flags **131 days (2.4%)**.
2. **Isolation Forest** on `[log_vol, |ret|, range_pct, vol_change, ret]` (contamination 1%): flags **54 days**, largely extreme-return *and* extreme-volume-day combinations (liquidity-run episodes).

**Representative genuine anomalies (mutual agreement):**

| Date | Daily return | Event context |
|---|---|---|
| 2013-04-11 | **−48.5%** | 2013 Mt.Gox bubble crash |
| 2013-11-18 | **+40.1%** | Late-2013 bubble blow-off |
| 2020-03-12 | **−39.0%** | COVID "Black Thursday", USD liquidity crunch |
| 2015-01-14/15 | −24%/+23% | Bitstamp outage + flash crash |
| 2017-07-20 | +26.9% | SegWit-activation rally |
| 2017-12-07 / 2013-12-19 | +22%/+32% | Cycle blow-off tops |

**Genuine anomalous vs normal variation:** crash flags are real market events, concentrated 2013–2020; since 2023 the only extreme clusters are swing events (e.g., early-2024 ETF aftermath, 2025-08 risk-off). Vol-clustering means 1-min "anomalies" (|logret|>3%) are mostly **normal variation inside high-vol regimes** — flagged as such rather than as errors. No data errors found.

---

## 10. KPI Analysis (dashboard design)

| KPI | Definition | Why it matters | Slice by | Visualization |
|---|---|---|---|---|
| Price & return | Close, daily/monthly return % | Core P&L tracking | Year, month, intraday | Candles + line (log) |
| **Realized volatility** | 30/90-day ann. σ | Risk budgeting; regime switching signal | Year, regime, hour | Rolling curve + band |
| **Drawdown** | (Close − running max)/max | Capital-at-risk, tail control | Cycle | Area fill |
| **Win rate** | % up days (rolling 30d) | Regime health vs drift | Year, weekday | Line 30d |
| **Volume activity** | 30d avg volume; vol-change | Liquidity & attention proxy | Year, hour | Bar + heatmap |
| **Vol-Volume link** | ρ(Volume, \|ret\|) | Volume = risk indicator | Window (rolling 1y) | Donut/gauge |
| **Regime mix** | % days in Bull/Base/Euphoria/Stress | Positioning & cycle stage | Year | Stacked bar |
| **Extreme-day frequency** | days |ret|>5%/10% per quarter | Tail-frequency surveillance | Pareto/bar |
| **GARCH volatility half-life** | persistence of 0.974, ~26d | Model uncertainty horizon | — | Stat card |

**Executive dashboard structure (c17):** Page 1 "Market Health" — log price, drawdown, 90d vol, rolling win rate, volume, regime-mix stacked bar. Filters: period, calendar year, weekday. Drill-down: minute/hour-of-day heatmap (c6), monthly calendar heatmap (c8), regime profile (c12).

---

## 11. Key Insights (impact-ranked)

1. **Direction is not predictable (predictive):** any OHLCV-only next-day model gives AUC ≈ 0.50–0.52; confidence deciles are flat. Any systematic "trade the daily signal" strategy from this file has no edge. *(Impact: high — kills a whole strategy class; evidence: full ML + TS sections.)*
2. **Volatility is the only reliable signal (statistical):** vol clusters with GARCH persistence 0.974 (26-day half-life); volume correlates with |returns| ρ=0.40 but with direction ρ=0.04. Risk, not return, is forecastable. *(Impact: high.)*
3. **Crashes are more violent than rallies (descriptive):** log-return skew −1.31; −48.5% day exists; ≥30% drawdowns recur every cycle (~every 2.5–3.5 yrs). *(Impact: high for position sizing.)*
4. **Historic drift is enormous but decaying (descriptive):** CAGR ~93% (2012–2026) but 2025–2026 annualized vol fell to ~42% and the 2025-10→2026-09 drawdown (−53%) is the largest since 2021-11. The "easy alpha" phase may be over. *(Impact: high for investors.)*
5. **Halving trade is a myth as an edge (statistical):** post-18m returns decayed 4,587% → 2,216% → 657% → 68% across halvings; Mann-Whitney significant (p=0.016) only through sample size, Cohen's d=0.07. *(Impact: medium-high — debunks narrative.)*
6. **Monday effect is real but shrinking (statistical/descriptive):** Monday +0.66% vs Thursday +0.06%; stronger in 2012–2018, faded since. *(Impact: medium.)*
7. **US-session timing is the liquidity sweet spot (statistical):** intraday vol doubles at 14–16 UTC vs 04–05 UTC; execution/impact models should weight these hours. *(Impact: medium.)*
8. **Regime asymmetry payoff (descriptive/statistical):** Stress days historically rebound (+1.05% next day) — contrarian optionality exists but with 8% daily vol, it is a *volatility trade*, not an edge. *(Impact: medium.)*
9. **Drawdown memory (analytical):** three >76% drawdowns in 14 years; equity-curve math means most "average-return" narratives mislead — median year win rate ≈ 51%. *(Impact: high for expectations-setting.)*
10. **Data is pristine for microstructure work (quality):** 100% continuous minutes for 14.7 years; 0 gaps — rare in financial datasets; 2012–2016 rows must be excluded for volume/liquidity work (95% zero-volume minutes in 2012). *(Impact: methodological.)*

---

## 12. Business Recommendations

1. **Stop chasing daily-direction alpha.** Reallocate R&D to volatility-regime forecasting (GARCH/ML vol), execution, and risk overlay — the areas with demonstrated, out-of-sample structure.
2. **Use volatility as the primary risk KPI.** Size positions by 30-day realized vol / GARCH forecast (persistence is measurable: ~26-day half-life). Expect vol spikes to *precede* moves, not accompany them.
3. **Build drawdown-aware expectations.** Given recurring ≥50% drawdowns, capital models should assume a −50% to −85% tail every multi-year cycle; no single-asset BTC allocation should rely on uninterrupted compounding.
4. **Exploit, don't ignore, calender microstructure:** schedule liquidity-sensitive operations (rebalancing, MM quoting) to 14–16 UTC window; treat Monday as historically favourable, Thursday as neutral.
5. **Sanitize the vintage:** exclude ≥2012–2016 volume stats from liquidity models (or volume-weight days); use ≥2017 by default and 2012+ only for price history.
6. **Contrarian optionality management:** after Stress-regime days, historical next-day mean is +1.05% — but never size it above a vol-aware cap (8% daily vol).
7. **Executive dashboards:** deploy §10 structure (vol, drawdown, regime-mix, win-rate) — not raw price — as the monitoring layer.
8. **Re-test on fresh regime before deployment:** the 2017+ era economics differ from 2012–2016; any strategy must be validated on the post-2020 sub-sample.

---

## 13. Limitations / Validation & Uncertainty

- **Single exchange, single feed.** Volume and price reflect one venue (2012–2026), not global BTC volume — especially in mirror/ETF-heavy post-2024 flows. Global volume today is dominated by venues absent from this file.
- **No fundamental/on-chain/flow covariates.** Funding rates, ETF flows, hash rate, macro rates, regulation (news) are outside the file; "BCI" movements are correctly treated as unexplained shocks (decomp residual σ=0.28 log).
- **Sample asymmetries:** only 4 full cycles; the 2024+ regime (post-ETF, halving 4) is under-sampled; 2026 is an *in-progress* drawdown — the −53% is not a completed episode.
- **Near-perfect data does not mean representativeness.** Early-2012 quotes are reference stamps with 0 trades (95% zero-volume minutes) — excluded/treated accordingly.
- **Observational, not experimental.** All calendar/seasonal findings are associations; the halving finding in particular suffers from narrative/selection bias (only 4 events), and announcement effects are inseparable from supply mechanics.
- **Survivorship-style bias is structural:** the dataset begins 2012 and "timestamp purity" cannot recover Mt.Gox-on-Kraken cleared volume; pre-2012 (negligible) lost; exchange-specific outages (2015 Bitstamp flash crash) appear as anomalies.
- **Uncertainty of point forecasts:** walk-forward RMSEs are dominated by unpredictable annual drift (≈47% log-scale); no reliable point-forecast band available from this data alone.
- **Statistical significance inflation:** with n≈7.7M rows, p-values near 0 carry little information; effect sizes and OOS results are the decision basis.

---

## 14. Executive Summary

Bitcoin (BTC/USD) **1-minute bar data, 2012-01-01 → 2026-09-21 (14.7 years, 7.74M bars)** received the full analytical treatment: audit, cleaning, EDA, statistics, ML, segmentation, time-series, anomaly detection, KPI design. The file is of exceptional quality — 0 missing, 0 duplicates, **100% contiguous minutes**.

**Most important findings**
1. BTC grew **16,263×** ($4.58 → $126,272 ATH on 2025-10-06; $81,319 at end) — CAGR ~93%/yr — but only 4 up-years-in-14 with violent cycles; largest drawdown **−84.9%** (2013→2017), and a **−53% drawdown is still unfolding** from the Oct-2025 peak.
2. **Daily returns are unpredictable.** Walk-forward ML (LR/RF/GBM, 2021–2026) tops out at AUC ≈0.52 vs 0.50 baseline; confidence-decile test is flat; AR/ARIMA forecasts never beat persistence; 30d-mean return forecast RMSE (4.00%) ≈ unconditional noise (4.02%). **OHLCV-only day-ahead direction trading has no demonstrable edge.**
3. **Volatility is real, forecastable structure.** GARCH(1,1) persistence 0.974 → ~26-day shock half-life; volume is a risk proxy (ρ=0.40 with |returns|) not a direction signal (ρ=0.04); annualized vol has secularly declined from 145% (2013) to ~42–46% (2025–26).
4. Returns are **leptokurtic (kurtosis 18–28)** with left-tail dominance (log-skew −1.31); crashes (−48.5% day in 2013; −39% COVID day 2020-03-12) are historically recurring risk events.
5. Regime analysis separates days into Bull (47%), Base (43%), Euphoria (5.4%, +9.9% avg), Stress (5.2%, −8.8% avg); Stress days historically rebound (+1.05% next day, but with 8% vol).
6. Calendar effects are **real but small**: Monday is the best weekday (+0.66%); intraday volatility peaks at 14–16 UTC; no monthly seasonality.
7. The **halving narrative is decaying**: post-halving 18-month returns 4,587% → 2,216% → 657% → 68% across the four halvings; statistically significant (p=0.016) but Cohen's d=0.07 — negligible practical edge.
8. Anomalies are concentrated in 2013–2020 (bubbles, Mt.Gox crash, COVID); the market has since matured, with the current 2025–26 drawdown the defining recent anomaly.

**Most important risks:** (a) assumption of continued historical drift — 2025–26 is the weakest macro period on record; (b) tail events recur (≥30% drawdowns ~every 2.5–3.5 yrs, two at −83%+); (c) vol-clustering means "quiet" periods precede spikes; (d) data covers one exchange, pre-2024 flows unobserved; (e) 2026 episode still in progress.

**Most important opportunities:** volatility-regime awareness as a portfolio/risk tool; execution timing to US-session hours; volume-based risk flags; disciplined, vol-scaled contrarian capture of Stress-day rebounds; and (negative opportunity) discontinuing naive daily-signal R&D.

**Key recommended actions:** deploy a vol/drawdown/regime executive dashboard (§10, c17); size positions by 30-day realized vol; model −50%+ drawdowns into capital planning; exclude 2012–2016 rows from liquidity analytics; treat all calendar/seasonal edges as weak and shrinking.

**Bottom line:** the dataset's value is **risk measurement and cycle literacy, not return prediction**. Its continuous 14.7-year record lets you know *what* to expect statistically (huge drift, fat tail risk, clustering) while the predictive evidence says *when* is not knowable from price and volume alone.

---
*All charts in `out/` (c1–c17). Scripts in `scripts/` (01–08) are reproducible end-to-end. Derived datasets in `out/*.parquet`.*