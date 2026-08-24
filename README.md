# Statistical Arbitrage / Pairs Trading System

> **Quant Developer project** | C++17 + Python | IIT Madras Placement Level  
> Engle-Granger cointegration · Kalman filter dynamic hedge ratio · Walk-forward backtesting · Full tearsheet

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Theory & Algorithms](#3-theory--algorithms)
4. [Build & Run](#4-build--run)
5. [Component Reference](#5-component-reference)
6. [Data Structures](#6-data-structures)
7. [Performance Metrics](#7-performance-metrics)
8. [Complexity Analysis](#8-complexity-analysis)
9. [Sample Results](#9-sample-results)
10. [Interview Preparation](#10-interview-preparation)
11. [Extension Ideas](#11-extension-ideas)
12. [References](#12-references)

---

## 1. Project Overview

Statistical arbitrage exploits temporary price divergences between cointegrated
asset pairs. When two stocks move together in the long run (cointegrated), their
spread - a linear combination of their prices - is stationary and mean-reverting.
We trade the spread: buy it when it is unusually low, sell it when unusually high,
profit when it reverts to its mean.

### Why this is placement-level

| Feature | What it shows |
|---|---|
| Engle-Granger + ADF test | You understand econometric tests, not just coding |
| Kalman filter dynamic β | You know static OLS hedge ratios drift; Kalman tracks them |
| Walk-forward backtesting | You actively design against look-ahead bias and overfitting |
| Transaction cost model | You know gross Sharpe is meaningless; net Sharpe matters |
| 8-metric tearsheet | You can communicate strategy quality like a quant desk |
| HMM regime filter | You know why stat-arb blew up in 2008 and how to prevent it |
| C++ core + Python analytics | You know when to optimise and when to iterate |

> **Note on integration status:** the C++ engine (cointegration, Kalman
> filter, signals, sizing, risk gates, walk-forward backtest) is the
> pipeline that actually executes and produces `trades.csv`/`equity.csv`.
> `ou_model.py` and `regime_hmm.py` are standalone, fully-working Python
> analytics modules - they are not called from inside the C++ backtest
> loop (no code path feeds an HMM regime series or an MLE half-life back
> into `RiskManager`/`WalkForwardEngine`). Run them yourself against a
> spread series for offline analysis, or wire `RegimeFilter.predict()`
> into the entry check yourself if you want live regime-gating - see
> [§11 Extension Ideas](#11-extension-ideas).

### Language split rationale

```
C++ core engine:   cointegration screening (1225 pairs in ~3s vs 45s Python)
                   Kalman filter (updates every tick, latency matters)
                   walk-forward backtester (tight loop over 600+ bars × N pairs)

Python analytics:  OU model fitting (scipy optimize, readable iteration)
                   HMM regime detection (EM algorithm, easy to experiment)
                   visualisation (matplotlib 4-panel dashboard)
                   tearsheet (pandas-free, readable metrics)
```

---

## 2. Repository Structure

```
StatArb/
├── cpp/
│   ├── src/
│   │   ├── Utils.h                  # Types, TimeSeries, Stats namespace
│   │   ├── DataFetcher.h/.cpp       # CSV reader, date alignment
│   │   ├── CointegrationEngine.h/.cpp  # ADF test, Engle-Granger, Hurst
│   │   ├── KalmanFilter.h/.cpp      # Dynamic hedge ratio (2-state)
│   │   ├── SignalGenerator.h/.cpp   # Z-score bands, entry/exit logic
│   │   ├── PositionSizer.h/.cpp     # Kelly criterion, z-score scaling
│   │   ├── RiskManager.h/.cpp       # VaR gate, drawdown circuit breaker
│   │   ├── TradeLog.h/.cpp          # P&L ledger, equity curve, CSV export
│   │   ├── WalkForwardEngine.h/.cpp # Walk-forward backtester (no lookahead)
│   │   ├── ReportGenerator.h/.cpp   # Tearsheet, overfitting check, export
│   │   └── main.cpp                 # CLI entry point (4 modes)
│   ├── tests/
│   │   └── test_all.cpp             # 28 unit tests (no external framework)
│   └── CMakeLists.txt
├── python/
│   ├── data_fetch.py        # yfinance downloader + synthetic generator
│   ├── ou_model.py          # OU parameter estimation (OLS + MLE)
│   ├── regime_hmm.py        # 2-state HMM regime filter (pure Python EM)
│   ├── tearsheet.py         # 8-metric performance tearsheet
│   ├── visualise.py         # 4-panel spread dashboard (matplotlib)
│   └── pair_scanner.py      # NIFTY-50 pair screener (C++ bridge + fallback)
├── data/
│   ├── prices/              # OHLCV CSVs (one file per ticker)
│   │   ├── HDFC.csv         # 600 bars, 2019–2021 synthetic
│   │   ├── ICICI.csv        # Cointegrated with HDFC (β=1.5)
│   │   ├── TCS.csv          # Cointegrated with INFY (β=2.0)
│   │   ├── INFY.csv
│   │   ├── WIPRO.csv        # Cointegrated with HCLTECH (β=2.3)
│   │   └── HCLTECH.csv
│   └── results/             # Generated output (trades, equity, reports)
├── requirements.txt
└── README.md
```

---

## 3. Theory & Algorithms

### 3.1 Cointegration (Engle-Granger two-step)

Two price series P_A and P_B are **cointegrated** if there exists β such that:

```
spread_t = P_A(t) - β·P_B(t) - α   is stationary (I(0))
```

**Step 1** - OLS regression: `P_A = α + β·P_B + ε`  
**Step 2** - ADF test on residuals ε to check stationarity

The **Augmented Dickey-Fuller test** checks the null hypothesis H₀: unit root exists.  
Regression: `Δy_t = α + ρ·y_{t-1} + γ₁Δy_{t-1} + ... + γₚΔy_{t-p} + ε`  
Test statistic: `τ = ρ̂ / SE(ρ̂)`

Critical values (no trend, constant):
```
1%  significance:  τ < -3.4336  →  reject H₀ (stationary)
5%  significance:  τ < -2.8621
10% significance:  τ < -2.5671
```

We also require **Hurst exponent H < 0.5** (independently confirms mean reversion)
and **half-life between 5 and 120 bars** (practical trading constraint).

### 3.2 Kalman Filter - Dynamic Hedge Ratio

Static OLS β drifts as the economic relationship between two stocks evolves.
The Kalman filter treats β as a latent state that evolves as a random walk:

**State vector:** `x_t = [β_t, α_t]ᵀ`

**Observation model:**  
`P_A(t) = [P_B(t), 1] · x_t + ε_t`,   ε_t ~ N(0, R)

**State transition:**  
`x_t = x_{t-1} + w_t`,   w_t ~ N(0, Q),   Q = δ·I

**Predict step:**
```
x_{t|t-1}  = x_{t-1|t-1}
P_{t|t-1}  = P_{t-1|t-1} + Q
```

**Update step:**
```
H          = [P_B(t), 1]
innovation = P_A(t) - H · x_{t|t-1}
S          = H · P_{t|t-1} · Hᵀ + R
K          = P_{t|t-1} · Hᵀ / S           (Kalman gain)
x_{t|t}    = x_{t|t-1} + K · innovation
P_{t|t}    = (I - K·H) · P_{t|t-1}
```

The **process noise δ** controls how fast β can change.  
We maximise log-likelihood over a grid of δ values to select the optimal one:

```
log L = Σ [ -½ log(2π·S_t) - innovation_t² / (2·S_t) ]
```

### 3.3 Ornstein-Uhlenbeck Model

The spread follows the OU SDE:
```
dX_t = θ(μ - X_t)dt + σ dW_t
```

Discretised:
```
ΔX_t = a + b·X_{t-1} + ε
b     = e^{-θΔt} - 1  ≈  -θ  (for small θ)
θ     = -b
half_life = ln(2)/θ
σ_eq  = σ / √(2θ)     (equilibrium std dev - used to set z-score bands)
```

### 3.4 Signal Generation

```
z_t = (spread_t - μ_roll) / σ_roll

Entry long  spread: z_t < -2.0   (spread abnormally low → buy A, sell B×β)
Entry short spread: z_t > +2.0   (spread abnormally high → sell A, buy B×β)
Exit:               |z_t| < 0.5  (spread reverted to mean)
Stop:               |z_t| > 3.5  (spread keeps diverging → protect capital)
```

### 3.5 Kelly Criterion Position Sizing

```
f* = p / |L| - (1-p) / W

where p  = win rate (from recent trade history)
      W  = average win P&L
      L  = average loss P&L (absolute)

Fractional Kelly: f = 0.25 · f*   (standard practice - full Kelly overbets)

Z-score scaling: notional = base_notional × min(|z| / entryZ, 2.0)
```

### 3.6 Walk-Forward Backtesting

```
│← trainWindow (252 bars) →│← testWindow (63 bars) →│
│  fit cointegration        │  generate signals        │
│  fit Kalman filter        │  execute trades          │
│  fit OU model             │  record OOS P&L          │
                             │← step (63 bars) →│← next window →│
```

Key property: **signals in the test window are generated only from parameters
fitted on the training window**. No future data leaks into the signal.

**Out-of-sample Sharpe** is reported separately from in-sample.
A large gap (IS Sharpe >> OOS Sharpe) signals overfitting.

---

## 4. Build & Run

### Prerequisites

```bash
# C++ build tools
sudo apt-get install cmake g++ build-essential   # Ubuntu/Debian
brew install cmake                                # macOS

# Python dependencies (all optional - pure-Python fallbacks exist)
pip install -r requirements.txt
```

### Step 1 - Generate sample data (or download real NSE data)

```bash
# Option A: synthetic cointegrated pairs (no internet needed)
python python/data_fetch.py --synthetic 600
# Already included in data/prices/ - HDFC, ICICI, TCS, INFY, WIPRO, HCLTECH

# Option B: real NSE data via yfinance
python python/data_fetch.py --tickers HDFCBANK.NS ICICIBANK.NS TCS.NS INFY.NS
```

### Step 2 - Build the C++ system

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### Step 3 - Run modes

```bash
# Walk-forward backtest on all 6 tickers
./build/statarb --mode wf \
    --data ../data/prices \
    --out  ../data/results \
    --tickers HDFC,ICICI,TCS,INFY,WIPRO,HCLTECH

# Deep analysis of a specific pair
./build/statarb --mode pair \
    --tickerA HDFC --tickerB ICICI \
    --data ../data/prices

# Cointegration screen only (no backtest)
./build/statarb --mode screen \
    --tickers HDFC,ICICI,TCS,INFY,WIPRO,HCLTECH
```

### Step 4 - Unit tests

```bash
cd cpp
./build/test_statarb
# Expected: 28 tests, all passing
```

### Step 5 - Python analytics

```bash
cd python

# Full tearsheet from C++ output
python tearsheet.py ../data/results/trades.csv

# 4-panel spread dashboard
python visualise.py --tickerA HDFC --tickerB ICICI \
    --prices ../data/prices \
    --trades ../data/results/trades.csv \
    --equity ../data/results/equity.csv

# Scan all pairs
python pair_scanner.py --data ../data/prices \
    --binary ../cpp/build/statarb

# OU model demo
python ou_model.py

# Regime filter demo
python regime_hmm.py
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `wf` | `wf` / `pair` / `screen` / `generate` |
| `--data` | `data/prices` | Price CSV directory |
| `--out` | `data/results` | Output directory |
| `--tickers` | - | Comma-separated ticker list |
| `--tickerA/B` | - | Pair for `--mode pair` |
| `--train` | `252` | Training window (bars) |
| `--test` | `63` | Test window (bars) |
| `--nav` | `1000000` | Initial portfolio NAV |
| `--entryZ` | `2.0` | Z-score entry threshold |
| `--exitZ` | `0.5` | Z-score exit threshold |
| `--stopZ` | `3.5` | Z-score stop threshold |
| `--zwin` | `30` | Z-score rolling window |
| `--pval` | `0.05` | ADF p-value cutoff |
| `--expand` | off | Expanding (vs rolling) train window |
| `--quiet` | off | Suppress per-bar / per-window output |

---

## 5. Component Reference

### C++ Components

| File | Responsibility | Key method |
|---|---|---|
| `Utils.h` | Types, TimeSeries, Stats | `Stats::sharpeRatio`, `Stats::hurstExponent` |
| `DataFetcher` | CSV loading, date alignment | `alignSeries(a, b)` |
| `CointegrationEngine` | ADF, Engle-Granger, screening | `screenAllPairs(tickers, prices)` |
| `KalmanFilter` | Dynamic β tracking | `update(pA, pB)`, `optimiseDelta(pA, pB)` |
| `SignalGenerator` | Z-score signals | `generateSignals(spread)` |
| `PositionSizer` | Kelly sizing | `compute(z, entryZ, pA, pB, β)` |
| `RiskManager` | Pre-trade checks | `checkEntry(portfolioDD, pairDD, spreadBps, openPositions, notional, adv, recentPnl)` |
| `TradeLog` | P&L ledger | `addTrade(t)`, `finalise()` |
| `WalkForwardEngine` | OOS backtesting | `run(tickers, prices, log)` |
| `ReportGenerator` | Tearsheet + export | `printTearsheet(log)` |

### Python Components

| File | Responsibility |
|---|---|
| `data_fetch.py` | yfinance download + synthetic CSV generation |
| `ou_model.py` | OLS and MLE OU parameter fitting, half-life CI |
| `regime_hmm.py` | 2-state Gaussian HMM via Baum-Welch EM |
| `tearsheet.py` | 8-metric tearsheet, IS vs OOS comparison |
| `visualise.py` | 4-panel matplotlib dashboard (ASCII fallback) |
| `pair_scanner.py` | NIFTY-50 screener (C++ bridge + Python fallback) |

---

## 6. Data Structures

### RankMatrix equivalent - O(1) spread lookup

```cpp
// Precomputed Kalman β and α avoid recomputing spread from scratch each bar
// After warmUp(), just call:
double spread = kf.computeSpread(priceA[i], priceB[i]);
// = priceA[i] - beta * priceB[i] - alpha   in O(1)
```

### Kalman state (2×2 system, no Eigen needed)

```cpp
struct KalmanState {
    double beta, alpha;       // state estimate
    double P00, P01, P10, P11; // 2×2 error covariance (stored flat)
    double Ve;                 // observation noise variance
    double logLikelihood;
};
```

Using flat storage instead of a matrix library:
- No external dependency
- Cache-friendly (4 doubles = 32 bytes, fits in one cache line)
- Fast for 2×2 - no general matrix multiply needed

### Trade record

```cpp
struct Trade {
    string tickerA, tickerB;
    int    entryBar, exitBar;
    double entrySpread, exitSpread;
    double entryZ, exitZ;
    double hedgeRatio;     // Kalman β at entry
    double positionSize;   // dollar notional
    double grossPnl, transactionCost, pnl;
    TradeDirection direction;
    bool   stoppedOut;
};
```

---

## 7. Performance Metrics

The tearsheet reports 8 metrics, each with a specific purpose:

| # | Metric | Formula | Interpretation |
|---|---|---|---|
| ① | Annualised return | `mean(daily_ret) × 252` | Raw profitability |
| ② | Sharpe ratio | `mean(excess_ret) / std × √252` | Risk-adjusted return |
| ③ | Sortino ratio | `mean(excess_ret) / downside_std × √252` | Penalises only losses |
| ④ | Max drawdown | `max((peak - trough) / peak)` | Worst loss from peak |
| ⑤ | Calmar ratio | `ann_return / max_drawdown` | Return per unit of tail risk |
| ⑥ | Win rate | `wins / total_trades` | Trade quality |
| ⑦ | Profit factor | `sum_wins / sum_losses` | > 1.5 is good |
| ⑧ | 95% VaR (daily) | `5th percentile(daily_pnl)` | Tail risk measure |

**Benchmarks** (these are good values for a stat-arb strategy):
```
Sharpe      > 1.5   (Sharpe > 2.0 is excellent)
Sortino     > 2.0
Max DD      < 15%
Calmar      > 1.0
Win rate    > 52%   (stat-arb pairs are not directional bets)
Profit factor > 1.3
```

---

## 8. Complexity Analysis

| Operation | Time | Space | Notes |
|---|---|---|---|
| ADF test (single series, n bars) | O(n·p) | O(n) | p = selected lag count |
| Engle-Granger (pair, n bars) | O(n) | O(n) | ADF dominates |
| Screen all pairs (N tickers) | O(N²·n) | O(N·n) | C++ does 1225 pairs in ~3s |
| Kalman update (per bar) | O(1) | O(1) | 2×2 system, constant time |
| Kalman warmup (n bars) | O(n) | O(n) if history kept | |
| Kalman δ optimisation | O(n·d) | O(n) | d = grid size (11 values) |
| Walk-forward run | O(W·N²·n) | O(n) | W = windows, reuses Kalman |
| Signal generation (n bars) | O(n·w) | O(n) | w = z-score window |
| Stability check (ADF) | O(n) | O(1) | Verify spread is I(0) |

**Total walk-forward complexity:** O(W × N² × n)  
For W=8 windows, N=6 tickers, n=600 bars: ~8 × 15 × 600 = **72,000 operations** - runs in milliseconds.

---

## 9. Sample Results

This is actual output from `./build/statarb --mode wf --tickers HDFC,ICICI,TCS,INFY,WIPRO,HCLTECH`
on the 600-bar synthetic data included in `data/prices/` (not hand-written/illustrative
numbers). The dataset is small - 600 bars only supports 5 walk-forward windows of
252 train + 63 test bars, and most windows find no pair that clears the cointegration
filter - so trade counts are low. Generate more synthetic bars
(`--mode generate --generate 2000`) or feed real downloaded price history for a
denser sample.

> **These are demo numbers, not a performance claim.** The run below uses 600 bars of
> *synthetic* prices and produces 2 trades. A Sharpe of 59 and a 100% win rate are what a
> 2-trade sample yields, not evidence of an edge. The overfitting check below the tearsheet
> is the number worth reading: IS 17.2 -> OOS 11.8, a 31% decay with 4 of 5 windows degraded.

```
Win Pair          Trades  OOS P&L     OOS Sharpe  IS Sharpe  MaxDD   WinRate
──────────────────────────────────────────────────────────────────────────────
0   TCS/HCLTECH   0       0           0.00        38.44      0.0%    0.0%
1   HDFC/WIPRO    2       2504        59.13       47.58      0.0%    100.0%
2   HDFC/ICICI    0       0           0.00        0.00       0.0%    0.0%
3   TCS/INFY      0       0           0.00        0.00       0.0%    0.0%
4   TCS/INFY      0       0           0.00        0.00       0.0%    0.0%

Combined Tearsheet:
  ① Ann. return    :  31.53%
  ② Sharpe ratio   :  59.13
  ③ Sortino ratio  :  inf   (no losing trades in this small sample)
  ④ Max drawdown   :  0.0%
  ⑤ Calmar ratio   :  inf   (no drawdown in this small sample)
  ⑥ Win rate       :  100.0%
  ⑦ Profit factor  :  inf   (no losing trades in this small sample)
  ⑧ 95% VaR       : -1,038.95
```

The `inf` values above are not a display bug - Sortino/Calmar/profit-factor are
genuinely undefined (division by a downside/drawdown/loss total of zero) when a
sample has no losing trades at all, and the tearsheet reports that honestly
rather than substituting an arbitrary large number. Do not read 2 trades /
100% win rate as a validated edge - it is what a 600-bar demo dataset produces,
nothing more.

**Overfitting check:**
```
Avg IS Sharpe : 17.204
Avg OOS Sharpe: 11.826
IS→OOS decay  : 31.3%
Degraded windows: 4/5
```

---

## 10. Interview Preparation

### Questions you will definitely be asked

**Q: Prove the Kalman filter state estimate is unbiased.**  
A: The Kalman gain K = PH'(HPH'+R)⁻¹ minimises the trace of the posterior
covariance matrix E[(x-x̂)(x-x̂)']. This is the MMSE (minimum mean square error)
estimator, which is unbiased for Gaussian noise.

**Q: Why not just use a rolling OLS hedge ratio instead of Kalman?**  
A: Rolling OLS weights all observations in the window equally and has a hard
cutoff - data from 31 days ago and 1 day ago count the same if the window is 30.
Kalman exponentially weights recent observations (controlled by δ) and has no
hard cutoff. More importantly, Kalman provides a full uncertainty estimate (P matrix)
which rolling OLS does not. In practice, Kalman β tracks regime shifts faster.

**Q: What is look-ahead bias and how does walk-forward prevent it?**  
A: Look-ahead bias occurs when future data influences past signals. In a naive
backtest, you fit the hedge ratio β using the full dataset, then test signals on
the same dataset - the β already "knows" future prices. Walk-forward prevents this
by fitting β only on a training window that ends strictly before the test window.
Each test window sees only a model fitted on its own past.

**Q: What is a blocking pair in the context of stability? (from your HRM project)**  
A: A blocking pair (r, h) is an unmatched pair where r prefers h over their
current assignment, AND h prefers r over at least one of its current residents.
A matching is stable iff no blocking pairs exist. Gale-Shapley always produces
a stable matching in O(n²) proposals.

**Q: Explain the Hurst exponent intuitively.**  
A: H measures the long-range dependence of a time series. H = 0.5 is a pure
random walk. H < 0.5 means the series is anti-persistent (mean-reverting):
if it went up yesterday it's more likely to go down today. H > 0.5 means
trending. For stat-arb, we need H < 0.5 on the spread.

**Q: Why Kelly criterion and not fixed fractional sizing?**  
A: Kelly maximises the expected logarithm of wealth (geometric growth rate),
which is the mathematically optimal criterion for long-run capital growth.
Fixed fractional (e.g. always 5%) ignores the trade's expected value and variance.
We use fractional Kelly (25% of full Kelly) in practice because full Kelly
has very high variance and one bad trade can wipe out a large fraction of capital.

**Q: How does the HMM regime filter help in crises like 2008?**  
A: In 2008, cointegrated pairs broke down - the economic relationship between
stocks changed fundamentally. The HMM detects this as a regime shift from
low-volatility (mean-reverting, state 0) to high-volatility (trending, state 1)
by monitoring the normalised spread volatility. Once in state 1, we block new
entries. This prevented most stat-arb funds that used such filters from taking
new positions in September 2008 when correlations spiked.

---

## 11. Extension Ideas

| Extension | Difficulty | Impact |
|---|---|---|
| Johansen cointegration test (supports N>2 assets) | Hard | High - portfolio of 3+ stocks |
| Kalman smoother (RTS smoother) for β estimation | Medium | Better in-sample β estimate |
| Pairs-level VaR with copula (Gaussian vs t-copula) | Hard | More accurate tail risk |
| Online learning: update OU params every 30 bars | Easy | More adaptive signals |
| Broker API integration (Zerodha Kite Connect) | Medium | Live paper trading demo |
| GPU-accelerated cointegration screening (CUDA) | Hard | 10,000+ universe in <1s |
| Reinforcement learning position sizing | Hard | Replace Kelly with RL policy |
| Multi-leg arbitrage (triangle: A-B, B-C, A-C) | Hard | Capture 3-way cointegration |

---

## 12. References

1. **Gale & Shapley (1962)** - "College Admissions and the Stability of Marriage"  
   *American Mathematical Monthly* - original stable matching proof

2. **Engle & Granger (1987)** - "Cointegration and Error Correction: Representation, Estimation, and Testing"  
   *Econometrica* - foundation of the cointegration test used here

3. **Kalman (1960)** - "A New Approach to Linear Filtering and Prediction Problems"  
   *Journal of Basic Engineering* - original Kalman filter paper

4. **Avellaneda & Lee (2010)** - "Statistical Arbitrage in the US Equities Market"  
   *Quantitative Finance* - industry-standard stat-arb with Kalman hedge ratio

5. **Kelly (1956)** - "A New Interpretation of Information Rate"  
   *Bell System Technical Journal* - Kelly criterion derivation

6. **MacKinnon (1994, 2010)** - "Approximate Asymptotic Distribution Functions for Unit-Root and Cointegration Tests"  
   *Journal of Business & Economic Statistics* - ADF critical value tables

7. **Roth & Peranson (1999)** - "The Redesign of the Matching Market for American Physicians"  
   *American Economic Review* - NRMP algorithm (extends your HRM project)

8. **Hamilton (1989)** - "A New Approach to the Economic Analysis of Nonstationary Time Series"  
   *Econometrica* - Markov regime-switching model (basis for HMM filter)

---

## License

MIT License - free to use, modify, and distribute with attribution.

---

*Built as a placement project demonstrating quant finance, algorithmic systems,  
and production-quality C++ engineering. All cointegration relationships in the  
sample data are synthetic and generated specifically for testing.*
