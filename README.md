# MarketResearcher — Autonomous Quantitative Research & Evolution System

<p align="center">
  <img src="https://img.shields.io/badge/Asset-XRP%2FUSDT-blue" />
  <img src="https://img.shields.io/badge/Exchange-Binance-yellow" />
  <img src="https://img.shields.io/badge/Capital-200%20USDC-green" />
  <img src="https://img.shields.io/badge/Strategies-13-purple" />
  <img src="https://img.shields.io/badge/GPU-Apple%20Metal-red" />
  <img src="https://img.shields.io/badge/Python-3.11-blue" />
</p>

Fully autonomous quantitative research platform for cryptocurrency trading on Binance. The system uses **genetic algorithm evolution** to discover, optimize, and validate trading strategy parameters over hundreds of generations, with walk-forward validation and regime-aware backtesting to combat overfitting.

Built on two architectural pillars:
- **autoresearch** (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)) — iterative experiment loop: hypothesis → mutation → backtest → accept/reject
- **MiroFish** (inspired by [666ghj/MiroFish](https://github.com/666ghj/MiroFish)) — multi-agent scenario simulation, regime classification, and catalyst detection

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Strategy Universe](#strategy-universe)
4. [Genetic Evolution Engine](#genetic-evolution-engine)
5. [Walk-Forward Validation](#walk-forward-validation)
6. [Fitness Function](#fitness-function)
7. [Regime Detection](#regime-detection)
8. [Monte Carlo Simulation](#monte-carlo-simulation)
9. [Feature Engineering](#feature-engineering)
10. [LSTM Prediction Model](#lstm-prediction-model)
11. [Risk Management](#risk-management)
12. [Experiment Results](#experiment-results)
13. [Post-Evolution Optimizations](#post-evolution-optimizations)
14. [Installation & Usage](#installation--usage)
15. [Next Steps](#next-steps)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    MarketResearcher Pipeline                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────────┐    │
│  │ Binance  │──▶│ Data Engine  │──▶│ Feature Engineering   │    │
│  │ API      │   │ (OHLCV+Cache)│   │ (35 features, no leak)│    │
│  └──────────┘   └──────────────┘   └───────────┬───────────┘    │
│                                                 │                │
│                    ┌────────────────────────────┤                │
│                    ▼                            ▼                │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐  │
│  │   MiroFish Scenario     │  │    Regime Classifier         │  │
│  │   Engine (7 agents)     │  │    (per-bar quantitative)    │  │
│  └────────────┬────────────┘  └──────────────┬───────────────┘  │
│               │                              │                  │
│               ▼                              ▼                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Autonomous Evolution Engine (GA)                │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │   │
│  │  │Tournament│▶│Crossover │▶│ Mutation │▶│Walk-Forward│  │   │
│  │  │Selection │ │(uniform) │ │(gaussian)│ │ Validation │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │   │
│  │                    ▼                                      │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │ Hall of Fame (diversity-enforced, max 4/strategy)│    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│               ┌──────────────┼──────────────┐                   │
│               ▼              ▼              ▼                   │
│  ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐      │
│  │ Monte Carlo    │ │ LSTM Model   │ │ Risk Manager     │      │
│  │ Simulation     │ │ (Attention)  │ │ (Circuit Breaker)│      │
│  └────────────────┘ └──────────────┘ └──────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                         │
│                    │  KVM Deployment  │                         │
│                    │  (Paper Trading) │                         │
│                    └──────────────────┘                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## System Components

| Module | File | Description |
|--------|------|-------------|
| **Data Ingestion** | `engine/data_ingestion.py` | OHLCV fetcher via ccxt with parquet cache, multi-exchange support |
| **Feature Engine** | `engine/features.py` | 35 backward-looking technical features, zero future leakage |
| **Backtester** | `engine/backtester.py` | Bar-by-bar backtester with walk-forward validator |
| **Metrics** | `engine/metrics.py` | Sharpe, Sortino, Calmar, VaR, CVaR, profit factor, expectancy |
| **Risk Manager** | `engine/risk_manager.py` | Circuit breaker, position sizing, Kelly criterion |
| **Monte Carlo** | `engine/monte_carlo.py` | Trade shuffling, return bootstrap, block bootstrap |
| **LSTM Model** | `engine/tf_model.py` | Multi-head attention LSTM on Metal GPU |
| **Scenario Engine** | `mirofish/scenario_engine.py` | Quantitative regime detection + multi-agent simulation |
| **Evolution Engine** | `auto_evolve.py` | Genetic algorithm with adaptive mutation and learning |
| **Strategies** | `strategies/*.py` | 13 active + 5 archived strategy implementations |
| **Config** | `config.py` | All system parameters as frozen dataclasses |

---

## Strategy Universe

### Active Strategies (13)

After evolution and pruning, the system uses 13 strategies in its genetic algorithm registry:

| # | Strategy | Type | Key Indicators | Status |
|---|----------|------|----------------|--------|
| 1 | `trend_following` | Trend | EMA + ADX + Fibonacci + RSI | Active |
| 2 | `momentum` | Momentum | RSI + ROC + ADX + acceleration | Active |
| 3 | `donchian_breakout` | Breakout | Donchian channels + ADX + volume | **Production Candidate** |
| 4 | `dual_ma` | Trend | Fast/Slow MA crossover + volume | Active |
| 5 | `keltner_breakout` | Breakout | Keltner channels + RSI + ADX | Active |
| 6 | `volatility_squeeze` | Volatility | BB/Keltner squeeze + momentum | **Production Candidate** |
| 7 | `supertrend` | Trend | ATR-based trailing stop + ADX | Active |
| 8 | `connors_rsi2` | Mean Reversion | 2-period RSI + streak + %rank | Active |
| 9 | `heikin_ashi_ema` | Trend | Heikin-Ashi candles + EMA | Active |
| 10 | `ichimoku_kumo` | Trend | Cloud + Tenkan/Kijun + Chikou | Active |
| 11 | `williams_cci` | Oscillator | Williams %R + CCI + BB squeeze | Active |
| 12 | `kama_trend` | Adaptive | Kaufman AMA + slope + ADX | Active (high risk) |
| 13 | `fisher_transform` | Oscillator | Fisher Transform + divergence | Active |

### Archived Strategies (5) — Removed after evolution (0% positive fitness)

| Strategy | Evaluations | Positive Rate | Reason |
|----------|-------------|---------------|--------|
| `volatility_breakout` | ~950 | 0% | No profitable parameter combination found |
| `mean_reversion` | ~950 | 0% | Regime filter too restrictive |
| `rsi_divergence` | ~950 | 0% | Signal too noisy for H1 timeframe |
| `vwap_reversion` | ~950 | 0% | VWAP less meaningful in 24/7 crypto |
| `obv_divergence` | ~950 | 0% | OBV divergence not predictive on XRP |

---

## Genetic Evolution Engine

The core optimization loop uses a genetic algorithm with tournament selection, uniform crossover, and Gaussian mutation operating over the full parameter space of all 13 strategies simultaneously.

### Genome Representation

Each genome is a tuple of strategy type and parameter dictionary:

$$\mathbf{g} = (s, \boldsymbol{\theta}_s) \quad \text{where} \quad s \in \mathcal{S}, \quad \boldsymbol{\theta}_s \in \mathbb{R}^{d_s}$$

where $\mathcal{S}$ is the set of 13 active strategies and $d_s$ is the dimensionality of strategy $s$'s parameter space.

### Tournament Selection

For each parent slot, $k=3$ individuals are sampled uniformly from the population. The fittest individual wins:

$$\text{parent} = \arg\max_{i \in \text{tournament}(k)} f(\mathbf{g}_i)$$

### Crossover (Uniform)

When two parents share the same strategy type, uniform crossover is applied parameter-by-parameter with probability $p_{\text{cross}} = 0.5$:

$$\theta_{\text{child}}^{(j)} = \begin{cases} \theta_{\text{parent}_1}^{(j)} & \text{with probability } 0.5 \\ \theta_{\text{parent}_2}^{(j)} & \text{with probability } 0.5 \end{cases}$$

### Gaussian Mutation

Each parameter is mutated with probability $p_{\text{mut}} = 0.3$. The mutation magnitude is proportional to the parameter's valid range:

$$\theta^{(j)} \leftarrow \text{clip}\left(\theta^{(j)} + \mathcal{N}\left(0, \, \sigma_j^2\right), \; \theta_{\min}^{(j)}, \; \theta_{\max}^{(j)}\right)$$

$$\sigma_j = 0.15 \times \left(\theta_{\max}^{(j)} - \theta_{\min}^{(j)}\right)$$

For integer-valued parameters, the result is rounded to the nearest integer. For boolean parameters, a coin flip with probability 0.3 toggles the value.

### Mandatory Exploration

To prevent premature convergence, 15% of each generation is replaced by individuals assigned round-robin across all 13 strategies with freshly randomized parameters:

$$|\mathcal{P}_{\text{explore}}| = \lceil 0.15 \times N_{\text{pop}} \rceil = \lceil 0.15 \times 40 \rceil = 6 \text{ individuals/generation}$$

### Hall of Fame Diversity Enforcement

The Hall of Fame stores up to 4 genomes per strategy, with a deduplication threshold based on normalized Euclidean distance:

$$d(\boldsymbol{\theta}_a, \boldsymbol{\theta}_b) = \sqrt{\frac{1}{d_s}\sum_{j=1}^{d_s} \left(\frac{\theta_a^{(j)} - \theta_b^{(j)}}{\theta_{\max}^{(j)} - \theta_{\min}^{(j)}}\right)^2}$$

A new genome is rejected from the HoF if $d(\boldsymbol{\theta}_{\text{new}}, \boldsymbol{\theta}_{\text{existing}}) < 0.10$ for any existing HoF member of the same strategy.

### Adaptive Learning

The evolution engine narrows parameter search ranges around successful regions as generations progress. Strategies with consistently positive fitness receive more population slots through proportional selection in the exploration phase.

---

## Walk-Forward Validation

All backtests use anchored walk-forward validation to ensure out-of-sample robustness.

### Fold Construction

Given total data of $T$ bars, we construct $K=5$ rolling folds:

$$\text{Fold}_i = \left[\text{Train}_i \;|\; \text{Val}_i \;|\; \text{Test}_i\right]$$

| Window | Duration | Bars (H1) |
|--------|----------|-----------|
| Train | 90 days | 2,160 |
| Validation | 15 days | 360 |
| Test (OOS) | 45 days | 1,080 |

Each fold slides forward by $\text{test\_bars}$ from the previous fold, creating non-overlapping test windows.

### Robustness Criteria

A strategy is declared **walk-forward robust** if all of the following hold simultaneously:

$$\text{Sharpe}_{\text{OOS}} \geq 0.8 \times \text{Sharpe}_{\min}$$

$$\text{MaxDD}_{\text{OOS}} \leq \text{DD}_{\text{tolerance}}$$

$$n_{\text{trades}} \geq n_{\min}$$

$$\text{Degradation}_{\text{OOS}} \leq 50\%$$

$$\text{PF}_{\text{OOS}} > 1.0$$

where Out-of-Sample degradation is measured as:

$$\text{Degradation}_{\text{OOS}} = 1 - \frac{\text{Sharpe}_{\text{OOS}}}{\text{Sharpe}_{\text{IS}}}$$

### Parameters Used

| Parameter | Value |
|-----------|-------|
| $\text{Sharpe}_{\min}$ | 1.0 |
| $\text{DD}_{\text{tolerance}}$ | 15% |
| $n_{\min}$ | 10 trades per fold |
| Max OOS Degradation | 50% |
| Commission | 0.1% (one-way taker) |
| Slippage | 5 basis points |

---

## Fitness Function

The fitness function balances in-sample performance, out-of-sample robustness, trade frequency, and risk:

$$f(\mathbf{g}) = \underbrace{0.5 \cdot \hat{S}_{\text{WF}}}_{\text{WF Sharpe}} + \underbrace{0.3 \cdot \hat{S}_{\text{IS}}}_{\text{IS Sharpe}} + \underbrace{0.1 \cdot \ln(1 + n_{\text{trades}})}_{\text{Trade Bonus}} - \underbrace{\lambda_d \cdot D_{\text{OOS}}}_{\text{Degradation Penalty}} - \underbrace{\lambda_{\text{dd}} \cdot P_{\text{DD}}}_{\text{Drawdown Penalty}} - \underbrace{P_{\text{low}}}_{\text{Low Trade Penalty}}$$

where:

- $\hat{S}_{\text{WF}}$ = Walk-forward aggregate Sharpe ratio (OOS)
- $\hat{S}_{\text{IS}}$ = In-sample Sharpe ratio
- $n_{\text{trades}}$ = Total number of trades in backtest

### Degradation Penalty

$$\lambda_d \cdot D_{\text{OOS}} = \begin{cases} 0 & \text{if } D_{\text{OOS}} \leq 0.3 \\ 0.5 \cdot (D_{\text{OOS}} - 0.3) & \text{if } D_{\text{OOS}} > 0.3 \end{cases}$$

### Drawdown Penalty

$$P_{\text{DD}} = \begin{cases} 0 & \text{if } \text{MaxDD} \leq 15\% \\ 0.3 \cdot (\text{MaxDD} - 0.15) & \text{if } \text{MaxDD} > 15\% \end{cases}$$

### Low Trade Count Penalty (Anti-Overfitting)

Strategies with fewer than 30 trades receive a linear penalty to discourage overfit solutions:

$$P_{\text{low}} = \begin{cases} 0.05 \times (30 - n_{\text{trades}}) & \text{if } n_{\text{trades}} < 30 \\ 0 & \text{if } n_{\text{trades}} \geq 30 \end{cases}$$

This penalizes up to $-1.5$ for zero-trade strategies, effectively eliminating solutions that look good on paper but have too few data points for statistical confidence.

---

## Regime Detection

The system classifies each bar into one of 5 market regimes using a quantitative decision tree. Strategies can filter trades by regime, improving signal quality.

### Regime Classification Algorithm

For each bar $t$, compute:

$$\text{BB}_{\text{pct}} = \text{Bollinger Bandwidth Percentile}_{20}$$

$$\text{ADX}_t = \text{Average Directional Index}_{14}$$

$$z_{\text{ret}} = \frac{r_t - \mu_{20}}{\sigma_{20}}$$

$$z_{\text{vol}} = \frac{V_t - \mu_{V,20}}{\sigma_{V,20}}$$

$$\text{VR} = \frac{\sigma_{\text{realized},5}}{\sigma_{\text{realized},20}}$$

### Decision Rules

$$\text{Regime}_t = \begin{cases}
\texttt{EXOGENOUS\_SHOCK} & \text{if } \text{VR} > 2.5 \text{ and } |z_{\text{ret}}| > 2.5 \\
\texttt{BREAKOUT} & \text{if } \text{BB}_{\text{pct}} < 15 \text{ and } z_{\text{vol}} > 2 \\
\texttt{TREND} & \text{if } \text{ADX}_t > 30 \\
\texttt{MEAN\_REVERSION} & \text{if } \text{ADX}_t < 20 \text{ and } |z_{\text{ret}}| > 1.5 \\
\texttt{LATERAL} & \text{otherwise}
\end{cases}$$

### MiroFish Multi-Agent Consensus

Seven simulated agent archetypes vote on market direction:

| Agent | Horizon | Behavior |
|-------|---------|----------|
| Whale Accumulator | Long | Buys dips, sells spikes |
| Momentum Trader | Short | Follows trends |
| Mean Reversion | Short | Fades extremes |
| Retail FOMO | Short | Panic buys/sells |
| Market Maker | Short | Widens spreads on volatility |
| News Reactive | Short | Reacts to extreme moves |
| Institutional Systematic | Medium | Follows ADX > 30 signals |

Net sentiment:

$$\text{Sentiment} = \frac{n_{\text{bull}} - n_{\text{bear}}}{n_{\text{total}}}$$

---

## Monte Carlo Simulation

Three simulation methods validate strategy robustness beyond single backtest results.

### Method 1: Trade Shuffling

Randomly permute the sequence of $n$ historical trades $N_{\text{sim}}$ times to destroy any serial correlation:

$$\boldsymbol{\pi} \sim \text{Uniform}(\mathcal{S}_n) \quad \Rightarrow \quad \{r_{\pi(1)}, r_{\pi(2)}, \dots, r_{\pi(n)}\}$$

For each permutation, reconstruct the equity curve:

$$E_t^{(\pi)} = E_0 \cdot \prod_{i=1}^{t} (1 + r_{\pi(i)})$$

### Method 2: Return Bootstrap

Sample $n$ trades with replacement from the empirical return distribution:

$$r_i^* \sim \hat{F}_n \quad \text{(i.i.d. bootstrap)}$$

$$E_t^* = E_0 \cdot \prod_{i=1}^{t} (1 + r_i^*)$$

This preserves the marginal distribution but destroys temporal dependencies.

### Method 3: Block Bootstrap

Preserve short-range dependencies by sampling contiguous blocks of length $b$:

$$B_k = \{r_{j_k}, r_{j_k+1}, \dots, r_{j_k+b-1}\} \quad \text{where } j_k \sim \text{Uniform}(1, n-b+1)$$

Concatenate blocks until reaching $n$ total trades.

### Output Statistics

For each simulation method with $N_{\text{sim}} = 1000$ iterations:

$$\hat{P}(\text{profit}) = \frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} \mathbb{1}\left[\text{PnL}^{(i)} > 0\right]$$

$$\hat{P}(\text{ruin}) = \frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} \mathbb{1}\left[\text{MaxDD}^{(i)} > 50\%\right]$$

$$\text{VaR}_{95} = \text{Percentile}_5\left(\{\text{PnL}^{(i)}\}_{i=1}^{N_{\text{sim}}}\right)$$

$$\text{CVaR}_{95} = \mathbb{E}\left[\text{PnL} \;|\; \text{PnL} \leq \text{VaR}_{95}\right]$$

---

## Feature Engineering

The system computes 35 features on each OHLCV bar, all strictly backward-looking (no future information leakage).

### Technical Indicators

$$\text{SMA}_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} C_{t-i}$$

$$\text{EMA}_n(t) = \alpha \cdot C_t + (1-\alpha) \cdot \text{EMA}_n(t-1), \quad \alpha = \frac{2}{n+1}$$

$$\text{ATR}_n(t) = \text{EMA}_n\left(\max(H_t - L_t, \; |H_t - C_{t-1}|, \; |L_t - C_{t-1}|)\right)$$

$$\text{RSI}_n(t) = 100 - \frac{100}{1 + \frac{\text{EMA}_n(\max(0, \Delta C))}{\text{EMA}_n(\max(0, -\Delta C))}}$$

$$\text{ROC}_n(t) = \frac{C_t - C_{t-n}}{C_{t-n}} \times 100$$

### ADX (Average Directional Index)

$$+\text{DM}_t = \max(H_t - H_{t-1}, 0) \cdot \mathbb{1}[H_t - H_{t-1} > L_{t-1} - L_t]$$

$$-\text{DM}_t = \max(L_{t-1} - L_t, 0) \cdot \mathbb{1}[L_{t-1} - L_t > H_t - H_{t-1}]$$

$$\text{ADX}_{14}(t) = \text{EMA}_{14}\left(\frac{|+\text{DI} - (-\text{DI})|}{+\text{DI} + (-\text{DI})} \times 100\right)$$

### Bollinger Bands

$$\text{BB}_{\text{upper}} = \text{SMA}_{20} + 2\sigma_{20}, \quad \text{BB}_{\text{lower}} = \text{SMA}_{20} - 2\sigma_{20}$$

$$\text{BB}_{\text{bandwidth}} = \frac{\text{BB}_{\text{upper}} - \text{BB}_{\text{lower}}}{\text{SMA}_{20}}$$

$$\%B = \frac{C_t - \text{BB}_{\text{lower}}}{\text{BB}_{\text{upper}} - \text{BB}_{\text{lower}}}$$

### Volatility Measures

**Realized Volatility:**

$$\sigma_{\text{realized},n} = \sqrt{\frac{252 \times 24}{n} \sum_{i=0}^{n-1} r_{t-i}^2}$$

**Garman-Klass Volatility:**

$$\sigma_{\text{GK}}^2 = \frac{1}{2}(\ln H - \ln L)^2 - (2\ln 2 - 1)(\ln C - \ln O)^2$$

### Volume Features

$$\text{Volume Ratio} = \frac{V_t}{\text{SMA}_{20}(V)}$$

$$\text{OBV}_t = \text{OBV}_{t-1} + V_t \cdot \text{sign}(C_t - C_{t-1})$$

$$\text{VWAP}_t = \frac{\sum_{i} C_i \cdot V_i}{\sum_{i} V_i}$$

### Microstructure Features

$$\text{Close Location} = \frac{C_t - L_t}{H_t - L_t}$$

$$\text{Buying Pressure} = C_t - \min(L_t, C_{t-1})$$

$$\text{Pressure Imbalance} = \frac{\text{Buying} - \text{Selling}}{\text{Buying} + \text{Selling}}$$

---

## LSTM Prediction Model

### Architecture: LSTM + Multi-Head Attention

The model predicts price direction and magnitude at 4 horizons simultaneously.

```
Input (60 x 35)
      |
      v
+--------------+
|  LSTM (64)   |-->  Hidden states (60 x 64)
+------+-------+
       v
+------------------+
| Multi-Head       |
| Attention (4h)   |-->  Context vector (64)
+------+-----------+
       v
+--------------+
| Dense (128)  |-->  ReLU + Dropout(0.3)
+--------------+
| Dense (64)   |-->  ReLU + Dropout(0.3)
+--------------+
| Dense (32)   |-->  ReLU
+------+-------+
       |
       +-->  Direction heads (4 x softmax) -> {up, down, neutral}
       +-->  Magnitude heads (4 x linear) -> expected % move
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Sequence length | 60 bars (2.5 days H1) |
| Input features | 35 |
| LSTM units | 64 |
| Dense layers | [128, 64, 32] |
| Dropout | 0.3 |
| Optimizer | Adam ($\eta = 10^{-3}$) |
| Epochs (early stop) | 100 (stopped at 17) |
| Batch size | 64 |
| Label threshold | 0.8% minimum move |
| Device | Apple Metal GPU |
| Total parameters | 62,924 |

### Attention Mechanism

$$\alpha_t = \frac{\exp(e_t)}{\sum_{j=1}^{T} \exp(e_j)}, \quad e_t = \mathbf{v}^T \tanh(\mathbf{W}_h \mathbf{h}_t + \mathbf{b})$$

$$\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t$$

### Test Results

| Horizon | Accuracy | Magnitude MAE |
|---------|----------|---------------|
| 1 hour | **80.94%** | 0.2065 |
| 6 hours | 47.09% | 0.9039 |
| 12 hours | 35.12% | 1.4019 |
| 24 hours | 46.14% | 1.9717 |

The 1-hour prediction is the primary signal used for trade confirmation.

---

## Risk Management

### Position Sizing: Volatility-Scaled with Kelly Constraint

$$\text{Size}_{\text{vol}} = \frac{\sigma_{\text{target}}}{\sigma_{\text{realized}}} \times \text{Equity}$$

$$\text{Size}_{\text{Kelly}} = f^* \times \text{Equity}, \quad f^* = \frac{p \cdot b - q}{b}$$

where $p$ is win rate, $q = 1-p$, $b$ is avg\_win / avg\_loss ratio. We use quarter-Kelly ($0.25 \times f^*$) for safety.

$$\text{Size}_{\text{final}} = \min\left(\text{Size}_{\text{vol}}, \; \text{Size}_{\text{Kelly}}, \; 0.15 \times \text{Equity}\right)$$

### Risk Constraints

| Rule | Threshold | Action |
|------|-----------|--------|
| Max concurrent positions | 3 | Reject new trades |
| Max daily loss | 3% of equity | Halt trading for the day |
| Circuit breaker drawdown | 10% of equity | Halt all trading |
| Min trade interval | 15 minutes | Reject rapid-fire trades |
| Max order size | 15% of equity | Cap position size |
| Time stop | 48 hours | Force close stale positions |

### Stop Loss / Take Profit (ATR-based)

$$\text{SL} = \text{Entry} \mp k_{\text{SL}} \times \text{ATR}_{14}$$

$$\text{TP} = \text{Entry} \pm k_{\text{TP}} \times \text{ATR}_{14}$$

where $k_{\text{SL}}$ and $k_{\text{TP}}$ are per-strategy evolved parameters (typically $k_{\text{SL}} \in [1.3, 4.3]$ and $k_{\text{TP}} \in [2.6, 7.9]$).

---

## Experiment Results

### Evolution Run #1: 10 Hours, 725 Generations, 29,040 Evaluations

**Configuration:** Population=40, Cores=10 (Mac M2 Pro), Data=365 days H1, 18 strategies (pre-pruning)

#### Per-Strategy Performance Summary

| Strategy | Evals | Positive Rate | Best Fitness | Best Sharpe | Best Trades | Best PnL |
|----------|-------|---------------|-------------|-------------|-------------|----------|
| **volatility_squeeze** | 5,375 | **77.58%** | 3.335 | 3.31 | 49 | $536 |
| **kama_trend** | 4,211 | **59.91%** | 3.529 | 3.37 | 13 | $367 |
| **donchian_breakout** | 2,739 | **50.31%** | 2.818 | 2.79 | 132 | $1,015 |
| fisher_transform | 1,104 | 5.16% | 2.645 | 1.92 | 43 | $281 |
| trend_following | 1,112 | 6.38% | 1.114 | 1.62 | 181 | $735 |
| momentum | 1,432 | 7.19% | 1.607 | 0.90 | 234 | $409 |
| ichimoku_kumo | 1,074 | 4.66% | 1.834 | 1.52 | 55 | $243 |
| dual_ma | 1,330 | 4.59% | 2.487 | 1.73 | 71 | $379 |
| connors_rsi2 | 1,162 | 0.43% | — | — | — | — |
| heikin_ashi_ema | 1,113 | 3.59% | — | — | — | — |
| supertrend | 1,192 | 3.10% | — | — | — | — |
| keltner_breakout | 1,136 | 3.70% | — | — | — | — |
| williams_cci | 1,096 | 2.28% | — | — | — | — |

#### Production Candidates (Walk-Forward Robust)

**1. Volatility Squeeze** — Best risk-adjusted performer

| Metric | Value |
|--------|-------|
| Sharpe (IS) | 3.31 |
| Sharpe (WF/OOS) | 3.94 |
| Profit Factor | 3.72 |
| Trades | 49 |
| Win Rate | 69.4% |
| Net PnL | $536.46 |
| Max DD | 62.0% |
| WF Robust | **Yes** |

Evolved parameters: `keltner_atr_mult=1.19, squeeze_min_bars=9, momentum_lookback=13, volume_surge=1.91, SL=3.80xATR, TP=5.25xATR`

**2. Donchian Breakout** — Highest absolute PnL with most statistical significance

| Metric | Value |
|--------|-------|
| Sharpe (IS) | 2.79 |
| Sharpe (WF/OOS) | 2.99 |
| Profit Factor | 1.89 |
| Trades | 132 |
| Win Rate | 44.7% |
| Net PnL | $1,014.96 |
| Max DD | 135.7% |
| WF Robust | **Yes** |

Evolved parameters: `entry_period=41, exit_period=10, volume_surge=1.55, adx_min=27.65, SL=2.68xATR, TP=7.86xATR`

#### Hall of Fame (Top 10)

| Rank | Strategy | Gen | Fitness | Sharpe IS | Sharpe WF | Trades | PnL | Robust |
|------|----------|-----|---------|-----------|-----------|--------|-----|--------|
| 1 | kama_trend | 657 | 3.529 | 3.365 | 4.527 | 13 | $367 | No |
| 2 | kama_trend | 468 | 3.513 | 2.995 | 4.514 | 13 | $331 | No |
| 3 | kama_trend | 296 | 3.397 | 3.316 | 4.251 | 18 | $402 | **Yes** |
| 4 | volatility_squeeze | 472 | 3.335 | 3.313 | 3.943 | 49 | $536 | **Yes** |
| 5 | volatility_squeeze | 417 | 3.295 | 3.527 | 3.738 | 47 | $654 | **Yes** |
| 6 | donchian_breakout | 144 | 2.818 | 2.787 | 2.986 | 132 | $1,015 | **Yes** |
| 7 | donchian_breakout | 694 | 2.621 | 2.559 | 2.921 | 123 | $906 | **Yes** |
| 8 | fisher_transform | 583 | 2.645 | 1.918 | 3.328 | 43 | $281 | **Yes** |
| 9 | dual_ma | 611 | 2.487 | 1.730 | 3.045 | 71 | $379 | **Yes** |
| 10 | ichimoku_kumo | 541 | 1.834 | 1.519 | 2.307 | 55 | $243 | **Yes** |

---

## Post-Evolution Optimizations

After analyzing the 10-hour evolution run, 5 optimizations were applied:

### 1. Dead Strategy Pruning
Removed 5 strategies with 0% positive fitness rate from the registry, freeing 16.4% of compute (4,765 wasted evaluations) for the remaining 13 strategies.

### 2. Hall of Fame Dedup Threshold Increase
Increased normalized parameter distance threshold from 0.05 to 0.10 to prevent near-clone flooding (especially kama_trend variants with <0.3% parameter differences occupying multiple HoF slots).

### 3. Low Trade Count Penalty
Added anti-overfitting penalty for strategies with $n_{\text{trades}} < 30$. This addresses the kama_trend problem where top HoF entries had only 13 trades (high Sharpe but low statistical significance).

### 4. Evolved Parameters Applied
Best-evolved parameters from 725 generations applied as new defaults for all 13 strategy files. Future evolution starts from these optimized seeds rather than hand-tuned defaults.

### 5. Bug Fixes
- **trend_following.py**: SMA to EMA fix (variable named `trend_ema` was using `.mean()` instead of `.ewm()`)
- **auto_evolve.py**: Shared PARAMS mutation contaminating parallel workers (fixed with `copy.deepcopy()` + `setattr()`)
- **mean_reversion.py**: Removed "unknown" from allowed_regimes
- **data_ingestion.py**: Added `fetchCurrencies: False` to prevent SAPI endpoint failure + public endpoint fallback
- **auto_evolve.py**: DataFrame truthiness bug in error handling (`if raw` to `raw.empty`)

---

## Installation & Usage

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (Metal GPU) or Linux
- Binance API key (read-only, for OHLCV data)
- ~4GB RAM for evolution with 10 cores

### Setup

```bash
# Clone
git clone git@github.com:Ramputink/MarketReasercher.git
cd MarketReasercher

# Create virtual environment
./setup_venv.sh

# Configure API keys
cp .env.example .env
# Edit .env with your Binance API key and secret
```

### Training Modes

```bash
# Full autonomous evolution (14 hours, recommended)
./train.sh --evolve

# Shorter evolution run (4 hours, for testing)
./train.sh --evolve --evolve-hours 4

# Custom population size
./train.sh --evolve --evolve-hours 14 --evolve-pop 60

# LSTM model training only
./train.sh --model-only

# Walk-forward validation only
./train.sh --walkforward

# Strategy parameter optimization (grid search)
./train.sh --optimize
./train.sh --optimize --opt-strategy donchian_breakout

# Full pipeline (research + model)
./train.sh

# Using with Claude Code as the agent
# In the repo directory, start Claude Code and prompt:
# "Read program.md and start a research session on XRP/USDT"
```

### Project Structure

```
MarketResearcher/
├── auto_evolve.py              # Genetic algorithm evolution engine (~1200 lines)
├── config.py                   # All system configuration (dataclasses)
├── train.sh                    # Master training script (all modes)
├── run.py                      # Research pipeline runner
├── optimize_strategies.py      # Grid/random search optimizer
├── deploy_kvm.sh               # KVM deployment for paper trading
├── setup_venv.sh               # Virtual environment setup
├── requirements_m2.txt         # Python dependencies (Metal GPU optimized)
├── program.md                  # Autonomous agent instructions
│
├── engine/
│   ├── data_ingestion.py       # Binance OHLCV via ccxt + parquet cache
│   ├── features.py             # 35 technical features (no future leakage)
│   ├── backtester.py           # Bar-by-bar backtester + walk-forward validator
│   ├── metrics.py              # Sharpe, Sortino, VaR, CVaR, PF, expectancy
│   ├── monte_carlo.py          # Trade shuffling, bootstrap, block bootstrap
│   ├── risk_manager.py         # Circuit breaker, Kelly sizing, position limits
│   ├── tf_model.py             # LSTM + Attention on Metal GPU
│   └── env_loader.py           # .env credential loader
│
├── strategies/                 # 13 active + 5 archived strategies
│   ├── trend_following.py
│   ├── momentum.py
│   ├── donchian_breakout.py    # PRODUCTION CANDIDATE
│   ├── volatility_squeeze.py   # PRODUCTION CANDIDATE
│   ├── dual_ma.py
│   ├── keltner_breakout.py
│   ├── supertrend.py
│   ├── connors_rsi2.py
│   ├── heikin_ashi_ema.py
│   ├── ichimoku_kumo.py
│   ├── williams_cci.py
│   ├── kama_trend.py
│   ├── fisher_transform.py
│   ├── mean_reversion.py       # [ARCHIVED]
│   ├── volatility_breakout.py  # [ARCHIVED]
│   ├── rsi_divergence.py       # [ARCHIVED]
│   ├── vwap_reversion.py       # [ARCHIVED]
│   └── obv_divergence.py       # [ARCHIVED]
│
├── mirofish/
│   ├── scenario_engine.py      # Regime detection + multi-agent simulation
│   └── ollama_engine.py        # LLM integration (Ollama llama3.1:8b)
│
├── autoresearch/
│   └── experiment_runner.py    # Automated experiment framework
│
├── models/                     # Trained model artifacts (gitignored)
├── reports/                    # Evolution reports, best genomes, trade logs
├── logs/                       # Training and evolution logs (gitignored)
├── data/                       # Cached OHLCV parquet files (gitignored)
└── checkpoints/                # Model checkpoints (gitignored)
```

---

## Next Steps

### Short Term (Next Evolution Run)

1. **Run 2nd evolution** with cleaned 13-strategy registry and anti-overfitting measures
2. **Validate convergence speed** — evolved params as seeds should produce faster convergence
3. **Monitor kama_trend** — verify the low-trade penalty is filtering overfit solutions ($n < 30$)

### Medium Term (Validation & Hardening)

4. **Monte Carlo validation** on production candidates (volatility_squeeze, donchian_breakout) — 1000 simulations per strategy
5. **Multi-asset cross-validation** on BTC/USDT, ETH/USDT, SOL/USDT to test generalization
6. **Regime-conditional analysis** — measure per-regime performance to build a regime-switching ensemble
7. **Slippage sensitivity analysis** — stress test at 10, 20, 50 bps slippage to find breakeven point

### Long Term (Deployment)

8. **Paper trading deployment** via KVM with live Binance data feed
9. **Ensemble construction** — combine top strategies with regime-based weighting:

$$w_s(t) = \frac{\text{Sharpe}_s(\text{regime}_t)}{\sum_{s' \in \mathcal{S}} \text{Sharpe}_{s'}(\text{regime}_t)}$$

10. **Live execution** with $200 USDC initial capital, quarter-Kelly sizing, 3% daily stop
11. **Continuous evolution** — weekly offline re-evolution with updated data to adapt to regime shifts
12. **Walk-forward monitoring** — automated alerts when live OOS degradation exceeds 30%

---

## License

Private research project. All rights reserved.

---

<p align="center">
  <i>Built with genetic algorithms, walk-forward validation, and a healthy fear of overfitting.</i>
</p>
