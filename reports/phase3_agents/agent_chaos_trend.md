# chaos_trend — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis

**Current logic summary.** `chaos_trend_strategy` is an adaptive trend-following strategy gated on the Hurst exponent. At each bar it:

1. Skips if a position is already open (one position at a time, no pyramiding).
2. Skips if `regime` is not in `["trend", "breakout"]` (external classifier gate).
3. Computes the Hurst exponent H via multi-scale Rescaled Range (R/S) analysis over the last `hurst_window` closes, using chunk sizes 16/32/64. Requires H ∈ [hurst_min, hurst_max].
4. Optionally enforces fractal dimension D = 2 − H < fractal_max (redundant with H filter but adds calibration flexibility).
5. Confirms direction via precomputed fast/slow EMA crossover (cached at module level).
6. Validates with a linear-regression slope over `momentum_period` closes; sign of slope must match EMA direction and |slope|/ATR must exceed `min_momentum_strength` (currently defaults to 0.0 — effectively inactive).
7. Optionally requires `volume_ratio` ≥ `volume_threshold` (1.2× SMA, relatively mild).
8. Emits a **long** signal when EMA is bullish, or a **short** signal when EMA is bearish.

**Does it emit shorts?** Yes. Lines 235-242 of the strategy file contain a fully symmetric short branch: when `bearish=True` (fast EMA < slow EMA) and all other conditions pass, the strategy emits `Signal(side="short", ...)` with `stop_loss = close + atr * stop_loss_atr_mult` and `take_profit = close - atr * take_profit_atr_mult`. This branch must be removed for Phase-3.

**Current STRATEGY_REGISTRY param_space** (from `auto_evolve.py` lines 200-213): hurst_window [50,150], hurst_min [0.52,0.70], hurst_max [0.75,0.95], ema_fast [8,20], ema_slow [20,50], adx_min [12.0,30.0], momentum_period [8,24], volume_threshold [0.8,2.0], use_fractal_filter (bool), fractal_max [1.35,1.55], stop_loss_atr_mult [1.5,5.0], take_profit_atr_mult [3.0,8.0].

**Edge hypothesis.** The Hurst filter is a genuinely meaningful regime filter for crypto — markets exhibit alternating trending (H > 0.55) and mean-reverting (H < 0.45) behavior. Conditioning entries on H > threshold plus EMA alignment reduces exposure to whipsaw periods. The R/S Hurst estimator is noisy at short scales (only 3 chunk sizes, only 100 bars), but directionally correct.

**Key weaknesses.**
- `min_momentum_strength` defaults to 0.0 (dead filter).
- Short branch violates Phase-3 constraints.
- R/S with 3 chunk sizes is statistically unstable; H estimates for adjacent bars can differ by 0.15+.
- EMA cache uses Python `id(df)` which is reused across walk-forward folds if prior DataFrames are GC'd — potential silent stale-cache bug.
- `fractal_max` filter is mathematically redundant with `hurst_min`.
- 48h time stop is likely too aggressive for a 3-ATR trend trade.

---

## 2. Spot Long-Only Edge Hypothesis

When BTC/ETH spot price shows statistically significant persistence (H > 0.55, confirmed by multi-scale R/S) AND the fast EMA is above the slow EMA AND the linear regression slope over the momentum period is positive AND volume is elevated versus the 20-bar mean, the market is in a self-reinforcing momentum state where trend continuation substantially exceeds reversal probability at the hourly scale.

In this regime, spot long positions entered at the above convergence of signals and sized via volatility targeting capture a right-skewed payoff: losses are capped at 2-4 ATR by the hard stop, while profits ride persistent momentum to 4-8 ATR targets. The long-or-cash structure exploits crypto's secular upward bias and eliminates the borrowing and funding costs associated with shorting. When none of these conditions hold, the strategy sits in cash — this reduces drawdown and avoids whipsaw losses that would otherwise accumulate during ranging or mean-reverting regimes.

---

## 3. Market Techniques

| Technique | Role in chaos_trend |
|---|---|
| Multi-scale R/S Hurst exponent | Primary regime filter: only trade persistent markets |
| Fractal dimension (D = 2 − H) | Orthogonal texture confirmation of trend quality |
| EMA crossover (fast/slow) | Directional filter: long bias requires fast > slow |
| Linear regression slope / ATR | Momentum strength and direction validator |
| ATR-based stop + take-profit | Volatility-normalized risk management |
| External regime label gate | Macro filter: skip consolidation and unknown regimes |
| Volume ratio filter | Participation confirmation: require elevated volume |
| Time stop | Cap dead-trade duration and free capital |

**Additional techniques recommended for Phase-3:**
- OBV divergence filter (OBV > OBV SMA = accumulation bias)
- Bollinger Band compression filter (bb_bw_percentile < 30 before breakout)
- BMSB macro gate (long-term bullish support band)
- Garman-Klass vol ceiling (avoid chasing parabolic vol spikes)
- RSI 14 long-only filter (RSI > 45 to avoid deeply oversold entries)

---

## 4. Useful Features

**Price / Return:**
- `dist_from_low_20`: confirm price in upper portion of recent range before entry
- `dist_from_high_20`: near 0 = near breakout high (momentum confirmation)
- `ret_zscore_20`: check return z-score is positive but not extreme (exhaustion avoidance)
- `roc_12`: independent 12-bar rate-of-change momentum confirmation
- `price_acceleration`: second derivative positive = accelerating uptrend

**Volatility:**
- `atr_14`: already used for stop/TP sizing
- `gk_vol`: Garman-Klass estimator; use as risk-off ceiling when gk_vol is in top 5% of 60-day distribution
- `bb_bandwidth` / `bb_bw_percentile`: squeeze detection before entries

**Volume:**
- `volume_ratio`: already used (tighten to 1.3-1.5)
- `volume_zscore`: complementary volume surge detection (50-bar normalization)
- `obv_divergence`: OBV above its 20-bar SMA = accumulation; add as optional filter

**Trend / Momentum:**
- `adx_14`: already used as trend strength gate
- `rsi_14`: add long-only gate (>45 to confirm bullish bias, not overbought >75)
- `rsi_7`: faster RSI for entry timing

**Market Regime / Macro:**
- `bmsb_bullish`: price above 20-SMA and 21-EMA of the long-term band; macro bullish gate
- Regime label: already used (trend/breakout)

**Microstructure:**
- `buying_pressure` / `close_location`: bar closed in upper range confirms buying interest at bar close
- `pressure_imbalance`: if positive (buying > selling pressure), supports long entry

**Funding/Sentiment (informative only, never as PnL):**
- If available, perp funding rate can be used as a risk-off filter: when funding is deeply negative (< -0.05% per 8h), reduce position size by 50% or skip entry entirely. This is a sentiment/positioning indicator only — not traded as carry PnL.

---

## 5. Parameters That Should Enter the Sweep/Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `hurst_window` | int | 50 | 150 | Rolling R/S lookback |
| `hurst_min` | float | 0.52 | 0.70 | Minimum H for persistent regime |
| `hurst_max` | float | 0.75 | 0.95 | Cap on H (regime shift detection) |
| `ema_fast` | int | 8 | 20 | Must be < ema_slow |
| `ema_slow` | int | 20 | 50 | Must be > ema_fast |
| `adx_min` | float | 15.0 | 35.0 | Trend strength gate |
| `momentum_period` | int | 8 | 24 | LR slope lookback |
| `min_momentum_strength` | float | 0.01 | 0.15 | Activate this dead filter |
| `volume_threshold` | float | 0.9 | 2.0 | Volume surge requirement |
| `fractal_max` | float | 1.35 | 1.55 | D = 2-H ceiling |
| `use_fractal_filter` | bool | — | — | Toggle fractal filter |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 3.0 | 8.0 | ATR multiplier for TP |
| `time_stop_hours` | int | 24 | 120 | Max hold duration |
| `rsi_long_min` | float | 35.0 | 55.0 | New: RSI floor for long-only |
| `require_bmsb` | bool | — | — | New: BMSB macro gate toggle |
| `require_obv_confirm` | bool | — | — | New: OBV divergence confirmation toggle |
| `dist_from_low_min` | float | 0.01 | 0.05 | New: min distance from 20-bar low |

---

## 6. Parameters That Should NOT Be Optimized

| Parameter | Reason |
|---|---|
| `require_ema_alignment` | Must always be True in long-only; disabling allows directionless entries |
| `require_regime` | Must always be True; disabling nullifies the macro filter that is the strategy's main structural guard |
| `allowed_regimes` | Fixed to ["trend", "breakout"]; adding "ranging" or "unknown" would defeat the regime logic |
| `require_volume_surge` (bool) | The toggle aspect should be fixed to True; only the threshold magnitude should be swept |
| `hurst_max` upper bound above 0.95 | R/S Hurst rarely exceeds 0.90 in real data; values above are fitting to estimation noise |
| `fractal_max` when `use_fractal_filter=False` | When fractal filter is off, fractal_max is meaningless; conditional on use_fractal_filter |

---

## 7. Long-Only Risk Management

**Stop-loss:** ATR-based hard stop: `stop_loss = entry_price - atr_14 * stop_loss_atr_mult`. Sweep `stop_loss_atr_mult` in [1.5, 4.0]. The backtester honors this when `current bar low <= stop_price`. Do not set below 1.5 ATR to avoid normal-volatility stop-outs.

**Take-profit:** `take_profit = entry_price + atr_14 * take_profit_atr_mult`. Enforce `take_profit_atr_mult > stop_loss_atr_mult + 0.5` (auto_evolve.py already does this check). Sweep in [3.0, 8.0]. At a 50% hit rate, a 2:1 TP/SL ratio achieves breakeven — target 3:1+ for positive expectancy.

**Trailing stop:** Not natively supported in the Signal interface. Approximation: exit on the next bar where fast EMA crosses below slow EMA while in a Hurst-confirmed trend (signal_exit mechanism in backtester). For explicit trailing, add a `trailing_atr_mult` param and re-evaluate stop per bar in the strategy when `position is not None`.

**Volatility targeting:** Use `sizing_method = "volatility_scaled"` in RiskConfig with `vol_target_annualized = 0.25`. This scales position size so a 1-ATR move equals 25% of annual vol budget. Position naturally shrinks in high-vol crypto regimes and expands in calm trending conditions.

**Max position size:** `max_position_pct = 0.80` of equity. Reduce to 0.50 when `gk_vol > 95th percentile of trailing 60-day distribution`. Signal strength `= clip(mom_strength * 5, 0, 1)` already provides proportional scaling.

**Cooldown:** Minimum 4 bars (4h on 1h data) after a stop-loss exit before re-entering. Extend to 8-12 bars after consecutive stop-loss exits (whipsaw guard).

**Risk-off regime gate:** Block new long entries when:
1. Regime not in ["trend", "breakout"] (already implemented).
2. `adx_14 < adx_min` (already implemented).
3. `gk_vol` > rolling 95th percentile of its 60-day history (parabolic vol — avoid chasing).
4. `bmsb_bullish` has been False for 3+ consecutive bars (macro downtrend — long-only edge disappears).
5. Optional: if perpetual funding rate available and funding < -0.05% per 8h (panic selling — reduce size 50%).

---

## 8. Relevant Metrics

| Metric | Target / Threshold | Notes |
|---|---|---|
| CAGR | > 20% | On 2024-06-07 → 2026-06-07 OOS window |
| Sharpe ratio | > 1.0 | Annualized, hourly equity curve |
| Sortino ratio | > 1.3 | More relevant for long-only asymmetric returns |
| Max drawdown % | < 25% | Hard cap for spot long-only |
| Calmar ratio | > 0.5 | CAGR / maxDD |
| Hit rate | 35-50% | Expected given wide TP/SL asymmetry |
| Profit factor | > 1.3 | Gross wins / gross losses |
| Avg trade PnL (net) | >> 2× round-trip cost | Ensure edge survives realistic costs |
| Turnover | 5-15 trades/month | Low due to strict filter stack |
| Market exposure | 20-45% | Fraction of bars with open long position |
| OOS degradation | < 40% | Sharpe_train vs Sharpe_OOS across WF folds |
| Per-year Sharpe stability | SD < 0.5 | Compute annual Sharpes on training window |
| Cost sensitivity | < 30% Sharpe drop at 2× costs | Confirms edge is not cost-fragile |
| Bootstrap maxDD (95th pct) | < 35% | 200 trade-sequence resample simulations |
| Ruin probability (MC) | < 5% | Monte Carlo across 200 simulations |
| Deflated Sharpe Ratio (DSR) | ≥ 0.95 | Deflated by cumulative trial count |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Drop the short branch.** Remove `elif bearish:` block (lines 235-242 of `chaos_trend.py`). Replace with `return None`. The strategy is now long-or-cash only.

2. **Fix `require_ema_alignment = True` and `require_regime = True`.** Remove these from the genome sweep (they should never be False in Phase-3).

3. **Activate `min_momentum_strength`.** Change default from `0.0` to `0.02` in PARAMS. Add to sweep: `("float", 0.01, 0.15)`.

4. **Extend `time_stop_hours` default to 72 and sweep range** from `("int", 24, 120)` to replace the current omission from the registry. Add `time_stop_hours` to `auto_evolve.py` chaos_trend `param_space`.

5. **Add `rsi_long_min` to PARAMS and sweep.** After the volume check, add: `if current.get('rsi_14', 50) < p.get('rsi_long_min', 45): return None`. Add `"rsi_long_min": ("float", 35.0, 55.0)` to registry.

6. **Add `require_bmsb` bool param.** Gate: `if p.get('require_bmsb', False) and not current.get('bmsb_bullish', True): return None`. Add `"require_bmsb": ("bool",)` to registry.

7. **Add `require_obv_confirm` bool param.** Gate: `if p.get('require_obv_confirm', False) and current.get('obv_divergence', 0) < 0: return None`. Add `"require_obv_confirm": ("bool",)` to registry.

8. **Add `dist_from_low_min` float param.** Gate: `if current.get('dist_from_low_20', 0) < p.get('dist_from_low_min', 0.0): return None`. Add `"dist_from_low_min": ("float", 0.01, 0.05)` to registry.

9. **Enforce `ema_fast < ema_slow` in auto_evolve.** Add a post-mutation/crossover check for chaos_trend identical to the existing TP>SL check: `if params["ema_fast"] >= params["ema_slow"]: params["ema_slow"] = params["ema_fast"] + 5`.

10. **Fix the EMA cache bug.** The module-level `_cache_id = id(df)` is unreliable across walk-forward folds. Replace with a checksum of `df['timestamp'].iloc[[0, -1]].values` or `len(df)` combined with the EMA params, to prevent stale cache hits across folds.

11. **Add `gk_vol` risk-off filter.** Precompute rolling 95th percentile of gk_vol over 60-bar window once per df; skip entry if `current.get('gk_vol', 0) > gk_vol_p95`. Add `gk_vol_ceiling_pct` param (`("float", 0.80, 0.99)`) to sweep.

12. **Update `STRATEGY_REGISTRY` in `auto_evolve.py`** for chaos_trend: add `rsi_long_min`, `time_stop_hours`, `require_bmsb`, `require_obv_confirm`, `dist_from_low_min`, `min_momentum_strength` (already in spec but default is 0.0). Remove `require_ema_alignment` and `require_regime` from param_space.

---

## 10. Methodological Risks

1. **R/S Hurst estimation noise.** Only 3 chunk sizes (16/32/64) over 100 bars means the log-log regression has at most 3 data points. H estimates for adjacent bars can differ by 0.15+, creating unstable entry signals. This noise means the H filter looks more discriminating in-sample (where it may accidentally correlate with known trending periods) than OOS. Mitigation: add chunk sizes [8, 48, 96] to increase regression robustness, or add a minimum H stability requirement (rolling SD of H < 0.05 over 10 bars).

2. **H filter as in-sample regime memorizer.** Sweeping `hurst_window` [50, 150] allows the genome to find the exact lookback that perfectly segments the training data into profitable and unprofitable regimes. This is pure curve-fitting — the "optimal" hurst_window may simply be the lag that aligns H transitions with known BTC bull/bear cycles in the training window.

3. **EMA cache stale-state bug (silent correctness risk).** `_cache_id = id(df)` can be reused by a different DataFrame object after GC within the walk-forward validator's fold loop. If the fold DataFrame has the same `id` as a prior (GC'd) fold, the strategy uses stale EMA values from the wrong fold. This produces wrong signals without any exception, biasing IS/OOS comparisons. Fix before any sweep results are trusted.

4. **Fractal dimension filter redundancy inflates search space.** D = 2 − H is a deterministic monotone function of H. Evolving both `hurst_min` and `fractal_max` simultaneously creates a 2D constraint on a 1D quantity. The genome can find many "distinct" param combinations that are mathematically equivalent (e.g., hurst_min=0.55 / fractal_max=1.45 vs hurst_min=0.57 / fractal_max=1.55). This inflates the apparent solution space and wastes evolution compute on redundant configurations.

5. **Low trade count in strict filter stacks.** The combined filter (Hurst + ADX + EMA + slope + volume + regime + proposed additions) is extremely selective. In 15-30 day walk-forward folds, the expected trade count may fall below the MIN_TRADES_FOR_POSITIVE_FITNESS=30 threshold, systematically penalizing high-selectivity configurations that may actually be robust. The evolution will bias toward weaker, more frequent signals to avoid the low_trade_penalty — perversely selecting for lower signal quality.

6. **Regime label dependency and hidden leakage.** The strategy fully gates on an external regime label (`_regime` column). If this regime classifier has any forward-looking features or is calibrated on the full training set (e.g., uses future data in its lookback windows), chaos_trend inherits that leakage. The regime labels must be verified as strictly computed using only data available at each bar before trust can be placed in any backtest results.

7. **Time stop and TP interaction overfitting.** `time_stop_hours` and `take_profit_atr_mult` are tightly coupled in determining trade outcome. A genome can achieve a high hit rate by setting time_stop_hours very short (24h) and take_profit_atr_mult very tight (3 ATR) — capturing small quick profits — but this configuration will fail OOS when market conditions change. Enforce joint constraint testing: ensure avg profitable trade exceeds 2× avg losing trade in all walk-forward folds.

8. **Volume ratio normalization period mismatch.** `volume_ratio` normalizes against a 20-bar SMA but `volume_zscore` uses a 50-bar window. Sweeping `volume_threshold` calibrates on the 20-bar norm; adding `volume_zscore` as an alternative filter without understanding this mismatch may create inconsistent behavior across different market regimes.

9. **Cost sensitivity at low turnover.** With 5-15 trades/month, round-trip commission + slippage per trade becomes a significant fraction of avg trade PnL. At 10 bps commission + 5 bps slippage each way (30 bps total), a 100-bar Hurst window strategy entering infrequently must generate > 30 bps avg trade PnL just to break even. Verify cost sensitivity explicitly before promoting any genome to candidate status.

10. **Single-position constraint limits recovery.** The strategy returns None when a position is already open (`if position is not None: return None`). In a strong trending regime, this prevents adding to a winner or re-entering after a time stop exits a profitable trend early. For long-only spot, a partial pyramid-in mechanism on confirmed H signals during an open position could substantially improve CAGR — worth exploring as a Phase-4 enhancement.

---

## Classification: CANDIDATE

The strategy's core logic — Hurst-gated trend following with EMA + slope confirmation — is economically sound and meaningfully differentiated from simple MA crossovers. The regime persistence filter has genuine theoretical grounding. The short branch is trivially removable. The main risks (estimation noise, cache bug, filter stack sparsity) are identifiable and fixable before the OOS lockbox is opened. With the short branch dropped, min_momentum_strength activated, the cache bug fixed, and RSI/BMSB/OBV filters added, chaos_trend is a viable Phase-3 candidate deserving a full sweep on pre-lockbox data.
