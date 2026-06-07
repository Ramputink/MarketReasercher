# Donchian Breakout — Phase 3 Spot Long-Only Research Report

## 1. Diagnosis

**Current Logic:** The strategy (`/strategies/donchian_breakout.py`) implements Turtle Trading Donchian channel breakout logic. At each bar it computes a rolling `entry_period`-bar high and low over the prior window. Entry conditions (both directions) require:
- ADX-14 >= `adx_min` (trend strength gate)
- `volume_ratio` >= `volume_surge_threshold` if `require_volume_confirmation=True`
- Channel width >= `atr_filter_mult * atr_14` if `use_volatility_filter=True`
- Regime in `allowed_regimes` if `require_breakout_regime=True`

Long entry: `close > channel_high` → Signal(side="long", strength=min((close-channel_high)/atr, 1.0), stop=close-2.66*atr, tp=close+7.85*atr, time_stop=48h)

Short entry: `close < channel_low` → Signal(side="short", ...) — **this branch exists and is active**.

**Does it emit shorts?** Yes. Lines 79-90 explicitly produce `side="short"` signals on downside channel breaks. This must be removed entirely for Phase 3.

**Exit logic bug:** Lines 35-36 (`if position is not None: return None`) unconditionally skip signal generation when a position is open. This means the `exit_period` channel is never actually checked — the strategy relies 100% on the backtester's stop/TP/time-stop exits. `exit_period` is currently a dead parameter.

**Evolved state:** Gen 144, 2739 GA evaluations, Sharpe=2.79 (WF=2.99), 132 trades. The evolved params (entry_period=40, exit_period=14, adx_min=27.6, volume_surge_threshold=1.50, stop_loss_atr_mult=2.66, take_profit_atr_mult=7.85) reflect optimization with both long AND short branches active. These params are invalidated for a long-only configuration and must be re-evolved.

**Likely edge:** Donchian breakouts on crypto legitimately capture momentum regime initiations — the first bar to close above an N-bar high in a trending market often precedes continued momentum. The ADX and volume gates reduce false breakouts. The wide TP (7.85x ATR) is appropriate for the positive-skew, fat-tail nature of crypto returns.

**Likely weakness:** The short side historically provides a substantial performance buffer during bear markets and ranging periods. Removing it eliminates this hedge. The strategy will underperform in choppy/bear markets unless a strong regime gate (e.g., BMSB) compensates.

---

## 2. Spot Long-Only Edge Hypothesis

Crypto spot assets exhibit persistent positive skew: the upper tail of returns is larger than the lower tail over multi-year horizons. A long-only Donchian breakout exploits this skew by entering at the precise moment that price exceeds a historical high — the opening of a momentum regime — and holding until a volatility-scaled stop or take-profit resolves the trade.

**Core hypothesis:** After a period of volatility compression (low BB bandwidth percentile), a breakout above the N-bar Donchian high accompanied by trend strength (ADX > threshold), volume surge, and bull-market macro conditions (BMSB above both lines) signals the beginning of a sustained directional move with above-average continuation probability. The strategy stays in cash (long-or-flat) otherwise, effectively timing entries to high-conviction moments and avoiding the negative compounding of short exposure on a spot asset where cost-of-carry, funding, and borrowing do not apply.

Expected profile: hit rate 40-55%, average win/loss ratio > 2.5x, positive skew, CAGR driven by a small number of large wins in genuine momentum regimes.

---

## 3. Market Techniques

1. **Donchian channel breakout** (N-bar high expansion) — primary entry trigger
2. **ATR-normalized stops and take-profit** — adapts risk-per-trade to realized volatility
3. **ADX trend-strength filter** — rejects false breakouts in ranging/oscillating markets
4. **Volume surge confirmation** — validates institutional participation at the breakout bar
5. **BMSB regime gate** — restricts long entries to macro bull phases (price above 20-SMA and 21-EMA on the weekly-equivalent band)
6. **BB bandwidth percentile pre-filter** — detects volatility compression before breakout (squeeze-then-expand pattern)
7. **RSI exhaustion filter** — avoids overbought entries that immediately mean-revert after breakout
8. **Close location confirmation** — ensures breakout bar shows intrabar conviction (closed near the high)
9. **Time stop** — flushes stalled positions and recycles capital
10. **Walk-forward expanding validation** on pre-lockbox data with 120-bar embargo

---

## 4. Useful Features

| Category | Feature | Usage |
|---|---|---|
| Trend | `adx_14` | Primary gate: ADX < threshold → reject |
| Trend | `bmsb_bullish` | Hard bull-market gate: False → stay flat |
| Trend | `bmsb_spread` | BMSB health; large positive spread = early-bull strength |
| Volatility | `atr_14` | Stop and TP anchor |
| Volatility | `atr_7` | Faster vol for detecting regime shifts |
| Volatility | `bb_bw_percentile` | Compression pre-filter (low = squeezed, breakout imminent) |
| Volatility | `bb_bandwidth` | Raw bandwidth for channel-width filter |
| Volatility | `gk_vol` | Cleaner realized vol estimator for position sizing |
| Volume | `volume_ratio` | Primary volume surge confirmation |
| Volume | `volume_zscore` | Stricter spike filter vs. 50-bar mean |
| Volume | `obv_divergence` | OBV vs. 20-SMA; positive divergence validates breakout |
| Momentum | `rsi_14` | Exhaustion filter: RSI > 75 → reject entry |
| Momentum | `roc_12` | Secondary momentum confirmation |
| Momentum | `dist_from_high_20` | Proximity to breakout level |
| Price/Return | `ret_zscore_20` | Avoid entries after extreme single-bar spikes |
| Price/Return | `bb_pct_b` | Near 1.0 confirms upper Bollinger band breakout |
| Microstructure | `buying_pressure` | Demand confirmation at breakout bar |
| Microstructure | `pressure_imbalance` | Net buying vs. selling within the breakout bar |
| Microstructure | `close_location` | Close near bar high (>= 0.6) = conviction |

Funding/sentiment (if available externally) may ONLY be used as an informational regime filter (e.g., elevated funding rate = crowded long = reduce size or require higher ADX) — never as a direct PnL source or short trigger.

---

## 5. Genome Parameters (Sweep)

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `entry_period` | int | 15 | 60 | Wider range captures both medium and long-term breakouts |
| `exit_period` | int | 5 | 20 | Currently dead (bug); fix exit logic then sweep |
| `adx_min` | float | 15.0 | 40.0 | Expand upper bound — evolved value 27.6 was near the 30.0 ceiling |
| `volume_surge_threshold` | float | 1.0 | 2.5 | Hard floor at 1.0 (at least average volume) |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | Tighten lower end vs. current 1.0 in param_space |
| `take_profit_atr_mult` | float | 3.0 | 10.0 | Expand upper bound for large-trend capture |
| `time_stop_hours` | int | 24 | 120 | Add to sweep — currently fixed at 48 |
| `use_volatility_filter` | bool | 0 | 1 | Keep as genome param |
| `atr_filter_mult` | float | 0.2 | 2.0 | Active only when use_volatility_filter=True |
| `bmsb_gate` | bool | 0 | 1 | NEW: require bmsb_bullish=True for entry |
| `rsi_max_entry` | float | 65.0 | 85.0 | NEW: RSI exhaustion ceiling |
| `bb_compression_pctile_max` | float | 20.0 | 60.0 | NEW: require volatility compression before entry |
| `close_location_min` | float | 0.4 | 0.8 | NEW: minimum intrabar close location at breakout bar |
| `require_positive_pressure` | bool | 0 | 1 | NEW: require pressure_imbalance > 0 |

---

## 6. Params NOT to Optimize

- **`require_volume_confirmation`** — fix True; removing volume confirmation systematically reduces signal quality and looks good in-sample only
- **`require_breakout_regime`** — fix True for long-only; the regime gate is the primary structural protection
- **`allowed_regimes`** — fix to `["breakout", "trend"]` (or add "bull" if regime classifier supports it); do not sweep regime labels
- **`time_stop_hours` relative to `entry_period`** — do not jointly constrain; sweep independently
- **`funding_bps_per_8h`** — not a strategy param; must be a fixed realistic assumption in backtester config, never tuned

---

## 7. Long-Only Risk Management

**Stop-loss:** Hard ATR-anchored stop at `entry_price - stop_loss_atr_mult * atr_14`, checked intra-bar against bar low. Sweep range 1.5-4.0x. Stops must be set at signal time and not subsequently loosened.

**Take-profit:** ATR-anchored at `entry_price + take_profit_atr_mult * atr_14`, checked intra-bar against bar high. Range 3.0-10.0x. The wide TP is intentional — breakout strategies derive P&L from a small number of large wins.

**Trailing stop (recommended addition):** After the position is profitable by >= 2x ATR, trail the stop at `highest_bar_close_since_entry - 1.5 * atr_14`. Implement as a genome bool `use_trailing_stop`. This protects gains on large momentum runs without the rigidity of a fixed TP.

**Volatility targeting:** Use `sizing_method=volatility_scaled` with `vol_target_annualized=0.25`. Size = min(equity * 0.25 / annual_vol, equity * 0.30). This normalizes risk-per-trade across high- and low-vol regimes.

**Max position size:** Hard cap at 30% of equity per trade. No leverage. Single open position at a time.

**Cooldown:** 48-hour cooldown after stop-loss exits before the next entry is permitted. After time-stop exits (not a loss), 12-hour cooldown. Implemented via `risk_manager.min_time_between_trades`.

**Risk-off gate:** Hard rules (any one triggers flat):
1. `bmsb_bullish = False` (price below BMSB)
2. `regime not in allowed_regimes`
3. Equity drawdown from peak > 20% → halt all new entries for remainder of simulation segment (circuit breaker, already in backtester's RiskManager)

---

## 8. Metrics

| Metric | Target | Notes |
|---|---|---|
| CAGR | > 20% net | On long-only equity curve |
| Sharpe | >= 1.5 | Annualized, hourly curve |
| Sortino | >= 2.0 | Penalizes downside only; breakout strategies should excel here |
| Max Drawdown | < 25% | Full-sample peak-to-trough; hard fail criterion |
| Calmar | > 1.0 | CAGR / maxDD |
| Hit rate | 40-55% | Lower than mean-reversion but with high R:R |
| Avg Win / Avg Loss | > 2.5x | Critical for breakout viability |
| Trades/month | 4-10 | Sufficient sample without excessive cost drag |
| Market exposure | 15-35% | Selective long-only strategy |
| Avg holding hours | Diagnostic | Cross-check vs. time_stop_hours |
| Cost sensitivity | Sharpe degrades < 30% at 2x costs | Hard realism check |
| WF OOS Sharpe | >= 1.0 mean across folds | Pre-lockbox validation |
| OOS degradation | < 40% | (train_sharpe - oos_sharpe) / train_sharpe |
| Per-year Sharpe | No year < 0 | Stability across regimes |
| Bootstrap 95th lower | > 0.5 | 1000 block-bootstrap resamples |
| Profit factor | > 1.5 net | Gross profit / gross loss |

---

## 9. Concrete Changes to Implement

1. **Remove the short branch** (lines 79-90): delete the `elif close < channel_low` block. Return None (stay flat) on downside channel breaks.

2. **Fix the exit logic bug**: when `position is not None`, compute the rolling `exit_period`-bar low and if `close < exit_channel_low`, return a Signal that triggers `signal_exit`. Remove the unconditional `return None` on line 36.

3. **Fix strength guard**: change `min((close - channel_high) / atr, 1.0)` to `min(max((close - channel_high) / atr, 0.0), 1.0)`.

4. **Add BMSB gate**: if `PARAMS["bmsb_gate"]` is True, check `current.get("bmsb_bullish", True)` and return None if False.

5. **Add RSI exhaustion filter**: if `current.get("rsi_14", 50) > PARAMS["rsi_max_entry"]`, return None.

6. **Add close location confirmation**: if `current.get("close_location", 1.0) < PARAMS["close_location_min"]`, return None.

7. **Add pressure imbalance check**: if `PARAMS["require_positive_pressure"]` and `current.get("pressure_imbalance", 0) <= 0`, return None.

8. **Add BB compression pre-filter**: when `use_volatility_filter=True`, also check `current.get("bb_bw_percentile", 100) < PARAMS["bb_compression_pctile_max"]` before allowing entry.

9. **Update `auto_evolve.py` STRATEGY_REGISTRY** param_space: add new params (`bmsb_gate`, `rsi_max_entry`, `bb_compression_pctile_max`, `close_location_min`, `require_positive_pressure`, `time_stop_hours`); expand `entry_period` to (15, 60); expand `adx_min` to (15.0, 40.0); expand `take_profit_atr_mult` to (3.0, 10.0).

10. **Re-evolve from scratch** on pre-2024-06-07 data only, with the short branch removed. Do not warm-start from the Gen 144 params — they are biased by the short branch.

---

## 10. Methodological Risks

1. **Short branch removal invalidates evolved params**: The published Sharpe=2.79 / WF=2.99 was achieved with both long and short branches. Dropping shorts changes the strategy fundamentally. The long-only variant must be treated as a new strategy for all performance attribution purposes.

2. **Exit logic bug makes exit_period a dead param**: The unconditional `return None` at lines 35-36 means the strategy never generates a signal_exit. All exits occur via stop/TP/time-stop. Fixing this bug changes strategy behavior and requires full re-evaluation.

3. **Thin trade sample after long/short split**: 132 total trades likely splits to ~65-70 longs. After re-evolution, per-fold samples in 15-day test windows may be < 10 trades — insufficient for reliable Sharpe estimation. Walk-forward results will have high variance.

4. **GA contamination risk**: If the GA's fitness evaluation used any data from 2024-06-07 onward (the lockbox), the evolved parameters are contaminated. Must audit the GA data window before using any evolved values as a warm-start.

5. **ADX ceiling problem**: The evolved ADX of 27.6 is near the param_space ceiling of 30.0, suggesting the optimizer was constrained. The true optimum may be > 30. With the range expanded to 40.0 in re-evolution, the solution may shift substantially.

6. **Time-stop dominance**: With only a 48h time stop and 7.85x ATR TP, many trades likely exit via time stop. If time-stop exits dominate, the effective holding period is short and the wide TP is decorative. This creates a brittle dependence on `time_stop_hours` as the primary return driver.

7. **Regime classifier dependency**: The strategy gates on `regime in allowed_regimes`, but the regime classifier is a separate black box. If the classifier was tuned on overlapping data, the combined system has hidden degrees of freedom beyond the visible PARAMS.

8. **Execution slippage at breakout bars**: Donchian breakouts enter at new N-bar highs — structurally momentum-chase entries where real fill quality is adversely selected. The 1-bar entry lag in the backtester helps but may not capture the full adverse selection at breakout moments in real markets.

9. **Bull market concentration**: Long-only Donchian outperforms in trending bull markets and generates flat-to-negative performance in 2022-style bear markets. The 2024-2026 OOS lockbox performance will be heavily path-dependent on macro regime. A strong BMSB gate is the primary mitigation.

10. **Overlapping rolling feature lookbacks**: If `entry_period` is expanded to 60 bars in re-evolution and a future feature (e.g., 100-bar BB percentile) is added, the current 120-bar embargo just barely covers both. Any further expansion of lookbacks requires increasing the embargo accordingly to avoid information leakage across fold boundaries.

---

## Classification: CANDIDATE

Donchian breakout is a **candidate** for Phase 3 long-only spot research. The structural edge — momentum capture at channel high breakouts with trend and volume confirmation — is well-supported by crypto market microstructure and positive-skew return distribution. The short branch removal is a clean surgical change. Key risks are the thin trade sample after removing shorts, the exit logic bug that must be fixed before re-evolution, and the need to re-validate that the regime gate and BMSB filter provide adequate bear-market protection. The strategy is not fragile by nature (the Donchian channel is a robust, interpretable construct), but the current implementation requires meaningful fixes before it can be trusted as a long-only candidate.
