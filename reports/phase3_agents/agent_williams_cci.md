# Williams %R + CCI — Phase-3 Spot Long-Only Research Report

**Strategy file:** `strategies/williams_cci.py`
**Registry status:** PRUNED in Round 2 (1.0% positive fitness rate across 36K evaluations)
**Classification: DOUBTFUL**

---

## 1. Diagnosis

### Current Logic
The strategy is a short-duration (24h) mean-reversion system with a conjunction of five entry gates:

1. **Regime gate:** `regime in ["lateral", "mean_reversion", "trend"]` — the allowed_regimes list is so permissive ("trend" is included) that this filter does almost nothing.
2. **ADX gate:** `adx_14 <= 35` — ensures the market is not in a strong trend, where mean-reversion fails.
3. **Volume spike:** `volume_ratio >= 1.7638` — requires a high-volume bar as a capitulation signal.
4. **BB squeeze:** `bb_bw_percentile <= 30` — requires the bandwidth to be in the lower 30th percentile, indicating pre-expansion compression.
5. **Dual oscillator extreme:** `WR <= -89.5 AND CCI <= -92.6` for long; `WR >= -21.7 AND CCI >= 148.9` for short.

Williams %R and CCI are both computed bar-by-bar inline (not from the prebuilt feature pipeline), which is correct but means their lookback windows are genome-free parameters that must be swept independently of any cached columns.

### Short Signal Presence
**Yes, the strategy emits short signals.** Lines 117-118 in the file set `side = "short"` when WR >= -21.6745 and CCI >= 148.9217. Stop-loss for shorts is set above entry price, take-profit below. The short branch also has its own strength formula (line 128). This entire branch must be removed for Phase-3.

### Evolved Parameters — Overfitting Indicators
The PARAMS block header states "19 trades, $109 PnL" over the evolved window. A Sharpe of 1.59 derived from 19 trades is not statistically meaningful (the standard error of the Sharpe estimate on N=19 is approximately 1/sqrt(19) ≈ 0.23, so the true Sharpe could easily be 0 at 95% confidence). The parameter precision (e.g., `wr_oversold=-89.5301`, four decimal places) is a textbook overfitting signature.

### Likely Edge
The genuine edge hypothesis: in a compressed-volatility environment (BB squeeze), a simultaneous extreme reading in two uncorrelated oscillators (WR detects price position within range; CCI detects deviation from a moving average of typical price) combined with a volume spike identifies a short-term capitulation bottom with a higher-than-random probability of a mean-reversion bounce. This is a legitimate intraday/short-swing microstructure pattern. However, extracting it requires enough trades to be statistically credible, which the current hyper-specific thresholds prevent.

### Likely Weakness
Filter stacking to five simultaneous binary conditions produces near-zero signal frequency. The strategy was pruned from the STRATEGY_REGISTRY precisely because it could not generate enough trades to survive the GA's minimum-30-trades fitness penalty. In Phase-3 long-only, removing the short branch halves signal frequency further.

---

## 2. Spot Long-Only Edge Hypothesis

In spot crypto markets, extreme dual-oscillator oversold conditions (Williams %R near -100 AND CCI near -100) occurring within a Bollinger Band squeeze and coinciding with a volume spike identify high-probability short-term capitulation events. The spot-long-only thesis:

- Buyers step in aggressively during the volume-spike bar while price is already deeply compressed.
- The BB squeeze (low bandwidth percentile) signals that the recent range has been narrow; the volume spike breaks the compression with directional momentum.
- Mean-reversion bounce is expected to the BB midline or an ATR-scaled level within 24-48 hours.
- The **long-or-cash** framing is absolute: when the overbought condition fires (the old short signal), the strategy does nothing and remains in cash.
- A **macro BMSB filter** (price above 20-SMA and 21-EMA) ensures the capitulation is a pullback within a bull structure, not a continuation of a bear trend. This is the most important long-only safety layer.
- A secondary **RSI_7 < 35** filter adds a fast confirming oversold signal from a different calculation method.

The strategy should be considered a **low-frequency, high-conviction mean-reversion** system targeting 1-4 trades per month at most.

---

## 3. Market Techniques

1. **Dual-oscillator capitulation trigger:** Williams %R extreme + CCI extreme — both must agree. WR measures price location within recent high-low range; CCI measures deviation from SMA of typical price. Their conjunction reduces false signals versus using either alone.
2. **Volatility compression gate (BB squeeze percentile):** Bollinger Band bandwidth below the 30th percentile of its 100-bar history signals a compression phase. Post-compression expansions tend to be sharper and more directional, giving mean-reversion bounces cleaner entries.
3. **Volume-spike confirmation:** Above-average volume on the signal bar (volume_ratio > threshold) identifies bars where genuine selling pressure or panic occurred rather than slow, volume-light drift, which is more likely to reverse.
4. **ATR-normalized stop/target:** Avoids fixed-pip stops that become irrelevant in volatility regime shifts. SL set below entry by ATR multiple; TP set above entry by larger ATR multiple (minimum 2:1 RR).
5. **BMSB macro trend filter:** The Bull Market Support Band (20-SMA / 21-EMA) filters out bear-market entries. Only engage long when price is above both MAs, i.e., in a macro uptrend structure.
6. **Time-stop discipline:** 24-48h maximum hold prevents position decay when the bounce fails to materialize. Mean-reversion setups that do not resolve quickly tend to become trending losers.
7. **ADX regime gate:** ADX < adx_max ensures the market is not in a strong directional trend, where mean-reversion consistently fails and oscillator signals give false oversold readings.
8. **Vol-targeting position sizing:** Garman-Klass volatility used to scale position notional so 1-ATR move = constant risk. Reduces size in high-vol bear markets automatically.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Momentum | `williams_r` (inline, wr_period 7-20) | Primary oversold detector |
| Momentum | `cci` (inline, cci_period 10-25) | Confirming oscillator |
| Momentum | `rsi_7` | Fast oversold confirmation |
| Trend | `adx_14` | Low-ADX mean-reversion gate |
| Trend | `bmsb_bullish` | Macro bull-structure filter (long-only safety) |
| Volatility | `bb_bw_percentile` | BB squeeze gate |
| Volatility | `bb_pct_b` | Position within BB; near 0 = oversold |
| Volatility | `gk_vol` | Vol targeting and regime detection |
| Volume | `volume_ratio` | Volume spike confirmation |
| Volume | `volume_zscore` | Normalized volume spike across vol regimes |
| Volume | `obv_divergence` | OBV rising while price falling = accumulation |
| Price | `dist_from_high_20` | Distance from 20-bar high; deep negative confirms oversold |
| Microstructure | `buying_pressure` | Intrabar buying pressure recovering on signal bar |
| Microstructure | `pressure_imbalance` | Pressure turning positive at potential bottom |
| Regime | `_regime` (lateral / mean_reversion) | Categorical regime gate |

Features **NOT** to use as signal: funding rate, perp open interest, any forward-looking label.

---

## 5. Parameters to Sweep (Genome)

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `wr_period` | int | 7 | 20 | WR lookback; shorter = more reactive |
| `wr_oversold` | float | -100.0 | -75.0 | WR entry threshold for long |
| `cci_period` | int | 10 | 25 | CCI lookback |
| `cci_long_threshold` | float | -150.0 | -50.0 | CCI must be below this for long |
| `bb_squeeze_percentile` | float | 15.0 | 50.0 | Max BB BW percentile for squeeze gate |
| `adx_max` | float | 20.0 | 45.0 | Max ADX before suppressing entry |
| `volume_spike_threshold` | float | 1.2 | 3.0 | Min volume_ratio for spike confirmation |
| `stop_loss_atr_mult` | float | 0.8 | 2.5 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 1.5 | 4.0 | ATR multiplier for TP (enforced > SL) |
| `time_stop_hours` | int | 12 | 72 | Max trade duration |
| `require_bb_squeeze` | bool | 0 | 1 | Toggle squeeze filter |
| `require_volume_spike` | bool | 0 | 1 | Toggle volume filter |
| `rsi_oversold_max` | float | 25.0 | 45.0 | Max RSI_7 at entry |
| `require_bmsb_bullish` | bool | 0 | 1 | Toggle BMSB macro filter |

---

## 6. Parameters NOT to Optimize

- **`wr_overbought`** — the entire short branch is dropped in Phase-3; this parameter is unused and should be removed from PARAMS.
- **`cci_short_threshold`** — same reason; the short signal does not exist in the long-only version.
- **`allowed_regimes`** — hardcode to `["lateral", "mean_reversion"]` (drop "trend" which contradicts the ADX filter); fixing this avoids regime-label snooping in the sweep.
- **`require_trend_regime`** — keep permanently True; toggling it off removes a principled safeguard and would invite overfitting to trending periods where mean-reversion appears to work by luck.
- **The exact threshold decimals of the evolved PARAMS** (wr_oversold=-89.5301, cci_long_threshold=-92.6157, etc.) — treat these as the interior of the sweep range, not as fixed truths. Using them as initialization seeds is fine; locking them is overfitting.

---

## 7. Long-Only Risk Management

### Stop Loss
Hard ATR stop: `entry_price - stop_loss_atr_mult * atr_14`. The evolved 1.32x ATR is a reasonable starting point. Stop is intrabar-checked against the low of each bar in the backtester (if `low <= stop_loss` → exit at stop_loss price). Sweep 0.8-2.5x ATR.

### Take Profit
ATR-based TP: `entry_price + take_profit_atr_mult * atr_14`. Sweep 1.5-4.0x ATR, enforcing TP > SL. Alternative: use the `bb_middle` column as a dynamic TP anchor (the mean-reversion target is the BB midline by construction). This avoids an extra parameter.

### Trailing Stop
Not recommended. The strategy's mean-reversion logic implies a single bounce target; trailing adds parameter complexity that overfits given thin trade counts. If implemented, activate only after unrealized gain exceeds 1x ATR (i.e., protect at breakeven once 1 ATR profitable).

### Volatility Targeting
Apply GK-vol targeting for position sizing: `size = min(equity * vol_target / gk_vol, equity * max_position_pct)`. This automatically reduces notional in high-volatility bear-market environments where mean-reversion fails most often. Target annualized vol = 20%.

### Maximum Position Size
Cap at 20% of equity (BacktestConfig.max_position_pct = 0.20). Vol-targeting will typically produce smaller notionals in normal conditions.

### Cooldown
Minimum 12-hour cooldown after any trade exit. Prevents re-entry during a continuing slide and reduces turnover-driven cost drag.

### Risk-Off Regime Gate
Gate all long entries on `bmsb_bullish = True` (price above both 20-SMA and 21-EMA). When `bmsb_bullish` is False, the market is in a structurally bearish posture and mean-reversion longs frequently become knife-catches. Additionally: `adx_14 <= adx_max` (already in code) and `_regime in ["lateral", "mean_reversion"]` (hardcode).

---

## 8. Relevant Metrics

| Metric | Target / Notes |
|---|---|
| CAGR | > 15% annualized over OOS (2024-06-07 → 2026-06-07) |
| Sharpe (WF-OOS) | >= 1.0; evolved IS Sharpe 1.59 must survive OOS. Standard error ~0.23 on N=19 means this is barely distinguishable from 1.0 |
| Sortino ratio | Report alongside Sharpe; downside-only denominator matters for a strategy with hard stops |
| Max drawdown % | Full-sample peak-to-trough; ruin check (equity never <= 0) |
| Calmar ratio | CAGR / maxDD; target > 0.5 |
| Hit rate | Expected > 55%; mean-reversion should win more often than lose given 2:1 RR target |
| Number of trades | **Minimum 30 per year (>= 60 total over OOS)** to be statistically credible; 19 evolved trades is the core problem |
| Average trade net PnL | Must be positive after full cost accounting (commission + slippage both legs) |
| Cost sensitivity | Double slippage_bps and re-run; alpha should survive given ATR-based TP is wide enough |
| OOS degradation | WF-OOS Sharpe vs IS Sharpe degradation < 30% |
| Per-year Sharpe stability | Run on each calendar year in 2019-2023 pre-lockbox; no single year should drive all returns |
| Bootstrap / MC ruin probability | < 5% across 1000 Monte Carlo simulations of trade sequence |
| Profit factor | > 1.3 OOS (gross wins / gross losses) |
| DSR | Deflated Sharpe Ratio >= 0.95 deflated by cumulative trial count |
| Exposure fraction | % of time in position; expected very low (< 15%); ensure cost of carry does not exceed alpha |

---

## 9. Concrete Sweep Changes

1. **Re-add `williams_cci` to `STRATEGY_REGISTRY`** in `auto_evolve.py` with the revised long-only `param_space` (14 params listed in Section 5). Remove `wr_overbought` and `cci_short_threshold` entirely from the registry entry.

2. **Remove the short branch from `williams_cci_strategy`**: Delete lines 117-118 (`elif wr >= p["wr_overbought"] and cci >= p["cci_short_threshold"]: side = "short"`) and lines 127-128 (short strength formula) and lines 137-138 (short stop/TP calculation). Only `side = "long"` can ever be returned.

3. **Add RSI_7 oversold filter** (line ~115, after CCI check): `if float(current.get("rsi_7", 100)) > p["rsi_oversold_max"]: return None`. Controlled by new param `rsi_oversold_max` (sweep 25-45).

4. **Add BMSB macro filter** (line ~116): `if p.get("require_bmsb_bullish") and not bool(current.get("bmsb_bullish", False)): return None`.

5. **Fix `allowed_regimes` to exclude "trend"**: Hardcode `allowed_regimes = ["lateral", "mean_reversion"]` in PARAMS; remove it from the sweep.

6. **Widen all threshold ranges** in STRATEGY_REGISTRY: `wr_oversold` from fixed -89.5 to sweep [-100.0, -75.0]; `cci_long_threshold` from fixed -92.6 to [-150.0, -50.0]; `volume_spike_threshold` from fixed 1.76 to [1.2, 3.0]; `bb_squeeze_percentile` from fixed 30.0 to [15.0, 50.0].

7. **Remove OBV divergence from main gate** (it is not currently used but was considered); keep as a metadata annotation only to avoid adding a correlated filter that worsens trade frequency.

8. **Enforce TP > SL constraint** in `random_genome` and `mutate_genome` (already enforced by the GA engine globally, but add an assertion in the strategy file's `__init__` guard at bar_idx check to catch misconfigured PARAMS).

9. **Run pre-lockbox sweep on 2019-01-01 → 2024-06-07 only**; the 2024-06-07 → 2026-06-07 lockbox is sealed throughout all sweeps. Walk-forward folds use the embargo_bars=120 gap (already configured).

10. **Raise `MIN_TRADES_FOR_POSITIVE_FITNESS`** to 40 for `williams_cci` specifically in the fitness function, or add a per-strategy trade-count penalty multiplier, to counteract the strategy's tendency to produce few trades.

---

## 10. Methodological Risks

1. **Extreme filter stacking / near-zero signal frequency:** Five simultaneous binary conditions produce very few qualifying bars. 19 trades over the evolution window yields a Sharpe standard error of ~0.23, making the IS Sharpe of 1.59 statistically indistinguishable from 1.0. In long-only mode (short branch dropped), signal frequency halves again. The strategy may simply not trade enough to be useful.

2. **Evolved threshold precision = overfitting signature:** Four-decimal-place thresholds (e.g., `wr_oversold=-89.5301`) on a 19-trade sample are a textbook in-sample fit to noise. These values must be widened in the sweep, not trusted as discovered parameters.

3. **Williams %R / CCI computed inline (not from feature pipeline):** The strategy recomputes WR and CCI bar-by-bar using raw `df["high"].iloc[start:bar_idx+1].values` slices. This is correct (no leakage) but means the genome's `wr_period` and `cci_period` are independent of any prebuilt pipeline cache. This is fine but adds per-bar computational cost and makes the code harder to vectorize.

4. **`bb_bw_percentile` 100-bar rolling window — embargo gap requirement:** The feature requires the WalkForwardValidator embargo to remain >= 120 bars (100-bar feature lookback + 20-bar safety margin). The engine already enforces `embargo_bars=120`. Any future feature addition with a longer window must raise this value, or leakage reappears.

5. **Regime label must be leak-free:** The `_regime` column used in the strategy gate must be computed without future data. Verify the regime classifier processes only past bars and that the OOS regime labels are produced identically to IS labels. If the regime classifier is trained on IS data and applied to OOS, this is not leakage; if it is retrained on OOS data, it is.

6. **Bitcoin/crypto bull-bias survivorship:** Mean-reversion longs from 2019-2023 disproportionately benefited from a structurally upward-trending market. The OOS lockbox (2024-2026) includes a more mature cycle with different volatility and structural regime. The BMSB filter partially mitigates knife-catch risk but does not remove the question of whether the edge survives regime change.

7. **Cross-asset generalization was a failure in Round 2:** The `cross_asset_xrp_only_penalty` was triggered, suggesting the strategy may have been overfit to the primary asset (XRP or BTC based on trading environment) and not generalized across ETH, SOL, etc. The long-only version should be validated cross-asset during in-sample sweep.

8. **Short removal impact on P&L attribution:** It is unknown from the evolved PARAMS whether the $109 PnL came predominantly from the 19 long or short trades. If the short side drove most of the gain, removing it could eliminate the apparent edge entirely. The long-only backtest should be run explicitly on the evolved PARAMS (short branch disabled) before investing sweep compute.

9. **Monte Carlo ruin risk on thin trade count:** With ~19-30 trades and a tight 24h time stop, a streak of 5-6 consecutive losers at 1.32x ATR SL can produce a 10-15% equity drawdown. Vol-targeting sizing is essential to prevent compounding of consecutive losers. The MC module in the engine is already wired (activated for fitness > 1.5, >= 15 trades).

10. **Tight time stop (24h) creates execution-timing sensitivity:** Whether the bounce occurs within 24 hours is partly a function of the bar timeframe. On hourly bars, 24h = 24 bars; on 4h bars, 24h = 6 bars. The time_stop_hours parameter is in hours (correctly) but the strategy should be tested on consistent timeframe data. A 24h stop on 4h bars gives very few bars for the position to resolve.

---

## Classification: DOUBTFUL

**Reason:** The strategy has a plausible and genuine mean-reversion logic for the long-only branch, but three structural problems make it doubtful for Phase-3:

1. The evolved version traded only 19 times — statistically insufficient to claim any edge.
2. It was already pruned from STRATEGY_REGISTRY (Round 2) for 1.0% positive fitness rate across 36K evaluations with both long AND short branches active.
3. Removing the short branch halves signal frequency in an already under-trading system.

**Path to candidacy:** Widen the oscillator thresholds, reduce filter stacking (make BB squeeze and volume spike toggleable, let evolution decide), add BMSB macro filter for long-only safety, and require >= 40 trades in any sweep window before accepting the genome as valid. If the strategy can achieve 40+ trades per in-sample period with a WF-OOS Sharpe >= 1.0, it deserves reconsideration as a genuine candidate. Without meeting the trade-count bar, it remains doubtful.
