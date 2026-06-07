# KAMA Trend — Phase 3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy mechanics:** `kama_trend_strategy` computes a Kaufman Adaptive Moving Average (KAMA) over a trailing 200-bar window at every evaluated bar. KAMA's core innovation is its Efficiency Ratio (ER = net direction / sum of bar-to-bar distances), which scales the smoothing constant between a fast alpha (2/(fast_sc+1)) and a slow alpha (2/(slow_sc+1)). When ER is high (price moving cleanly in one direction), KAMA tracks price closely. When ER is low (choppy), KAMA is nearly stationary.

Entry logic uses two sub-cases:
- **Flip entry:** KAMA direction changed from falling to rising (kama_rising_now AND NOT kama_rising_prev), price is above KAMA, and slope over signal_period > threshold.
- **Crossover entry:** slope > 2x threshold, AND the previous bar's close was below KAMA while current close is above (bullish crossover).

The mirror of both conditions for `side="short"` is present in lines 125–131.

**Does it emit shorts? YES.** Lines 125–131 explicitly set `side = "short"` when KAMA turns down, price crosses below, and slope < -threshold. This must be removed for Phase 3.

**Risk management:** ATR-based stop (2.28x ATR) and take-profit (3.18x ATR), plus a 48-hour time stop. Only one position open at a time; no re-entry while in position.

**Gatekeeping:** ADX >= 16.7 (evolved), regime in ["trend", "breakout", "lateral"] (essentially all non-bear regimes), optional volume_ratio filter (currently disabled).

**Evolved stats:** IS Sharpe 3.32, WF Sharpe 4.25, **only 18 trades** — this is the dominant concern. With 4,211 genetic evaluations selecting on 18 trades, the reported performance is very likely an artifact of lucky sequence selection rather than genuine edge discovery.

**Edge hypothesis (gross level):** KAMA's adaptive smoothing is conceptually sound — it avoids the lag of a fixed MA while filtering noise better than a fast EMA. The dual-condition entry (flip OR strong crossover) is selective and reduces whipsaws relative to a pure crossover system. The weakness is a near-complete absence of secondary confirmation and a very short IS trade history.

---

## 2. Spot Long-Only Edge Hypothesis

When KAMA's efficiency ratio is elevated for multiple consecutive bars, price is in a regime of genuine directional momentum rather than noise. In crypto spot markets (BTC/ETH), such trending phases arrive episodically but reliably (halving cycles, macro risk-on windows, ETF-inflow periods) and tend to persist for days to weeks once established.

**Hypothesis:** Enter long when KAMA transitions from flat/declining to rising (ER begins climbing, adaptive smoothing accelerates) AND spot price is trading above KAMA. Exit to cash (not short) when KAMA flattens or reverses, or when stop/TP/time-stop fires.

This captures the initiation phase of trending moves, not their overbought extension. It is naturally long-or-cash because the short branch (which previously exploited KAMA decline) is simply dropped — the strategy has no way to profit from downside, so it waits in cash during bear moves. The residual edge in long-only mode comes from KAMA's ability to identify low-noise momentum windows that have above-average continuation probability.

---

## 3. Market Techniques

- **Adaptive moving average trend-following (KAMA/ER):** core mechanism, retain and refine
- **ATR-normalized stop-loss and take-profit:** ensures consistent per-trade risk in volatility-adjusted units
- **1-bar entry lag (next-bar open fill):** already enforced by backtester; critical for avoiding same-bar look-ahead
- **ADX-based regime filter:** retains entries in directional markets, discards noise; complement with BMSB macro gate
- **Volatility-scaled position sizing:** normalize exposure by realized vol so position risk is consistent across regimes
- **Walk-forward expanding-window validation:** minimum 5 folds, each ≥ 15-day OOS window, with 120-bar embargo
- **Bootstrap / Monte Carlo on trade sequences:** with historically low trade counts, bootstrap CI on Sharpe is mandatory before trusting any result
- **Post-stop cooldown period:** 4-bar pause after stop-loss exit prevents immediate re-entry into whipsaws
- **BMSB macro regime gate:** binary filter (price above 20W SMA + 21W EMA at proper hourly-scaled periods) to suppress entries in macro bear environments

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Trend | `adx_14` | Already in use; primary trend-strength gate; retain |
| Trend | `bmsb_bullish` | Macro bull regime filter (NOTE: periods must be scaled to hourly, ~3360/3528 bars) |
| Trend | `roc_12` | Secondary momentum confirmation of KAMA direction |
| Momentum | `rsi_14` | Upper gate to avoid buying overbought tops (e.g., RSI > 75) |
| Momentum | `price_acceleration` | Positive second derivative confirms momentum is building |
| Volatility | `atr_14` | Already in use for stop/TP; also position sizing denominator |
| Volatility | `bb_bandwidth` | Low bandwidth = compression; rising bandwidth post-compression aligns with KAMA flip |
| Volatility | `bb_bw_percentile` | Percentile rank; entering after extreme compression (<25th pctile) is a valid breakout filter |
| Volatility | `gk_vol` | Cross-check for extreme-volatility day detection; reduce size when gk_vol spikes |
| Volume | `volume_ratio` | Relative volume vs 20-bar SMA; trend entries should have above-average volume |
| Volume | `volume_zscore` | Spike detection; very high z (> 2) on entry bar adds conviction |
| Volume | `obv_divergence` | OBV above its 20-bar SMA signals accumulation backing the trend |
| Price/return | `ret_zscore_20` | Avoid entries on parabolic spikes (z > 3); mean-reversion risk |
| Market regime | `bb_pct_b` | Entry above 0.5 confirms upside momentum within the band |
| Drawdown filter | `dist_from_high_20` | Entry within 5% of 20-bar high confirms trend strength; avoid entries in recovery rallies from deep holes |
| Microstructure | `buying_pressure` | Bar-level buyer dominance (close near high) on the signal bar |
| Microstructure | `pressure_imbalance` | 10-bar cumulative buy/sell imbalance as a trend confirmation filter |

**Funding/sentiment note:** Funding rate data (if available as a feature) should be used ONLY as an informative filter — e.g., suppress entries when funding is extremely negative (shorts getting paid, extreme fear) as a potential tail-risk indicator. It must never be used as a direct PnL source.

---

## 5. Parameters That Should Enter the Sweep / Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `er_period` | int | 8 | 25 | Core KAMA window; extend ceiling from 20 to 25 |
| `fast_sc` | int | 2 | 6 | Fast smoothing constant; wider upper bound |
| `slow_sc` | int | 20 | 40 | Slow smoothing constant; do NOT co-sweep with fast_sc in early rounds |
| `signal_period` | int | 3 | 10 | KAMA slope measurement window |
| `slope_threshold_pct` | float | 0.02 | 0.20 | Minimum slope for trend qualification |
| `adx_min` | float | 10.0 | 30.0 | ADX floor; allow stricter trend filter |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 2.0 | 6.0 | ATR multiplier for TP; enforce TP > SL always |
| `time_stop_hours` | int | 24 | 120 | Currently fixed at 48; add to genome |
| `rsi_max_entry` | float | 65.0 | 85.0 | NEW: upper RSI gate, avoid buying overbought |
| `bb_bw_pctile_min` | float | 0.0 | 40.0 | NEW: compression filter at entry; 0 disables it |
| `volume_threshold` | float | 0.6 | 1.5 | Only sweep when require_volume is fixed True |

---

## 6. Parameters That Should NOT Be Optimized

- **`require_trend_regime` (bool) + `allowed_regimes` (list):** The regime label itself is a heuristic output of a separate classifier. Optimizing which regimes to include on 18 trades = selecting the regime label combination that happened to align with the 18 lucky trades. Remove both from the genome. Replace with the `bmsb_bullish` binary gate.
- **`require_volume` (bool):** A boolean gate that when co-swept with `volume_threshold` creates a 2-param interaction on a tiny trade sample. Fix `require_volume=True`, sweep only `volume_threshold`.
- **`fast_sc` and `slow_sc` simultaneously:** They interact nonlinearly through the KAMA smoothing equation. Co-sweeping both on the existing trade count is a textbook overfitting trap. Fix `slow_sc=30` as the default in early rounds; only unlock it in later rounds once trade count exceeds 50.

---

## 7. Long-Only Risk Management

**Stop-loss:** `entry_price - stop_loss_atr_mult * atr_14`. Enforce floor of 1.5x ATR to prevent stops so tight they are triggered by normal intrabar volatility. Set at signal time; held fixed until hit, TP fires, or time stop fires.

**Take-profit:** `entry_price + take_profit_atr_mult * atr_14`. Enforce TP/SL ratio > 1.3 at parameter level (raise error if take_profit_atr_mult / stop_loss_atr_mult < 1.3). Current evolved ratio is 3.18/2.28 = 1.4 — acceptable.

**Trailing stop (optional):** Once a trade is 1.5x ATR in profit, ratchet the stop up to entry_price + 1x ATR (lock in partial profit). Continue trailing at 1.5x ATR below rolling high. This converts time stops on profitable trending trades into profit-protection stops.

**Volatility targeting:** Target 15% annualized portfolio volatility. Position size = equity * (0.15 / annualized_vol), where annualized_vol = (atr_14 / close) * sqrt(365 * 24). Cap at `max_position_pct = 0.80`.

**Max position size:** 80% of equity. The remaining 20% cash buffer prevents a single trade from being fully all-in and provides slippage cushion.

**Cooldown:** 4-bar minimum cooldown after a stop-loss exit. If two consecutive stop-losses occur, extend cooldown to 12 bars. This prevents chasing in whipsaw conditions.

**Risk-off macro gate:** Suppress all new long entries when `bmsb_bullish = False` (price below 20W SMA or 21W EMA). This is the primary bear-market filter. No position is opened; any existing open position exits on its own stop/TP/time-stop; no new entry is queued. NOTE: the `bmsb_sma` and `bmsb_ema` periods in `features.py` are currently 20 and 21 bars — on hourly data this is 20 hours, not 20 weeks. The periods must be corrected to 3360 (20 weeks * 168 hours) and 3528 (21 weeks * 168 hours) for the BMSB feature to function as intended.

---

## 8. Relevant Metrics

| Metric | Target / Use |
|---|---|
| CAGR | Annualized compound return on OOS (2024-06 to 2026-06); target > 20% |
| Sharpe ratio | Annualized, RF = 0; target > 1.2 OOS; reject < 0.8 |
| Sortino ratio | Downside-only deviation; more appropriate for long-only; target > 1.5 |
| Max drawdown | Full-sample continuous run; hard cap 25% for spot long-only |
| Calmar ratio | CAGR / max drawdown; target > 1.0 |
| Hit rate | % winning trades; with 1.4 reward:risk, acceptable down to 40% |
| Profit factor | Gross wins / gross losses; target > 1.5 |
| Trade count | OOS minimum 30 trades; reject any config below this threshold |
| Avg trade duration | Hours held on average; detect whether time_stop dominates |
| Turnover | Round-trips per month; monitor cost sensitivity |
| Exposure ratio | Fraction of bars in-position; track separately from return |
| Cost sensitivity | Rerun OOS with 2x commission+slippage; confirm Sharpe degrades < 30% |
| Bootstrap Sharpe CI | 1000-resample CI on OOS trade PnL; reject if 5th pctile Sharpe < 0.5 |
| Per-year Sharpe | 2024, 2025, 2026 (partial) — confirm no single-year dominance |
| WF OOS degradation | OOS Sharpe / IS Sharpe per fold; target ratio > 0.6 |
| Max consecutive losses | Operational risk indicator; flag if > 5 consecutive losses |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove short branch:** Delete lines 125–131 in `kama_trend.py`. Replace with `# Phase 3: short signals dropped — exit to cash`. The `elif price_below` block must not exist. Update `Signal` return to only emit `side="long"`.

2. **Add RSI upper gate:** After the ADX check block, insert: `if float(current.get("rsi_14", 50)) > p["rsi_max_entry"]: return None`. Add `"rsi_max_entry": 75` to PARAMS. Add `"rsi_max_entry": ("float", 65.0, 85.0)` to STRATEGY_REGISTRY param_space.

3. **Add BMSB macro gate:** After the regime check, insert: `if not bool(current.get("bmsb_bullish", True)): return None`. Remove `require_trend_regime` and `allowed_regimes` from PARAMS entirely. Note the bmsb period bug (see Section 10) must be fixed in `features.py` first.

4. **Fix `require_volume=True`:** Hard-code `require_volume = True` in PARAMS; remove from genome. Keep `volume_threshold` in both PARAMS and the genome.

5. **Add `time_stop_hours` to genome:** In `auto_evolve.py` STRATEGY_REGISTRY for `kama_trend`, add `"time_stop_hours": ("int", 24, 120)`.

6. **Add `rsi_max_entry` and `bb_bw_pctile_min` to genome:** Add both to STRATEGY_REGISTRY param_space. Add `"bb_bw_pctile_min"` to PARAMS with default 0 (disabled), and add the check: `if p["bb_bw_pctile_min"] > 0 and current.get("bb_bw_percentile", 100) < p["bb_bw_pctile_min"]: return None` (entry is only allowed after compression).

7. **Extend er_period ceiling to 25** in STRATEGY_REGISTRY (currently 20).

8. **Fix slow_sc strategy:** For initial sweep rounds, fix `slow_sc = 30` in PARAMS (do not include in genome). Add it back to genome only in a second round once trade count is confirmed sufficient (>50 OOS trades).

9. **Enforce trade count floor:** In `WalkForwardValidator.validate()`, add to the robustness check: `and aggregate.total_trades >= 30`. Update log messages to report when rejection was due to low trade count.

10. **Add bootstrap Sharpe CI to sweep output:** After each genome evaluation, run 1000-resample bootstrap on OOS trade PnL; log the mean, 5th pctile, and 95th pctile Sharpe alongside the point estimate. Flag as "low confidence" if 5th pctile < 0.5.

11. **Remove `allowed_regimes` and `require_trend_regime`** from PARAMS and STRATEGY_REGISTRY entirely.

---

## 10. Methodological Risks

1. **Dominant risk — low trade count (18 IS trades):** With 4,211 genetic evaluations searching over 8+ parameters, the observed IS Sharpe of 3.32 on 18 trades is almost certainly a lucky sequence selected by the optimizer. The t-stat for a Sharpe ≥ 1.0 requires ~50+ trades; at 18 trades, any Sharpe estimate has enormous confidence intervals. WF_Sharpe=4.25 is even more suspect since small WF folds may each have 0–3 trades.

2. **Parameter soup on tiny trade samples:** The genome simultaneously searches `fast_sc`, `slow_sc`, `slope_threshold_pct`, `adx_min`, `er_period` — five interacting parameters — on 18 trades. This is textbook overfitting. Each additional free parameter requires roughly 10 independent trades of power to constrain.

3. **Regime label overfitting:** `allowed_regimes = ["trend", "breakout", "lateral"]` effectively passes almost all bars. The regime classifier's own accuracy is not validated OOS. If the regime classifier itself overfits the training period, this filter provides false confidence.

4. **KAMA re-computation in O(n) per bar:** The KAMA is re-computed from scratch over a 200-bar trailing window at every bar evaluation. In a continuous walk-forward, this means the KAMA value at bar `i` and bar `i-1` are computed from partially overlapping windows but with different starting seeds, creating slight numerical inconsistencies. In a streaming live implementation, the KAMA would be accumulated incrementally. The two can diverge, and this divergence must be audited.

5. **Stop-price vs. fill-price inconsistency:** The stop-loss level is computed at the signal bar's close price (`close - mult * atr`), but the actual entry fill is at the next bar's open. If the next open gaps (common in crypto on weekend liquidation events), the effective ATR stop distance changes, making per-trade risk inconsistent with the intended formula.

6. **WF fold granularity:** With 15-day OOS test windows and ~18 total IS trades, individual folds may have 0–2 trades each. Averaging Sharpe ratios across near-empty folds produces numerically meaningless aggregates. The WF framework must enforce a minimum trade count per fold before including that fold in the aggregate.

7. **Survivorship bias in genetic generation 296:** After 296 generations of selection on pre-lockbox history (2019–2024), the evolved params likely reflect structural features of the 2021 bull run and 2023 BTC recovery — two periods with unusually clean KAMA signals (high ER, low noise). The OOS lockbox (2024–2026) includes post-ETF-approval macro context and different volatility dynamics.

8. **BMSB period bug in features.py:** `bull_market_support_band()` uses `sma_period=20` and `ema_period=21` — interpreted as 20 and 21 **bars**, which on hourly data is 20 hours (~1 day). The intended Bitcoin Bull Market Support Band uses 20-week SMA and 21-week EMA, requiring periods of 3360 and 3528 hourly bars respectively. The current implementation produces a feature that is meaningless as a macro filter. This bug affects any strategy that uses `bmsb_bullish` as a regime gate.

9. **Same-bar KAMA measurement vs. next-bar entry:** The KAMA slope used for entry qualification is measured through and including the signal bar's close. The entry fill occurs at the next bar's open. On breakout entries (slope > 2x threshold + crossover condition), the gap between signal-bar close and next-bar open can be 0.5–2% adverse, meaning the effective entry premium is not captured in the backtest cost model.

10. **No confirmation of OOS lockbox integrity:** There is no automated check that the 2-year OOS window (2024-06-07 to 2026-06-07) was never touched during the 296-generation genetic evolution. This should be verified by auditing the date ranges used in `auto_evolve.py` before accepting the lockbox as clean.
