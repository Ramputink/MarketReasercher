# Mean Reversion Strategy — Phase 3 Spot Long-Only Analysis

## 1. Diagnosis

**File:** `strategies/mean_reversion.py`
**Status in auto_evolve.py STRATEGY_REGISTRY:** PRUNED (Round 1, 0% positive fitness across 29,000 evaluations)

### Current Logic

The strategy is a post-spike counter-trend system with the following entry pipeline:

1. **Spike detection** (lines 103-111): measures `(close[-1] / close[-spike_lookback_candles] - 1) * 100`. Qualifies both upside (`is_spike_up`) and downside (`is_spike_down`) spikes of `>= spike_min_move_pct` (currently 1.5%).
2. **Z-score overextension** (lines 114-121): requires `abs(ret_zscore_20) >= zscore_entry_threshold` (2.0) in the direction of the spike.
3. **Bollinger Band distance** (lines 123-129): for spike-up, requires `bb_pct_b >= max_bb_pct_b_short` (0.95); for spike-down, requires `bb_pct_b <= min_bb_pct_b_long` (0.05).
4. **Volume exhaustion** (lines 132-141): requires volume declining bar-over-bar (vol_ratio dropping vs previous bar's vol_ratio).
5. **Wick rejection** (lines 143-155): **INERT BUG** — both branches end with `pass`, not `return None`. This filter has zero effect regardless of `require_wick_rejection=True`.
6. **RSI filter** (lines 157-163): for spike-up requires `rsi_14 >= rsi_overbought` (75); for spike-down requires `rsi_14 <= rsi_oversold` (25).
7. **ADX filter** (lines 165-169): blocks entry when `adx_14 > max_adx_for_entry` (25.0).
8. **Regime gate** (lines 97-98): only trades in `["mean_reversion", "lateral"]` regimes.

### Short Positions: Yes, Extensively

The dominant signal is **`side = "short" if is_spike_up else "long"`** (line 172). In a crypto bull market, upside spikes dominate, meaning this strategy is primarily a short-seller. The position-management early-exit (lines 86-92) also emits `exit_side = "short" if position.side == "long"` — another short signal. Both are forbidden in Phase 3.

### Likely Edge / Weakness

The theoretical edge is panic-driven overshoots that partially recover. In practice, crypto is a structurally trending asset, and shorting upside momentum in a bull market is a consistent loss-maker. The strategy's 0% positive fitness in Round 1 confirms the edge does not survive realistic costs and walk-forward OOS testing. The spike-down long branch is the only Phase-3-compatible direction but fires rarely due to stacked filters and the rarity of genuine panic flushes.

---

## 2. Spot Long-Only Edge Hypothesis

**Hypothesis:** In liquid crypto markets (BTC, ETH), violent downside spikes of >= 1.5–3% over 3–10 bars that simultaneously satisfy (a) ret_zscore_20 <= -2.0, (b) bb_pct_b <= 0.05, (c) RSI-14 <= 25, and (d) volume_zscore >= 2.0 (panic climax) produce short-lived mispricings that partially recover within 6–24 hours. The spot long entry captures this recovery.

The position is held until the z-score reverts toward zero (abs(ret_zscore_20) < zscore_exit_threshold), the ATR-based take-profit is reached, or the time stop fires.

The strategy operates as **long-or-cash**: no position is held outside of identified spike-exhaustion setups. Entry is only permitted in `mean_reversion` or `lateral` regimes, and only when the macro bull market support band (bmsb_sma) is within 15% of current price (indicating this is a dip, not capitulation in a bear market).

**Caveat:** Given the Round 1 pruning evidence, this hypothesis is weak. The spike-down long is the residual path after eliminating the short branch, and it is unlikely to generate sufficient alpha on its own without fundamental restructuring. Classification is therefore `doubtful`.

---

## 3. Market Techniques

1. **Post-crash mean reversion (long-only):** Buy downside spikes showing volume climax and hammer candle formation in the spot market.
2. **Oversold bounce entry:** Dual-confirmation RSI-7 < 20 + bb_pct_b < 0 reduces false entries in trending-down markets.
3. **Volatility regime gating:** Only trade when bb_bandwidth_percentile is 30th–70th percentile (ranging, not breakout or pre-compression).
4. **Exhaustion candle detection:** Require `close_location > 0.55` on the spike bar (hammer/dragonfly), replacing the broken inert wick-rejection filter.
5. **Volume climax confirmation:** Require `volume_zscore > 2.0` on the spike bar to distinguish genuine panic from low-liquidity drift.
6. **Macro structure filter (BMSB):** Only enter when price is within 15% of bull market support band to block structural bear market false longs.
7. **Strict time-stop discipline:** 12–24 hour maximum hold. Mean reversion resolves quickly or not at all.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Mean-reversion | `ret_zscore_20` | Primary entry trigger: must be <= -zscore_entry_threshold |
| Mean-reversion | `bb_pct_b` | Price must be below lower BB (< 0.05) |
| Mean-reversion | `dist_from_low_20` | Additional stretch confirmation |
| Momentum | `rsi_7` | Fast oversold (<20); more selective than rsi_14 for short-term bounces |
| Momentum | `rsi_14` | Secondary confirmation; also used as exit trigger (reversion above 40) |
| Volatility | `atr_14` | Stop/TP placement |
| Volatility | `bb_bandwidth` | Regime filter; avoid extremes (trending or pre-breakout) |
| Volatility | `gk_vol` | Garman-Klass vol for volatility-targeting position sizing |
| Volatility | `vol_ratio` | Short/long-term vol ratio to detect vol expansion after spike |
| Volume | `volume_zscore` | Panic volume climax confirmation (> 2.0 required) |
| Volume | `volume_ratio` | Volume declining on entry bar (exhaustion) |
| Volume | `buying_pressure` | Should be recovering (> 0.5) on entry bar |
| Microstructure | `close_location` | Must be > 0.55 on spike bar (hammer pattern) — replaces broken wick filter |
| Microstructure | `pressure_imbalance` | Buying - selling pressure; positive = recovery |
| Trend | `adx_14` | Must be < max_adx_for_entry to block strong downtrend entries |
| Trend | `bmsb_sma`, `bmsb_ema` | Macro bullish structure gate |
| Regime | `_regime` | Only trade in mean_reversion or lateral regimes |
| Price accel | `price_acceleration` | Second derivative; should be turning positive (decelerating decline) |

---

## 5. Parameters for the Sweep/Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `zscore_entry_threshold` | float | 1.5 | 3.5 | Overextension threshold for spike-down entry |
| `zscore_exit_threshold` | float | 0.1 | 0.8 | Z-score reversion exit trigger |
| `spike_lookback_candles` | int | 3 | 10 | Spike detection window |
| `spike_min_move_pct` | float | 1.0 | 4.0 | Minimum % decline to qualify as spike |
| `rsi_oversold` | float | 15.0 | 35.0 | RSI-14 oversold threshold |
| `max_adx_for_entry` | float | 15.0 | 35.0 | ADX ceiling (block strong trends) |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | ATR multiplier for hard stop below entry |
| `take_profit_atr_mult` | float | 1.0 | 4.0 | ATR multiplier for take-profit above entry |
| `time_stop_hours` | int | 6 | 36 | Max trade duration |
| `min_volume_zscore` | float | 0.5 | 3.0 | NEW: panic volume floor on spike bar |
| `min_close_location` | float | 0.4 | 0.75 | NEW: hammer candle close location floor |

---

## 6. Parameters NOT to Optimize (Overfitting Traps)

- **`bb_period`** — Fix at 20. Standard definition; sweeping creates collinear parameter space with minimal information gain.
- **`bb_std`** — Fix at 2.0. Standard Bollinger. Current 2.5 in PARAMS is already curve-fit to historical data.
- **`zscore_lookback`** — Fix at 20. Matches the `ret_zscore_20` feature computed by the pipeline. Changing requires feature recomputation per trial (not supported).
- **`require_reversion_regime`** — Fix at True. Core hypothesis filter.
- **`allowed_regimes`** — Fix as `["mean_reversion", "lateral"]`. Sweeping regime membership creates combinatorial explosion and overfits to historical regime distribution.
- **`volume_decline_threshold`** — Fix at 0.7. Bar-to-bar ratio comparison; fine-tuning this produces negligible signal differentiation.
- **`require_volume_decline`** — Fix at True. Structurally sound exhaustion condition.
- **`max_bb_pct_b_short`** — Remove entirely. Dead code after eliminating the spike-up short branch.
- **`wick_rejection_ratio`** — Remove entirely. Replace with `min_close_location` which actually works.

---

## 7. Long-Only Risk Management

- **Stop:** Hard stop at `entry_price - stop_loss_atr_mult * atr_14`. Must be placed below the spike bar's low, requiring `stop_loss_atr_mult >= 2.0` in most cases. ATR-adaptive (not fixed %).
- **Take-profit:** `entry_price + take_profit_atr_mult * atr_14`. Minimum risk:reward 1:1 enforced. Z-score exit (abs(ret_zscore_20) < zscore_exit_threshold) provides alternative early exit.
- **Trailing stop:** Not recommended. Mean reversion targets are short and defined. Instead, partial exit: 50% off at 1.5x ATR gain, remainder to full TP or time stop.
- **Volatility targeting:** Use `volatility_scaled` sizing via RiskManager: `size = equity * (vol_target_annualized / gk_vol)`, capped at `max_position_pct`. Target annualized vol = 15–25%.
- **Max position size:** 20% of equity per trade. Counter-trend strategy; aggressive sizing amplifies sequence-of-loss risk.
- **Cooldown:** Mandatory 6-hour cooldown after any exit to prevent re-entry into the same spike event (crypto echoes).
- **Risk-off gate:** Block entries when: (a) `_regime` not in `[mean_reversion, lateral]`; (b) `adx_14 > max_adx_for_entry`; (c) price is > 15% below `bmsb_sma` (structural bear market); (d) rolling 10-day equity drawdown > 10% (circuit breaker via RiskManager).

---

## 8. Relevant Metrics

| Metric | Target / Note |
|---|---|
| CAGR | > 15% annualized; rare signals make this difficult |
| Sharpe ratio | > 0.8 on WF OOS; > 1.2 unlikely given regime mismatch |
| Sortino ratio | > 1.0; more appropriate than Sharpe for asymmetric returns |
| Max drawdown | < 20% on full pre-lockbox history |
| Calmar ratio | > 0.8 (CAGR / maxDD) |
| Hit rate | > 55% required; bounded wins, unbounded time-stop losses |
| Avg trade duration | Track vs time_stop_hours; persistent exceedance = failed trades |
| Turnover | 2–10 trades/month; < 2 makes CAGR targets unreachable |
| Market exposure % | 5–20% of bars in position; > 40% implies gate too permissive |
| Cost sensitivity | Sharpe drop < 30% from frictionless to 5 bps commission + 3 bps slippage |
| OOS degradation % | < 50% per backtester max_oos_degradation_pct |
| Bootstrap/MC | 1000 resampled paths; report 5th-percentile CAGR and max DD |
| Per-year stability | Report CAGR/Sharpe by calendar year; identify era-specific performance |
| Profit factor | > 1.3 after costs |
| Signal frequency | > 1 trade per 500 bars minimum; stacked filters may starve the strategy |

---

## 9. Concrete Sweep Changes

1. **Eliminate the short branch:** Replace `side = "short" if is_spike_up else "long"` with `if is_spike_up: return None; side = "long"`. Delete all `is_spike_up` code paths.
2. **Fix the exit signal emission:** The position-management block (lines 82-93) must NOT emit a `side="short"` signal. Replace with a sentinel exit: return `Signal(side="long", strength=0.0, ...)` or restructure to return None and handle exit via the z-score check differently (e.g., set a flag). Audit that the backtester's pending_entry queue does not misinterpret an exit signal as a new entry.
3. **Fix the broken wick rejection filter:** Lines 148-154 currently `pass` on both branches. Replace with: `if is_spike_down and close_loc < p.get("min_close_location", 0.5): return None`.
4. **Add volume_zscore gate:** After the spike detection block, add: `if current.get("volume_zscore", 0) < p.get("min_volume_zscore", 1.0): return None`.
5. **Add BMSB macro filter:** `if close < current.get("bmsb_sma", close) * 0.85: return None` (blocks entries > 15% below bull market support band).
6. **Add `min_volume_zscore` and `min_close_location` to PARAMS dict** with documented defaults.
7. **Remove dead params:** Delete `max_bb_pct_b_short` and `wick_rejection_ratio` from PARAMS.
8. **Add strategy back to STRATEGY_REGISTRY** in `auto_evolve.py` with the long-only param_space: `zscore_entry_threshold`, `zscore_exit_threshold`, `spike_lookback_candles`, `spike_min_move_pct`, `rsi_oversold`, `max_adx_for_entry`, `stop_loss_atr_mult`, `take_profit_atr_mult`, `time_stop_hours`, `min_volume_zscore`, `min_close_location`.
9. **Test signal frequency** before running full evolution: run a dry loop counting signals per year from 2019–2024; if < 50 signals in 5 years, the strategy is too rare to evaluate statistically and needs loosened filters.

---

## 10. Methodological Risks

1. **Round 1 pruning is a hard prior:** 29,000 evaluations with 0% positive fitness is strong evidence the edge does not exist in the original structure. Long-only conversion removes the dominant (short) branch, leaving only the residual spike-down long path which was always the minority signal. The probability of Phase 3 success is materially lower than for unpruned strategies.
2. **Regime label leakage risk:** The `_regime` column uses EMA-smoothed ADX and BB bandwidth that have rolling lookbacks. The WalkForwardValidator's 120-bar embargo should contain this, but `classify_regime_quantitative` in `mirofish/scenario_engine.py` must be audited to confirm no feature uses a window longer than 120 bars.
3. **Spike threshold non-stationarity:** `spike_min_move_pct = 1.5%` is a fixed percentage applied across regimes with very different volatilities (2019 BTC vol ~80% annual, 2023 ~40%). In low-vol regimes this threshold is rarely crossed; in high-vol regimes it fires on noise. Should normalize to `N * (atr_14 / close * 100)` to make the trigger regime-adaptive.
4. **Crash continuation vs reversal ambiguity:** Spike-down longs risk catching falling knives in genuine downtrends. The 2022 BTC bear market had dozens of -5% bars that all looked like oversold RSI + BB extension entries and all continued lower. The BMSB filter helps but is not sufficient; the ADX filter at 25 is also permissive (a strong downtrend can have ADX 25–35).
5. **Statistical fragility from low trade count:** After all stacked filters, expected frequency is < 2–3 trades per month. Over 5 years of pre-lockbox history (~60 months) this gives roughly 100–180 total signals. The 2-year OOS lockbox yields at most 50 trades — far too few for stable performance estimation; the reported Sharpe will have confidence intervals of ± 0.5 or wider.
6. **Filter collinearity:** `ret_zscore_20`, `bb_pct_b`, and `rsi_14` all measure the same concept (price deviation from mean) over slightly different lookbacks. They may provide ~1.2 independent bits of information while filtering as if they provide 3, causing signal starvation without proportional quality improvement.
7. **Exit signal backtester interaction:** The position-management early-exit emits `side = "short"` to trigger the backtester's `signal.side != position.side` exit check. After the long-only fix, the exit must be re-implemented carefully. If the sentinel signal accidentally queues a new short entry via `pending_entry`, the backtest will show phantom short trades. This must be explicitly tested.
8. **Parameter era instability:** Mean reversion behavior differs dramatically between 2019–2021 (volatile bull), 2022 (structural bear), and 2023–2026 (recovery). Parameters calibrated on the full pre-lockbox history may overfit to the 2021 high-vol regime. Walk-forward must report per-year breakdown.
