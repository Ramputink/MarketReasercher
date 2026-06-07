# Volatility Breakout — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy file:** `/strategies/volatility_breakout.py`

The strategy implements a two-phase logic:

**Phase A — Compression detection.** `bb_bw_percentile` (rolling percentile rank of BB bandwidth over 100 bars) must be below `bandwidth_percentile_threshold` (default 25.0) for at least `min_compression_bars` (default 3) consecutive bars. This ensures that the market has been genuinely "coiling" before the breakout is considered.

**Phase B — Breakout confirmation.** Price must close above `bb_upper` (long signal) or below `bb_lower` (short signal), with:
- `volume_ratio >= volume_surge_threshold` (default 1.5x SMA-20 volume)
- `atr_14 / atr_5bars_ago >= atr_expansion_ratio` (default 1.15) — ATR expansion filter is ON
- Optional `pressure_imbalance` microstructure gate (currently OFF)
- Regime gate: regime must be in `["breakout", "trend"]`

**Short signals are present.** When `close < bb_lower`, `side="short"` is emitted with an inverted stop/TP. This is a hard Phase-3 violation and must be removed.

**Evolutionary history.** The strategy was pruned in Round 1 of the genetic optimizer (auto_evolve.py line 64) after 29K evaluations with 0% positive fitness — the worst result in the registry. It does not appear in the current STRATEGY_REGISTRY. This is a significant prior against the strategy's edge in its original form.

**Known weaknesses:**
- The `breakout_confirmation_candles` parameter is effectively dead code — the confirmation branch contains bare `pass` statements, so setting this to any value above 1 has no behavioral effect. Any IS optimization over this param is pure noise.
- The 24-hour time stop is very short for a strategy that requires multi-bar compression to form, meaning the strategy frequently exits before the breakout fully resolves.
- Symmetric long/short design in a bull-biased spot crypto market suppresses long-side alpha by allocating risk budget to short trades that do not benefit from the spot upward drift.

---

## 2. Spot Long-Only Edge Hypothesis

After a multi-bar Bollinger Band bandwidth compression, crypto spot markets tend to resolve upward with disproportionate momentum during bull-regime phases. The specific edge in long-only spot:

1. **Compression creates a coiled spring.** When realized volatility collapses (bb_bw_percentile < 25), market participants reduce positioning; the resulting thin order book amplifies any directional catalyst.
2. **Upside breakouts are more sustained in spot.** Spot holders have no margin; downside breakouts in perp markets are partially driven by liquidation cascades that self-limit (liquidations stop when margin is exhausted). Upside breakouts in spot are amplified by FOMO buying and short-covering on perps without a symmetric self-limiting mechanism.
3. **Regime gate captures structural bull-market phases.** Restricting to `bmsb_bullish=True` (price above 20-period SMA and 21-period EMA) or regime in `["breakout","trend"]` ensures compression breakouts occur within a macro uptrend, where the asymmetry between up-breakout size and down-breakout size is largest.
4. **Volume + ATR expansion confirms real breakouts.** False breakouts (wicks above the band that immediately reverse) are filtered by requiring both a volume surge (1.5-2.5x average) and ATR expansion (>1.15x vs 5 bars ago), reducing noise trade frequency.

The strategy becomes long-or-cash: **long on upside compression breakout with all filters satisfied; cash otherwise** (including when downside breakout occurs).

---

## 3. Market Techniques

- **Bollinger Band squeeze / NR7 compression detection** via `bb_bw_percentile` rolling rank
- **ATR-expansion breakout confirmation** (ATR_now / ATR_5ago ratio gate)
- **Volume surge confirmation** at breakout bar (`volume_ratio` vs 20-bar SMA baseline)
- **Bull-market support band (BMSB)** macro regime gate to restrict entries to uptrend phases
- **ADX trend-strength filter** to prefer breakouts in already-trending environments
- **OBV divergence confirmation** — OBV making new highs alongside price supports long breakout authenticity
- **Garman-Klass volatility** as secondary vol-expansion signal
- **Pressure imbalance microstructure** filter at the breakout bar (long-only: require imbalance > 0)
- **Walk-forward optimization** restricted to pre-lockbox history (2019-2024-06-07); sealed OOS = last 2 years
- **ATR-based volatility-targeted position sizing** to normalize risk across compression/expansion regime transitions

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Volatility | `bb_bw_percentile` | Primary compression signal |
| Volatility | `bb_bandwidth` | Raw input to percentile calculation |
| Volatility | `atr_14` | Breakout expansion gate + stop/TP sizing |
| Volatility | `atr_7` | Faster expansion detection secondary |
| Volatility | `gk_vol` | Garman-Klass realized vol; confirm true vol expansion |
| Volume | `volume_ratio` | Breakout authenticity; primary volume gate |
| Volume | `volume_zscore` | Secondary surge confirmation vs 50-bar baseline |
| Volume | `obv_divergence` | OBV above its SMA = volume trend supports long |
| Price/Return | `bb_upper` | Breakout threshold for long entry trigger |
| Price/Return | `bb_pct_b` | Near 1.0 post-compression = strong long signal |
| Price/Return | `dist_from_high_20` | Near 0 = testing resistance before breakout |
| Trend | `adx_14` | Trend strength; >20 at breakout = directional commitment |
| Trend | `bmsb_bullish` | Boolean macro regime gate |
| Momentum | `rsi_14` | Entry quality gate; avoid >80 (overbought) |
| Momentum | `roc_12` | Positive aligns with long breakout direction |
| Momentum | `price_acceleration` | Second derivative; acceleration supports breakout |
| Market Regime | `regime` (external) | `"breakout"` or `"trend"` filter |
| Microstructure | `pressure_imbalance` | Buying > selling pressure at breakout bar |
| Microstructure | `close_location` | Close near bar high (>0.7) confirms bullish bar |

**Funding/sentiment note:** Funding rates may only be used as an informational filter (e.g., "avoid longs when perp funding is extremely negative, signaling macro bearish positioning") — never as a direct PnL source. This is not currently in the strategy and does not need to be added for Phase 3.

---

## 5. Parameters for the Sweep / Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `bandwidth_percentile_threshold` | float | 10.0 | 30.0 | Core compression tightness; narrow range to avoid overfitting |
| `min_compression_bars` | int | 2 | 6 | Duration of squeeze before entry allowed |
| `volume_surge_threshold` | float | 1.2 | 2.5 | Breakout volume authenticity |
| `require_atr_expansion` | bool | 0 | 1 | ATR expansion gate; default ON is likely correct |
| `atr_expansion_ratio` | float | 1.05 | 1.35 | ATR_now / ATR_5ago minimum |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | Wider stops suit compression entries |
| `take_profit_atr_mult` | float | 2.0 | 6.0 | Generous TP for post-compression moves |
| `time_stop_hours` | int | 12 | 72 | Trade duration limit; needs sweep (currently fixed at 24) |
| `adx_min_at_entry` | float | 10.0 | 30.0 | Trend strength gate at breakout bar |
| `require_bmsb_bullish` | bool | 0 | 1 | Macro uptrend gate |
| `rsi_max_entry` | float | 65.0 | 85.0 | Overbought avoidance |
| `use_pressure_imbalance` | bool | 0 | 1 | Microstructure long-filter |
| `pressure_imbalance_threshold` | float | 0.05 | 0.30 | Minimum imbalance when gate is ON |

---

## 6. Parameters NOT to Optimize (Overfitting Traps)

- **`bb_period` (fix at 20):** Standard Bollinger convention. Sweeping this against `bandwidth_percentile_threshold` creates collinearity — the percentile baseline changes meaning as the period changes, making results non-stationary across regimes.
- **`bb_std` (fix at 2.0):** Changing std changes what constitutes a "breakout" — different std values measure different quantiles of the distribution. Optimizing this is equivalent to fitting the threshold to IS data without a principled anchor.
- **`breakout_confirmation_candles` (fix at 1, or implement properly):** The current confirmation logic is dead code (bare `pass` statements). Optimizing it would produce random noise in the sweep. It must be either correctly implemented or removed before any sweep is run.
- **`allowed_regimes` list (fix, do not enumerate subsets):** With only ~3 possible regime values, sweeping over subset combinations (2^3 = 8 options) is a direct IS overfit with no OOS generalization guarantee. Replace with the explicit `require_bmsb_bullish` boolean and `adx_min_at_entry` float which have continuous search spaces.

---

## 7. Long-Only Risk Management

| Component | Specification |
|---|---|
| **Stop loss** | Hard ATR stop: `entry_price - stop_loss_atr_mult * atr_14`; computed at signal bar, applied from fill bar open |
| **Take profit** | Fixed ATR target: `entry_price + take_profit_atr_mult * atr_14`; enforce `take_profit_atr_mult > stop_loss_atr_mult` in genome validation |
| **Trailing stop** | Optional: after unrealized gain > 1.5x ATR, trail stop to `breakeven + 0.5 * atr_14`; locks in partial profit on extended breakouts |
| **Volatility targeting** | `volatility_scaled` sizing: target annualized vol 40-60%; `size = equity * (vol_target / annual_vol)`, capped at `max_position_pct` |
| **Max position size** | `max_position_pct = 0.80`; spot-only (no leverage); `signal_strength` scalar reduces size below cap |
| **Cooldown** | `min_bars_between_trades = 4h`; prevents re-entry immediately after a false breakout stop-out during the same compression episode |
| **Risk-off gate** | Skip entry if: (a) regime not in allowed set AND `bmsb_bullish=False`; (b) `adx_14 < adx_min_at_entry`; (c) circuit-breaker fires if equity drawdown from peak > 15% (uses existing `RiskManager` mechanism in backtester) |

---

## 8. Relevant Metrics

| Metric | Target / Notes |
|---|---|
| CAGR | Primary return metric; computed on 2024-06-07 to 2026-06-07 OOS lockbox equity curve |
| Sharpe ratio | Must exceed 1.0 net of all costs to be considered for promotion to Phase 4 |
| Sortino ratio | More relevant than Sharpe for skewed breakout return distributions |
| Max drawdown % | Full-sample peak-to-trough via `WalkForwardValidator.full_sample_max_dd_pct` (already implemented) |
| Calmar ratio | CAGR / maxDD; target > 0.5 |
| Hit rate | % winning trades; expected 40-55%; below 35% = edge is marginal even with good R:R |
| Profit factor | Gross wins / gross losses; must exceed 1.2 net of costs |
| Average trade duration | Should be well below `time_stop_hours`; high time-stop % = strategy failing to resolve |
| Exit reason distribution | TP% > SL%; time_stop% < 30% indicates healthy breakout resolution |
| Turnover / trade frequency | Trades per month; compression + multi-filter stack is naturally low-frequency; need >= 5 trades/month for statistical power in 2-year OOS |
| Exposure time | % time in market; expected 10-25% for this filter stack; acceptable for spot |
| Cost sensitivity | Re-run OOS with 2x and 3x commission+slippage; Sharpe degradation should be < 30% |
| Bootstrap / MC | Resample trade P&L sequence 1000x; report 5th-percentile Sharpe and maxDD |
| Per-year Sharpe stability | Annual Sharpe for each calendar year in pre-lockbox data; any strongly negative year is a red flag |
| OOS degradation % | `WalkForwardValidator.oos_degradation_pct`; target < 40% |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove the short branch entirely:** Delete `breakout_short` detection, the `"short"` side assignment, and short-side stop/TP. Replace with `return None` when `close < bb_lower`.
2. **Add `require_bmsb_bullish` to PARAMS and to a new STRATEGY_REGISTRY entry** as `("bool",)`; gate long entries on `bmsb_bullish == True` when enabled.
3. **Add `adx_min_at_entry` to PARAMS** (default 15.0) and param_space as `("float", 10.0, 30.0)`.
4. **Add `rsi_max_entry` to PARAMS** (default 75.0) and param_space as `("float", 65.0, 85.0)`.
5. **Add `volatility_breakout` back to STRATEGY_REGISTRY** in auto_evolve.py — the long-only version has a structurally different edge hypothesis from the pruned long+short version and must be re-evaluated from scratch.
6. **Fix the dead `breakout_confirmation_candles` logic** — either implement properly (all bars in recent window above `bb_upper` for long) or remove entirely and fix at 1. Do not include in the param_space until the logic is verified.
7. **Relax `allowed_regimes`** to include `"unknown"` when `require_bmsb_bullish=True` and `bmsb_bullish=True` — recovers valid breakout bars the regime classifier misses.
8. **Add `time_stop_hours` to param_space** as `("int", 12, 72)` — currently hard-coded at 24 and not evolvable.
9. **Add optional OBV divergence confirmation:** new bool param `use_obv_confirm`; when True, require `obv_divergence > 0` at breakout bar.
10. **Fix signal strength calculation:** the `pressure_imbalance` contribution to strength is unconditional but should only apply when `use_pressure_imbalance=True`; the current code adds a strength bonus regardless of the gate setting.

---

## 10. Methodological Risks

1. **Round 1 evolutionary pruning as a negative prior.** 29K evaluations across the long+short design found 0% positive fitness. While the long-only hypothesis differs structurally, the base-rate probability of genuine edge is lower than for strategies that survived. The Phase-3 analysis must be treated as exploratory, not confirmatory.

2. **Dead confirmation logic invalidates IS results.** The `breakout_confirmation_candles` parameter has a dead code path (bare `pass`). Any past IS optimization or Sharpe reported with this parameter set to values > 1 is unreliable — the strategy behaved identically regardless of this setting.

3. **Fixed ATR lag (5 bars) in expansion filter.** The ATR expansion check uses a hard-coded 5-bar lag (`bar_idx - 5`) that was never parameterized. This arbitrary choice may be an accidental IS overfit that does not generalize.

4. **Percentile baseline contamination risk.** `bb_bandwidth_percentile` uses a 100-bar rolling window. The WalkForwardValidator embargo is correctly set to 120 bars, but any custom train/val split shorter than 120 bars would contaminate the percentile rank across the boundary.

5. **Signal strength creates implicit sizing overfit.** The additive strength bonuses (0.2 for vol_ratio > 2.0, 0.15 for bw_pct < 10, 0.15 for pressure_imbalance > 0.5) were set manually and never swept. These bonuses modulate position size through the strength scalar, creating an implicit IS fit to whatever regime rewarded larger sizes.

6. **Short time stop vs. compression resolution time.** The 24-hour default time stop may be too short for a strategy that requires 3+ bars of compression to form; the full breakout move in crypto can take 48-96 hours to play out. This creates a systematic truncation of winners.

7. **External regime classifier leakage risk.** The `regime` parameter is injected externally. If the regime classifier was calibrated or trained on data that overlaps the lockbox OOS window (2024-06-07 onward), the regime gate introduces an indirect information leakage channel.

8. **BMSB feature timeframe mismatch.** The `bmsb_bullish` feature in `features.py` uses `sma_period=20` and `ema_period=21` in the native data timeframe. On hourly data these are 20-hour and 21-hour windows — not 20-week and 21-week as intended by the "Bull Market Support Band" concept. For the macro uptrend gate to be meaningful, BMSB should be computed on daily aggregates (or the periods scaled to ~3360 and ~3528 for hourly data as the docstring notes).

9. **Low expected trade count in OOS window.** With compression + volume + ATR + regime + BMSB + RSI + ADX filters all stacked, expected signal frequency is low. If fewer than ~30 trades occur in the 2-year lockbox OOS window, bootstrap confidence intervals on Sharpe will be very wide and the metric will not be statistically distinguishable from zero.

10. **Volume ratio normalization and time-of-day bias.** `volume_ratio = volume / SMA20_volume` is computed on hourly OHLCV. This normalizes by recent history but does not account for time-of-day seasonality in crypto volume (Asian session vs. US session). The surge threshold may trigger systematically more in low-volume periods (thin book = smaller absolute volume needed to exceed the ratio threshold), creating a time-of-day bias in entry timing.

---

## Classification: DOUBTFUL

The strategy carries genuine structural appeal (compression → breakout is a real phenomenon in crypto) but faces three compounding concerns that justify "doubtful" rather than "candidate": (1) the evolutionary optimizer found 0% positive fitness in 29K evaluations of the original design; (2) the code contains dead logic (`breakout_confirmation_candles`) that makes any prior IS results unreliable; and (3) the BMSB feature has a likely timeframe mismatch that must be corrected before the macro regime gate has any meaning. The long-only conversion is mandatory, achievable, and removes the structural mismatch between a bull-biased spot market and a symmetric long/short design — this is the single most important change. If the dead code is fixed, the BMSB period mismatch corrected, and the long-only genome re-run from scratch with walk-forward validation, the strategy deserves a clean evaluation. It should not be promoted based on any previously reported IS metrics.
