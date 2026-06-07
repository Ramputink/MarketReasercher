
# connors_rsi2 — Phase 3 Spot Long-Only Research Report

**Classification: DOUBTFUL**
Strategy is convertible to long-only but carries structural issues (dead exit logic, RS period mismatch, short-branch dependency for its bear-market edge) that require fixes before it is viable for OOS evaluation.

---

## 1. Diagnosis

**Current logic.** `connors_rsi2_strategy` implements Larry Connors' RSI(2) approach on hourly OHLCV data. At each bar (bar_idx >= 200, no open position), it:

1. Computes a custom simple-mean RSI(2) over the last 3 closes (a non-Wilder variant).
2. Computes EMA(200) of close prices as the trend filter.
3. Requires ADX(14) >= 15 (when `require_adx=True`).
4. Emits `side="long"` when `rsi2 < 10` and `close > ema_200`.
5. Emits `side="short"` when `rsi2 > 90` and `close < ema_200`.

**Does it emit shorts?** Yes. The short branch is structurally complete and symmetric — it fires on RSI(2) overbought in downtrends with inverted stop/TP. This must be removed for Phase 3.

**Dead exit code.** `exit_threshold_long = 65.0` exists in PARAMS but is never read at runtime. The guard `if position is not None: return None` on lines 53-54 prevents the strategy from evaluating its own exit RSI condition, so all exits are driven entirely by ATR-stop, TP, or 24-hour time stop — not by the mean-reversion completion signal Connors originally intended.

**RSI inconsistency.** The local `_rsi_n()` function uses `np.mean` of gains/losses (simple average), while `features.py` uses Wilder EWM (`ewm(alpha=1/period)`). These produce meaningfully different RSI values, especially for short periods, creating a subtle inconsistency if the feature pipeline RSI is ever used as a substitute.

**Evolution history.** Pruned from `STRATEGY_REGISTRY` in Round 2 after 36K evaluations with only 1.2% positive-fitness rate. However, this rate was computed on the bi-directional version whose short branch likely consumed losing trades in crypto bull markets (2023-2024 data dominates the recent period). The long-only reformulation has a genuinely different trade distribution.

**Likely edge.** RSI(2) dip-buying within EMA(200) uptrends exploits liquidation-cascade overshoots in crypto, which are real and systematic. The edge exists but is narrow: avg holding time of ~24h is too short for multi-day crypto pullbacks; the EMA(200) on hourly bars is only an ~8-day MA (not the weekly trend most associate with it); and the regime gate is too permissive (allows lateral and mean_reversion regimes where the signal is noisy).

**Likely weakness.** High turnover (many RSI(2) signals per year on hourly data), cost sensitivity, absent RSI-based exit causing premature time-stops, and the strategy being effectively untested in its intended form (due to dead exit logic).

---

## 2. Spot Long-Only Edge Hypothesis

The long-only edge: **short-duration oversold flushes (RSI(2) < 10) in confirmed intermediate-term uptrends produce positive expected returns over the next 48-120 hours in BTC/ETH spot**, because crypto uptrends are punctuated by sharp but temporary liquidation-driven sell-offs that statistically mean-revert. The signal becomes high-quality when:
- The BMSB (Bull Market Support Band) is bullish (price above both 20-SMA and 21-EMA).
- The flush bar shows extreme negative `pressure_imbalance` (capitulation).
- `gk_vol` is not in its top decile (which would indicate structural breakdown, not a dip).

The short branch (RSI(2) > 90 in downtrends) is dropped entirely. When no long condition is met, the strategy holds cash/USDC.

---

## 3. Market Techniques

1. **Mean-reversion within trend**: the canonical Connors application — enter oversold dips only when the larger trend is intact (EMA gate).
2. **Multi-timeframe trend confirmation**: extend the trend filter to a longer horizon (EMA(500) or BMSB) to distinguish bull-market dips from bear-market dead-cat bounces.
3. **ATR-normalized exits**: keep ATR-based stops and TP, but extend the TP multiplier range and add a breakeven trailing mechanism.
4. **Regime gating**: restrict entries to `regime == 'trend'` only; the current permissive set (`trend`, `lateral`, `mean_reversion`) dilutes signal quality.
5. **Volatility-scaled position sizing**: reduce size when `gk_vol` is elevated (structural break risk), increase when volatility is compressed and trend is confirmed.
6. **RSI-based exit**: implement the intended `exit_threshold_long` exit — exit the long when RSI(rsi_period) recovers above 60-70, replacing the dead time-only exit.
7. **Capitulation-volume confirmation**: require `volume_zscore > 1.5` on the flush bar to distinguish genuine capitulation from slow drift below RSI(2)=10.

---

## 4. Useful Features

| Category | Feature | Usage |
|---|---|---|
| Trend | `bmsb_bullish` | Hard gate — only enter when True (medium-term uptrend confirmed) |
| Trend | `dist_from_high_20` | Confirm pullback depth; require < -0.03 (at least 3% off 20-bar high) |
| Momentum | `rsi_7` | Exit trigger: exit long when `rsi_7 > 60` (implements dead `exit_threshold_long`) |
| Momentum | `ret_zscore_20` | Entry filter: require < -1.5 (flush bar is statistically extreme) |
| Momentum | `price_acceleration` | Negative value on entry bar confirms continued selling pressure being faded |
| Volatility | `atr_14` | Already used for stop/TP sizing |
| Volatility | `gk_vol` | Regime gate: skip entries when `gk_vol > 80th pct of 90-day rolling history` |
| Volatility | `bb_pct_b` | Below 0.2 = price near lower BB, corroborates RSI(2) oversold |
| Volatility | `bb_bw_percentile` | Avoid entries at extreme-high bandwidth (suggests trend exhaustion not dip) |
| Volume | `volume_zscore` | Capitulation confirmation: spike > 1.5 on flush bar |
| Volume | `buying_pressure` | Low value at entry (bearish flush), rising value as recovery begins |
| Volume | `pressure_imbalance` | Extreme negative at entry → recovery pattern expected |

---

## 5. Genome Parameters (Sweep)

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `rsi_period` | int | 2 | 5 | RSI look-back; 2 is Connors canonical, 3-4 less noisy on hourly crypto |
| `entry_threshold_long` | float | 5.0 | 20.0 | Tighter = rarer but higher-quality signals |
| `exit_threshold_long` | float | 55.0 | 80.0 | RSI level for mean-reversion exit (currently dead, must be wired up) |
| `trend_ema_period` | int | 100 | 300 | EMA period for trend filter; 200 is canonical |
| `adx_min` | float | 10.0 | 30.0 | Minimum ADX to require directional trend |
| `stop_loss_atr_mult` | float | 1.0 | 3.5 | ATR multiplier for hard stop |
| `take_profit_atr_mult` | float | 1.5 | 6.0 | ATR multiplier for take-profit |
| `time_stop_hours` | int | 48 | 168 | Hold-time cap; 24h default is too short |
| `require_volume` | bool | 0 | 1 | Whether to require volume confirmation |
| `volume_threshold` | float | 0.8 | 2.0 | `volume_ratio` threshold (>1.0 = capitulation spike required) |

---

## 6. Parameters NOT to Optimize

- **`allowed_regimes`**: Hard-code to `['trend']` only. Expanding this list always increases in-sample trade count, leading to trivial overfitting; the genome should never control it.
- **`entry_threshold_short` / `exit_threshold_short`**: The short branch is removed in Phase 3; these params have no function and must not enter the genome.
- **`require_adx`**: Fixed `True`. Making it a bool genome param converges to `False` (more trades) in-sample while degrading OOS quality.
- **`require_trend_regime`**: Fixed `True`. Removing the regime gate inflates in-sample trade count without OOS benefit.

---

## 7. Long-Only Risk Management

**Stop loss**: Hard ATR(14)-based stop at `entry_price - stop_loss_atr_mult * atr_14`. Minimum floor at `entry_price * 0.94` (6% hard cap) to protect against hourly-bar gap risk.

**Take profit**: Fixed ATR-based TP at `entry_price + take_profit_atr_mult * atr_14`. Enforce TP/SL ratio >= 1.5 to maintain positive expectancy given typical RSI(2) hit rates of ~55%.

**Trailing stop**: Once unrealised gain exceeds `1.0 * atr_14`, move stop to breakeven. Continue trailing at `1.5 * atr_14` below rolling trade high. Captures mean-reversion bounce without giving back gains.

**Volatility targeting**: Scale position size using `gk_vol`. Target 1% daily portfolio vol per trade. When `gk_vol > 80th pct of 90-day rolling history`, reduce position size by 50%. Cap at `max_position_pct` regardless.

**Max position size**: 25% of equity per trade. Mean-reversion trades carry fatter left tails during structural breaks.

**Cooldown**: Minimum 24-hour cooldown between closing a long and opening a new long, to avoid re-entering the same flush before resolution.

**Risk-off gate**: Skip all entries unless:
1. `bmsb_bullish == True` (medium-term uptrend confirmed), AND
2. `regime == 'trend'`, AND
3. `dist_from_high_20 > -0.20` (not more than 20% below 20-bar high, which would suggest a structural break rather than a dip).

---

## 8. Relevant Metrics

| Metric | Target / Filter |
|---|---|
| CAGR (net of costs) | >= 15% on 2019-2024 IS |
| Sharpe ratio | >= 0.8; benchmark vs BTC buy-and-hold |
| Sortino ratio | >= 1.2 (preferred metric for long-only) |
| Maximum drawdown | Hard filter: reject genome if maxDD > 35% |
| Calmar ratio | >= 0.5 |
| Hit rate | 50-60%; below 45% flags strategy failure |
| Avg trade return (net) | Must be > 0.3% after all costs |
| Turnover | < 100 round-trips/year on hourly data |
| Exposure | 15-35% of bars in a position |
| Cost sensitivity | Edge must survive 2x costs with Sharpe >= 0.6 |
| Walk-forward OOS degradation | IS-to-OOS Sharpe ratio; flag if degradation > 40% |
| Bootstrap / Monte Carlo | Trade-order shuffle x1000; report 5th-pct Sharpe |
| Per-year stability | Annual Sharpe each year 2019-2023; flag if >= 2 years negative |
| Regime breakdown | Hit rate and avg return by regime label separately |

---

## 9. Concrete Sweep Changes

1. **Drop the short branch** entirely: remove `elif is_downtrend and rsi2 > p["entry_threshold_short"]` and all short-side stop/TP arithmetic. The strategy returns `None` when no long condition fires.
2. **Wire up the RSI exit**: remove `if position is not None: return None` on lines 53-54 and instead check `if position is not None: [compute rsi2]; if rsi2 >= p["exit_threshold_long"]: return Signal(side="exit_long" or equivalent)`. This implements Connors' intended exit and activates the dead `exit_threshold_long` param.
3. **Replace `_rsi_n` with pre-computed `rsi_7`** for the exit trigger, or switch to Wilder EWM smoothing consistent with `features.py`. Remove the bespoke inconsistent implementation.
4. **Fix `allowed_regimes`** to `['trend']` only in PARAMS; remove it from any genome param_space.
5. **Extend `time_stop_hours`** default from 24 to 72; genome sweep range 48-168.
6. **Add `bmsb_bullish` gate**: after the ADX check, add `if not bool(current.get("bmsb_bullish", False)): return None`.
7. **Add `gk_vol` regime gate**: compute a rolling 90-day 80th percentile of `gk_vol` and skip entry when current `gk_vol` exceeds it.
8. **Add to genome**: `exit_threshold_long` (55.0-80.0), `time_stop_hours` (48-168), `volume_threshold` as sweep parameter.
9. **Remove from genome**: `entry_threshold_short`, `exit_threshold_short`, `require_trend_regime`.
10. **Re-add connors_rsi2 to `STRATEGY_REGISTRY`** with the cleaned long-only param_space. The Round 2 pruning was on the bi-directional version; the long-only reformulation with the above changes is a materially different strategy.

---

## 10. Methodological Risks

1. **Cost drag from high turnover**: RSI(2) fires frequently on hourly crypto data. Even at 0.10% round-trip (0.05% commission + 0.05% slippage), hundreds of trades per year can fully consume the gross mean-reversion edge if avg gross return is below 0.3% per trade. Must validate cost sensitivity aggressively.

2. **EMA(200) on hourly bars is ~8 days, not weeks**: The trend filter many practitioners associate with EMA(200) is weekly-scale. On hourly bars, EMA(200) is a short-term trend MA, not a structural bull/bear filter. The BMSB gate (which covers ~3-4 week horizon) is more appropriate as the structural filter.

3. **Pruning evidence is not applicable**: The 1.2% positive-fitness figure was measured on the bi-directional version. The long-only version has a completely different trade distribution. The pruning decision should not bias evaluation of the reformulated strategy.

4. **Dead exit logic = unintended strategy**: Because `exit_threshold_long` was never connected, all prior backtests of connors_rsi2 used a purely ATR/time-based exit, not the Connors RSI-recovery exit. This means no backtested version has ever matched the strategy's stated design intent. Any comparison to prior runs is misleading.

5. **2022 bear-market risk**: RSI(2) dip-buying is catastrophic in structural bear markets. Any IS window that underweights 2022 (e.g., 2019-2021 only) will dramatically overestimate expected CAGR. Walk-forward windows must include at least one full bear-market sub-period.

6. **Strength signal is neutralised**: The strength calculation maps `(entry_threshold_long - rsi2) / 20` to [0, 0.5], added to 0.5, giving range [0.5, 1.0]. For RSI(2)=0 strength=1.0; for RSI(2)=10 strength=0.5. This effectively means no entry ever has strength < 0.5, so the volatility-scaled sizing is compressed into a narrow range and adds minimal risk management benefit. Consider a non-linear (log or exponential) strength mapping or a wider normalization window.

7. **Simple-mean vs Wilder EWM RSI inconsistency**: The `_rsi_n` function uses `np.mean` of gains/losses over the last `period` bars — a fundamentally different calculation from Wilder's exponential smoothing. For a 2-period RSI this difference is small (only 2 deltas), but extending `rsi_period` to 4-5 in the sweep will produce values meaningfully different from what practitioners mean by RSI(4-5). All genome sweep runs using `rsi_period > 2` will be testing a non-standard indicator.
