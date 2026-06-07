# RSI Divergence — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis

**Current logic.** The strategy detects price/RSI divergence over a rolling `divergence_lookback` window (default 14 bars). The window is split into two equal halves. Bullish divergence fires when the second-half minimum price is lower than the first-half minimum, while the second-half minimum RSI is higher — price makes a lower low, RSI makes a higher low. Bearish divergence does the reverse. Both require the magnitude of the price and RSI moves to exceed thresholds (`min_price_delta_pct`, `min_rsi_delta`), and for RSI to be in an extreme zone at signal time. An ADX cap prevents signals in strong trends. Optional volume-contraction confirmation exists but defaults to False.

**Does it emit shorts?** Yes. The bearish-divergence branch (lines 91-115) explicitly generates `Signal(side="short", ...)` with a stop above price and take-profit below. This is directly incompatible with Phase-3 and must be removed entirely.

**Evolutionary history.** The strategy was pruned from the active STRATEGY_REGISTRY in Round 1 of the evolutionary optimizer, alongside four other strategies, after achieving **0% positive fitness** across approximately 29,000 evaluations. It is currently commented out as a historical note. This is the most critical piece of information: no parameter combination in a generous search space produced a viable strategy.

**Edge hypothesis (what the bullish branch attempts).** RSI divergence attempts to capture selling exhaustion — price hits a new low but momentum (as measured by RSI) is already recovering. This is a genuine pattern in technical analysis with theoretical basis in momentum mean-reversion. In practice, crypto's dominant trending character and high intraday volatility mean the setup fires frequently but bounces are shallow and often fail.

**Weakness.** The split-window min/max method is a poor approximation of swing-pivot divergence. It fires whenever any transient dip appears in either half, regardless of structural significance. RSI extreme thresholds of 35/65 are generous for crypto (RSI can hover near 70 for weeks in bull runs). The 0% fitness result confirms these weaknesses are not fixable with parameter tuning alone.

---

## 2. Spot Long-Only Edge Hypothesis

In ranging or early-recovery crypto regimes, when price retests a prior low with decreasing selling momentum (RSI makes a higher low), the setup signals exhaustion of sellers. If this occurs near structural support (Bollinger lower band, VWAP, BMSB), with volume contracting on the re-test leg, and OBV not confirming the price low, the probability of a mean-reversion bounce increases materially.

The long-only thesis: enter spot long when a bullish RSI divergence is confirmed by support proximity and volume dry-up, hold for a target of 3-4x ATR (prior swing high), exit via hard stop or time stop if the bounce fails within 18-72 hours. The strategy is in cash otherwise — no shorts, no leverage.

The bearish divergence branch is dropped entirely. A bearish signal converts to "exit to cash" if already long (handled by the backtester's signal-exit logic when a signal with a different side is returned), but since the strategy already skips execution when `position is not None`, the practical change is simply deleting the bearish branch.

---

## 3. Market Techniques

1. **Bullish RSI divergence (long branch only):** price lower low + RSI higher low in the lookback window, with RSI below `rsi_extreme_low` at signal time.
2. **ATR-based adaptive stops and take-profits:** already implemented; sweep the multipliers.
3. **ADX cap for mean-reversion context:** prevent entries when trend strength is high.
4. **Volume contraction on re-test leg:** `volume_ratio < volume_decline_ratio` confirms sellers are exhausted, not accelerating.
5. **Bollinger Band lower-band proximity:** `bb_pct_b` near 0 adds structural support confirmation.
6. **BMSB macro regime gate:** only trade longs when price is in or re-testing the Bull Market Support Band — avoids entries in macro bear markets.
7. **VWAP distance filter:** entries below or near `vwap_20` avoid chasing.
8. **OBV co-confirmation:** require `obv_divergence > 0` so that volume-flow is not also making a new low.
9. **Pressure imbalance timing:** `buying_pressure` should be rising at the signal bar (close near high of the bar).
10. **Time-based exit:** 18-72 hour time stop as primary exit when the take-profit is not reached.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Momentum | `rsi_14` | Primary divergence oscillator; current-bar gate (`< rsi_extreme_low`) |
| Momentum | `rsi_7` | Faster RSI for entry timing refinement after divergence detection |
| Trend | `adx_14` | Cap at `adx_max` to enforce mean-reversion context |
| Volatility | `atr_14` | Stop-loss and take-profit distance denominator |
| Volatility | `atr_7` | Tighter stop in low-vol regimes |
| Volatility | `bb_pct_b` | Price proximity to Bollinger lower band (support confirmation) |
| Volatility | `bb_bandwidth` | Low bandwidth = compression phase before potential bounce |
| Volatility | `gk_vol` | Regime filter; extreme GK vol signals gap risk, avoid |
| Volume | `volume_ratio` | Contraction on re-test: `< volume_decline_ratio` |
| Volume | `obv_divergence` | OBV vs OBV_SMA; positive confirms bullish price divergence |
| Volume | `volume_zscore` | Spike detector; avoid entries during volume-spike re-tests |
| Trend | `bmsb_bullish` | Macro long-only gate; only long when above BMSB |
| Mean-reversion | `dist_from_low_20` | Oversold context; near 20-bar low supports the divergence setup |
| Mean-reversion | `ret_zscore_20` | Extreme negative z-score confirms recent selling pressure |
| Microstructure | `pressure_imbalance` | Should turn positive at divergence confirmation bar |
| Microstructure | `buying_pressure` | Close near bar high at signal bar = bullish confirmation |
| Price | `vwap_20` | Support/resistance reference; below VWAP = better long entry |
| Market regime | `regime` string | Gate on `lateral` and `mean_reversion` only |

---

## 5. Genome Parameters (Sweep)

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `divergence_lookback` | int | 8 | 30 | Window length for split divergence detection |
| `rsi_extreme_low` | float | 25.0 | 45.0 | RSI must be below this for bullish entry |
| `min_price_delta_pct` | float | 0.3 | 3.0 | Min % price decline for lower-low qualification |
| `min_rsi_delta` | float | 2.0 | 15.0 | Min RSI improvement for divergence qualification |
| `adx_max` | float | 20.0 | 45.0 | ADX cap for mean-reversion context |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | ATR multiplier for hard stop |
| `take_profit_atr_mult` | float | 2.0 | 6.0 | ATR multiplier for take-profit (must exceed stop) |
| `time_stop_hours` | int | 18 | 72 | Max hold duration in hours |
| `require_volume_decline` | bool | 0 | 1 | Toggle volume contraction filter |
| `volume_decline_ratio` | float | 0.5 | 0.95 | Volume_ratio threshold for contraction |
| `require_obv_confirm` | bool | 0 | 1 | Require OBV divergence co-confirmation |
| `require_bmsb` | bool | 0 | 1 | Gate entries on bmsb_bullish flag |
| `bb_pct_b_max` | float | 0.2 | 0.6 | Max bb_pct_b allowed (near lower band) |

---

## 6. Parameters NOT to Optimize

- **`rsi_period`:** Fixed at 14. Sweeping it creates implicit look-ahead bias because the same bars construct both the RSI series and the divergence window; the interaction is non-trivial and the extra degree of freedom buys nothing except overfitting.
- **`allowed_regimes`:** Fixed at `["lateral", "mean_reversion"]`. Sweeping the regime set (e.g., including/excluding "trend") directly overfits to the specific regime-classifier behavior across a limited training history.
- **`require_regime`:** Always `True`. Disabling it removes the only meaningful environmental filter.
- **`divergence_lookback` + `rsi_period` swept simultaneously:** Their product determines the total warmup, and sweeping both creates a multiplicative search space that is too large for meaningful convergence.

---

## 7. Long-Only Risk Management

**Stop-loss:** Hard ATR stop — `close - atr_14 * stop_loss_atr_mult`. Range 1.5-4.0x ATR. Checked against bar low each subsequent bar. Already implemented correctly in the backtester.

**Take-profit:** Fixed ATR target — `close + atr_14 * take_profit_atr_mult`. Range 2.0-6.0x ATR. Must exceed stop multiplier to maintain positive expected R:R.

**Trailing stop:** Optional — activate only once unrealized gain exceeds 1.5x ATR, trail at 1.0x ATR below rolling high. Protects partial gains without cutting winners too early.

**Volatility targeting:** Use `volatility_scaled` sizing method. Target 15% annualized portfolio volatility. When `gk_vol` is above the 80th percentile of the trailing 90-day distribution, reduce position size to 50% of normal.

**Maximum position size:** 25% of equity per trade (`max_position_pct = 0.25`). Given expected low hit-rate (~45-55%), full-kelly sizing is inappropriate.

**Cooldown:** Minimum 24-hour cooldown between trades (`min_time_between_trades_hours = 24`). Prevents re-entering the same failing structure immediately after a stop-out.

**Risk-off gate (three layers):**
1. Regime must be `lateral` or `mean_reversion` (not `trend`).
2. `bmsb_bullish = True` (macro bull-market context for long-only entries) — optionally sweepable as `require_bmsb`.
3. Circuit-breaker: if portfolio drawdown exceeds 15% from equity peak, halt new entries. Already supported by the risk_manager circuit-breaker logic in the backtester.

---

## 8. Relevant Metrics

| Metric | Target / Notes |
|---|---|
| CAGR | >15% annualized on 2024-2026 OOS |
| Sharpe ratio | >0.8; below 0.6 is unacceptable given costs |
| Sortino ratio | >1.0 (long-only: upside variance is not a risk) |
| Max drawdown | Hard limit 20%; consecutive losses in trending markets are the key risk |
| Calmar ratio | >0.5 (CAGR / MaxDD) |
| Hit rate | 45-55% expected; below 40% = no edge |
| Average trade PnL net | Must be positive net of commission (0.04%) + slippage (5bps) |
| Trades per month | 2-6 expected; below 2 = insufficient statistical power |
| Market exposure | <30% time in market (mostly cash for a mean-reversion strategy) |
| Cost sensitivity | Re-run at 2x commission + 2x slippage; Sharpe must remain positive |
| Bootstrap 5th-pct Sharpe | Run 500 equity-curve resamples on OOS; report 5th-pct Sharpe |
| Per-year Sharpe stability | Decompose OOS into 2024 and 2025; both must be positive |
| OOS degradation | Target <30% train-to-OOS Sharpe degradation |

---

## 9. Concrete Sweep Changes to Implement

1. **Delete the bearish-divergence branch** (lines 91-115 of `rsi_divergence.py`). Replace with a single comment noting Phase-3 compliance.
2. **Fix `allowed_regimes`** to `["lateral", "mean_reversion"]` in PARAMS; remove from the genome param_space.
3. **Add `require_obv_confirm` parameter** (default False): `if require_obv_confirm and current.get('obv_divergence', 0) <= 0: return None`.
4. **Add `require_bmsb` parameter** (default False): `if require_bmsb and not current.get('bmsb_bullish', True): return None`.
5. **Add `bb_pct_b_max` parameter** (default 1.0): `if current.get('bb_pct_b', 0.5) > bb_pct_b_max: return None`.
6. **Add `rsi_divergence` entry to STRATEGY_REGISTRY** in `auto_evolve.py` with the genome param_space defined in Section 5.
7. **Fix signal strength** to incorporate `bb_pct_b`: `strength = min((abs(rsi_delta) / 20) * (1.0 - max(0, current.get('bb_pct_b', 0.5) - 0.2)), 1.0)`.
8. **Change divergence window split** from symmetric 50/50 halves to 33/67: first third establishes the reference low, final two-thirds contain the divergence check. This creates more temporal separation between the reference and the test.
9. **Add regime-stability check**: require regime to have been `lateral` or `mean_reversion` for at least 3 consecutive bars before entry (simple rolling consistency gate).
10. **Remove the `rsi_7` fallback** in the divergence window (`rsi_col` selection): require `rsi_14` explicitly and return None if it is missing, ensuring consistent feature usage.

---

## 10. Methodological Risks

1. **Round-1 evolutionary null result is decisive.** Zero positive fitness across 29,000 evaluations is not a sampling artifact — it means no parameter combination in a generous search space produced a profitable strategy. Structural code changes (not just parameter tightening) are necessary before re-introduction.

2. **Split-window divergence is not swing-pivot divergence.** Half-window min/max comparisons fire on any transient dip/peak. They produce a high rate of false divergences relative to true swing-pivot divergences. A proper zigzag or rolling-extremum implementation would be more reliable but requires structural refactoring.

3. **Low trade count creates wide confidence intervals.** With all filters active, expect 2-4 trades per month. Over the 2-year OOS window, this yields 50-100 trades — below the ~200 trades needed for statistically robust Sharpe estimation. Bootstrap CIs will be wide and the "true" edge is underdetermined.

4. **Regime label dependency.** The strategy gates on an externally provided regime string. The regime classifier's accuracy, lag, and volatility of switching are not modeled. A misclassified trending market labeled as "lateral" bypasses the primary safety filter.

5. **ATR stop in volatile crypto mean-reversion.** A 1.5-4x ATR stop on a mean-reversion trade places the stop below recent lows. In high-volatility crypto, a normal counter-trend extension easily hits the stop before the bounce materializes, creating a sequence of losses even when the divergence pattern eventually proved correct.

6. **RSI EWM look-back at walk-forward boundaries.** The RSI implementation uses an exponentially weighted moving average (EWM) with infinite memory (min_periods=14 only sets the minimum, not a cap on the kernel). RSI values near train/test boundaries retain influence from distant historical bars, creating a subtle cross-boundary dependency not fully addressed by the 120-bar embargo.

7. **1-bar execution lag eliminates the best entry.** Divergence bounces often begin within the signal bar itself. The 1-bar fill delay (required by the backtester for honest execution) means the actual entry is at the following bar's open, frequently after the initial bounce has already occurred. Combined with slippage, this structural disadvantage may consume the thin mean-reversion edge.

8. **Boolean filter combinatorial overfitting.** Three new sweepable boolean parameters (`require_obv_confirm`, `require_bmsb`, `require_volume_decline`) combined with `bb_pct_b_max` create 8 discrete filter combinations on top of 9 continuous parameters. The genetic algorithm will find the boolean combination that best fits the training folds, which is likely to be spurious.

---

## Classification: DOUBTFUL

The bullish-divergence branch has theoretical merit as a mean-reversion signal, but the Round-1 evolutionary evidence (0% positive fitness across 29K evaluations) is a strong prior against any version of this strategy being viable in its current form. The split-window divergence detection is methodologically weak, the 1-bar execution lag is particularly costly for mean-reversion setups, and the expected trade count is too low for robust OOS evaluation. The strategy can be re-introduced with the structural changes described above, but should be treated as a low-priority research candidate with a high probability of further pruning. It should not be allocated significant evolutionary compute budget until at least a 3-month walk-forward validation on pre-2024 data shows a positive Sharpe with >50 trades per fold.
