# dual_ma — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy mechanics:** `dual_ma` implements a classic fast/slow exponential moving-average crossover on the close price. At each bar (if flat and regime allows), the strategy computes EMA(fast_period) and EMA(slow_period) over a trailing window of `slow_period + 5` bars. A **golden cross** (fast EMA crosses above slow EMA) generates a LONG entry; a **death cross** (fast crosses below slow) generates a SHORT entry.

**Current evolved parameters (gen 447):** EMA(20)/EMA(46), ADX ≥ 26.5, volume_ratio ≥ 1.07, price above slow EMA, regime ∈ {trend, breakout}, stop at 3.48×ATR, TP at 5.64×ATR, time stop 36h.

**Does it emit shorts? YES.** Lines 89–99 of `dual_ma.py` emit `Signal(side="short")` on every death cross. This branch must be eliminated for Phase-3 compliance. The death cross should become a cash signal only.

**Edge:** Trend-following. The ADX filter restricts entries to confirmed directional regimes; volume confirmation filters out low-participation crossovers. The wide ATR stops give trades room to breathe. The regime gate reduces entries in choppy/bear periods.

**Weakness:** The core MA crossover is a lagging indicator by construction — entries occur after the move is partially underway. With only 41 evolved trades over multi-year data, the statistical base is insufficient to distinguish genuine alpha from curve-fitting. The OOS walk-forward Sharpe (3.13) exceeding IS Sharpe (1.84) is a statistical implausibility that signals the test window coincided with a favorable regime, not structural generalization. Parameter values carry multiple decimal places of genetic precision (adx_min=26.5476, volume_threshold=1.0685, stop_loss_atr_mult=3.4827) — classic overfitting signatures.

**EMA calculation artifact:** The strategy recalculates EMA from scratch on a sub-slice `df["close"].iloc[bar_idx - slow_period - 1 : bar_idx + 1]` on every bar (lines 56–59). This means the EMA warm-up history is only `slow_period + 5` bars, not the full price history. The resulting EMA values differ (sometimes materially at the beginning of each sub-window) from a full-history EMA, creating subtle signal timing inconsistencies.

---

## 2. Spot Long-Only Edge Hypothesis

Crypto assets exhibit persistent multi-week uptrends driven by adoption cycles, macro risk-on flows, halving-related supply shocks, and institutional rotation. A golden-cross signal on EMA(fast)/EMA(slow) — conditioned on confirmed trend strength (ADX above threshold), elevated volume (confirming broad participation), and price already above the slow MA — identifies the moment when a trend is both measured and broadly supported.

The long-only edge is: **enter when the trend is confirmed, step to cash when it inverts.** A death cross does not justify shorting; it justifies capital preservation. The combination of regime filter, ADX gate, and BMSB macro overlay can suppress most bear-market re-entries, keeping the strategy in USDC during structural downtrends while capturing the upside convexity of crypto bull cycles.

The asymmetry that benefits long-only spot: crypto up-moves are faster and larger than down-moves on average (positive skew in bull cycles). A trend-following strategy that avoids being short during violent upside reversals can capture this skew while limiting drawdown to the time between the crossover reversal and the time-stop exit.

---

## 3. Market Techniques

1. **EMA golden/death cross trend-following (long-or-flat only)** — core signal; death cross becomes flat-to-cash transition
2. **ADX trend-strength gating** — filter choppy environments where crossovers produce whipsaw; keep ADX minimum as evolved structural filter
3. **Volume-ratio confirmation** — distinguish breakout-quality crossovers from low-participation noise signals
4. **Regime classification gate** — engine-level trend/breakout regime restricts entries to high-probability environments
5. **Bull Market Support Band (BMSB)** — 20-period SMA / 21-period EMA macro overlay; price below both bands signals structural bear; suspend all new longs
6. **RSI overbought filter** — reject entries where RSI-14 > 70 to avoid chasing momentum peaks; improves average entry price
7. **BB bandwidth percentile filter** — avoid entries during volatility compression (low BB bandwidth percentile < 20); direction unclear, breakout quality low
8. **ATR-scaled stops and take-profits** — volatility-adaptive risk management; prevents fixed-dollar stops being gapped in high-vol environments
9. **Time stop (36-120h)** — prevent indefinite capital lock-up in stalled trades; free capital for next valid setup
10. **Volatility-targeted position sizing** — scale position inversely to realized volatility (gk_vol); reduces size in explosive vol regimes, increases in calm trending ones

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Trend | `adx_14` | Primary trend-strength gate (already used) |
| Trend | `bmsb_bullish` | Macro regime overlay; suspend longs when False |
| Trend | `roc_12` | Rate-of-change momentum confirmation |
| Momentum | `rsi_14` | Overbought filter at entry (reject RSI > 70) |
| Momentum | `price_acceleration` | Confirm crossover backed by accelerating price |
| Volume | `volume_ratio` | Primary volume confirmation (already used) |
| Volume | `obv_divergence` | Secondary check: OBV trend agrees with price |
| Volume | `volume_zscore` | Detect genuine volume spikes vs. noisy ratio |
| Volatility | `atr_14` | Stop/TP scaling (already used) |
| Volatility | `bb_bw_percentile` | Avoid entries during volatility compression |
| Volatility | `gk_vol` | Garman-Klass vol for volatility-targeted sizing |
| Mean-reversion | `bb_pct_b` | Reject entries at extreme stretch (bb_pct_b > 0.9) |
| Market regime | engine `regime` param | trend/breakout gate (already used) |
| Drawdown | `dist_from_high_20` | Breakouts near 20-bar highs are higher quality |
| Microstructure | `pressure_imbalance` | Buying pressure - selling pressure corroborates bullish crossover |
| Return | `ret_zscore_20` | Filter extreme-zscore entries (already overbought) |

---

## 5. Parameters to Enter the Sweep/Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `fast_period` | int | 5 | 30 | Slightly wider than registry [5,25] |
| `slow_period` | int | 20 | 80 | Extended upper bound for slower trends; enforce fast < slow |
| `signal_type` | choice | — | — | ["sma", "ema"] |
| `adx_min` | float | 15.0 | 35.0 | Explore sensitivity around evolved 26.5 |
| `volume_threshold` | float | 0.8 | 2.0 | Same as registry |
| `stop_loss_atr_mult` | float | 1.5 | 5.0 | Wider upper bound for crypto volatility |
| `take_profit_atr_mult` | float | 2.0 | 8.0 | Same as registry |
| `time_stop_hours` | int | 24 | 120 | NEW: evolved 36h too short for multi-day moves |
| `rsi_max_entry` | float | 60.0 | 80.0 | NEW: overbought rejection filter |
| `bmsb_filter` | bool | — | — | NEW: macro regime gate toggle |
| `bb_bw_percentile_min` | float | 10.0 | 40.0 | NEW: compression filter |
| `pullback_tolerance_pct` | float | 0.1 | 1.0 | Price-above-slow tolerance |

---

## 6. Parameters That Should NOT Be Optimized

- **`require_adx`** — fix to `True`; ADX filtering is structurally sound; the boolean adds a degree of freedom with no theoretical benefit to removing it
- **`require_volume`** — fix to `True`; volume confirmation is a well-grounded structural filter; sweeping it only adds noise trades
- **`require_price_above_slow`** — fix to `True`; entering below the slow MA invites mean-reversion entries disguised as trend entries
- **`require_trend_regime`** — fix to `True`; always gate on regime
- **`allowed_regimes`** — fix to `["trend", "breakout"]`; sweeping across 2^4 regime subsets massively overfits to historical regime label sequences
- **`time_stop_hours` at sub-hour granularity** — if swept, round to 12-hour increments (24, 36, 48, 60, 72, 96, 120h) to avoid curve-fitting to specific historical bar sequences
- **`adx_min` at 4-decimal precision** — the evolved value 26.5476 is an overfitting artifact; round to nearest 0.5 in any sweep

---

## 7. Long-Only Risk Management

**Hard stop:** `entry_price - stop_loss_atr_mult × atr_14`; floor at 1.5× ATR minimum to prevent hairline stops being gapped through. Already implemented in Signal constructor.

**Take-profit:** Fixed ATR-scaled TP already implemented. For long-only spot where upside can be multi-hundred percent, consider partial take: 50% at 3× ATR, trail remainder. This avoids full exit at the first TP in a strong bull run.

**Trailing stop:** After price moves 2× ATR in favor, trail stop at `entry_price + 1× ATR` (breakeven + cushion). This captures extended trend moves that would otherwise be fully exited at fixed TP.

**Volatility targeting:** Compute annualized vol from `gk_vol`; set `position_size = min(equity × 0.20 / asset_annualized_vol, equity × 0.80)`. Reduces size in explosive vol regimes. Do not sweep the vol target itself — fix at 20% annualized.

**Max position size:** 80% of equity in spot (no leverage). Reserve 20% USDC as cost/drawdown buffer. `max_position_pct = 0.80` in risk config.

**Cooldown:** 12-hour minimum between any exit and the next entry signal evaluation. Extend to 24 hours after a stop-loss exit specifically (stop-outs often indicate a failed breakout in a choppy market where a second immediate entry would likely also fail).

**Risk-off regime gate:** Dual gate — (1) `bmsb_bullish == False`: suspend all new long entries; (2) engine `regime` ∉ {trend, breakout}: suspend new entries. Both gates must be True simultaneously for an entry to be considered. This provides a structural bear-market shield without requiring shorting.

---

## 8. Relevant Metrics

| Metric | Target | Notes |
|---|---|---|
| CAGR | > 20% annualized | Over OOS 2024-06-07 to 2026-06-07 window |
| Sharpe ratio | > 1.2 after all costs | Hourly returns, annualized |
| Sortino ratio | > 1.5 | Long-only: only downside vol penalized |
| Max drawdown | < 25% | Full equity curve, peak-to-trough |
| Calmar ratio | > 0.8 | CAGR / maxDD |
| Hit rate | 40–55% | Wide ATR stops/TPs; below 35% = stop-hunting |
| Avg trade PnL net | > 0 after costs | 5bps commission + 5bps slippage per side minimum |
| Trade count | ≥ 80 OOS trades | Statistical floor; 41 IS trades is insufficient |
| Exposure time | 30–60% | Spot long-only; expect significant cash periods |
| Cost sensitivity | Sharpe > 0.8 at 15bps commission + 20bps slippage | Stress test |
| WF Sharpe stability | std(fold Sharpe) < 1.0 | High variance = sub-period dependency |
| Bootstrap 5th-pct Sharpe | > 0.5 | 1000 resample iterations on trade returns |
| Per-year Sharpe | Positive in ≥ 4 of 5 years | Especially 2022 bear year |
| OOS degradation | < 50% IS to OOS | `compute_oos_degradation` from engine |

---

## 9. Concrete Sweep Changes

1. **Remove the short branch** (lines 89–99 of `dual_ma.py`): delete the `elif fast_prev >= slow_prev and fast_now < slow_now:` block entirely. A death cross while flat = stay flat. A death cross while long = emit `Signal(side="short")` ONLY to trigger the `signal.side != position.side` exit path in the backtester (line 302 of `backtester.py`) — this exits the long cleanly without opening a short position (the engine will set `position = None` and not open a new position since `pending_entry` is only set in the new-entry branch).

2. **Fix structural booleans:** Hard-code `require_adx = True`, `require_volume = True`, `require_price_above_slow = True`, `require_trend_regime = True`, `allowed_regimes = ["trend", "breakout"]` as constants in `PARAMS`. Remove them from `STRATEGY_REGISTRY["dual_ma"]["param_space"]` in `auto_evolve.py`.

3. **Add BMSB macro gate:** In the entry logic, after the regime check, add: `if p.get("bmsb_filter", True) and not current.get("bmsb_bullish", True): return None`.

4. **Add RSI overbought filter:** After ADX/volume checks, add: `if current.get("rsi_14", 50) > p.get("rsi_max_entry", 70.0): return None`.

5. **Add BB bandwidth percentile filter:** After RSI check, add: `if current.get("bb_bw_percentile", 50) < p.get("bb_bw_percentile_min", 20.0): return None`.

6. **Fix EMA calculation:** Replace the per-bar sub-slice EMA recalculation with pre-computed EMA columns in the features DataFrame (computed once, full history, by `build_all_features`). This eliminates the warm-up artifact and ensures consistent signal timing.

7. **Extend genome ranges in `auto_evolve.py`:** Update `STRATEGY_REGISTRY["dual_ma"]["param_space"]` to add `time_stop_hours: ("int", 24, 120)`, `rsi_max_entry: ("float", 60.0, 80.0)`, `bmsb_filter: ("bool",)`, `bb_bw_percentile_min: ("float", 10.0, 40.0)`, `pullback_tolerance_pct: ("float", 0.1, 1.0)`. Extend `slow_period` upper bound from 60 to 80.

8. **Enforce fast < slow constraint in genome evaluation:** In the genetic algorithm's fitness evaluation wrapper, skip any genome where `fast_period >= slow_period` and assign worst-case fitness (−∞ Sharpe).

9. **Set `funding_bps_per_8h = 0.0`** in the Backtester when running spot strategies (no perp funding costs apply to spot holdings). The current default of 1.0 bps/8h would incorrectly penalize spot simulations.

10. **Round evolved parameter precision:** In sweep initialization, round `adx_min` to nearest 0.5 and `volume_threshold` to nearest 0.05 to prevent overfitting to decimal-place artifacts from previous genetic runs.

---

## 10. Methodological Risks

1. **Thin trade count:** 41 trades over the evolved IS run yields a Sharpe standard error of ≈1/√41 ≈ 0.16. The reported IS Sharpe of 1.84 has a 95% CI of roughly [1.5, 2.2]. The WF Sharpe of 3.13 on a similar or smaller trade count is statistically implausible — it almost certainly reflects a favorable regime coincidence, not structural alpha.

2. **OOS Sharpe > IS Sharpe is a red flag:** Genuine OOS performance degrades relative to IS. When WF OOS Sharpe (3.13) exceeds IS Sharpe (1.84), it typically indicates the test window happened to fall in a strong trending period (e.g., 2023 Q4 BTC rally) that the strategy was inadvertently tuned to avoid unfavorable periods in training, not that it generalizes better out of sample.

3. **EMA warm-up artifact (existing code bug):** The strategy recomputes EMA on a sub-slice of length `slow_period + 5` at every bar. This produces different EMA values than a full-history computation (the first few values of the exponential filter differ materially). Signals can fire or not fire on specific bars depending on this artifact, creating a fragile signal boundary that may not persist in live trading where the EMA has full history.

4. **Genetic overfitting at decimal precision:** Parameters evolved to 4+ decimal places (adx_min=26.5476, volume_threshold=1.0685, stop_loss_atr_mult=3.4827) indicate the optimizer found a precise hole in the historical data, not a robust parameter region. These should be regularized by rounding and confirmed to be robust to ±10% perturbation.

5. **Regime label leakage:** If the regime classifier is calibrated on the full dataset (including the OOS lockbox), entries conditioned on `regime == "trend"` inherit that leakage. The regime classification pipeline must be verified to use only pre-lockbox calibration.

6. **Parameter dimensionality vs. trade count:** With ~12 genome parameters and 41 trades, the effective degrees of freedom per trade is insufficient for reliable parameter estimation. Fixing 5 structural parameters reduces the genome to ~7 free params, but this remains high given the trade count. Target ≥80 OOS trades before trusting parameter estimates.

7. **2022 bear market survivorship bias:** A strategy that correctly filters out most 2022 entries via regime/ADX gates will show low drawdown in 2022 but also near-zero activity — which means the equity curve in 2022 just follows the cash curve. This makes the strategy look "safe" when it is actually just inactive. The 2024–2026 OOS window is another bull period, so the strategy may look strong again for the same structural reason (trending asset, trend-following strategy) rather than for genuine skill.

8. **Cost model for spot:** The backtester `funding_bps_per_8h` defaults to 1.0 bps/8h (a perpetual swap cost). If not explicitly set to 0.0 for spot backtests, this silently inflates costs and makes spot strategies appear worse than they actually are — or conversely, if the cost was already excluded, IS metrics are slightly optimistic relative to live execution costs.

9. **Single-asset concentration:** The strategy is tested on one asset (presumably BTC or ETH). Cross-asset validation (same parameter set applied to ETH, SOL, etc.) is needed to confirm generalization beyond curve-fitting to one asset's specific volatility regime and trend structure.

10. **Time stop at 36h interacts with funding-period rhythm:** The 36h time stop is not aligned with the standard 8h perp funding period or any natural market rhythm. For spot it is irrelevant for funding, but 36h appears to be a curve-fitted value that happens to expire trades at favorable moments in the historical data. Extending to 48h or 72h and confirming robustness to this change is a required sanity check.

---

## Summary Classification: **CANDIDATE**

The dual_ma strategy has a structurally sound long-only edge hypothesis (trend-following with quality filters in a trending asset class). The core logic is convertible to Phase-3 by dropping the short branch. The primary concerns are: thin trade count undermining statistical confidence, genetic overfitting at decimal precision, the EMA warm-up artifact, and the implausibly high WF OOS Sharpe that demands scrutiny. These are fixable issues, not fundamental flaws. The strategy merits Phase-3 sweep development with the modifications described above, but should be stress-tested on 2022 bear market data and validated with ≥80 OOS trades before promotion.
