# Keltner Breakout — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis of the Strategy

### Current Logic
The strategy builds a Keltner Channel at each bar: the center is an EMA of `close` over `ema_period` bars (evolved: 19), and the upper/lower bands are offset by `atr_mult × atr_14` (evolved: 3.404). Entry conditions for both directions require:
- ADX_14 >= `adx_min` (evolved: 14.76) — trend strength gate
- `volume_ratio` >= `volume_threshold` (evolved: 1.98) — volume surge confirmation
- Regime ∈ {breakout, trend} — regime gate
- If `consecutive_bars_outside > 1`: prior bar must also have been outside the band (currently set to 1, so this check is inert)

**Long entry** (lines 69-90): `close > upper_band` AND `rsi_14 >= rsi_long_min` (evolved: 50.99). Stop-loss is anchored at `ema_center` (not a fixed ATR below entry). Take-profit is `close + take_profit_atr_mult × atr_14` (evolved: 6.24×). Time stop: 36 hours.

**Short entry** (lines 92-107): `close < lower_band` AND `rsi_14 <= rsi_short_max` (evolved: 45.40). Same stop/TP structure mirrored.

### Does It Emit Shorts?
**Yes — explicitly.** Lines 92-107 create a `Signal(side="short", ...)` when price breaks below the lower Keltner band. This branch must be removed entirely for Phase-3.

### Likely Edge
The long-side edge is momentum continuation: a close outside a wide (3.4× ATR) Keltner Channel during a trending/breakout regime with elevated volume is a genuinely informative signal in crypto. Crypto assets exhibit fat-tailed momentum; once a meaningful channel is broken with volume, continuation probability exceeds 50% on the next 12-36 bars.

### Likely Weakness
- **Stop-loss anchor bug**: `stop_loss=ema_center` creates variable-dollar risk per trade. When the breakout is small (close just above upper), ema_center may be close to entry, giving a tight stop. When the breakout is large, ema_center is far below, giving a loose stop. Position sizing under `volatility_scaled` then becomes inconsistent.
- **Small trade count**: 142 trades over ~7 years (~20/year) is statistically thin. Walk-forward fold Sharpe estimates are high-variance.
- **Boundary pressure**: `atr_mult` evolved to 3.404 against a prior ceiling of 3.5 — optimizer was likely constrained by the range boundary.

---

## 2. Spot Long-Only Edge Hypothesis

In crypto spot markets, prices that break convincingly above a wide ATR-scaled Keltner Channel (3–4.5× ATR) during confirmed trending/breakout regimes — with simultaneously elevated ADX, RSI above 50, and above-average volume — tend to exhibit short-term momentum continuation. The channel acts as a squeeze-release detector: wide channels filter out small-amplitude noise, and a close outside requires real directional velocity. Volume confirmation validates institutional participation rather than retail stop-hunting.

In a spot long-only framework, the strategy allocates to the spot asset under these conditions and holds USDC otherwise. The short branch is dropped entirely. The downward break signal becomes "no trade, stay in cash," which is appropriate: spot-only strategies benefit from BTC/ETH's long-run upward drift, so aggressive cash periods during downside breaks are correct — the opportunity cost is low because drawdowns in crypto tend to mean-revert or cascade. Being in cash during downside Keltner breaks preserves capital without the perp-funding drag of a short position.

---

## 3. Market Techniques That Can Benefit This Strategy

1. **Keltner Channel volatility breakout (ATR-based, trend-following variant)** — core mechanism
2. **ADX trend-strength filter** — prevents false breakouts in ranging/choppy markets
3. **RSI momentum confirmation gate** — RSI > 50 ensures underlying momentum aligns with the breakout direction
4. **Volume surge confirmation (volume_ratio)** — validates breakout with capital inflows
5. **Regime-conditional trading** — entry only in breakout/trend regimes; cash in mean-reverting/ranging regimes
6. **ATR-normalized stop-loss and take-profit** — volatility-adaptive exit management
7. **Time stop** — prevents indefinite holding in stalled breakouts
8. **Pre-breakout squeeze detection (bb_bw_percentile)** — Bollinger bandwidth percentile < 30 before breakout increases post-breakout momentum (squeeze → expansion)
9. **Bull Market Support Band (bmsb_bullish)** — macro long-term bull regime filter to avoid longs in structural downtrends
10. **OBV divergence confirmation** — ensures buying pressure (OBV above SMA) aligns with price breakout
11. **Volatility-targeted position sizing** — naturally scales down size during high-vol environments where breakouts fail more often

---

## 4. Useful Features

| Category | Feature | Rationale |
|---|---|---|
| Trend | `adx_14` | Primary trend-strength gate — already in use |
| Trend | `bmsb_bullish` | Macro bull/bear regime composite flag |
| Momentum | `rsi_14` | Momentum confirmation for long entries — already in use |
| Momentum | `rsi_7` | Faster RSI for more responsive confirmation |
| Momentum | `roc_12` | 12-bar rate of change as secondary momentum signal |
| Momentum | `ret_zscore_20` | Z-scored returns to filter overextended entries |
| Volume | `volume_ratio` | Volume surge confirmation — already in use |
| Volume | `volume_zscore` | Z-scored volume spike (more robust normalization) |
| Volume | `obv_divergence` | OBV vs its SMA — confirms buying pressure behind breakout |
| Volatility | `atr_14` | Channel width and stop/TP scaling — already in use |
| Volatility | `atr_7` | Shorter ATR for detecting local volatility expansion |
| Volatility | `gk_vol` | Garman-Klass realized vol for vol-targeting and regime |
| Volatility | `bb_bw_percentile` | Pre-breakout compression (low = squeeze before expansion) |
| Volatility | `bb_bandwidth` | Raw bandwidth — volatility expansion after breakout |
| Price | `dist_from_high_20` | Distance from 20-bar high — gauge breakout vs resistance |
| Price | `close_location` | Where close fell within the breakout bar (> 0.7 = strong close) |
| Microstructure | `buying_pressure` | Confirms strong closing within bar on breakout |
| Microstructure | `pressure_imbalance` | Directional buy/sell pressure differential |

Funding/sentiment features (perp funding rates, open interest, etc.) may be used **only as informative filters** (e.g., elevated funding rate = crowded long = reduce position size) and must never contribute to PnL as carry.

---

## 5. Parameters That Should Enter the Sweep / Genome

| Parameter | Type | Range | Note |
|---|---|---|---|
| `ema_period` | int | [10, 40] | EMA center of Keltner Channel |
| `atr_mult` | float | [1.5, 4.5] | Channel width; extended ceiling from 3.5 to 4.5 |
| `rsi_long_min` | float | [45.0, 65.0] | RSI threshold for long momentum confirmation |
| `adx_min` | float | [10.0, 30.0] | Trend-strength gate |
| `volume_threshold` | float | [1.0, 3.0] | Volume surge multiplier; widened upper bound |
| `stop_loss_atr_mult` | float | [1.0, 4.0] | ATR multiple from entry price (fix anchor from ema_center) |
| `take_profit_atr_mult` | float | [2.0, 10.0] | Wide TP for trend-riding; extended upper bound |
| `time_stop_hours` | int | [12, 96] | Trade duration limit |
| `rsi_lookback` | int | [7, 21] | RSI period (currently hardcoded 14) |
| `bb_squeeze_filter` | bool | [0, 1] | Require pre-breakout BB squeeze |
| `obv_confirm` | bool | [0, 1] | Require OBV above its SMA at entry |
| `bmsb_filter` | bool | [0, 1] | Require bmsb_bullish macro gate |

---

## 6. Parameters That Should NOT Be Optimized (Overfitting Traps)

- **`consecutive_bars_outside`** (= 1): Has zero effect at value 1. Increasing to 2+ drastically reduces trade count. Fix at 1.
- **`require_momentum`** (= True): Disabling degrades signal quality with no theoretical justification. Hardcode True.
- **`require_close_outside`** (= True): Required for honest bar-close confirmation. Hardcode True.
- **`volume_confirmation`** (= True): Volume confirmation is a structural risk filter. Hardcode True.
- **`require_regime`** (= True): Regime gating is the primary drawdown protection mechanism. Hardcode True.
- **`allowed_regimes`** (= ["breakout", "trend"]): Theoretically motivated; changing adds overfitting surface without mechanistic basis. Hardcode.
- **`rsi_short_max`**: Short-side parameter — irrelevant post-conversion. Remove entirely from PARAMS and genome.

---

## 7. Long-Only Risk Management

### Stop-Loss
Fix the stop anchor: `stop_loss = entry_price - stop_loss_atr_mult × atr_14`. The current `ema_center` anchor creates variable dollar risk. The backtester fills at the next bar's open, so `stop_loss` must be recalculated using the fill price, not the signal-bar close.

### Take-Profit
ATR-anchored: `take_profit = entry_price + take_profit_atr_mult × atr_14`. Genome sweeps [2.0, 10.0]. Consider a staged exit: sell 50% at 3× ATR, trail the remainder.

### Optional Trailing Stop
Activate trailing once position is `trail_activation_atr_mult × atr_14` in profit. Trail at `max(entry_stop, current_price - trail_distance × atr_14)`. Sweep `trail_activation_atr_mult` as a float parameter [1.5, 4.0] gated by a bool `use_trailing`.

### Volatility Targeting
Use `sizing_method=volatility_scaled` with `vol_target_annualized` in [0.10, 0.25]. This naturally reduces position size during high-volatility breakouts (which have higher failure rates) and increases size during calm trending conditions. Use `gk_vol` as the realized vol estimate for the sizing formula.

### Maximum Position Size
`max_position_pct = 0.95` for spot long (near-full allocation). Vol-targeting will hold realized size below this in high-vol regimes. Hard cap at 0.95 to retain a small USDC buffer for entry costs.

### Cooldown
After a stop-loss exit, impose a cooldown of `2 × ema_period` bars before re-entry. At ema_period=19 on hourly data, this is 38 hours. Wire through `risk_manager.min_time_between_trades_hours`. This prevents rapid re-entry into a failing breakout environment.

### Risk-Off Regime Gate
1. Regime gate (breakout, trend) — already implemented; keep hardcoded.
2. `bmsb_bullish` filter — no longs when price is below the Bull Market Support Band (macro bear structure).
3. Circuit-breaker: halt new entries if rolling 20-day equity drawdown from peak exceeds 20%.
4. Extreme volatility gate: if `gk_vol` is in the top 10th percentile of its 6-month rolling distribution, skip entry or reduce size by 50%. Extreme vol often precedes cascade liquidations.

---

## 8. Relevant Metrics

| Metric | Target / Purpose |
|---|---|
| CAGR | ≥ 20% annualized for spot long-only candidate |
| Sharpe Ratio | ≥ 0.8 minimum, ≥ 1.0 for promotion; compute annualized with risk-free = 0 |
| Sortino Ratio | Most relevant for long-only; target ≥ 1.2 (upside vol is desirable) |
| Max Drawdown % | Hard limit ≤ 30%; full-sample peak-to-trough on equity curve |
| Calmar Ratio | Target ≥ 0.7 (CAGR / maxDD) |
| Hit Rate | Expected 40–55% for trend-following breakout; < 35% suggests TP is too wide |
| Average Trade PnL | Must be positive net of 2× commission + slippage |
| Trade Count | Minimum 30 OOS trades aggregated across folds for statistical validity |
| Exposure % | Target 30–60% bars in position; very low exposure = missing bull drift |
| Cost Sensitivity | Re-run at 2× and 3× commission/slippage; edge must survive 3× |
| WF OOS Sharpe | OOS Sharpe vs in-sample Sharpe degradation < 40% |
| Bootstrap Sharpe CI | 1000 resamples; lower 5th percentile > 0 |
| Per-Year Sharpe | Compute for each calendar year 2019–2023 (pre-lockbox); reject if any year < −0.3 |
| Profit Factor | Target > 1.4 (gross wins / gross losses) |
| Exit Reason Distribution | Diagnose % of exits by stop_loss / take_profit / time_stop / signal_exit; time_stop > 50% indicates TP is unreachable |
| Average Bars Held | Diagnoses time_stop dominance; should be < time_stop_hours on average |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove short signal branch**: delete lines 92-107 in `keltner_breakout.py`. When `close < lower`, return `None`.
2. **Remove `rsi_short_max`** from `PARAMS` and from `auto_evolve.py` `param_space`.
3. **Fix stop-loss anchor**: change `stop_loss=ema_center` to `stop_loss=entry_price - stop_loss_atr_mult * atr` in the Signal construction. Since entry_price is only known at the fill bar (next open), pass `atr` in the signal metadata and recalculate the stop in the fill logic, or use `close - stop_loss_atr_mult * atr` as the best available estimate at signal time.
4. **Add `bmsb_filter` param** (bool): if True, require `current["bmsb_bullish"] == True` before entering. Wire after ADX check.
5. **Add `obv_confirm` param** (bool): if True, require `current["obv_divergence"] > 0` at entry.
6. **Add `bb_squeeze_filter` param** (bool): if True, require `bb_bw_percentile < 30` at any of the prior 5 bars.
7. **Add `rsi_lookback` param** (int, [7, 21]): select between pre-computed `rsi_7` and `rsi_14` based on param value (or compute inline for other values).
8. **Extend `atr_mult` sweep range** to [1.5, 4.5] in `auto_evolve.py` — prior ceiling of 3.5 was binding.
9. **Extend `take_profit_atr_mult` sweep range** to [2.0, 10.0].
10. **Add `time_stop_hours`** to genome sweep [12, 96].
11. **Add trailing stop capability**: add `use_trailing` (bool) and `trail_activation_atr_mult` (float, [1.5, 4.0]) to allow the genome to discover whether trailing improves Sortino.
12. **Set `sizing_method=volatility_scaled`** for this strategy's genome evaluations; add `vol_target` as genome param (float, [0.10, 0.25]).
13. **Enforce lockbox**: all genome sweeps and walk-forward folds must use only data before 2024-06-07.

---

## 10. Methodological Risks

1. **Small trade count / statistical fragility**: 142 trades over 7 years (~20/year) is thin. Walk-forward folds of 15 days may contain 0–3 trades each, making fold-level Sharpe estimates statistically meaningless. Require a minimum of 30 OOS trades aggregated across all folds before declaring a genome robust.

2. **Stop-loss anchor bug**: `stop_loss=ema_center` creates variable-dollar risk per trade and inconsistent vol-targeted sizing. This must be fixed before the sweep — the current evolved params were shaped by this inconsistency and may not transfer to the corrected version.

3. **Parameter boundary pressure**: `atr_mult` evolved to 3.404 against a prior ceiling of 3.5, and `take_profit_atr_mult` to 6.239 against a ceiling of 8.0. Parameters near boundaries indicate optimizer constraint. The reported WF_Sharpe=2.30 may be a lower bound if the true optimum lies outside the prior ranges.

4. **Regime label leakage**: regime labels (breakout, trend) must be computed strictly from past data. If the regime classifier uses a rolling window longer than the embargo gap (120 bars), the first val/test bars of each fold have contaminated regime labels. Verify regime computation lookback does not exceed 120.

5. **Extreme regime concentration**: `volume_threshold=1.98` + wide `atr_mult=3.404` + ADX gate may yield so few qualifying bars that the backtest is dominated by a handful of extreme events (COVID recovery rally, 2021 cycle peak, 2024 ETF rally). The apparent edge may be concentrated in 3–5 individual trades, making it non-generalizable.

6. **Time stop dominance**: if > 50% of exits are `time_stop` at 36 hours, it signals the take-profit at 6.24× ATR is rarely reached. This means the strategy's risk-reward is actually governed by time decay, not TP/SL parameters. Must diagnose exit reason distribution before interpreting TP optimization results.

7. **Crypto structural break risk**: evolved parameters span 2019–2026 across multiple structural regimes. A walk-forward including 2022 deleveraging bear data may produce regime-dependent parameters that fail if the macro cycle turns again. Per-year Sharpe stability check across 2019–2023 is essential.

8. **OBV/volume feature collinearity**: adding both `obv_confirm` and `volume_threshold` as genome params creates redundant volume signals. The optimizer may find configurations that double-count volume evidence, producing brittle edge that is sensitive to exchange-specific volume reporting methodologies.

9. **EMA recomputation at signal time**: the strategy recomputes EMA inline at each bar (lines 57-58) rather than using a pre-computed feature column. This is computationally expensive in the sweep and introduces a subtle difference from the `ema` feature in `features.py` (which uses `min_periods=period`). Standardize to use a single pre-computed Keltner column to avoid inconsistency.

---

## Classification: CANDIDATE

The keltner_breakout strategy presents a genuine and theoretically motivated long-side edge (momentum continuation post-channel-breakout with volume and trend confirmation), evolved walk-forward metrics above baseline (WF_Sharpe=2.30), and a sensible regime gate. The short branch is easily removed. However, the stop-loss anchor bug, small trade count, and possible boundary pressure on `atr_mult` must be corrected before the sweep results can be trusted. With these fixes and the expanded genome described above, this strategy merits a full Phase-3 sweep.
