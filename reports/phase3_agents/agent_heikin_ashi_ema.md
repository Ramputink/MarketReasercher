# Strategy Analysis Report: heikin_ashi_ema — Phase 3 Spot Long-Only Conversion

**Classification: DOUBTFUL**
Rationale: pruned in Round 2 at 0.4% positive fitness across 36K evaluations (long+short combined); short branch removal further reduces signal frequency; HA window reset bug undermines the smoothing premise; only 53 total trades in evolution history.

---

## 1. Diagnosis

### What the strategy does
`heikin_ashi_ema` combines two signal sources:
1. **EMA crossover** (fast EMA span=21, slow EMA span=45) — detects trend direction via bullish/bearish cross or alignment.
2. **Heikin Ashi color confirmation** — smoothed candle color (bullish = HA close > HA open) validates EMA direction.

Entry conditions (long): EMA fast > slow (aligned or fresh cross) AND current HA bar is green AND (optionally) N consecutive green HA bars. Entry conditions (short): mirror image — EMA fast < slow AND current HA bar is red.

Filters: ADX >= 24.97 (trend strength), volume_ratio >= 1.07 (volume confirmation), regime in ["trend", "breakout"] (macro gate).

Exit: ATR-based stop (3.56x ATR below entry), ATR-based take-profit (5.64x ATR above), 48-hour time stop.

### Does it emit shorts?
**Yes.** Lines 121–122 and 126 explicitly set `side = "short"` when bearish EMA alignment coincides with bearish HA color. Lines 151–153 compute `stop_loss = close + stop_loss_atr_mult * atr_val` and `take_profit = close - take_profit_atr_mult * atr_val` for the short case. The Signal object returned carries `side="short"`. This is fully forbidden in Phase 3.

### Edge hypothesis
The edge is noise-reduced trend continuation: HA smoothing reduces whipsaw signals that a raw EMA cross produces. When both EMA alignment and HA color agree, the probability of a genuine trend continuation episode is elevated relative to a pure EMA signal.

### Weaknesses
- HA window reset per call: `_compute_heikin_ashi` initializes `ha_open[0]` from scratch on each invocation, so the HA open values do not accumulate true history. This is a significant logic flaw.
- Extremely few trades: 53 trades over ~7 years = ~7.6/year. Long-only branch alone may generate 20–35 long trades, too few for robust statistics.
- **Pruned in Round 2 at 0.4% positive fitness.** This is the hardest evidence against the strategy.
- EMA computed on rolling slice, not full series — introduces initialization error in crossover detection.
- ADX value of 24.9722 is a precision-overfit evolved parameter.

---

## 2. Spot Long-Only Edge Hypothesis

In spot BTC/ETH, momentum regimes produce prolonged directional runs where Heikin Ashi candles stay consistently green for 5–20+ consecutive bars. When this HA persistence coincides with a bullish EMA alignment, the probability of a continued upside move over the next 12–72 hours exceeds the base rate. The long-or-cash version:
- Enters long when HA color and EMA confirm bullish momentum.
- Returns to cash when bearish HA/EMA conditions arise (rather than going short).
- Uses ATR-based stops to limit drawdown during false trend signals.

This is a plausible edge but lacks direct empirical support in the existing evolution data — prior positive-fitness rate of 0.4% is the critical caveat.

---

## 3. Market Techniques That Benefit This Strategy

| Technique | Rationale |
|-----------|-----------|
| Heikin Ashi candle smoothing | Reduces noise; green HA requires sustained buying over prior bars |
| Dual EMA crossover | Low-lag trend direction signal with fresh-cross priority |
| ADX trend gate | Filters out choppy/range-bound environments where HA flips frequently |
| Volume ratio confirmation | Ensures institutional participation validates the move |
| ATR-based stop and TP | Adapts risk management to current volatility regime |
| Time stop (hours) | Prevents dead exposure when neither stop nor TP is reached |
| Regime classifier gate | Macro filter for trend/breakout vs mean-reversion/choppy |
| Bull Market Support Band | Long-term macro filter; only take longs above BMSB |
| Consecutive HA bar filter | Reduces false positives from single-bar HA color changes |
| Volatility-targeted sizing | Reduces size in high-vol spike regimes (GK vol target) |

---

## 4. Useful Features

### Price / Return
- `ret_zscore_20`: avoid entries during anomalous return spikes (fat-tail filter)
- `dist_from_high_20`: avoid chasing entries near 20-bar highs (overextended)

### Volatility
- `atr_14`: existing stop/TP sizing — retain
- `bb_bw_percentile`: low percentile (compression) before entry suggests pending breakout
- `gk_vol`: Garman-Klass vol for position sizing target

### Volume
- `volume_ratio`: existing confirmation filter — retain
- `obv_divergence`: OBV above 20-SMA confirms demand-side accumulation
- `volume_zscore`: abnormal spike validates breakout

### Trend / Momentum
- `adx_14`: existing trend gate — retain
- `roc_12`: rate-of-change over 12 bars as secondary momentum
- `rsi_14`: overbought guard (reject long when RSI > 75); confirmation (prefer RSI 45–65 at entry)
- `bmsb_bullish`: macro bull regime gate (price above both BMSB lines)
- `bmsb_spread`: spread between SMA/EMA BMSB lines signals trend health

### Microstructure
- `buying_pressure`: (close - low) / range; confirms intra-bar buying dominance
- `pressure_imbalance`: net buying vs selling pressure as entry quality filter

### Market Regime (informative only — NOT as direct PnL)
- External regime classifier output used as existing `regime` parameter — keep as gate only

---

## 5. Parameters That SHOULD Enter the Sweep / Genome

| Parameter | Type | Low | High | Notes |
|-----------|------|-----|------|-------|
| `ema_fast` | int | 8 | 30 | Fast EMA span; broadened from evolved value of 21 |
| `ema_slow` | int | 25 | 80 | Slow EMA span; enforced > ema_fast + 10 at runtime |
| `require_consecutive_ha` | int | 1 | 5 | Consecutive bullish HA bars; higher = stricter |
| `adx_min` | float | 15.0 | 40.0 | Trend gate; rounded from overfit 24.9722 |
| `volume_threshold` | float | 0.7 | 2.0 | Volume ratio gate |
| `stop_loss_atr_mult` | float | 1.5 | 5.0 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 2.0 | 8.0 | ATR multiplier for TP; enforce TP/SL >= 1.3 |
| `time_stop_hours` | int | 24 | 120 | Max holding duration |
| `require_volume` | bool | 0 | 1 | Toggle volume gate |
| `rsi_max_entry` | float | 65.0 | 85.0 | New: overbought guard |
| `bmsb_require_bullish` | bool | 0 | 1 | New: macro regime gate via BMSB |

---

## 6. Parameters That Should NOT Be Optimized

| Parameter | Reason |
|-----------|--------|
| `require_ha_color_flip` | Evolved to False after 1073 evals; re-optimizing risks curve-fitting entry style to a specific regime artifact |
| `allowed_regimes` | Only 4–6 discrete combinations; optimizing which regime strings to allow will overfit to regime labeling artifacts |
| `require_trend_regime` | Disabling this defeats the primary choppy-market filter and will overfit during trending training periods |
| HA lookback window | Implicitly `n_consec + 2`; changing it alters HA open initialization subtly across all entries; fix at n_consec + 3 |
| Strength formula weights | The 0.5 base + 0.2 cross + 0.15 ADX>30 weights scale position size; optimizing them is equivalent to curve-fitting leverage to past vol regimes |

---

## 7. Long-Only Risk Management

### Stop Loss
Hard ATR stop: `close - stop_loss_atr_mult * atr_14`. Default 2.5x ATR for Phase 3 (tighter than evolved 3.56x). Stop is set at signal time and must never be moved adversely. The backtester tests `bar_low <= stop_loss` each bar for long exits.

### Take Profit
ATR take-profit: `close + take_profit_atr_mult * atr_14`. Default 4.0x ATR. Minimum reward/risk ratio of 1.3 enforced (TP_mult / SL_mult >= 1.3 at all times). Backtester tests `bar_high >= take_profit`.

### Trailing Stop
Optional: after unrealized gain exceeds 1.5x ATR, trail stop to breakeven + 0.5x ATR. Prevents full reversal after a significant favorable move. This requires post-entry position management logic not currently in the strategy code.

### Volatility Targeting
Use `RiskConfig.sizing_method = 'volatility_scaled'` with `vol_target_annualized = 0.20`. The backtester's `_compute_position_size` method already supports this mode (line 189–196 of backtester.py). When `gk_vol` is elevated (crypto spike), position size is automatically reduced.

### Maximum Position Size
25% of equity per trade (`config.max_position_pct = 0.25`). With signal strength 0.5–1.0, effective size is 12.5%–25%. No leverage in spot Phase 3.

### Cooldown
- Minimum 6 hours between new entry signals (RiskConfig min_time_between_trades).
- Post-stop-loss: enforce 12-hour cooldown to prevent immediate re-entry into a continuing adverse move.
- The strategy's existing `if position is not None: return None` blocks concurrent positions but has no post-loss cooldown logic — this must be added.

### Risk-Off Regime Gate
1. `bmsb_bullish == True` required (if `bmsb_require_bullish=True` param) — price above both BMSB lines.
2. `regime in ["trend", "breakout"]` — existing gate; keep mandatory.
3. `rsi_14 < rsi_max_entry` — overbought guard; default threshold 75.
4. Circuit breaker in backtester halts new entries if full-sample drawdown exceeds `max_drawdown_pct`.

---

## 8. Relevant Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| CAGR | >= 15% | Annualized, compound growth rate |
| Sharpe ratio | >= 1.0 | Annualized, risk-free = 0; prior IS was 1.36 combined long+short |
| Sortino ratio | >= 1.2 | Downside deviation only; more appropriate for long-only |
| Max drawdown | <= 25% | Peak-to-trough on equity curve; backtester computes full-sample DD |
| Calmar ratio | >= 0.5 | CAGR / maxDD |
| Hit rate | >= 45% | With TP/SL ratio ~1.6, breakeven hit rate is ~38% |
| Turnover | >= 2 trades/month | Evolved strategy had ~0.6/month — too low for robust statistics |
| Market exposure | 20%–50% | % of time in long position |
| Average trade duration | 12–96 hours | For hourly-bar trend continuation |
| Cost sensitivity | Sharpe stable at 2x costs | Re-run with doubled commission + slippage to verify edge survival |
| Per-year Sharpe stability | Positive in >= 3 of 5 calendar years | Rolling annual Sharpe on OOS equity |
| Bootstrap/MC 5th-percentile Sharpe | >= 0.5 | 1000-trial trade sequence resample; critical given ~30 long-only trades |
| OOS degradation | <= 30% degradation | WF_Sharpe / IS_Sharpe >= 0.70 |
| Profit factor | >= 1.3 | Gross profit / gross loss |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove short branch entirely**: delete `side = "short"` assignments (lines 122, 126) and short-side stop/TP calculations (lines 151–153). Add `if side == "short": return None` as a safety guard.

2. **Add rsi_max_entry filter**: after ADX check, add `if float(current.get("rsi_14", 50)) > p.get("rsi_max_entry", 75): return None`.

3. **Add bmsb_require_bullish filter**: after regime check, add `if p.get("bmsb_require_bullish", False) and not bool(current.get("bmsb_bullish", True)): return None`.

4. **Fix HA window initialization**: pre-compute a standing HA series on the full DataFrame at strategy startup (using a module-level cache keyed by `id(df)`), replacing the per-call `_compute_heikin_ashi` that resets state every invocation.

5. **Register in STRATEGY_REGISTRY**: re-add `heikin_ashi_ema` to `auto_evolve.py` as a long-only Phase 3 entry with the updated `param_space` table from Section 5.

6. **Add OBV confirmation toggle**: add optional `if p.get("require_obv_confirm", False) and float(current.get("obv_divergence", 0)) < 0: return None`.

7. **Switch to volatility-scaled sizing**: set `RiskConfig.sizing_method = "volatility_scaled"` and `vol_target_annualized = 0.20` in the sweep configuration.

8. **Add post-loss cooldown**: track last stop-loss exit timestamp in a module-level variable; return None if elapsed hours < `p.get("cooldown_hours_after_loss", 12)`.

9. **Add BMSB strength bonus**: if `bmsb_bullish` is True, add 0.1 to signal strength (capped at 1.0) to increase size during confirmed macro bull regimes.

10. **Enforce minimum trade count**: in the sweep fitness function, discard any genome producing fewer than 20 long trades over the pre-lockbox training window — insufficient trades make the Sharpe estimate unreliable.

---

## 10. Methodological Risks

1. **HA window reset bug**: `_compute_heikin_ashi` resets `ha_open[0]` every call from a fresh window. True HA requires continuous averaging across the full history. This bug means the strategy's HA signals may differ substantially from what a standard HA indicator would produce — the backtest result does not accurately represent a real HA-based system.

2. **Critical: pruned at 0.4% positive fitness**: the strongest empirical signal against this strategy is its Round 2 pruning result. With 36K+ evaluations across long+short combined, fewer than 1 in 250 genomes produced positive fitness. Re-admitting to Phase 3 requires new evidence (long-only conditional analysis) that is not currently available.

3. **Insufficient trade count**: 53 total trades over ~7 years in the full long+short version. Removing shorts may leave ~25–35 long trades — a sample too small to distinguish edge from luck. Sharpe confidence intervals at N=30 are approximately ±0.6 at 95% confidence.

4. **Precision-overfit ADX**: `adx_min = 24.9722` is a hallmark of numerical overfitting. The 4-decimal value has no economic interpretation and will not generalize. Must be rounded and treated as an approximate threshold.

5. **EMA initialization on rolling slice**: computing EMA via `.ewm(span=X, adjust=False)` on a slice starting `ema_slow + 10` bars before `bar_idx` introduces a systematic initialization offset. The EMA values will differ from a full-history EMA, and the crossover detection at line 87 may produce false signals near the slice start.

6. **Regime gate leakage risk**: if the external regime classifier used to compute the `regime` parameter has any rolling feature with a lookback that crosses the OOS boundary, the regime gate becomes a leakage channel. The embargo_bars in WalkForwardValidator (120 bars) should cover this, but must be verified.

7. **OOS lockbox contamination**: the evolved PARAMS (including `adx_min=24.9722`, `stop_loss_atr_mult=3.5641`) were produced by Round 2 evolution. If the evolution data window included any bars from the 2024–2026 OOS lockbox period, these params carry contamination. The lockbox seal date must be verified against the evolution data range before using these evolved params as Phase 3 defaults.

8. **Wide ATR multiples reduce TP hit rate**: with SL=3.56 ATR and TP=5.64 ATR, the take-profit is very far from entry. Hourly BTC/ETH typically mean-reverts within 2–4 ATR. The practical TP hit rate may be well below 30%, meaning most exits are time-stops or stop-losses, making the reward/risk ratio theoretical rather than realized.

9. **Single-regime concentration risk**: the `allowed_regimes = ["trend", "breakout"]` filter means the strategy is entirely inactive during ranging/choppy or mean-reversion regimes. In 2022–2023 prolonged bear/sideways periods, the strategy generates zero signals. While this avoids losses, it also means zero contribution to long-term compounding during 30–40% of calendar time.
