# CryptoResearchLab — Full Project Documentation

*Living document. Maintained by the autonomous improvement loop. Last major update: 2026-06-22.*

## 1. What this project is

An autonomous quantitative research lab for crypto trading strategies. It discovers,
tunes (via genetic evolution), and **honestly validates** trading strategies on
Binance OHLCV data. The guiding principle is **anti-overfitting**: a strategy only
"counts" if it survives out-of-sample, walk-forward, and realistic costs. The lab does
research and backtesting only — **no live trading, no orders, no exchange keys used for
execution.**

### Honest north star
Earlier work proved that a legitimate **10x-in-6-months is not achievable** on this data
without cheating (in-sample fitting, single-window cherry-picking, or leverage with no
liquidation modeling). See `docs/HONEST_RETURNS_FEASIBILITY.md`. The realistic honest
ceiling is modest per-6-month multiples intraday, but **large multi-year multiples are
achievable on the long-horizon trend-or-cash side (BMSB)** because it rides real bull
markets while sitting out bear drawdowns.

## 2. Architecture

```
Binance (ccxt) ─► data_ingestion ─► parquet cache
                                      │
                          build_all_features (67 features, causal)
                                      │
                       classify_regime_quantitative (per-bar regime)
                                      │
   ┌──────────────────────────────────┼───────────────────────────────────┐
   │ engine/backtester.py (FROZEN)    │  strategies/*.py (strategy_fn)     │
   │ event-driven, costs, stops/TP    │  emit Signal(side, sl, tp, ...)    │
   └──────────────────────────────────┴───────────────────────────────────┘
                                      │
        WalkForwardValidator   +   tools/honest_eval.py (OOS + liquidation)
                                      │
                 auto_evolve.py (genetic search) ─► reports/, logs/
```

### Locked vs editable
- **FROZEN (never edit):** `engine/backtester.py`, `engine/metrics.py`,
  `engine/features.py`, `engine/risk_manager.py`, `config.py`,
  `mirofish/scenario_engine.py`. Protected by the golden regression harness.
- **Editable:** `strategies/*.py`, `auto_evolve.py` (registry/search), `tools/`, `docs/`.

## 3. The backtest engine (key facts)

- Interface: `strategy_fn(df, bar_idx, position, regime) -> Optional[Signal]`. No
  future leakage (strategy sees bar_idx and earlier).
- Costs: 0.1% commission per side + 5 bps slippage. Always applied.
- Sizing: `max_position_pct` caps notional as a fraction of equity. Default config is
  ultra-conservative (0.10, vol-targeted, no leverage, one position at a time).
- **Leverage caveat:** the engine fills stops exactly at the stop price and does NOT
  model gap-through / liquidation. XRP had a single −49.75% bar → a leveraged long is
  liquidated at ~2×. Therefore any leveraged return MUST be measured with
  `tools/honest_eval.py`, which models per-trade Maximum Adverse Excursion → realistic
  liquidation. Never quote a raw leveraged backtest multiple.

## 4. Honest evaluation: `tools/honest_eval.py`

The reporting gate. Provides:
- `featured(asset)` — features + causal regime for an asset.
- `leverage_frontier(...)` — liquidation-aware multiple/DD across leverages.
- `evaluate_holdout(...)` — **genuine** out-of-sample: random-search params on the
  first 60% (train), freeze, measure on the unseen 40% (test). Never uses the stored
  full-period genome for selection (that would be a contaminated holdout).
Rule: report the OOS, liquidation-aware number — not the in-sample full-period one.

## 5. Strategies

### Part A — Intraday micro-volatility (long-only, 1h)
- `mr_vwap_reversion` — buy dislocations BELOW VWAP measured in ATR units, only when
  NOT trending hard (ADX/regime filter), RSI oversold, buyers stepping in. Mean
  reversion (limits-to-arbitrage). *Status: thin/negative OOS — under search.*
- `vol_expansion_long` — long when realized volatility expands out of contraction and
  price breaks the recent high with momentum + volume. Time-series momentum.
  *Status: thin/negative OOS — under search.*
- Legacy strategies: trend_following, donchian_breakout, volatility_squeeze,
  vol_regime_arb, ichimoku_kumo, fisher_transform, kama_trend, chaos_trend, dual_ma,
  keltner_breakout, lstm_pattern (note: most are XRP-overfit — see suitability matrix).

### Part B — Long horizon (long-only, daily)
- `bmsb_long` — **real Bull Market Support Band** (20-week SMA + 21-week EMA, i.e.
  140d/147d on daily). Long while price holds above the band; to cash on a decisive
  close below. Rides bull markets, avoids bear drawdowns. *Status: beats buy&hold on
  BTC/ETH/ADA with ~half the drawdown; best honest performer in the project.*

## 6. Honest results so far

**Part B BMSB vs buy&hold (8 years daily, L=1 spot, real costs):**

| coin | BMSB | buy&hold | BMSB DD | B&H DD | beats? |
|---|---|---|---|---|---|
| BTC | 17.2x | 9.9x | 42% | 77% | ✅ |
| ETH | 22.0x | 6.3x | 56% | 79% | ✅ |
| ADA | 7.4x | 1.5x | 73% | 95% | ✅ |
| SOL | 23.3x | 48.0x | 72% | 96% | ❌ (lagged a vertical bull) |
| BNB | 21.1x | 60.6x | 87% | 76% | ❌ |

**Part A intraday (confirmed negative, Cycle 2):** OOS parameter search across 10 coins
(train 60% → unseen 40%, L=1) — `mr_vwap_reversion` OOS mean **0.930x** (1/10 coins
>1.02x), `vol_expansion_long` OOS mean **0.902x** (0/10), with a clear train→test
collapse (overfit). No robust long-only intraday edge after costs. Next angle: a
cross-sectional long-top-k momentum basket (APT), which trades relative strength across
coins rather than single-asset timing.

**Coin × strategy suitability** (see `docs/COIN_SUITABILITY_MATRIX.md`): legacy genomes
are XRP-overfit (only beat 1.0 on XRP). Mechanism-justified matches: trend→BTC/ETH,
squeeze→volatile coins, vol_regime_arb = broadest generalizer.

## 7. Data

- 1h: 15 liquid USDT pairs, 365 days (bearish sample).
- Daily: 8 majors, up to 3000 days (8 years) — enables real BMSB and captures full
  bull/bear cycles (BTC buy&hold 9.9x, ETH 6.3x, SOL 48x, BNB 61x over the period).
- Golden gate uses a **pinned** snapshot (`data/pinned_XRP_USDT_1h.parquet`) so it
  never drifts when the live cache refreshes.

## 8. How to run

```bash
# Honest evaluation of any strategy (leverage frontier + OOS holdout)
./venv/bin/python tools/honest_eval.py --strategy vol_regime_arb --asset XRP

# Genetic evolution substrate (full CPU + Metal GPU)
./venv/bin/python auto_evolve.py --hours 10 --pop-size 40 --cores 9

# Golden regression gate (must stay green after any engine/strategy change)
./venv/bin/python tests/test_backtester_regression.py --check
```

## 9. Mapping the 7 classic finance papers → what we actually use

- **Markowitz MPT (1952):** portfolio construction, vol-targeting, risk weighting.
- **APT (Ross 1976):** multi-factor / cross-sectional momentum selection.
- **CAPM (Sharpe):** beta-to-BTC awareness; vol targeting.
- **Limits to Arbitrage (Shleifer–Vishny 1997):** why short-horizon reversions persist
  (the rationale for `mr_vwap_reversion`).
- **Black–Scholes (1973):** not options — its volatility lineage (realized-vol
  estimators) feeds sizing and the vol-expansion trigger.
- **Fama EMH (1970):** the null hypothesis → enforces out-of-sample discipline.
- **Agency Theory (Jensen–Meckling 1976):** corporate finance; **not applicable** to
  systematic crypto trading.

## 10. Roadmap (driven by the overnight loop)

- Part B: vol-targeted sizing, weekly-resampled band, multi-coin BMSB portfolio,
  second-holdout re-validation.
- Part A: hard parameter search; honestly retire it if no OOS edge emerges; consider a
  cross-sectional long-top-k basket (APT) as the more promising intraday angle.
- Engine: real evolution resume (compounding multi-run search); optional liquidation
  modeling inside the live path.
