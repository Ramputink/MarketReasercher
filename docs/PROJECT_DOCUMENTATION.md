# CryptoResearchLab — Full Project Documentation (FINAL REFERENCE)

*Maintained by the autonomous improvement loop. Last update: 2026-06-23.*
*Companion docs: `OVERNIGHT_LOOP_LEDGER.md` (per-cycle reasoning), `HONEST_RETURNS_FEASIBILITY.md`
(the 10x proof), `COIN_SUITABILITY_MATRIX.md` (coin×strategy).*

---

## 0. TL;DR / FINAL RESULTS

- **Goal asked:** a system that backtests to ≥10x in 6 months, long-only crypto on Binance,
  without cheating.
- **Honest verdict:** **10x-in-6-months is NOT achievable honestly.** Every backtest that
  reaches it relies on in-sample fitting, single-window cherry-picking, or leverage with no
  liquidation modeling. Proven and adversarially verified (`HONEST_RETURNS_FEASIBILITY.md`).
- **What DOES work (the real, defensible edge):** a **long-horizon, long-only, trend-or-cash
  strategy on the real Bull Market Support Band (BMSB)**, run as a **breadth-scaled multi-coin
  portfolio**. It beats buy-and-hold AND roughly halves drawdown.

### Final results table (real costs: 0.1% commission + 5 bps slippage; daily; long-only)

| Strategy / config | window | multiple | maxDD | vs buy&hold |
|---|---|---|---|---|
| BMSB single-coin BTC | 8y full | 17.2x | 42% | 9.9x / DD 77% ✅ |
| BMSB single-coin ETH | 8y full | 22.0x | 56% | 6.3x / DD 79% ✅ |
| **BMSB portfolio (6 majors)** | 8y full | **3.55x*** | **46%** | eq-wt B&H 2.04x / DD 82% ✅ |
| **BMSB portfolio + breadth** | 8y full | **4.12x** | **40%** | ✅ better return & DD |
| **BMSB portfolio + breadth** | recent 40% (OOS) | **1.087x** | **30%** | eq-wt B&H 0.745x / DD 67% ✅ |
| Part A intraday (any variant) | OOS | <1.0x | — | ❌ no edge |

*Portfolio "3.55x full" uses the correct daily-return-compounding equal-weight calc (Cycle 6);
an earlier mean-of-curves calc reported 5.12x. Both show: beats B&H, much lower DD.*

**Recommended deployment:** breadth-scaled equal-weight BMSB portfolio across BTC/ETH/BNB/SOL/
ADA/LINK (`tools/bmsb_breadth.py`). Honest expectation: roughly market-beating long-term return
with ~half the drawdown of buy-and-hold — NOT 10x/6mo.

---

## 1. What this project is

An autonomous quant research lab: it discovers, tunes (genetic evolution), and **honestly
validates** crypto trading strategies on Binance OHLCV. Guiding principle: **anti-overfitting** —
a strategy only counts if it survives out-of-sample, walk-forward, and realistic costs. Research
and backtesting only — **no live trading, no orders, no execution keys.**

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
   WalkForwardValidator + tools/honest_eval.py (OOS + liquidation) + portfolio tools
                                      │
                 auto_evolve.py (genetic search) ─► reports/, logs/
```

**Frozen (never edit; golden-protected):** `engine/backtester.py`, `engine/metrics.py`,
`engine/features.py`, `engine/risk_manager.py`, `config.py`, `mirofish/scenario_engine.py`.
**Editable:** `strategies/*.py`, `auto_evolve.py` (registry/search), `tools/`, `docs/`.

## 3. Engine facts

- Interface: `strategy_fn(df, bar_idx, position, regime) -> Optional[Signal]`. No future leakage.
- Costs: 0.1% commission/side + 5 bps slippage, always applied.
- Sizing: `max_position_pct` caps notional as a fraction of equity (default 0.10, vol-targeted,
  no leverage, one position at a time).
- **Leverage caveat (critical):** the engine fills stops exactly at the stop price and does NOT
  model gap-through / liquidation. XRP printed a single −49.75% bar → a leveraged long is
  liquidated at ~2×. So ANY leveraged return must be measured with `tools/honest_eval.py`
  (models per-trade Maximum Adverse Excursion → realistic liquidation). Never quote a raw
  leveraged backtest multiple.

## 4. Honest evaluation tooling

- `tools/honest_eval.py` — liquidation-aware leverage frontier + **genuine** train/test OOS
  holdout (random-searches params on the first 60%, freezes, measures on the unseen 40%).
- `tools/bmsb_search.py` — one global BMSB rule via train/test (buffers/confirm).
- `tools/bmsb_portfolio.py` — equal-weight multi-coin BMSB portfolio.
- `tools/bmsb_breadth.py` — breadth-scaled portfolio + 2nd-holdout re-validation.
- `tools/bmsb_weekly_check.py` — true weekly band vs daily-140/147 approximation.
- `tools/xsec_momentum.py`, `tools/partA_search.py` — Part-A (intraday) analyses (negative).

## 5. Strategies

### Part A — Intraday micro-volatility (long-only, 1h) — CLOSED, no edge
- `mr_vwap_reversion` — buy ATR-scaled dislocations below VWAP when not trending hard
  (limits-to-arbitrage). OOS mean 0.930x; 1/10 coins >1.02x.
- `vol_expansion_long` — long on volatility expansion + breakout + momentum (TS momentum).
  OOS mean 0.902x; 0/10; train→test collapse (overfit).
- Cross-sectional long-top-k momentum basket (APT) — OOS 0.674x (loss), does not beat cash.
- **Why closed:** in a broad bear, long-only short-horizon strategies are always in the market
  and get chopped; the only long-only bear defense is CASH = the BMSB trend filter (Part B).

### Part B — Long horizon (long-only, daily) — THE EDGE
- `bmsb_long` — the REAL Bull Market Support Band (20-week SMA + 21-week EMA = 140d/147d on
  daily; the legacy feature builder mislabeled 20/21 BARS as "BMSB" — fixed). Long while price
  holds above the band, to cash on a decisive close below. Options (all off by default, tested
  & rejected as washes): `enable_vol_target`, `dip_entry`.
- **Portfolio** (`bmsb_portfolio.py`) — equal-weight BMSB across majors; diversification cuts DD.
- **Breadth-scaled** (`bmsb_breadth.py`) — scale exposure by % of majors above their band
  (lagged 1d, causal). Recommended deployment.

### Legacy strategies (mostly XRP-overfit — see COIN_SUITABILITY_MATRIX.md)
trend_following, donchian_breakout, volatility_squeeze, vol_regime_arb, ichimoku_kumo,
fisher_transform, kama_trend, chaos_trend, dual_ma, keltner_breakout, lstm_pattern.

## 6. Overnight loop findings (every accept/reject with numbers)

| Cycle | Part | Change | Result | Decision |
|---|---|---|---|---|
| 1 | B | BMSB buffer/confirm search | OOS wash (tuned 1.393x vs default 1.417x) | REJECT |
| 2 | A | intraday single-asset OOS search | OOS 0.90–0.93x, train→test collapse | REJECT |
| 3 | B | vol-targeted sizing (Barroso–Santa-Clara) | OOS wash (Calmar 0.944 vs 0.931) | REJECT |
| 4 | A | cross-sectional momentum basket (APT) | OOS 0.674x loss; Part A CLOSED | REJECT |
| 5 | B | multi-coin BMSB portfolio | beats B&H + cuts DD both windows | **ACCEPT** |
| 6 | B | breadth-scaled exposure | OOS Calmar↑ both windows; recent +; 2nd-holdout ok | **ACCEPT** |
| 7 | B | weekly band vs daily approx | 0.3% OOS diff (immaterial) | VALIDATE (keep daily) |
| 8 | B | buy-the-dip-to-band entry | OOS wash (Calmar 0.840 vs 0.836) | REJECT |

## 7. How to run

```bash
# Recommended deployment: breadth-scaled multi-coin BMSB portfolio
./venv/bin/python tools/bmsb_breadth.py
# Single-coin / portfolio BMSB
./venv/bin/python tools/bmsb_portfolio.py
# Honest leverage + OOS evaluation of any strategy
./venv/bin/python tools/honest_eval.py --strategy vol_regime_arb --asset XRP
# Genetic evolution substrate (full CPU + Metal GPU)
./venv/bin/python auto_evolve.py --hours 10 --pop-size 40 --cores 9
# Golden regression gate (must stay 12/12; uses PINNED data)
./venv/bin/python tests/test_backtester_regression.py --check
```

## 8. Data

- 1h: 15 liquid USDT pairs, 365 days (a broadly bearish sample).
- Daily: 8 majors, up to 3000 days (8 years) — captures full bull/bear cycles; required for BMSB.
- Golden gate uses a **pinned** snapshot (`data/pinned_XRP_USDT_1h.parquet`) so it never drifts
  when the live cache refreshes.

## 9. Honest limitations

- The 1h sample is one bearish year — Part A negatives are strongest there; a different regime
  could differ, but long-only short-horizon will always struggle when the market falls.
- BMSB lags in straight-vertical bull runs (it's trend-or-cash); it wins by avoiding bear DD.
- Multiple-testing: many coin×param trials were run; accepted edges are mechanism-justified and
  re-validated on a 2nd holdout, but treat absolute numbers as indicative, not guarantees.
- Backtest ≠ live. No execution, slippage in fast markets, or exchange risk is fully modeled.

## 10. The 7 classic finance papers → what we actually use

- **Markowitz MPT (1952):** the BMSB portfolio + breadth-scaling (diversification, risk control).
- **APT (Ross 1976):** cross-sectional momentum basket (tested, negative here).
- **CAPM (Sharpe):** beta-to-BTC awareness; vol targeting (tested).
- **Limits to Arbitrage (Shleifer–Vishny 1997):** rationale for `mr_vwap_reversion` (tested).
- **Black–Scholes (1973):** its volatility lineage (realized-vol estimators) feeds sizing/triggers.
- **Fama EMH (1970):** the null hypothesis → enforces the out-of-sample discipline used throughout.
- **Agency Theory (Jensen–Meckling 1976):** corporate finance; **not applicable** to this bot.
