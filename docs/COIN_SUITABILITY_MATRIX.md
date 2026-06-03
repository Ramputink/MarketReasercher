# Coin × Strategy Suitability Matrix

**Date:** 2026-06-03 · **Universe:** 15 liquid Binance USDT pairs, 365d 1h · **Branch:** `auto/12h-loop`
**Goal:** find which coin each strategy is most suited to, to raise ROI — **honestly** (out-of-sample, liquidation-aware, multiple-testing flagged).

> Macro context: this was a brutal bear year. Buy & hold over the period: TRX 1.23x and BNB 0.96x
> were the only survivors; everything else fell hard (DOT 0.27x, ADA 0.31x, AVAX 0.39x, SOL 0.47x,
> DOGE 0.48x, XRP 0.56x, BTC 0.64x, ETH 0.72x). Long-biased strategies face a headwind everywhere.

## Reading 1 — Apply-as-is (the overfitting smoking gun)

XRP-tuned best genomes applied to each coin (genuinely out-of-sample for non-XRP). Full-period
multiple at 3× leverage, liquidation-aware. **The XRP column lights up; almost everything else is
< 1 (a loss).** A real, transferable edge would work across coins — it doesn't. This is hard proof
that the stored genomes are **curve-fit to XRP**, not durable alpha.

| strategy | XRP | ETH | BTC | SOL | DOGE | (typical other coin) |
|---|---|---|---|---|---|---|
| ichimoku_kumo | **5.90** | 5.42 | 0.56 | 1.08 | 1.17 | mostly < 0.6 |
| volatility_squeeze | **3.39** | 0.69 | 0.72 | 0.69 | 0.56 | mostly < 0.8 |
| trend_following | **2.61** | 1.27 | 0.53 | 1.93 | 0.02 | mostly < 0.5 |
| vol_regime_arb | **2.16** | 1.08 | 0.80 | 1.22 | 1.10 | ~1.0 (most consistent) |
| fisher_transform | **2.12** | 1.24 | 1.08 | 0.86 | 1.12 | ~1.0–1.3 |
| donchian_breakout | 0.67 | 0.32 | 0.35 | 0.04 | 0.24 | < 0.5 everywhere |
| chaos_trend | 1.07 | 0.50 | 0.33 | 0.12 | 0.09 | < 0.7 |

## Reading 2 — Out-of-sample (re-tuned per coin, tested on unseen 40%)

Re-search params on each coin's first 60%, test on its unseen last 40%, best survivable leverage at
DD ≤ 50%. This is the honest "best achievable per coin." Numbers collapse toward ~1.0 with scattered
winners — the classic signature of multiple-testing across 7×15 = 105 cells.

**Best coin per strategy (OOS) and its "native habitat":**

| strategy | best OOS coin | OOS mult | why it fits (honest read) |
|---|---|---|---|
| chaos_trend | BNB | 2.93x @5× | single standout — treat as luck until re-confirmed |
| volatility_squeeze | DOGE | 2.12x @5× | squeeze-breakout likes high-volatility meme coins |
| vol_regime_arb | DOGE | 1.95x @5× | **most consistent generalizer** (positive on most coins) |
| trend_following | ETH | 1.68x @2× | wants sustained trends → majors (ETH 1.68, BTC 1.59); 0.00 on choppy ADA/DOGE/BCH |
| fisher_transform | LINK | 1.68x @5× | mean-reversion oscillator → range-bound alts |
| ichimoku_kumo | ETH | 1.15x @3× | huge in-sample XRP (5.9x) **evaporates OOS** → overfit |
| donchian_breakout | BCH | 1.16x @3× | breakouts failed in the bear chop everywhere |

**Best strategy per coin (OOS):** BTC→trend_following (1.59x) · ETH→volatility_squeeze (1.72x) ·
BNB→chaos_trend (2.93x) · DOGE→volatility_squeeze (2.12x) · SOL→trend_following (1.29x) ·
LINK→fisher_transform (1.68x) · others 1.0–1.3x.

## Honest conclusions

1. **The existing genomes are XRP-overfit.** They only beat 1.0 on XRP; on other coins applied as-is
   they mostly lose. Any "great" XRP backtest number is in-sample, not a forward expectation.
2. **Re-tuning per coin gives scattered ~1.0–2.9x OOS** — but with 105 cells tested, some winners are
   statistical noise. The signal you can *defend* on mechanism: trend_following→trending majors
   (BTC/ETH), volatility_squeeze→volatile coins (DOGE/ETH), vol_regime_arb→broadest generalizer.
3. **A diversified "best-coin-per-strategy" portfolio averages ~1.8x over ~5 months** — and that 1.8x
   is *optimistically biased* by selecting the winners. **None reaches 10x.** It confirms the prior
   verdict: 10x-in-6-months is not honestly reachable; the honest ceiling is ~1.5–2x per ~6 months.
4. **To turn this into real ROI without cheating:** (a) re-validate each per-coin pick on a *second*
   untouched holdout (kills multiple-testing winners), (b) evolve genomes *per coin* rather than
   reusing XRP's, (c) match strategy to coin by mechanism (trend↔trending, squeeze↔volatile), and
   (d) size by realistic liquidation risk, not idealized stops.

*Method notes: liquidation modeled via per-trade Max Adverse Excursion (XRP had a −49.75% bar →
liquidation at 2× there); costs = 0.1% commission + 5bps slippage + perp funding drag; all "best"
numbers are out-of-sample. kama_trend excluded (run failed; weak 18-trade strategy). Full per-coin
tables in the tmp `suit_*.json` artifacts.*
