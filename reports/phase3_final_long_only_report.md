# Phase 3 — Spot Long-Only — Final Report

**Date:** 2026-06-08 · **Spec:** `3.0.0-p3` · **Lockbox:** 2024-06-07 → 2026-06-07 (sealed, touched once)

## Objective
Build spot **long-only** (long-or-cash, no shorts / leverage / perps / borrow / funding-PnL) strategies, evolved on all available history (2019-04 → 2024-06) and judged once on the sealed last 2 years.

## What was done
- **Preflight (Fase 0):** 21 per-strategy research agents → orchestrator plan → 3 adversarial design testers. The testers caught real blockers (1-year snapshot, silent unsealed-fallback leak, LSTM models trained through the lockbox, funding/long-only not threaded into the walk-forward validator).
- **Iter 1 — infrastructure hard-gates:** `long_only` engine flag (drops every non-long entry; short signal vs an open long exits to cash), threaded into the WF validator too; perp funding forced to 0; `load_data` made fail-closed (raises instead of silently leaking); multi-year snapshot rebuilt (2019→2026), lockbox = last 730 days. Verified end-to-end: short-emitting strategies produce **0 non-long trades, 0 funding**.
- **Iters 2-6 — evolution:** genetic search of all candidate strategies, long-only, with walk-forward selection on pre-lockbox data only. Iter 4 added a **macro risk-off gate** (long only while price is above its multi-week trend; A/B-validated: +86% Sharpe/trade, lower drawdown on identical genomes). Iter 5-6 enabled compounding warm-start.
- **Iter 7-8 — freeze:** in-sample pre-screen (≥40 trades, +Sharpe, survives 2× costs, +cross-asset, per-year stability). Exactly **1 of 10** strategies passed (`keltner_breakout`); `kama_trend` carried as exploratory.

## Final sealed-lockbox result (2-year OOS)

Buy & hold XRP/USDT over the window: **+131%** total, **−71% max drawdown**.

| Candidate | Tier | Trades | CAGR | Sharpe | maxDD | Calmar | Cross-asset OOS Sharpe | Acceptance |
|---|---|---|---|---|---|---|---|---|
| `keltner_breakout` | primary | 33 | +0.5% | 0.18 | −3.7% | 0.14 | **−0.13** (median) | 5/9 → **REJECT** |
| `kama_trend` | exploratory | 10 | +0.9% | 0.44 | −2.5% | −0.21 | −0.21 | 6/9 → exploratory |

### Why the primary candidate is rejected
1. **Too few trades OOS** — 33 (< 40). The macro gate made it ultra-selective; the sample is too small to trust.
2. **Does not generalize across assets** — median cross-asset OOS Sharpe is **negative** (−0.13). It fits XRP, not a transferable edge — a hard rejection criterion.
3. **Economically marginal** — CAGR ~0.5%, i.e. essentially flat. It is "mostly in cash with tiny gains."

### The one genuinely positive observation
Both candidates held a **−3 to −4% max drawdown vs buy-and-hold's −71%**. The macro risk-off gate *works as risk control* — it sits in cash through crashes. But the strategies are so selective and flat that this risk control protects almost nothing worth holding. Low risk on a ~0% return is not an edge.

## Verdict
**No spot long-only technical strategy in this universe passes the acceptance criteria on the sealed 2-year OOS lockbox.** This is consistent with the entire project: technical signals — long-only, short, and market-neutral — show no robust, transferable, certifiable edge on crypto. The single real signal the project ever found was Phase-2 funding-carry, which is structurally a short/perp strategy and is explicitly out of scope here.

The result is *honest*, not a bug: the infrastructure is verified leak-free and long-only, the lockbox was sealed and read once, and the macro gate demonstrably improved candidates in-sample. The strategies simply have no durable long-only edge.

## Recommendation: **DISCARD** (do not paper-trade these candidates)
- **Do not** deploy or paper-trade `keltner_breakout`/`kama_trend`: negative cross-asset OOS and ~0% return make them indistinguishable from holding cash.
- **Keep the infrastructure**: the long-only engine gate, sealed multi-year lockbox, fail-closed data path, and macro risk-off feature are sound and reusable.
- **Most promising real direction** (separate research, requires revisiting the no-short constraint): the Phase-2 funding-contrarian signal was the only edge with a pulse (7/8 positive years OOS). A spot-compatible adaptation would be a **long-only "buy capitulation" filter** — use extreme-negative funding as an *entry feature* for spot longs in risk-on regimes — rather than the long-only technical strategies tested here.
- If staying strictly long-only-technical: the honest conclusion is there is no edge to deploy; **continue research** only with a structurally different signal class, not more parameter tuning of these indicators.

## Artifacts
`reports/phase3_lockbox_2y_long_only_results.json` · `reports/phase3_lockbox_2y_scorecard.csv` · `reports/phase3_lockbox_2y_equity_curves.json` · `reports/phase3_locked_candidates.json` · `reports/phase3_frozen_candidates.json` · 21 agent reports + orchestrator plan + preflight testers in `reports/phase3_agents/`.
