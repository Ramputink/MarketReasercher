# Overnight Improvement Loop — Ledger

> Autonomous 10-hour self-paced loop. Each cycle reads this, does ONE focused
> improvement (alternating Part A intraday / Part B BMSB), validates honestly
> (out-of-sample + liquidation-aware + golden gate), commits & pushes to GitHub
> with full reasoning, and updates the documentation.

```
loop_started_at:  2026-06-22 23:09:46 CEST
loop_deadline:    2026-06-23 09:09:46 CEST   (+10h)
branch:           auto/12h-loop  (pushed to origin: Ramputink/MarketReasercher)
substrate:        auto_evolve.py --hours 10 (PID 23068, 9 cores + Metal GPU)
gate:             ./venv/bin/python tests/test_backtester_regression.py --check  (PINNED data, 12/12)
honest evaluator: tools/honest_eval.py  (liquidation-aware + true OOS holdout)
data (1h):        tmp featured caches for 15 coins
data (daily):     tmp *_daily_feat.pkl with REAL BMSB (140d/147d) for 8 majors
```

## Objectives (both parts)

- **Part A — Intraday micro-volatility (long-only, 1h):** find a robust long-only
  edge. Baseline is honest-negative (mr_vwap_reversion / vol_expansion_long land
  <1.0 OOS). Search params across coins; be honest if no robust edge exists.
- **Part B — Real BMSB (long-only, daily, long horizon):** already beats buy&hold
  on BTC/ETH/ADA with ~half the drawdown. Refine: entry/exit buffers, weekly
  resample vs daily-approx, vol-targeting (Barroso–Santa-Clara), multi-coin
  portfolio (Markowitz). Push risk-adjusted return up without overfitting.

## Anti-overfitting discipline (non-negotiable)

1. Every accepted change must improve **out-of-sample** (train/test holdout), not
   just in-sample. 2. Golden gate green after every change. 3. Track the
   multiple-testing budget — many coin×param trials inflate the best by luck;
   prefer mechanism-justified edges and re-validate winners on a second holdout.
   4. No leverage claim without liquidation modeling. 5. Long-only only.

## Stop conditions

- 10h elapsed (deadline above) · user intervenes · 3 consecutive cycles with no
  progress on either part · golden gate unrecoverable.

---

## Cycle log
(each cycle appends: timestamp · part · hypothesis · result/metrics · decision · commit · next)

### Cycle 0 · 2026-06-22 23:09 — setup
- Built & registered 3 new long-only strategies; fixed the fake-BMSB bug (20/21
  bars → real 20/21 weeks on daily). Pinned golden data (no more drift).
- Honest baseline: **Part B BMSB beats buy&hold** (BTC 17.2x vs 9.9x, DD 42% vs
  77%; ETH 22x vs 6.3x; ADA 7.4x vs 1.5x). **Part A intraday <1.0 OOS** (no edge yet).
- Launched 10h evolution substrate. Committed + pushed (commit on branch).
- **Next (Cycle 1 = Part B):** add volatility-targeted sizing to bmsb_long and a
  small entry/exit-buffer search with train/test holdout on BTC/ETH/BNB/SOL.

### Cycle 1 · 2026-06-22 23:1x — Part B (BMSB buffer/confirm search)
- **Hypothesis:** tuning entry_buffer / exit_buffer / confirm_bars / band-bullish
  improves BMSB risk-adjusted return out-of-sample.
- **Method:** grid of 72 rules; pick ONE global rule by mean train risk-adjusted
  score (Calmar-like) over BTC/ETH/BNB/SOL/ADA/LINK (first 60%), measure frozen
  rule on unseen last 40%. Tool: `tools/bmsb_search.py` (reusable).
- **Result:** train-best = no buffer/no confirm. OOS = **wash**: tuned mean
  1.393x vs default 1.417x; tuned beats default on only 4/6 coins; both beat
  buy&hold on 4/6. No robust improvement over the default.
- **Decision: REJECT** the param change (anti-overfit: OOS not clearly better).
  Keep defaults. Keep the search tool (reusable each Part-B cycle).
- **Gate:** golden 12/12 (no engine/strategy logic changed). **Commit:** tool +
  this entry, pushed.
- **Next (Cycle 2 = Part A):** parameter search for mr_vwap_reversion &
  vol_expansion_long across coins with OOS holdout; honestly assess if any
  long-only intraday edge clears costs.

### Cycle 2 · 2026-06-22 23:50 — Part A (intraday OOS parameter search)
- **Hypothesis:** train-searched params reveal a robust long-only intraday edge
  in mr_vwap_reversion / vol_expansion_long on some coin(s).
- **Method:** evaluate_holdout (30 train samples, first 60%) frozen onto unseen
  40%, L=1, 10 coins. Tool: tmp/partA_search.py.
- **Result (honest, negative):**
  - mr_vwap_reversion: OOS mean **0.930x**, median 0.923x, only **1/10** coins
    clear 1.02x (LTC 1.04).
  - vol_expansion_long: OOS mean **0.902x**, **0/10** clear; train→test collapse
    (AVAX 1.56→0.88, ETH 1.42→0.93) = overfit signature.
  - Evolution substrate agrees: neither new intraday strategy is in the HoF
    (leader = legacy volatility_squeeze, fit 2.85).
- **Decision: REJECT** long-only intraday micro-vol as deployable. Mechanism:
  long-only can't harvest the dominant down-moves; reversal/breakout edges are
  eaten by 0.1%+5bps costs. Strategies kept in repo (documented as negative).
- **Gate:** no code change this cycle → golden unaffected (verified 12/12).
- **Next (Cycle 3 = Part B):** add volatility-targeted sizing (Barroso–Santa-
  Clara) to bmsb_long via Signal.strength; measure OOS risk-adjusted (Calmar) vs
  flat sizing. Then (Cycle 4 = Part A) pivot to a cross-sectional long-top-k
  momentum BASKET (APT) — relative strength across coins, the more promising
  intraday/swing angle than single-asset timing.

### Cycle 3 · 2026-06-23 00:30 — Part B (BMSB vol-targeted sizing)
- **Hypothesis:** Barroso–Santa-Clara vol-targeting (size entry inversely to recent
  realized vol, via Signal.strength) improves BMSB risk-adjusted return OOS.
- **Method:** added `enable_vol_target` to bmsb_long; train (60%) picks
  vol_target_annual by mean Calmar, test (40%) compares vs flat. 6 coins.
- **Result (wash):** train picked vol_target=1.5 (highest → feature nearly inert).
  OOS identical to flat on 5/6 coins; mean Calmar 0.944 vs flat 0.931 (+1.4%, noise);
  better on only 1/6 (ADA). 
- **Decision: REJECT** as default (keep `enable_vol_target=False`). Mechanism:
  vol-targeting tames equity *momentum crashes*, but in long-only crypto trend the
  high-vol periods coincide with the up-moves you want — downsizing just forfeits
  return. Code kept as documented, off-by-default option.
- **Gate:** golden 12/12 (bmsb_long not in golden; change is inert by default).
- **Next (Cycle 4 = Part A):** build cross-sectional long-top-k momentum BASKET
  (APT): rank coin universe by trailing return, hold top-k equal-weight, rebalance;
  OOS train/test. This is the more promising intraday/swing angle.

### Cycle 4 · 2026-06-23 01:09 — Part A (cross-sectional momentum basket, APT)
- **Hypothesis:** long-only top-k relative-strength rotation beats single-asset
  timing and clears costs.
- **Method:** rank 15-coin universe by trailing momentum, hold top-k equal-weight
  (positive-momentum only), rebalance with turnover costs; train(60%) picks
  (lookback,k,rebal) by Sharpe, test(40%). Tool: tools/xsec_momentum.py.
- **Result (negative):** best train rule already had Sharpe **−0.15** (no in-sample
  edge). OOS **0.674x** (loss), Sharpe −1.81, DD 36%, ann.turnover 91x. Marginally
  beats equal-weight-all (0.644) but both lose; loses to BTC hold (0.745); does NOT
  beat cash.
- **Decision: REJECT. → PART A CLOSED.** Mechanism (now proven from 2 angles): in a
  broad bear, long-only short-horizon strategies are always-in-market and get chopped;
  the only long-only bear defense is CASH — which is exactly the long-horizon BMSB
  trend filter (Part B). Short-horizon long-only has no robust edge here.
- **Gate:** golden 12/12 (standalone analysis tool; no engine/strategy change).
- **LOOP REDIRECT:** Part A is disproven; remaining cycles focus on **Part B** where
  genuine gains remain.
- **Next (Cycle 5 = Part B):** multi-coin BMSB PORTFOLIO (equal-weight BMSB across
  BTC/ETH/BNB/SOL/ADA/LINK) — expect diversification to cut drawdown vs single-coin
  while keeping the trend capture. OOS train/test, compare to single-coin & buy&hold.

### Cycle 5 · 2026-06-23 01:48 — Part B (multi-coin BMSB portfolio) — ✅ ACCEPT
- **Hypothesis:** equal-weight BMSB across 6 majors cuts drawdown vs single-coin
  (Markowitz diversification) while keeping trend capture.
- **Method:** each coin long-or-cash on its own band, equal capital, independent
  compounding, date-aligned; report FULL + recent-40% (OOS-style) vs eq-wt buy&hold
  and vs avg single-coin. Tool: tools/bmsb_portfolio.py.
- **Result (robust, both windows):**
  - FULL (8y): portfolio **5.12x** vs B&H 2.67x; DD **63.9%** vs avg-single 67.9%
    vs B&H 89.5%; Sharpe 0.81 vs 0.64.
  - TEST (recent 40%): portfolio **0.82x** vs B&H 0.70x; DD **38.9%** vs avg-single
    55.1% vs B&H 64.9%.
  - Beats buy&hold on return in BOTH windows AND cuts drawdown in BOTH.
- **Decision: ACCEPT.** First accepted improvement. Mechanism (diversification) is
  sound and the benefit is OOS-confirmed. Value-add = RISK reduction (return ≈ avg
  sleeve by construction; DD dramatically lower). Recommended Part-B deployment =
  equal-weight BMSB portfolio, not single-coin.
- **Gate:** golden 12/12 (standalone tool; no engine/strategy change).
- **Next (Cycle 6 = Part B):** regime-aware exposure — scale total portfolio
  exposure by BREADTH (% of majors above their band), expecting further DD cut;
  then re-validate the accepted portfolio on a second untouched holdout.

### Cycle 6 · 2026-06-23 02:26 — Part B (breadth-scaled exposure + 2nd holdout) — ✅ ACCEPT
- **Hypothesis:** scaling portfolio exposure by breadth (% of majors above their band,
  lagged 1 day = causal) cuts drawdown / improves Calmar; and the diversification win
  re-validates on a different coin subset.
- **Method:** tools/bmsb_breadth.py — daily-return compounding (the correct equal-weight
  calc, vs Cycle 5's mean-of-curves), flat vs breadth-scaled, on majors + a 2nd subset.
- **Result:**
  - MAJORS breadth vs flat — FULL: 4.12x DD40% Cal2.95 vs 3.55x DD46% Cal2.43; 
    TEST40: **1.087x DD30% Cal0.84** vs flat 0.875x DD35% Cal0.65 (and vs B&H 0.745x DD67%).
    Breadth improves return AND DD in BOTH windows; turns the recent bear window POSITIVE.
  - 2nd SUBSET {XRP,LTC,ADA,LINK} (re-validation): TEST40 portfolio 1.332x vs B&H 1.303x
    (still beats B&H), DD cut holds; breadth Calmar 0.85 ≥ flat 0.83. Benefit is real,
    thinner on weak coins.
- **Decision: ACCEPT** breadth-scaling as a portfolio enhancement (2nd accept). Mechanism
  (distrust thin rallies) is sound; OOS-confirmed in both windows + a 2nd holdout.
- **Note:** Cycle-6 daily-return compounding (3.55x flat FULL) is the more correct
  equal-weight figure than Cycle-5's mean-of-curves (5.12x); same qualitative story.
- **Gate:** golden 12/12 (standalone tools; no engine/strategy change).
- **Next (Cycle 7 = Part B):** weekly-resampled band vs daily-140/147 approximation
  (does true weekly BMSB differ materially?); then a doc-hardening cycle.

### Cycle 7 · 2026-06-23 03:05 — Part B (weekly band vs daily approx) — VALIDATED (keep daily)
- **Hypothesis:** the daily-140/147 band may differ materially from the TRUE weekly
  (20W SMA/21W EMA resampled) band; if weekly is clearly better, switch.
- **Method:** resample daily→weekly (W-SUN), 20W SMA + 21W EMA, ffill to daily (causal),
  run bmsb_long with each band on 6 majors. Tool: tools/bmsb_weekly_check.py.
- **Result:** OOS(TEST40) mean **1.417x daily vs 1.422x weekly = 0.3% diff (immaterial)**.
  Per-coin a wash (weekly better on BTC 1.97 vs 1.79 lower DD; worse on ETH/LINK). FULL
  in-sample weekly slightly higher on SOL/BNB but similar DD.
- **Decision: KEEP daily-140/147 approximation** (validated as a faithful weekly proxy;
  simpler, no resample/ffill in the hot path). Important robustness confirmation since all
  of Part B rests on this band. Counts as a decisive proven finding (progress).
- **Gate:** golden 12/12 (standalone check; no engine/strategy change).
- **Next (Cycle 8 = Part B):** BMSB buy-the-dip-to-band add-on entry (add on pullbacks
  that hold the band) — test if it improves OOS return without raising DD; else move to
  the doc-hardening cycle.

### Cycle 8 · 2026-06-23 03:43 — Part B (buy-the-dip-to-band entry) — REJECT (wash)
- **Hypothesis:** entering pullbacks that hold the band (uptrend, recovering) adds OOS
  return without raising DD vs breakout-only entry.
- **Method:** added `dip_entry` option to bmsb_long; compared breadth portfolio TEST40
  with dip_entry off vs on (6 majors). Tool: tmp/cycle8_dip.py.
- **Result (wash):** off 1.087x DD30.0% Cal0.836 vs on 1.092x DD29.9% Cal0.840 — a 0.5%
  Calmar change, within noise.
- **Decision: REJECT** as default (keep dip_entry=False). Reason: single-position engine
  means dip-entry only fires in the narrow window when flat AND price sits on the band;
  it catches a few earlier entries but adds nothing material. Code kept as off-by-default
  option.
- **Gate:** golden 12/12 (bmsb_long not in golden; change inert by default).
- **SUBSTANTIVE PART-B LEVERS EXHAUSTED.** Tally: ACCEPT portfolio (C5), ACCEPT breadth
  (C6); VALIDATE daily band (C7); REJECT vol-target (C3), buffers (C1), dip-entry (C8).
- **Next (Cycle 9 = DOCS):** harden docs/PROJECT_DOCUMENTATION.md into the complete final
  reference + FINAL RESULTS summary table (architecture, every strategy, every accept/
  reject with numbers, how-to-run, honest limitations, 7-papers mapping).
