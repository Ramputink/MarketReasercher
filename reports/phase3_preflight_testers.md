# Phase 3 — Preflight Design Testers


## Tester: Phase-3 preflight design review: long-only compliance + methodological/lockbox leakage, verified against the actual repo (engine/backtester.py, auto_evolve.py, benchmark/data_lockbox.py, benchmark/spec.py, mirofish/scenario_engine.py, engine/features.py, models/*meta.json).  —  verdict: **revise**

# Phase-3 PREFLIGHT DESIGN REVIEW — Spot Long-Only + Leakage Lens

**Verdict: REVISE** (two concrete leakage blockers + two enforcement-wiring blockers). The long-only architecture and genome design are sound; the blockers are about WHERE the safety switches are actually wired, not the intent.

## What I verified in the repo (not taken on faith)
- `engine/backtester.py`: opens shorts on `side=='short'`; opposite-side signal vs open position routes to `signal_exit` to cash (line 302) — confirms the 'short-as-exit-to-cash' mechanic works. Entry is queued and filled at `i+entry_lag_bars` OPEN (default 1) — no same-bar look-ahead (confirmed lines ~334/385). **No `long_only` guard exists today** (grep of `side ==`/`side !=` shows only execution logic).
- `funding_bps_per_8h` is a **Backtester constructor arg, default 1.0 (perp cost)** — NOT a `BacktestConfig` field. `fitness_worker` (auto_evolve.py ~526) and **`WalkForwardValidator` (engine/backtester.py:512)** both build `Backtester(...)` with no funding arg.
- `benchmark/spec.py:48` `lockbox_days: int = 120`; `benchmark/snapshots/manifest.json` currently `120`. Evolution loads `get_evolution_window(BENCHMARK_SPEC)` -> `DataLockbox.in_sample()` = `df[timestamp < lockbox_cutoff_ts]` (data_lockbox.py:196/229). The accessor is **`in_sample()`**, not `in_sample_view()` as the plan's md names it.
- `auto_evolve.py:519-523` wrapper is `def strategy_fn(d,i,p)` and is the single central call site; backtester calls it with 3 positional args.
- Fitness = `0.5*is_train_sharpe + 0.3*is_val_sharpe + 0.1*log(trades) + cross_term - penalties`; **WF OOS test Sharpe carries ZERO weight** — good anti-leakage property, confirmed (lines 540-606). `MIN_TRADES_FOR_POSITIVE_FITNESS=30`. DSR deflated by `n_trials_now = prior_trials + total_evaluated + 1` (line 977).
- Regime is causal: `classify_regime_quantitative(df, bar_idx)` reads `df.iloc[bar_idx]` only; underlying features (`bb_bandwidth_percentile`) use `rolling(...).apply(pctile)` — rolling, not whole-series. No future leak in regime labels.
- BMSB feature is 20/21-bar (`features.py:295-296`), confirming the plan's caveat that it is ~20h not weeks; the 480-bar `bmsb_macro_bullish` does **not exist yet**.
- Short-only params confirmed present in the registry: `rsi_short_max` (127), `contraction_rsi_overbought` (229), `rsi_overbought` (262), etc. — deletion is real work.
- Only TP>SL repair exists (auto_evolve.py ~410/438). Ordering repairs (fast<slow, tenkan<kijun) the plan relies on do **not** exist yet.

## BLOCKERS

### B1 (HARD LEAKAGE) — Pretrained models span the lockbox
`models/lstm_*_training_meta.json`: `trained_at=2026-03-29`, `train_candles=51194` (1h) ≈ 5.8 years ≈ mid-2020 -> 2026-03-29. **That training window runs straight through the sealed lockbox (2024-06-07 -> 2026-06-07).** Any strategy consuming these LSTM/cluster artifacts (`lstm_pattern`, plus cluster-derived features) is contaminated. The plan defers this to an iteration-7 'provenance audit' — too late and too soft. This must be a Phase-3-wide gate: no model trained past 2024-06-07 may be loaded by any evaluated strategy. Either retrain with a hard `train_end < lockbox_cutoff_ts` (and record it in the meta) or exclude `lstm_pattern` + cluster_variant features from Phase 3.

### B2 (ENFORCEMENT WIRING) — funding=0 and long_only=True don't reach the WF validator
Because `funding_bps_per_8h` and the proposed `long_only` are **Backtester constructor args**, and `WalkForwardValidator` builds its **own** `Backtester` at line 512, a config-only change is a no-op for every walk-forward fold. Since the GA selects on `is_train_sharpe`/`is_val_sharpe` (fold metrics), missing this means: (a) perp funding silently charged on spot during selection, and (b) potential short trades inside selection folds. Must thread both through `WalkForwardValidator.__init__` -> internal Backtester, plus the fitness_worker Backtester, sweep/run.py, and benchmark/runner. The wrapper filter (layer 2) is the real long-only safety net for the validator path because the validator receives the same `strategy_fn` closure — keep it, and ALSO add the engine flag for defense-in-depth.

### B3 (SEAL) — lockbox_days is still 120
`BenchmarkSpec` default and the frozen `manifest.json` are both 120. Must set 730 in `benchmark/spec.py` AND rebuild snapshots so `lockbox_cutoff_ts` recomputes to 2024-06-07. Without the rebuild, `in_sample()` will keep exposing 2024-06 -> 2026 data to the GA. Smoke test must assert `max(timestamp of evolution window) < 2024-06-07`.

### B4 (CORRECTNESS) — promised repairs/doc mismatches
Ordering repairs (fast<slow, tenkan<kijun, ema_fast<ema_slow) are assumed by the genome but not implemented (only TP>SL is). The md names `in_sample_view()`; the real method is `in_sample()`. Fix both before implementation to avoid a wrong-method/no-op edit.

## PASS-grade aspects
- **Long-only architecture**: belt-and-suspenders (engine flag + central wrapper filter + fitness -999 + code-level short-branch removal + funding=0) is the right shape and physically prevents shorts at the entry block once wired. Net exposure bounded [0, 0.80] via long-or-cash + `max_position_pct` FIXED 0.80, no leverage/borrow. Funding used only as cost (set to 0), never as PnL.
- **Genome breadth**: SHARED_LONG_CORE + per-strategy trigger-only block, max 3 evolvable booleans, derived (not swept) senkou/displacement, frozen `allowed_regimes` per archetype — all materially reduce overfit surface. Reasonable.
- **Search bounds**: SL [1.5,4.5], TP [2.0,9.0] with TP>=SL+0.5 (repair confirmed), time_stop [12,120], adx [10,40], vol_target [0.12,0.25] — all sane for 1h crypto.
- **No-lockbox-optimization design intent**: WF OOS Sharpe is zero-weight in fitness (verified), single sealed read in iteration 8, regime labels causal. The intent is correct — it's the wiring (B2/B3) and pretrained-model contamination (B1) that currently break it in practice.
- **Reproducibility**: deterministic regime/features, fixed seeds in MC (seed=42), manifest fingerprint/hash available, cumulative DSR n_trials tracked. Recommend recording manifest_hash + cumulative n_trials at the sealed-OOS read.

## Required fixes before GO
1. Global model-provenance gate (B1) ahead of iteration 2; retrain or drop lstm/cluster.
2. Pass `funding_bps_per_8h=0.0` and `long_only=True` into the WF validator's internal Backtester + all other call sites (B2); smoke-assert `sum(funding)==0` and zero short trades.
3. Set `lockbox_days=730`, rebuild snapshots, assert evolution window ends < 2024-06-07 (B3).
4. Implement the ordering-repair blocks + the registry unit test (every entry has full SHARED_LONG_CORE, zero short-only keys) (B4).
5. Seed zero short-era genomes for short-edge strategies; one-shot guard on the iteration-8 lockbox read.

With B1-B4 fixed, this plan is long-only compliant and leakage-clean.

## Tester: Phase-3 preflight design review: genome breadth, search bounds, no last-2-year optimization, param specificity, long-only and lockbox integrity — verified against the actual repo.  —  verdict: **revise**

# Phase-3 PREFLIGHT DESIGN REVIEW — Spot Long-Only

**Verdict: REVISE** (blocking issues are in genome breadth + implementability of the dedup/merge; long-only and lockbox design are sound.)

Lens: genome not too broad, bounds reasonable, no optimization over the last 2 years, params not hyper-specific. All claims below were verified against the actual repo.

## 1. Long-only compatibility — PASS
- Backtester `strategy_fn(df,i,position)` is called at the entry block (line 385) and the position-management block (line 301). The single central wrapper at `auto_evolve.py:519-523` is 3-arg and injects regime — the proposed `return None` filter for `signal.side != 'long'` drops cleanly in here. **Implementable as described.**
- A short signal while LONG already routes to `signal_exit` (line 302-304) = exactly the required "treat short as exit-to-cash" rule.
- `funding_bps_per_8h` default is **1.0** (perp cost charged on any open position, lines 247-261). Setting it to 0.0 for Phase-3 is correct and structurally bans funding-as-PnL. Net exposure bounded [0, 0.80] by long-or-cash + `max_position_pct`.
- Belt-and-suspenders (engine flag + wrapper + fitness -999 + code-level strip) is appropriate. `long_only` does not exist anywhere yet — net-new, low risk.

## 2. Methodological leakage — PASS
- `load_data(use_lockbox=True)` -> `get_evolution_window()` -> `DataLockbox.in_sample()` keeps only bars `< lockbox_cutoff_ts`, with a **hard assert** (`auto_evolve.py:1493-1497`) that the in-sample max ts is below the cutoff.
- `_regime` is computed **per-bar from a single row** (`classify_regime_quantitative` reads only `df.iloc[bar_idx]`, scenario_engine.py:122) over precomputed **rolling/causal** features (bb_bw_percentile etc. are rolling, features.py). No cross-lockbox leak from regime labeling.
- Fitness uses TRAIN+VAL Sharpe only; **OOS WF test folds carry ZERO selection weight** — strong anti-leakage property; preserve it.
- Cross-asset confirm also drops each asset's lockbox tail (lines 858-861). Good.
- One residual to enforce in code: warm-start seeds are short-era PARAMS; `warm_start_max_frac` caps at 0.5 of pop (line 943). The plan's "seed at most ONE genome" must be enforced by lowering this cap, else the dedup/anti-overlap intent leaks.

## 3. Genome design breadth — REVISE (blocking)
- **Collision:** `stop_loss_atr_mult` / `take_profit_atr_mult` appear **~23x inline** across existing param_space + PARAMS (e.g. vol_regime_arb lines 228-229); `adx_min` 8x, `volume_threshold` 5x. The plan injects SHARED_LONG_CORE on top but only explicitly enumerates deletion of *short-only* keys — it never states the helper strips these duplicated inline exec/gate keys. As written, the merge collides or double-counts, so the genome is **not actually deduped/narrowed**.
- **No total-dimension cap:** vol_regime_arb has 14 signal keys, chaos_trend 12, lstm_pattern 13. After +~13 shared keys, the widest genomes hit ~25+ evolvable dimensions. The only stated breadth control is "max 3 booleans." With pop 60-80 / 8-12 gens, 25-D search is under-sampled and overfit-prone.

## 4. Search bounds — PASS
- SL [1.5,4.5], TP [2.0,9.0] with engine-enforced TP>=SL+0.5; time_stop [12,120]; adx [10,40]; vol_ratio [0.7,2.5]; vol_target [0.12,0.25]; max_position 0.80 fixed. All reasonable, none hyper-specific. Ordering-constraint repair (fast<slow, tenkan<kijun) mirroring the existing TP>SL repair is implementable.
- `time_stop_hours` currently appears 0x in param_space — confirms it is hard-coded per strategy today; promoting it to a shared evolvable key is a genuine improvement, not new breadth.

## 5. No optimization over last 2 years — PASS
- lockbox_days default is **120** (spec.py:48); plan correctly sets **730**. In-sample = `timestamp < cutoff`. Sealed OOS touched exactly once in iteration 8 for reporting only. WF/expanding validation runs on pre-lockbox history only. Design is correct; just verify the 730 value and re-freeze the snapshot so the cutoff_ts reflects 2024-06-07.

## 6. Reproducibility — PASS
- Frozen snapshot is hashed (sha256 per symbol) with a manifest; DSR deflated by REAL cumulative n_trials (auto_evolve.py:972-984); per-strategy HoF + checkpoint carried. Deterministic synthetic data path exists for harness self-test. Scorecard metrics enumerated. Reproducible provided seeds/snapshot hash are logged per run.

## 7. Implementability — REVISE (blocking)
- The wrapper filter, engine flag, fitness net, lockbox=730, funding=0 are all straightforward against the current code.
- **Gap:** the build-helper merge will collide with the ~23 inline exec/gate key copies unless an explicit strip step precedes injection. The plan's unit test only checks shared-keys-present + short-keys-absent; it does **not** assert inline exec/gate keys are removed, so the most likely bug passes the test silently.

## Blocking issues (must fix before evolution)
1. Build-helper must DELETE inline exec/gate/size keys from every param_space AND PARAMS before injecting SHARED_LONG_CORE (collision on stop_loss_atr_mult ×23, take_profit_atr_mult ×23, adx_min ×8, volume_threshold ×5, rsi_overbought/oversold).
2. Impose an explicit total evolvable-dimension ceiling per genome (~<=18); trim vol_regime_arb / chaos_trend / lstm_pattern signal blocks accordingly.
3. Strengthen the unit test to assert zero collisions (no Layer-B key in the shared keyset) + boolean-count<=3, not just presence/absence.

## Sign-off
Long-only and lockbox/anti-leakage architecture is verified sound and implementable. The plan is **not ready to run as written** purely because the dedup/merge mechanics collide with pervasive inline exec keys and the genome-breadth ceiling is unbounded for the widest strategies. Fix the three blocking items, re-run the smoke evolution (assert zero short trades, sealed lockbox, no key collisions), then proceed.

## Tester: Phase-3 preflight: reproducibility (snapshot/manifest/seeds) AND implementability of every proposed change against the repo as-is. Adversarial leakage/long-only audit.  —  verdict: **revise**

# Phase-3 Preflight Review — Reproducibility & Implementability Lens

**Verdict: REVISE** (long-only design is sound; the data/lockbox foundation is not implementable as written)

## Scope
Adversarial preflight against the repo as-is. I read engine/backtester.py, auto_evolve.py (registry, wrapper, fitness, load_data, EvolutionEngine), benchmark/data_lockbox.py, benchmark/spec.py, the frozen manifest.json, engine/features.py, and strategy signatures. Focus: are reports reproducible (snapshot/manifest/seeds/asserts) and is every proposed change implementable today, with no secret long-only or lockbox violation.

## What the plan gets RIGHT (verified)
- **Long-only enforcement is solid and implementable.** The engine entry block is exactly at `backtester.py:384` (`signal = strategy_fn(df, i, None)`); dropping `signal.side!='long'` there is trivial. The opposite-side-exits-to-cash path already exists at line 302 (`signal.side != position.side -> signal_exit`). The central wrapper is exactly at `auto_evolve.py:519-523`. The fitness safety net is feasible because `Trade.to_dict()` emits `side`. `funding_bps_per_8h=0.0` structurally bans funding-as-PnL. Defense-in-depth is appropriate. **PASS.**
- **No methodological leakage in the FITNESS objective.** Verified at `auto_evolve.py:603-604` + comment 582-585: selection = `0.5*is_train_sharpe + 0.3*is_val_sharpe + ...`; the walk-forward TEST (OOS) Sharpe is computed but ZERO-weight. Preserve this. **PASS.**
- **Genome breadth & search bounds reasonable.** SHARED_LONG_CORE + per-strategy trigger split is clean; `random_genome/mutate_genome/crossover` iterate `param_space` generically (so shared keys evolve with no per-strategy code). Ordering repairs mirror the existing TP>=SL repair. "Max 3 evolvable booleans" caps the filter-combo surface. **PASS.**
- **Short-only params really exist** in the registry (e.g. `rsi_short_max` line 127, `contraction_rsi_overbought` line 229) — deletion targets are real.
- **BMSB caveat is correct.** `bmsb_sma/ema` are 20/21-bar (~20h) at `features.py:133-140` — NOT macro. The proposed `close.rolling(480).mean()` fix is well-founded.
- Strategies already accept `regime=` and carry `allowed_regimes` (e.g. trend_following.py:45-46), so freezing regimes as constants is implementable.

## BLOCKING issues (reproducibility / implementability)

### 1. The frozen snapshot is 365 days, not 2019->2026 — the "2-year sealed lockbox" is impossible on it
`benchmark/snapshots/manifest.json`: every symbol has `n_bars=8760`, `first_ts=2025-06-07`, `last_ts=2026-06-07`, `lockbox_days=120`, `lockbox_cutoff_ts=2026-02-07`. The plan's premise ("~2019->2026", "last 2 years 2024-06-07->2026-06-07 sealed", "set lockbox_days=730") cannot hold:
- `in_sample()` = `df[timestamp < last_ts - lockbox_days*86.4M]`.
- With `lockbox_days=730`, cutoff = 2024-06-08, which is **before** `first_ts` (2025-06-07) -> **in-sample is EMPTY**, the whole dataset becomes lockbox.
- Result at runtime: either the hard assert at `auto_evolve.py:1494` fires, or worse, the silent fallback (issue 3) loads an unsealed full series.

### 2. `history_days=365` is hardcoded and caps any rebuild
`benchmark/spec.py:38` `history_days=365`; the builder fetches `since_days=spec.history_days` (`data_lockbox.py:108`). Re-snapshotting WITHOUT raising `history_days` still yields 1 year. Both `history_days` (to ~2555 for 7y) and `lockbox_days` (to 730) must change **together**, then a network refetch + `python -m benchmark snapshot --force` must run (re-hashing all parquet sha256). This multi-step data prerequisite is absent from priority_ranking and iteration 1.

### 3. Silent leakage fallback in `load_data` (the biggest secret-leak vector)
`auto_evolve.py:1488-1517`: any exception from the lockbox path is caught and replaced by a full-series LIVE load explicitly described as "NOT lockbox-sealed". After a re-snapshot, the `DataLockbox` sha256 integrity guard (`data_lockbox.py:172-181`) raises on byte mismatch; that raise is swallowed -> an unsealed load that DOES contain 2024-06->2026-06. The plan's iter-1 smoke test never asserts the sealed PATH was taken. This is a concrete route to a secret lockbox violation.

### 4. Cross-asset in-sample frames also empty at lockbox_days=730
`auto_evolve.py:853-862` keys cross-asset in-sample on the same `lockbox_days`. At 730 each frame empties; the `len<200` guard (line 862) silently SKIPS every cross asset, so the cross-asset generalization term vanishes from fitness with no error — quietly weakening the anti-overfit objective the plan depends on.

### 5. Re-admitting pruned strategies is heavier than "gated re-admission"
`STRATEGY_REGISTRY` has ONLY 11 active entries. The 10 strategies scheduled for iters 6-8 (momentum, connors_rsi2, vwap_reversion, williams_cci, supertrend, obv_divergence, rsi_divergence, mean_reversion, heikin_ashi_ema, volatility_breakout) are NOT registered. Each needs a full new registry block + param_space + SHARED_LONG_CORE merge + short-branch surgery — medium effort, not the low-effort framing implied.

## Required revisions before any evolution runs
1. Add an explicit DATA-PREP step 0: raise `history_days` (>=2555) AND `lockbox_days` (730) together, then `python -m benchmark snapshot --force`; assert `first_ts<=2019`, `lockbox_cutoff_ts == last_ts-730d (~2024-06-07)`, and in-sample bar count > 0.
2. Make the lockbox non-optional for Phase 3: snapshot/integrity failure must RAISE, never fall back to the unsealed live load. Hard-fail if the fallback branch is reached.
3. Iter-1 smoke test must POSITIVELY assert the sealed path ran (log line present, N>0, `df.timestamp.max() < cutoff`, cutoff==2024-06-07) and cross-asset frames non-empty.
4. Reset stale DSR `n_trials`/warm-start seeds computed on the old 365d/short-era window.
5. Reframe iters 6-8 as registry-reconstruction tasks, budgeted accordingly.

## Sign-off matrix
- long_only_ok: PASS
- leakage_ok: FAIL (silent unsealed fallback + cross-asset term can vanish)
- genome_breadth_ok: PASS
- search_limits_ok: PASS
- no_lockbox_opt_ok: FAIL (730-day lockbox empties in-sample on current snapshot; fallback can leak)
- reproducibility_ok: FAIL (snapshot does not match the documented 2019-2026 window; manifest must be rebuilt and re-hashed)
- implementability_ok: FAIL (data window + registry reconstruction prerequisites missing)

The long-only architecture can ship as designed. Do not start iteration 1 until the multi-year snapshot exists, the lockbox is made fail-closed, and the smoke test affirmatively proves the sealed path with a 2024-06-07 cutoff.