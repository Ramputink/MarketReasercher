# Phase-3 Orchestrator Plan — SPOT LONG-ONLY

## 0. Repo-grounded facts (verified)
- `engine/backtester.py`: `Signal.side in {long,short}`; backtester opens shorts when `side=='short'` and treats opposite-side signal vs open position as `signal_exit` to cash (line ~302). Entry is queued and filled at `i+entry_lag_bars` open (default 1) — no same-bar look-ahead. `funding_bps_per_8h` default **1.0 (perp cost)** — must be 0 for spot.
- `auto_evolve.py:519-523`: the **single central wrapper** that calls `strategy_fn_ref(d,i,p, regime=_bar_regime(d,i))`. `_regime` is precomputed causally via `mirofish.scenario_engine.classify_regime_quantitative(df, bar_idx=i)` (uses `df.iloc[bar_idx]`).
- `random_genome`/`mutate_genome`/`crossover` iterate `param_space` generically and already repair `TP>SL` (lines 372/411/439). So shared genome keys evolve with **zero per-strategy code**.
- Fitness = `0.5*IS_train_Sharpe + 0.3*IS_val_Sharpe + trade_bonus + cross_term - degrad/dd/low_trade/cross penalties`. **OOS WF Sharpe is computed but ZERO-weight** — good anti-leakage property to preserve. `MIN_TRADES_FOR_POSITIVE_FITNESS=30`. DSR deflated by real cumulative `n_trials`.
- `benchmark/data_lockbox.py`: `lockbox_days` default **120**; `in_sample_view()` = `df[timestamp < lockbox_cutoff_ts]` is the seal. Must be **730** for the 2-year OOS lockbox.

## 1. Long-only enforcement (belt-and-suspenders)
1. **Engine flag** `long_only=True` in `Backtester`: at the new-entry block drop any `signal.side!='long'`; keep the `signal_exit` path so a short signal while long exits to cash.
2. **Wrapper filter** at `auto_evolve.py:519-523`: return `None` for non-long signals.
3. **Fitness safety net**: any trade with `side!='long'` => fitness `-999`.
4. **Code-level**: strip short branches in each strategy; delete short-only params.
5. **Spot honesty**: `funding_bps_per_8h=0.0` — also structurally bans funding-as-PnL. Net exposure bounded `[0, 0.80]`.

## 2. Anti-leakage / lockbox
- `lockbox_days=730`; all evolution/sweep uses `DataLockbox.in_sample_view()` (pre-2024-06-07). Internal validation = `WalkForwardValidator(train90/val15/test45, embargo 120)` on pre-lockbox history only.
- Sealed 2-year lockbox touched exactly **once**, in iteration 8, for reporting — never for tuning.
- Audit `_regime` and any model (lstm) training cutoff < 2024-06-07.

## 3. Common genome + dedup
- `SHARED_LONG_CORE` (exec/gate/size keys) merged into every registry entry by a build-helper; per-strategy block holds only the unique trigger. Unify BMSB (new `bmsb_macro_bullish` ~480-bar feature), RSI gates, ADX min/max (via `adx_polarity`), volume, vol-target sizing, and one engine-side trailing rule. Freeze `allowed_regimes` per archetype as non-evolvable constants. Discard all short-era evolved PARAMS as finals (may seed one genome only).

## 4. Tiers
- **Candidate (re-evolve, primary budget)**: trend_following, donchian_breakout, dual_ma, chaos_trend, kama_trend, keltner_breakout, ichimoku_kumo, vol_regime_arb, volatility_squeeze.
- **Doubtful (gated re-admission after code fix + >=40-trade gate)**: momentum, fisher_transform, connors_rsi2, lstm_pattern, vwap_reversion, williams_cci, supertrend.
- **Fragile (structural rewrite or stay dropped)**: obv_divergence, rsi_divergence, mean_reversion, heikin_ashi_ema.

## 5. Priority (impact/risk)
Hard-gates first (long-only flag, lockbox=730, spot costs) — high impact, low risk. Then shared genome + param cleanup, regime freeze, BMSB macro feature, full candidate re-evolution, shared sizing/trailing, trade-floor + early-kill, functional exit-bug fixes, and finally gated doubtful re-admission.

## 6. Eight 1-hour FULL cycles
Each iteration = hypothesize -> implement -> run evolution/sweep -> validate (WF + cross-asset + bootstrap/DSR) -> save HoF -> commit -> relaunch. Iter 1 infra+scaffold; 2-5 candidates by cluster + consolidation; 6-7 gated doubtful; 8 fragile triage + final portfolio + single sealed-OOS scorecard.

## 7. Acceptance per strategy (pre-lockbox WF, costs on)
- >=40 trades (sparse set), IS Sharpe>0, WF degradation <40%, maxDD<25%, Calmar>0.5, profit factor>1.3, bootstrap 5th-pct Sharpe>0, DSR pass, cost-sensitivity (2x) Sharpe drop <30%, per-year Sharpe stability (no >=2 negative years). Only survivors enter the final long-only portfolio and the sealed-OOS read.