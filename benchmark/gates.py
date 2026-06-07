"""
benchmark.gates — the gauntlet
==============================

Eleven independent gates. Each attacks a specific way a backtest lies. A
candidate is IRONCLAD only if every mandatory gate passes on the frozen lockbox.

Each gate is a pure function:  gate(ctx) -> GateResult
where `margin` is normalised so >=0 means "passed, with this much headroom".

    G01  lockbox_performance   — does it even work on never-seen data?
    G02  oos_degradation       — did the edge survive leaving the training set?
    G03  deflated_sharpe       — is it distinguishable from the best of N noise trials?  ★
    G04  no_ruin_drawdown      — full-sample DD bounded, account never blown
    G05  cost_stress           — does it survive 4x slippage + higher fees?
    G06  execution_lag         — does it survive realistic 1-bar-lag fills?
    G07  monte_carlo           — bootstrap: low ruin prob, high profit prob
    G08  multi_asset           — same params generalise to other coins?
    G09  param_stability       — is it a robust basin, not a knife-edge optimum?
    G10  significance          — is the mean trade return statistically > 0?
    G11  concentration         — not one lucky trade / one lucky regime
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from benchmark.harness import ConservativeBacktester, ExecModel
from benchmark.scorecard import GateResult
from benchmark.candidate import Candidate
from benchmark import statistics as st


# ─────────────────────────────────────────────────────────────────────────────
# Shared context
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GateContext:
    candidate: Candidate
    spec: object
    in_sample: pd.DataFrame
    lockbox: pd.DataFrame
    cross_lockboxes: dict           # symbol -> df
    bar_hours: float = 1.0

    # caches so gates that need the same backtest don't recompute it
    _cache: dict = field(default_factory=dict)

    def base_exec(self, **overrides) -> ExecModel:
        s = self.spec
        kw = dict(
            commission_rate=s.base_commission_rate,
            slippage_bps=s.base_slippage_bps,
            funding_bps_per_8h=s.funding_bps_per_8h,
            entry_lag_bars=s.entry_lag_bars,
            position_pct=s.position_pct,
            initial_capital=s.initial_capital,
            ruin_equity_floor=s.ruin_equity_floor,
            warmup_bars=s.warmup_bars,
            bar_hours=self.bar_hours,
        )
        kw.update(overrides)
        return ExecModel(**kw)

    def run(self, df: pd.DataFrame, exec_model: ExecModel,
            param_overrides: Optional[dict] = None):
        fn = self.candidate.build_strategy_fn(param_overrides)
        return ConservativeBacktester(exec_model).run(df, fn, self.candidate.strategy)

    def lockbox_base(self):
        """Cached canonical lockbox backtest at base costs / 1-bar lag."""
        if "lockbox_base" not in self._cache:
            self._cache["lockbox_base"] = self.run(self.lockbox, self.base_exec())
        return self._cache["lockbox_base"]


def _gr(spec, name, passed, margin, summary, detail=None, error=None) -> GateResult:
    g = spec.gates.get(name)
    return GateResult(
        name=name, passed=passed,
        mandatory=(g.mandatory if g else True),
        weight=(g.weight if g else 1.0),
        margin=float(margin), summary=summary,
        detail=detail or {}, error=error,
    )


def _safe(spec, name, fn):
    """Run a gate, converting any crash into a hard fail (never a silent pass)."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return _gr(spec, name, passed=False, margin=-1.0,
                   summary="gate crashed", error=f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# G01 — Lockbox performance (true out-of-sample)
# ─────────────────────────────────────────────────────────────────────────────

def g01_lockbox_performance(ctx) -> GateResult:
    s = ctx.spec
    name = "g01_lockbox_performance"
    def _():
        r = ctx.lockbox_base()
        checks = {
            "sharpe": (r.sharpe_annualised, s.g01_min_sharpe),
            "profit_factor": (r.profit_factor if np.isfinite(r.profit_factor) else 999, s.g01_min_profit_factor),
            "trades": (r.n_trades, s.g01_min_trades),
            "net_pnl": (r.net_pnl, s.g01_min_net_pnl),
        }
        ok = (not r.ruined
              and r.sharpe_annualised >= s.g01_min_sharpe
              and (r.profit_factor >= s.g01_min_profit_factor)
              and r.n_trades >= s.g01_min_trades
              and r.net_pnl > s.g01_min_net_pnl)
        margin = min(
            r.sharpe_annualised / max(s.g01_min_sharpe, 1e-9) - 1.0,
            (r.n_trades / max(s.g01_min_trades, 1)) - 1.0,
        )
        return _gr(s, name, ok, margin,
                   f"OOS Sharpe={r.sharpe_annualised:.2f} PF={r.profit_factor:.2f} "
                   f"trades={r.n_trades} pnl=${r.net_pnl:.0f}"
                   + (" RUINED" if r.ruined else ""),
                   detail={"summary": r.summary(), "thresholds": {k: v[1] for k, v in checks.items()}})
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G02 — Out-of-sample degradation (in-sample -> lockbox)
# ─────────────────────────────────────────────────────────────────────────────

def g02_oos_degradation(ctx) -> GateResult:
    s = ctx.spec
    name = "g02_oos_degradation"
    def _():
        is_r = ctx.run(ctx.in_sample, ctx.base_exec())
        oos_r = ctx.lockbox_base()
        sr_is = is_r.sharpe_annualised
        sr_oos = oos_r.sharpe_annualised
        if sr_is <= 0:
            return _gr(s, name, False, -1.0,
                       f"in-sample Sharpe non-positive ({sr_is:.2f}); cannot certify edge")
        degradation = 1.0 - sr_oos / sr_is
        ok = degradation <= s.g02_max_degradation and sr_oos > 0
        margin = (s.g02_max_degradation - degradation) / max(s.g02_max_degradation, 1e-9)
        return _gr(s, name, ok, margin,
                   f"IS Sharpe={sr_is:.2f} -> OOS Sharpe={sr_oos:.2f}  degradation={degradation*100:.0f}% "
                   f"(max {s.g02_max_degradation*100:.0f}%)",
                   detail={"sr_is": sr_is, "sr_oos": sr_oos, "degradation": degradation})
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G03 — Deflated Sharpe Ratio (multiple-testing correction)  ★ headline gate
# ─────────────────────────────────────────────────────────────────────────────

def g03_deflated_sharpe(ctx) -> GateResult:
    s = ctx.spec
    name = "g03_deflated_sharpe"
    def _():
        r = ctx.lockbox_base()
        if r.n_trades < 5:
            return _gr(s, name, False, -1.0,
                       f"only {r.n_trades} trades — cannot compute a meaningful DSR")
        n_trials = ctx.candidate.n_trials or s.default_n_trials
        # None -> statistics uses the principled 1/sqrt(T) null dispersion.
        trials_std = ctx.candidate.trials_sharpe_std
        dsr = st.deflated_sharpe_ratio(r.trade_returns, n_trials, trials_std)
        ok = dsr["dsr"] >= s.g03_min_dsr
        margin = (dsr["dsr"] - s.g03_min_dsr) / max(1.0 - s.g03_min_dsr, 1e-9)
        return _gr(s, name, ok, margin,
                   f"DSR={dsr['dsr']:.3f} (min {s.g03_min_dsr:.2f}) | "
                   f"SR/trade={dsr['sr_hat']:.3f} vs noise-max SR*={dsr['sr_star']:.3f} "
                   f"over N={n_trials:,} trials",
                   detail=dsr)
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G04 — No ruin + bounded full-sample drawdown
# ─────────────────────────────────────────────────────────────────────────────

def g04_no_ruin_drawdown(ctx) -> GateResult:
    s = ctx.spec
    name = "g04_no_ruin_drawdown"
    def _():
        full = pd.concat([ctx.in_sample, ctx.lockbox], ignore_index=True)
        r = ctx.run(full, ctx.base_exec())
        ruined = r.ruined
        dd = r.max_drawdown_pct
        ok = (not (ruined and s.g04_forbid_ruin)) and dd <= s.g04_max_drawdown_pct
        margin = (s.g04_max_drawdown_pct - dd) / max(s.g04_max_drawdown_pct, 1e-9)
        if ruined:
            margin = -1.0
        return _gr(s, name, ok, margin,
                   f"full-sample MaxDD={dd:.1f}% (cap {s.g04_max_drawdown_pct:.0f}%)"
                   + (" — ACCOUNT RUINED" if ruined else ""),
                   detail={"max_dd_pct": dd, "ruined": ruined, "n_trades": r.n_trades})
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G05 — Cost stress (slippage 4x + commission 1.5x)
# ─────────────────────────────────────────────────────────────────────────────

def g05_cost_stress(ctx) -> GateResult:
    s = ctx.spec
    name = "g05_cost_stress"
    def _():
        stressed = ctx.base_exec(
            slippage_bps=s.g05_stress_slippage_bps,
            commission_rate=s.base_commission_rate * s.g05_stress_commission_mult,
        )
        r = ctx.run(ctx.lockbox, stressed)
        pf = r.profit_factor if np.isfinite(r.profit_factor) else 999
        ok = (not r.ruined
              and r.sharpe_annualised >= s.g05_min_sharpe_under_stress
              and pf >= s.g05_min_profit_factor_under_stress)
        # also sweep to estimate breakeven slippage
        breakeven = _breakeven_slippage_bps(ctx)
        margin = r.sharpe_annualised / max(s.g05_min_sharpe_under_stress, 1e-9) - 1.0
        return _gr(s, name, ok, margin,
                   f"@ {s.g05_stress_slippage_bps:.0f}bps + {s.g05_stress_commission_mult:.1f}x fees: "
                   f"Sharpe={r.sharpe_annualised:.2f} PF={pf:.2f} | breakeven≈{breakeven}",
                   detail={"stressed": r.summary(), "breakeven_slippage_bps": breakeven})
    return _safe(s, name, _)


def _breakeven_slippage_bps(ctx) -> str:
    """Find the slippage (bps) at which the lockbox edge dies (PF<=1)."""
    for bps in (5, 10, 20, 30, 50, 75, 100):
        r = ctx.run(ctx.lockbox, ctx.base_exec(slippage_bps=float(bps)))
        pf = r.profit_factor if np.isfinite(r.profit_factor) else 999
        if pf <= 1.0 or r.ruined:
            return f"<{bps}bps"
    return ">100bps"


# ─────────────────────────────────────────────────────────────────────────────
# G06 — Execution-lag dependence
# ─────────────────────────────────────────────────────────────────────────────

def g06_execution_lag(ctx) -> GateResult:
    s = ctx.spec
    name = "g06_execution_lag"
    def _():
        lag = ctx.lockbox_base()                                   # entry_lag_bars from spec (1)
        nolag = ctx.run(ctx.lockbox, ctx.base_exec(entry_lag_bars=0))
        sr_lag, sr_nolag = lag.sharpe_annualised, nolag.sharpe_annualised
        pf_lag = lag.profit_factor if np.isfinite(lag.profit_factor) else 999
        if sr_nolag > 0:
            degradation = 1.0 - sr_lag / sr_nolag
        else:
            degradation = 0.0 if sr_lag >= 0 else 1.0
        ok = (sr_lag >= s.g06_min_sharpe_with_lag
              and pf_lag > 1.0
              and degradation <= s.g06_max_lag_degradation)
        margin = (s.g06_max_lag_degradation - degradation) / max(s.g06_max_lag_degradation, 1e-9)
        return _gr(s, name, ok, margin,
                   f"realistic(lag=1) Sharpe={sr_lag:.2f} vs cheat(lag=0)={sr_nolag:.2f}  "
                   f"dependence={max(degradation,0)*100:.0f}%",
                   detail={"sr_lag": sr_lag, "sr_nolag": sr_nolag, "degradation": degradation})
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G07 — Monte Carlo robustness (bootstrap of trade pnls)
# ─────────────────────────────────────────────────────────────────────────────

def g07_monte_carlo(ctx) -> GateResult:
    s = ctx.spec
    name = "g07_monte_carlo"
    def _():
        from engine.monte_carlo import monte_carlo_trades
        r = ctx.lockbox_base()
        if r.n_trades < 15:
            return _gr(s, name, False, -1.0,
                       f"only {r.n_trades} trades — Monte Carlo not trustworthy")
        pnls = r.trades["pnl_net"].to_numpy()
        mc = monte_carlo_trades(
            pnls, initial_capital=s.initial_capital,
            n_simulations=s.g07_n_sims, method="bootstrap",
            ruin_dd_pct=s.g07_ruin_dd_pct, seed=s.seed,
        )
        ok = (mc.prob_ruin <= s.g07_max_prob_ruin
              and mc.prob_profit >= s.g07_min_prob_profit)
        margin = min(
            (s.g07_max_prob_ruin - mc.prob_ruin) / max(s.g07_max_prob_ruin, 1e-9),
            (mc.prob_profit - s.g07_min_prob_profit) / max(1.0 - s.g07_min_prob_profit, 1e-9),
        )
        return _gr(s, name, ok, margin,
                   f"P(profit)={mc.prob_profit*100:.0f}% (min {s.g07_min_prob_profit*100:.0f}%) "
                   f"P(ruin)={mc.prob_ruin*100:.1f}% (max {s.g07_max_prob_ruin*100:.0f}%) "
                   f"worst5%=${mc.pnl_p5:.0f}",
                   detail={"prob_profit": mc.prob_profit, "prob_ruin": mc.prob_ruin,
                           "pnl_p5": mc.pnl_p5, "max_dd_p95": mc.max_dd_p95})
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G08 — Multi-asset generalisation (same params, other coins)
# ─────────────────────────────────────────────────────────────────────────────

def g08_multi_asset(ctx) -> GateResult:
    s = ctx.spec
    name = "g08_multi_asset"
    def _():
        if not ctx.cross_lockboxes:
            return _gr(s, name, False, -1.0,
                       "no cross-asset data in snapshot — cannot test generalisation")
        per = {}
        sharpes = []
        n_profitable = 0
        for sym, df in ctx.cross_lockboxes.items():
            r = ctx.run(df, ctx.base_exec())
            per[sym] = {"sharpe": r.sharpe_annualised, "pf": r.profit_factor,
                        "trades": r.n_trades, "net_pnl": r.net_pnl, "ruined": r.ruined}
            sharpes.append(r.sharpe_annualised)
            if r.net_pnl > 0 and not r.ruined and r.n_trades >= 10:
                n_profitable += 1
        median_sr = float(np.median(sharpes)) if sharpes else -1.0
        ok = (n_profitable >= s.g08_min_assets_profitable
              and median_sr >= s.g08_min_median_sharpe)
        margin = (n_profitable - s.g08_min_assets_profitable) / max(len(per), 1)
        return _gr(s, name, ok, margin,
                   f"{n_profitable}/{len(per)} cross-assets profitable, median Sharpe={median_sr:.2f}",
                   detail=per)
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G09 — Parameter-neighbourhood stability (no knife-edge optima)
# ─────────────────────────────────────────────────────────────────────────────

def g09_param_stability(ctx) -> GateResult:
    s = ctx.spec
    name = "g09_param_stability"
    def _():
        space = ctx.candidate.numeric_param_space()
        if not space:
            return _gr(s, name, True, 0.0,
                       "no numeric params to perturb — stability trivially holds",
                       detail={"note": "strategy has no float/int params in registry"})
        rng = np.random.default_rng(s.seed)
        base = dict(ctx.candidate.params)
        sharpes = []
        n_profitable = 0
        for _i in range(s.g09_n_neighbours):
            override = {}
            for pname, (kind, lo, hi) in space.items():
                cur = base.get(pname)
                if cur is None:
                    # fall back to midpoint if candidate didn't pin this param
                    cur = (lo + hi) / 2.0
                step = (hi - lo) * s.g09_perturbation_pct
                val = float(cur) + rng.uniform(-step, step)
                val = min(max(val, lo), hi)
                override[pname] = int(round(val)) if kind == "int" else val
            r = ctx.run(ctx.lockbox, ctx.base_exec(), param_overrides=override)
            sharpes.append(r.sharpe_annualised)
            pf = r.profit_factor if np.isfinite(r.profit_factor) else 999
            if pf > 1.0 and not r.ruined:
                n_profitable += 1
        frac = n_profitable / s.g09_n_neighbours
        median_sr = float(np.median(sharpes))
        ok = frac >= s.g09_min_fraction_profitable and median_sr >= s.g09_min_median_sharpe
        margin = (frac - s.g09_min_fraction_profitable) / max(s.g09_min_fraction_profitable, 1e-9)
        return _gr(s, name, ok, margin,
                   f"±{s.g09_perturbation_pct*100:.0f}% neighbourhood: {frac*100:.0f}% profitable "
                   f"(min {s.g09_min_fraction_profitable*100:.0f}%), median Sharpe={median_sr:.2f}",
                   detail={"fraction_profitable": frac, "median_sharpe": median_sr,
                           "n_neighbours": s.g09_n_neighbours})
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G10 — Statistical significance of the edge
# ─────────────────────────────────────────────────────────────────────────────

def g10_significance(ctx) -> GateResult:
    s = ctx.spec
    name = "g10_significance"
    def _():
        r = ctx.lockbox_base()
        if r.n_trades < s.g10_min_trades:
            return _gr(s, name, False, -1.0,
                       f"only {r.n_trades} trades (need {s.g10_min_trades}) — underpowered")
        t = st.tstat_pvalue(r.trade_returns)
        ok = t["tstat"] >= s.g10_min_tstat and r.n_trades >= s.g10_min_trades
        margin = (t["tstat"] - s.g10_min_tstat) / max(s.g10_min_tstat, 1e-9)
        return _gr(s, name, ok, margin,
                   f"mean-return t-stat={t['tstat']:.2f} (min {s.g10_min_tstat:.1f}), "
                   f"p={t['pvalue']:.4f}, n={r.n_trades}",
                   detail=t)
    return _safe(s, name, _)


# ─────────────────────────────────────────────────────────────────────────────
# G11 — Concentration / regime-fragility
# ─────────────────────────────────────────────────────────────────────────────

def g11_concentration(ctx) -> GateResult:
    s = ctx.spec
    name = "g11_concentration"
    def _():
        r = ctx.lockbox_base()
        if r.n_trades < 5 or r.net_pnl <= 0:
            return _gr(s, name, False, -1.0,
                       f"net_pnl=${r.net_pnl:.0f} over {r.n_trades} trades — nothing to certify")
        pnls = r.trades["pnl_net"].to_numpy()
        total = pnls.sum()
        # single-trade concentration (use max positive contribution / total)
        top_trade_share = float(pnls.max() / total) if total > 0 else 1.0
        # single-regime concentration
        regime_pnl = r.regime_pnl
        pos_regimes = {k: v for k, v in regime_pnl.items() if v > 0}
        top_regime_share = (max(pos_regimes.values()) / sum(pos_regimes.values())
                            if pos_regimes else 1.0)
        ok = (top_trade_share <= s.g11_max_single_trade_pnl_share
              and top_regime_share <= s.g11_max_single_regime_pnl_share)
        margin = min(
            (s.g11_max_single_trade_pnl_share - top_trade_share) / max(s.g11_max_single_trade_pnl_share, 1e-9),
            (s.g11_max_single_regime_pnl_share - top_regime_share) / max(s.g11_max_single_regime_pnl_share, 1e-9),
        )
        return _gr(s, name, ok, margin,
                   f"top trade={top_trade_share*100:.0f}% of pnl (max {s.g11_max_single_trade_pnl_share*100:.0f}%), "
                   f"top regime={top_regime_share*100:.0f}% (max {s.g11_max_single_regime_pnl_share*100:.0f}%)",
                   detail={"top_trade_share": top_trade_share, "top_regime_share": top_regime_share,
                           "regime_pnl": regime_pnl})
    return _safe(s, name, _)


# Ordered registry the runner iterates over.
ALL_GATES = [
    g01_lockbox_performance,
    g02_oos_degradation,
    g03_deflated_sharpe,
    g04_no_ruin_drawdown,
    g05_cost_stress,
    g06_execution_lag,
    g07_monte_carlo,
    g08_multi_asset,
    g09_param_stability,
    g10_significance,
    g11_concentration,
]
