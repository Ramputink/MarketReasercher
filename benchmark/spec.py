"""
benchmark.spec — The Constitution
=================================

Frozen thresholds for the benchmark gauntlet. Bumping ANY threshold REQUIRES
bumping `SPEC_VERSION`, because changing a threshold makes old scorecards
incomparable to new ones. The version travels inside every scorecard so you can
never accidentally compare apples (spec 1.0.0) to oranges (spec 1.1.0).

Thresholds are deliberately strict. The whole point is that passing is *hard*
and therefore *meaningful*. If everything passes, the gauntlet is broken — make
it harder, do not relax it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import json


# Bump this whenever ANY value below changes.
# 3.0.0-p3: Phase-3 spot long-only data contract — multi-year history with a
# sealed 2-year (730d) lockbox, replacing the 365d/120d Phase-1/2 contract.
SPEC_VERSION = "3.0.0-p3"


@dataclass(frozen=True)
class GateSpec:
    """Per-gate configuration + whether it is a hard gate (mandatory to pass)."""
    enabled: bool = True
    mandatory: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class BenchmarkSpec:
    """
    Immutable benchmark configuration. Frozen dataclass => hashable, comparable,
    and impossible to mutate mid-run.
    """
    version: str = SPEC_VERSION

    # ── Data / lockbox contract ────────────────────────────────────────────
    primary_symbol: str = "XRP/USDT"
    cross_symbols: tuple = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT")
    timeframe: str = "1h"
    # Phase-3: pull the full available multi-year history (~7y) so a 2-year
    # lockbox still leaves years of in-sample data for evolution.
    history_days: int = 2600
    # The FINAL `lockbox_days` are held out and NEVER given to evolution.
    # Phase-3 seals the last 2 years (730d) as the spot long-only OOS lockbox.
    lockbox_days: int = 730
    # Bars of warm-up discarded at the start of every segment (indicator burn-in).
    warmup_bars: int = 60

    # ── Execution realism (the harness uses these, not the research backtester) ─
    base_commission_rate: float = 0.001       # 0.1% per side, taker
    base_slippage_bps: float = 5.0            # realistic average
    funding_bps_per_8h: float = 0.0           # Phase-3 SPOT: no perp funding (long-or-cash)
    entry_lag_bars: int = 1                   # fill at NEXT bar open — kills look-ahead
    position_pct: float = 0.10                # fixed notional fraction (comparable across candidates)
    ruin_equity_floor: float = 0.0            # equity <= this => account ruined, stop
    initial_capital: float = 10_000.0

    # ── Statistical trial budget (for the Deflated Sharpe gate) ─────────────
    # How many genomes the evolution evaluated. Larger => harder DSR bar.
    # Override per-candidate from the actual evolution report when available.
    default_n_trials: int = 30_000
    # Spread of trial Sharpes. 0.0 => statistics uses the principled 1/sqrt(T)
    # null dispersion (per-trade units). Supply the measured value per-candidate
    # via Candidate.trials_sharpe_std when the evolution run reports it.
    default_trials_sharpe_std: float = 0.0

    # ── Gate thresholds ─────────────────────────────────────────────────────
    # G01 Lockbox performance (true OOS)
    g01_min_sharpe: float = 0.8               # bar-level annualised, on the lockbox
    g01_min_profit_factor: float = 1.20
    g01_min_trades: int = 30
    g01_min_net_pnl: float = 0.0

    # G02 OOS degradation (in-sample -> lockbox)
    g02_max_degradation: float = 0.50         # 1 - SR_oos/SR_is must be <= this

    # G03 Deflated Sharpe Ratio (multiple-testing correction). THE headline gate.
    g03_min_dsr: float = 0.95                 # P(true trade-Sharpe > 0 | N trials)

    # G04 No-ruin + full-sample drawdown
    g04_max_drawdown_pct: float = 25.0        # full-sample, NOT fold-level
    g04_forbid_ruin: bool = True              # equity must never cross the floor

    # G05 Cost stress
    g05_stress_slippage_bps: float = 20.0     # must still work at 4x base slippage
    g05_stress_commission_mult: float = 1.5   # and 1.5x commission
    g05_min_sharpe_under_stress: float = 0.3
    g05_min_profit_factor_under_stress: float = 1.05

    # G06 Execution-lag dependence (edge must survive realistic 1-bar-lag fills)
    g06_min_sharpe_with_lag: float = 0.5
    g06_max_lag_degradation: float = 0.60     # SR must not collapse >60% vs zero-lag

    # G07 Monte Carlo (bootstrap of trade pnls)
    g07_n_sims: int = 2000
    g07_max_prob_ruin: float = 0.05
    g07_min_prob_profit: float = 0.85
    g07_ruin_dd_pct: float = 50.0

    # G08 Multi-asset generalisation (same params, other coins)
    g08_min_assets_profitable: int = 2        # of cross_symbols
    g08_min_median_sharpe: float = 0.0

    # G09 Parameter-neighbourhood stability (no knife-edge optima)
    g09_perturbation_pct: float = 0.10        # +/-10% on each numeric param
    g09_n_neighbours: int = 24
    g09_min_fraction_profitable: float = 0.60 # >=60% of neighbours must keep PF>1
    g09_min_median_sharpe: float = 0.3

    # G10 Statistical significance of the edge
    g10_min_tstat: float = 2.0                # t-stat of mean trade return
    g10_min_trades: int = 30

    # G11 Concentration / regime-fragility
    g11_max_single_trade_pnl_share: float = 0.35   # no trade > 35% of net pnl
    g11_max_single_regime_pnl_share: float = 0.80  # not all profit from one regime

    # ── Gate registry: which gates run and which are mandatory ──────────────
    gates: dict = field(default_factory=lambda: {
        "g01_lockbox_performance":   GateSpec(mandatory=True,  weight=2.0),
        "g02_oos_degradation":       GateSpec(mandatory=True,  weight=1.5),
        "g03_deflated_sharpe":       GateSpec(mandatory=True,  weight=3.0),
        "g04_no_ruin_drawdown":      GateSpec(mandatory=True,  weight=2.0),
        "g05_cost_stress":           GateSpec(mandatory=True,  weight=2.0),
        "g06_execution_lag":         GateSpec(mandatory=True,  weight=1.0),
        "g07_monte_carlo":           GateSpec(mandatory=True,  weight=1.5),
        "g08_multi_asset":           GateSpec(mandatory=True,  weight=2.0),
        "g09_param_stability":       GateSpec(mandatory=True,  weight=1.5),
        "g10_significance":          GateSpec(mandatory=True,  weight=1.5),
        "g11_concentration":         GateSpec(mandatory=False, weight=1.0),
    })

    # Reproducibility
    seed: int = 42

    def fingerprint(self) -> str:
        """Stable content hash of the spec — identifies the 'constitution'."""
        payload = json.dumps(self._plain(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _plain(self) -> dict:
        d = asdict(self)
        # GateSpec -> plain dict for hashing/serialisation
        d["gates"] = {k: asdict(v) for k, v in self.gates.items()}
        return d

    def to_dict(self) -> dict:
        return self._plain()


# The single canonical spec instance used everywhere.
BENCHMARK_SPEC = BenchmarkSpec()
