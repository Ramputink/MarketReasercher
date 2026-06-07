#!/usr/bin/env python3
"""
CryptoResearchLab — Autonomous Evolutionary Strategy Optimizer
==============================================================
Genetic algorithm-based strategy evolution that runs for N hours autonomously.
Learns from its own mistakes, narrows the search space, and logs everything.

Architecture:
  1. GENOME = (strategy_type, param_dict)
  2. FITNESS = walk-forward robustness score (not just in-sample Sharpe)
  3. POPULATION evolves via tournament selection, crossover, mutation
  4. HALL OF FAME tracks all-time best genomes per strategy
  5. LEARNING: parameter ranges narrow around successful regions
  6. REGIME-AWARE: all backtests use per-bar regime labels

Usage:
    python auto_evolve.py                    # Run for 14 hours (default)
    python auto_evolve.py --hours 4          # Run for 4 hours
    python auto_evolve.py --pop-size 50      # Population of 50
    python auto_evolve.py --cores 8          # Use 8 CPU cores
"""
import argparse
import copy
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

MP_CTX = mp.get_context("spawn")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LabConfig, BacktestConfig, RiskConfig, DataConfig, TimeFrame, EvolutionConfig
from engine.data_ingestion import DataIngestionEngine
from engine.env_loader import has_binance_keys
from engine.features import build_all_features
from engine.backtester import Backtester, WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("auto_evolve")


# ═══════════════════════════════════════════════════════════════
# STRATEGY REGISTRY — all available strategies + their param spaces
# ═══════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    # NOTE: Round 1 pruning (10h/29K-eval evolution, 0% positive fitness):
    #   volatility_breakout, mean_reversion, rsi_divergence, vwap_reversion, obv_divergence
    # NOTE: Round 2 pruning (14h/36K-eval evolution, <1.5% positive fitness):
    #   supertrend (0.3%), momentum (0.5%), heikin_ashi_ema (0.4%), connors_rsi2 (1.2%), williams_cci (1.0%)
    # 10 strategies removed total.
    # NOTE: Round 3 additions: chaos_trend (Hurst/fractal), vol_regime_arb (GK vol z-score)
    # NOTE: Round 4 addition: lstm_pattern (K-Means clustering + LSTM sequence prediction)
    # NOTE: Round 5 upgrade: multi-variant clustering (7 variants: kmeans, hierarchical, bisecting)
    #   with evolvable cluster_variant param. Evolution selects best algo + granularity.
    # 11 active strategies.
    "trend_following": {
        "module": "strategies.trend_following",
        "function": "trend_following_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "adx_threshold": ("float", 15.0, 40.0),
            "fib_zone_tolerance_pct": ("float", 0.5, 3.0),
            "stop_loss_atr_mult": ("float", 1.0, 5.0),
            "take_profit_atr_mult": ("float", 2.0, 8.0),
            "min_volume_ratio": ("float", 0.5, 1.5),
            "rsi_lower_bound": ("int", 25, 45),
            "rsi_upper_bound": ("int", 55, 75),
        },
    },
    # ── momentum PRUNED (Round 2) ──
    "donchian_breakout": {
        "module": "strategies.donchian_breakout",
        "function": "donchian_breakout_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "entry_period": ("int", 10, 50),
            "exit_period": ("int", 5, 25),
            "volume_surge_threshold": ("float", 0.8, 2.5),
            "adx_min": ("float", 10.0, 30.0),
            "atr_filter_mult": ("float", 0.2, 1.5),
            "use_volatility_filter": ("bool",),
            "stop_loss_atr_mult": ("float", 1.0, 5.0),
            "take_profit_atr_mult": ("float", 2.0, 8.0),
        },
    },
    "dual_ma": {
        "module": "strategies.dual_ma",
        "function": "dual_ma_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "fast_period": ("int", 5, 25),
            "slow_period": ("int", 15, 60),
            "signal_type": ("choice", ["sma", "ema"]),
            "require_adx": ("bool",),
            "adx_min": ("float", 10.0, 35.0),
            "require_volume": ("bool",),
            "volume_threshold": ("float", 0.8, 2.0),
            "stop_loss_atr_mult": ("float", 1.0, 4.0),
            "take_profit_atr_mult": ("float", 2.0, 8.0),
        },
    },
    "keltner_breakout": {
        "module": "strategies.keltner_breakout",
        "function": "keltner_breakout_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "ema_period": ("int", 10, 40),
            "atr_mult": ("float", 1.0, 3.5),
            "rsi_long_min": ("float", 40.0, 60.0),
            "rsi_short_max": ("float", 40.0, 60.0),
            "adx_min": ("float", 10.0, 30.0),
            "volume_threshold": ("float", 0.8, 2.0),
            "stop_loss_atr_mult": ("float", 1.0, 4.0),
            "take_profit_atr_mult": ("float", 2.0, 8.0),
        },
    },
    "volatility_squeeze": {
        "module": "strategies.volatility_squeeze",
        "function": "volatility_squeeze_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "keltner_atr_mult": ("float", 1.0, 2.5),
            "squeeze_min_bars": ("int", 2, 10),
            "momentum_lookback": ("int", 6, 24),
            "volume_surge_threshold": ("float", 0.8, 2.5),
            "stop_loss_atr_mult": ("float", 1.0, 4.0),
            "take_profit_atr_mult": ("float", 2.0, 8.0),
        },
    },
    # ── supertrend, connors_rsi2, heikin_ashi_ema PRUNED (Round 2) ──
    "ichimoku_kumo": {
        "module": "strategies.ichimoku_kumo",
        "function": "ichimoku_kumo_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "tenkan_period": ("int", 7, 12),
            "kijun_period": ("int", 20, 35),
            "senkou_b_period": ("int", 40, 65),
            "displacement": ("int", 20, 35),
            "require_chikou_confirm": ("bool",),
            "require_tk_cross": ("bool",),
            "adx_min": ("float", 10.0, 25.0),
            "stop_loss_atr_mult": ("float", 2.0, 5.0),
            "take_profit_atr_mult": ("float", 3.0, 7.0),
        },
    },
    # ── williams_cci PRUNED (Round 2) ──
    "kama_trend": {
        "module": "strategies.kama_trend",
        "function": "kama_trend_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "er_period": ("int", 8, 20),
            "fast_sc": ("int", 2, 5),
            "slow_sc": ("int", 25, 35),
            "signal_period": ("int", 3, 8),
            "slope_threshold_pct": ("float", 0.02, 0.15),
            "adx_min": ("float", 8.0, 25.0),
            "stop_loss_atr_mult": ("float", 1.5, 4.0),
            "take_profit_atr_mult": ("float", 2.5, 6.0),
        },
    },
    "fisher_transform": {
        "module": "strategies.fisher_transform",
        "function": "fisher_transform_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "period": ("int", 8, 20),
            "signal_threshold": ("float", 1.2, 2.8),
            "require_divergence": ("bool",),
            "divergence_lookback": ("int", 5, 15),
            "adx_filter": ("bool",),
            "adx_max": ("float", 30.0, 50.0),
            "stop_loss_atr_mult": ("float", 1.0, 3.0),
            "take_profit_atr_mult": ("float", 2.0, 5.0),
        },
    },
    # ── NEW strategies (Round 3 additions) ──
    "chaos_trend": {
        "module": "strategies.chaos_trend",
        "function": "chaos_trend_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "hurst_window": ("int", 50, 150),
            "hurst_min": ("float", 0.52, 0.70),
            "hurst_max": ("float", 0.75, 0.95),
            "ema_fast": ("int", 8, 20),
            "ema_slow": ("int", 20, 50),
            "adx_min": ("float", 12.0, 30.0),
            "momentum_period": ("int", 8, 24),
            "volume_threshold": ("float", 0.8, 2.0),
            "use_fractal_filter": ("bool",),
            "fractal_max": ("float", 1.35, 1.55),
            "stop_loss_atr_mult": ("float", 1.5, 5.0),
            "take_profit_atr_mult": ("float", 3.0, 8.0),
        },
    },
    "vol_regime_arb": {
        "module": "strategies.vol_regime_arb",
        "function": "vol_regime_arb_strategy",
        "params_dict": "PARAMS",
        "param_space": {
            "gk_fast_period": ("int", 5, 20),
            "gk_slow_period": ("int", 30, 80),
            "gk_baseline_period": ("int", 60, 150),
            "expansion_zscore_threshold": ("float", -2.5, -0.8),
            "expansion_min_bars_compressed": ("int", 3, 10),
            "expansion_momentum_period": ("int", 6, 20),
            "expansion_adx_min": ("float", 10.0, 28.0),
            "contraction_zscore_threshold": ("float", 1.5, 3.0),
            "contraction_rsi_oversold": ("float", 20.0, 40.0),
            "contraction_rsi_overbought": ("float", 60.0, 80.0),
            "enable_contraction_mode": ("bool",),
            "volume_threshold": ("float", 0.8, 2.0),
            "stop_loss_atr_mult": ("float", 1.5, 4.5),
            "take_profit_atr_mult": ("float", 2.5, 7.0),
        },
    },
    # ── LSTM + K-Means pattern recognition (requires pre-trained models) ──
    # NOTE: Round 5 — Multi-variant clustering with evolvable variant selection.
    # Evolution discovers which clustering algorithm + granularity works best.
    # Variants: kmeans_8/20/50, hier_20/50 (hierarchical tree), bisect_20/50 (divisive tree).
    "lstm_pattern": {
        "module": "strategies.lstm_pattern",
        "function": "lstm_pattern_strategy",
        "params_dict": "PARAMS",
        # Optional heavy deps. If any are missing the engine drops this strategy
        # at startup (one warning) instead of failing per-bar 200k times.
        "requires": ["sklearn", "tensorflow"],
        "param_space": {
            "cluster_variant": ("choice", [
                "kmeans_8", "kmeans_20", "kmeans_50",
                "hier_20", "hier_50",
                "bisect_20", "bisect_50",
            ]),
            "min_cluster_prob": ("float", 0.45, 0.75),
            "min_lstm_confidence": ("float", 0.35, 0.65),
            "min_combined_confidence": ("float", 0.45, 0.70),
            "cluster_weight": ("float", 0.2, 0.6),
            "lstm_weight": ("float", 0.4, 0.8),
            "adx_min": ("float", 10.0, 28.0),
            "require_volume": ("bool",),
            "volume_threshold": ("float", 0.8, 2.0),
            "rsi_oversold": ("float", 15.0, 35.0),
            "rsi_overbought": ("float", 65.0, 85.0),
            "stop_loss_atr_mult": ("float", 1.5, 4.5),
            "take_profit_atr_mult": ("float", 2.5, 7.0),
        },
    },
}


def _available_strategies() -> list:
    """
    Active strategies whose optional heavy dependencies are importable in THIS
    environment. A strategy declaring `requires` (e.g. lstm_pattern needs sklearn
    + tensorflow) is dropped — with a single clear warning — when a dep is missing,
    instead of failing inside every bar of every genome (which floods the log and
    wastes ~1/N of the compute budget). Run in the full venv to include them all.
    """
    import importlib.util
    available = []
    for name, spec in STRATEGY_REGISTRY.items():
        missing = [m for m in spec.get("requires", []) if importlib.util.find_spec(m) is None]
        if missing:
            logger.warning(
                f"Strategy '{name}' disabled this run: missing {', '.join(missing)} "
                f"(install them / use the full venv to include it)."
            )
            continue
        available.append(name)
    return available


# ═══════════════════════════════════════════════════════════════
# GENOME — a strategy configuration that can be evolved
# ═══════════════════════════════════════════════════════════════

@dataclass
class Genome:
    strategy: str
    params: dict
    fitness: float = -999.0
    sharpe: float = 0.0
    pf: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    net_pnl: float = 0.0
    max_dd: float = 0.0
    wf_sharpe: float = 0.0
    wf_degradation: float = 100.0
    wf_robust: bool = False
    # In-sample train/validation Sharpe (the GA's actual selection objective — STEP 2)
    is_train_sharpe: float = 0.0
    is_val_sharpe: float = 0.0
    # Cross-asset generalisation (STEP 5)
    cross_asset_median_sharpe: float = 0.0
    cross_asset_profitable: int = 0
    # Deflated Sharpe Ratio on in-sample trade returns (STEP 3)
    dsr: float = 0.0
    dsr_pass: bool = False
    generation: int = 0
    parent_ids: list = field(default_factory=list)
    genome_id: str = ""

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "params": self.params,
            "fitness": self.fitness,
            "sharpe": self.sharpe,
            "pf": self.pf,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "net_pnl": self.net_pnl,
            "max_dd": self.max_dd,
            "wf_sharpe": self.wf_sharpe,
            "wf_degradation": self.wf_degradation,
            "wf_robust": self.wf_robust,
            "is_train_sharpe": self.is_train_sharpe,
            "is_val_sharpe": self.is_val_sharpe,
            "cross_asset_median_sharpe": self.cross_asset_median_sharpe,
            "cross_asset_profitable": self.cross_asset_profitable,
            "dsr": self.dsr,
            "dsr_pass": self.dsr_pass,
            "generation": self.generation,
            "genome_id": self.genome_id,
        }


def random_param_value(spec):
    """Generate a random parameter value from its specification."""
    ptype = spec[0]
    if ptype == "float":
        return round(random.uniform(float(spec[1]), float(spec[2])), 4)
    elif ptype == "int":
        lo, hi = int(spec[1]), int(spec[2])
        if lo > hi:
            lo, hi = hi, lo
        return random.randint(lo, hi)
    elif ptype == "bool":
        return random.choice([True, False])
    elif ptype == "choice":
        return random.choice(spec[1])
    return None


def random_genome(strategy: str, generation: int = 0) -> Genome:
    """Create a random genome for a strategy."""
    spec = STRATEGY_REGISTRY[strategy]
    params = {}
    for pname, pspec in spec["param_space"].items():
        params[pname] = random_param_value(pspec)
    # Ensure take_profit > stop_loss
    if "take_profit_atr_mult" in params and "stop_loss_atr_mult" in params:
        if params["take_profit_atr_mult"] <= params["stop_loss_atr_mult"]:
            params["take_profit_atr_mult"] = params["stop_loss_atr_mult"] + 1.0
    return Genome(
        strategy=strategy,
        params=params,
        generation=generation,
        genome_id=f"G{generation}_{strategy[:3]}_{random.randint(1000,9999)}",
    )


def mutate_genome(genome: Genome, mutation_rate: float = 0.3, generation: int = 0) -> Genome:
    """Mutate a genome's parameters with Gaussian noise."""
    spec = STRATEGY_REGISTRY[genome.strategy]
    new_params = copy.deepcopy(genome.params)

    for pname, pspec in spec["param_space"].items():
        if pname not in new_params:
            continue
        if random.random() > mutation_rate:
            continue

        ptype = pspec[0]
        if ptype == "float":
            lo, hi = pspec[1], pspec[2]
            sigma = (hi - lo) * 0.15
            val = new_params[pname] + random.gauss(0, sigma)
            new_params[pname] = round(max(lo, min(hi, val)), 4)
        elif ptype == "int":
            lo, hi = int(pspec[1]), int(pspec[2])
            delta = max(1, int((hi - lo) * 0.2))
            val = int(new_params[pname]) + random.randint(-delta, delta)
            new_params[pname] = int(max(lo, min(hi, val)))
        elif ptype == "bool":
            new_params[pname] = not new_params[pname]
        elif ptype == "choice":
            new_params[pname] = random.choice(pspec[1])

    # Fix TP > SL constraint
    if "take_profit_atr_mult" in new_params and "stop_loss_atr_mult" in new_params:
        if new_params["take_profit_atr_mult"] <= new_params["stop_loss_atr_mult"]:
            new_params["take_profit_atr_mult"] = new_params["stop_loss_atr_mult"] + 0.5

    child = Genome(
        strategy=genome.strategy,
        params=new_params,
        generation=generation,
        parent_ids=[genome.genome_id],
        genome_id=f"G{generation}_{genome.strategy[:3]}_{random.randint(1000,9999)}",
    )
    return child


def crossover_genomes(g1: Genome, g2: Genome, generation: int = 0) -> Genome:
    """Crossover two genomes of the same strategy (uniform crossover)."""
    assert g1.strategy == g2.strategy
    new_params = {}
    for key in g1.params:
        if key in g2.params:
            new_params[key] = g1.params[key] if random.random() < 0.5 else g2.params[key]
        else:
            new_params[key] = g1.params[key]
    for key in g2.params:
        if key not in new_params:
            new_params[key] = g2.params[key]

    # Fix TP > SL
    if "take_profit_atr_mult" in new_params and "stop_loss_atr_mult" in new_params:
        if new_params["take_profit_atr_mult"] <= new_params["stop_loss_atr_mult"]:
            new_params["take_profit_atr_mult"] = new_params["stop_loss_atr_mult"] + 0.5

    return Genome(
        strategy=g1.strategy,
        params=new_params,
        generation=generation,
        parent_ids=[g1.genome_id, g2.genome_id],
        genome_id=f"G{generation}_{g1.strategy[:3]}_{random.randint(1000,9999)}",
    )


# ═══════════════════════════════════════════════════════════════
# FITNESS EVALUATION (in separate process via spawn)
# ═══════════════════════════════════════════════════════════════

def _worker_init():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Disable Metal GPU on Apple Silicon — prevents GPU contention across spawn workers
    os.environ["TF_METAL_DEVICE_PLACEMENT"] = "false"
    os.environ["METAL_DEVICE_WRITABLE"] = "0"
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass


def _trade_returns_from_df(trades_df):
    """Per-trade returns (pnl_net / notional size) from a backtester trades_df.

    These are *per-trade* (per-observation) returns — the correct unit for the
    Deflated Sharpe Ratio (do NOT pass annualised values)."""
    import numpy as np
    import pandas as pd
    if not isinstance(trades_df, pd.DataFrame) or len(trades_df) == 0:
        return np.array([])
    if "pnl_net" not in trades_df.columns:
        return np.array([])
    size = trades_df["size"].abs() if "size" in trades_df.columns else None
    if size is not None and (size > 0).any():
        ret = trades_df["pnl_net"].values / size.replace(0, np.nan).values
        ret = ret[np.isfinite(ret)]
        return np.asarray(ret, dtype=float)
    return np.asarray(trades_df["pnl_net"].values, dtype=float)


def evaluate_genome(args_tuple):
    """Evaluate a genome via walk-forward validation. Runs in spawn process."""
    (genome_dict, df_path, bt_cfg, risk_cfg, train_days, val_days, test_days,
     cross_paths, evo_knobs) = args_tuple

    import sys, os, copy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    import pandas as pd
    import numpy as np
    from config import BacktestConfig, RiskConfig
    from engine.backtester import Backtester, WalkForwardValidator

    df = pd.read_pickle(df_path)
    bt_config = BacktestConfig(**bt_cfg)
    risk_config = RiskConfig(**risk_cfg)

    strategy_name = genome_dict["strategy"]
    params = genome_dict["params"]
    genome_id = genome_dict["genome_id"]

    try:
        reg = STRATEGY_REGISTRY[strategy_name]
        mod = importlib.import_module(reg["module"])
        strategy_fn_ref = getattr(mod, reg["function"])
        params_dict_ref = copy.deepcopy(getattr(mod, reg["params_dict"]))
        params_dict_ref.update(params)
        setattr(mod, reg["params_dict"], params_dict_ref)

        def _bar_regime(d, i):
            return d.iloc[i].get("_regime", "unknown") if i < len(d) else "unknown"

        def strategy_fn(d, i, p):
            return strategy_fn_ref(d, i, p, regime=_bar_regime(d, i))

        # FULL (in-sample) backtest — used for reporting + DSR trade returns
        backtester = Backtester(bt_config, risk_config)
        trades_df, eq, metrics = backtester.run(df, strategy_fn, strategy_name)

        # WALK-FORWARD validation
        validator = WalkForwardValidator(bt_config, risk_config,
                                         train_days=train_days,
                                         val_days=val_days,
                                         test_days=test_days)
        wf_result = validator.validate(df, strategy_fn, strategy_name)
        wf_agg = wf_result["aggregate"]
        wf_degrad = wf_result.get("oos_degradation", 100.0)
        wf_robust = wf_result.get("is_robust", False)

        # ── In-sample TRAIN / VALIDATION Sharpe (the GA's selection objective) ──
        # STEP 2: the GA must NOT optimise on the walk-forward TEST (OOS) folds.
        # We aggregate the per-fold TRAIN and VAL Sharpes instead. wf_agg (the OOS
        # test aggregate) is still computed/reported but carries ZERO selection weight.
        fold_results = wf_result.get("fold_results", [])
        train_sharpes = [f["train"].sharpe_ratio for f in fold_results
                         if getattr(f["train"], "total_trades", 0) >= 3]
        val_sharpes = [f["val"].sharpe_ratio for f in fold_results
                       if getattr(f["val"], "total_trades", 0) >= 3]
        is_train_sharpe = float(np.mean(train_sharpes)) if train_sharpes else (
            metrics.sharpe_ratio if metrics.total_trades >= 5 else -5.0)
        is_val_sharpe = float(np.mean(val_sharpes)) if val_sharpes else is_train_sharpe

        # ── Cross-asset generalisation (STEP 5) ──
        # Same params, evaluated on the IN-SAMPLE windows of the cross assets
        # (their lockbox tails stay sealed — see EvolutionEngine._freeze_cross_assets).
        cross_sharpes = []
        if evo_knobs.get("cross_asset_generalisation", True) and cross_paths:
            sample_n = int(evo_knobs.get("cross_asset_sample_n", len(cross_paths)))
            for cpath in cross_paths[:max(1, sample_n)]:
                try:
                    cdf = pd.read_pickle(cpath)
                    _, _, cmetrics = backtester.run(cdf, strategy_fn,
                                                    f"{strategy_name}_xa")
                    if cmetrics.total_trades >= 5:
                        cross_sharpes.append(float(cmetrics.sharpe_ratio))
                except Exception:
                    continue
        cross_median = float(np.median(cross_sharpes)) if cross_sharpes else 0.0
        cross_profitable = int(sum(1 for s in cross_sharpes if s > 0))

        # ════════════════════════════════════════════════════════════════════
        # FITNESS (STEP 2 + STEP 5) — IN-SAMPLE ONLY selection objective.
        #
        #   fitness =  0.50 * is_train_sharpe          (in-sample train fit)
        #            + 0.30 * is_val_sharpe            (in-sample validation fit)
        #            + 0.10 * log(trades)              (trade-count bonus)
        #            + cross_w * clip(cross_median,..) (cross-asset generalisation)
        #            - degrad_penalty                  (WF degradation, robustness)
        #            - dd_penalty                      (full-sample drawdown)
        #            - low_trade_penalty               (<30 trades)
        #            - xrp_only_penalty                (profitable ONLY on XRP)
        #
        # NOTE: the walk-forward TEST (OOS) Sharpe `wf_agg.sharpe_ratio` is
        # COMPUTED and REPORTED below but has ZERO weight here. The degradation
        # term only penalises instability; it does not reward OOS profit.
        # ════════════════════════════════════════════════════════════════════
        MIN_TRADES_FOR_POSITIVE_FITNESS = 30
        trade_bonus = math.log(max(metrics.total_trades, 1)) * 0.1
        degrad_penalty = max(0, wf_degrad - 30) * 0.02
        dd_penalty = max(0, metrics.max_drawdown_pct - 5) * 0.1
        low_trade_penalty = max(0, (MIN_TRADES_FOR_POSITIVE_FITNESS - metrics.total_trades) * 0.05) if metrics.total_trades < MIN_TRADES_FOR_POSITIVE_FITNESS else 0.0

        cross_w = float(evo_knobs.get("cross_asset_weight", 0.25))
        xrp_only_pen = float(evo_knobs.get("cross_asset_xrp_only_penalty", 0.30))
        cross_term = 0.0
        cross_penalty = 0.0
        if evo_knobs.get("cross_asset_generalisation", True) and cross_sharpes:
            # Reward positive median cross-asset Sharpe (clipped to keep it bounded).
            cross_term = cross_w * max(-1.0, min(1.0, cross_median))
            # Penalise genomes that work only on the primary asset.
            if cross_median <= 0.0:
                cross_penalty = xrp_only_pen

        fitness = (is_train_sharpe * 0.5
                   + is_val_sharpe * 0.3
                   + trade_bonus
                   + cross_term
                   - degrad_penalty
                   - dd_penalty
                   - low_trade_penalty
                   - cross_penalty)

        # ── Deflated Sharpe Ratio on in-sample TRADE returns (STEP 3) ──
        # Deflated by the REAL cumulative number of genome evaluations so far.
        dsr_val = 0.0
        try:
            from benchmark.statistics import deflated_sharpe_ratio
            tr = _trade_returns_from_df(trades_df)
            n_trials = max(int(evo_knobs.get("n_trials", 1)), 1)
            if len(tr) >= int(evo_knobs.get("dsr_min_trades", 10)):
                dsr_val = float(deflated_sharpe_ratio(tr, n_trials, None)["dsr"])
        except Exception:
            dsr_val = 0.0

        # Monte Carlo simulation for top genomes only (fitness > 1.5 to avoid
        # wasting compute on mediocre genomes during evolution).
        # Reduced to 200 simulations — sufficient for ruin/profit probability.
        mc_result = None
        if fitness > 1.5 and metrics.total_trades >= 15:
            try:
                from engine.monte_carlo import monte_carlo_trades
                trade_pnls = np.array([t.pnl_net for t in trades_df]) if hasattr(trades_df, '__iter__') and len(trades_df) > 0 else None
                if trade_pnls is None and isinstance(trades_df, pd.DataFrame) and len(trades_df) > 0:
                    trade_pnls = trades_df["pnl_net"].values if "pnl_net" in trades_df.columns else None
                if trade_pnls is not None and len(trade_pnls) >= 15:
                    mc = monte_carlo_trades(
                        trade_pnls, initial_capital=200.0,
                        n_simulations=200, method="bootstrap", seed=42,
                    )
                    mc_result = mc.to_dict()
                    # Penalize if Monte Carlo shows high ruin probability
                    if mc.prob_ruin > 0.1:
                        fitness -= mc.prob_ruin * 0.5
                    # Bonus for high probability of profit
                    if mc.prob_profit > 0.8:
                        fitness += 0.1
            except Exception:
                pass

        return {
            "genome_id": genome_id,
            "strategy": strategy_name,
            "params": params,
            "fitness": fitness,
            "sharpe": metrics.sharpe_ratio,
            "sortino": metrics.sortino_ratio,
            "pf": metrics.profit_factor,
            "trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "net_pnl": metrics.net_pnl,
            "max_dd": metrics.max_drawdown_pct,
            "wf_sharpe": wf_agg.sharpe_ratio,
            "wf_trades": wf_agg.total_trades,
            "wf_degradation": wf_degrad,
            "wf_robust": wf_robust,
            "is_train_sharpe": is_train_sharpe,
            "is_val_sharpe": is_val_sharpe,
            "cross_asset_median_sharpe": cross_median,
            "cross_asset_profitable": cross_profitable,
            "dsr": dsr_val,
            "monte_carlo": mc_result,
        }

    except Exception as e:
        return {
            "genome_id": genome_id,
            "strategy": strategy_name,
            "error": str(e),
            "fitness": -999.0,
        }


# ═══════════════════════════════════════════════════════════════
# LEARNING ENGINE — tracks what works and narrows search space
# ═══════════════════════════════════════════════════════════════

class LearningEngine:
    """Tracks successful parameter ranges and biases future mutations."""

    def __init__(self):
        self.successes = defaultdict(list)  # strategy -> list of successful param dicts
        self.failures = defaultdict(list)   # strategy -> list of failed param dicts
        self.best_ranges = {}               # strategy -> narrowed param ranges
        self.strategy_scores = defaultdict(list)  # strategy -> list of fitness scores

    def record(self, genome: Genome):
        """Record a genome's result for learning."""
        if genome.fitness > 0:
            self.successes[genome.strategy].append(genome.params)
        else:
            self.failures[genome.strategy].append(genome.params)
        self.strategy_scores[genome.strategy].append(genome.fitness)

    def get_promising_strategies(self, top_n: int = 5) -> list[str]:
        """Return strategies sorted by average fitness."""
        avg_scores = {}
        for strat, scores in self.strategy_scores.items():
            if len(scores) >= 3:
                avg_scores[strat] = np.mean(sorted(scores, reverse=True)[:5])
            else:
                avg_scores[strat] = np.mean(scores) if scores else -10
        ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked[:top_n]]

    def get_learned_ranges(self, strategy: str) -> dict:
        """Get narrowed parameter ranges based on successful genomes."""
        successes = self.successes.get(strategy, [])
        if len(successes) < 3:
            return {}

        spec = STRATEGY_REGISTRY.get(strategy, {}).get("param_space", {})
        narrowed = {}
        for pname, pspec in spec.items():
            ptype = pspec[0]
            values = [s.get(pname) for s in successes if pname in s and s[pname] is not None]
            if len(values) < 2:
                continue
            if ptype in ("float", "int"):
                lo_orig, hi_orig = pspec[1], pspec[2]
                lo_new = max(lo_orig, np.percentile(values, 10))
                hi_new = min(hi_orig, np.percentile(values, 90))
                # Keep at least 30% of original range
                min_range = (hi_orig - lo_orig) * 0.3
                if hi_new - lo_new < min_range:
                    center = np.mean(values)
                    lo_new = max(lo_orig, center - min_range / 2)
                    hi_new = min(hi_orig, center + min_range / 2)
                if ptype == "int":
                    lo_new, hi_new = int(round(lo_new)), int(round(hi_new))
                    if lo_new > hi_new:
                        lo_new, hi_new = hi_new, lo_new
                else:
                    lo_new, hi_new = float(lo_new), float(hi_new)
                narrowed[pname] = (ptype, lo_new, hi_new)
        return narrowed

    def smart_random_genome(self, strategy: str, generation: int) -> Genome:
        """Generate a genome biased toward successful parameter ranges."""
        narrowed = self.get_learned_ranges(strategy)
        spec = STRATEGY_REGISTRY[strategy]
        params = {}

        for pname, pspec in spec["param_space"].items():
            if pname in narrowed and random.random() < 0.7:
                params[pname] = random_param_value(narrowed[pname])
            else:
                params[pname] = random_param_value(pspec)

        if "take_profit_atr_mult" in params and "stop_loss_atr_mult" in params:
            if params["take_profit_atr_mult"] <= params["stop_loss_atr_mult"]:
                params["take_profit_atr_mult"] = params["stop_loss_atr_mult"] + 1.0

        return Genome(
            strategy=strategy,
            params=params,
            generation=generation,
            genome_id=f"G{generation}_{strategy[:3]}_{random.randint(1000,9999)}",
        )


# ═══════════════════════════════════════════════════════════════
# MAIN EVOLUTION LOOP
# ═══════════════════════════════════════════════════════════════

class EvolutionEngine:
    """Autonomous evolutionary optimizer."""

    def __init__(
        self,
        df: pd.DataFrame,
        max_hours: float = 14.0,
        pop_size: int = 40,
        max_workers: int = None,
        train_days: int = 90,
        val_days: int = 15,
        test_days: int = 45,
    ):
        self.df = df
        self.max_hours = max_hours
        self.pop_size = pop_size
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.train_days = train_days
        self.val_days = val_days
        self.test_days = test_days

        self.config = LabConfig()
        self.bt_config = asdict(self.config.backtest)
        self.risk_config = asdict(self.config.risk)
        self.evo_config = self.config.evolution

        self.learning = LearningEngine()
        self.hall_of_fame = []  # Top 20 all-time best genomes (diverse)
        self.hall_of_fame_per_strategy = defaultdict(list)  # per-strategy top 5
        self.generation = 0
        self.total_evaluated = 0
        self.total_robust = 0
        # Cumulative trials from PRIOR warm-started runs. The Deflated Sharpe must
        # deflate against the size of the WHOLE compounding search (across loop
        # cycles), not just this run — otherwise the bar gets dishonestly easier
        # every cycle as warm-start carries survivors forward.
        self.prior_trials = self._load_prior_trials()
        self.strategies_available = _available_strategies()
        # Minimum exploration: at least 15% of pop goes to underexplored strategies
        self.min_explore_pct = 0.15

        # Save df for workers
        os.makedirs("/tmp/crypto_evolve", exist_ok=True)
        self.df_path = "/tmp/crypto_evolve/df.pkl"
        df.to_pickle(self.df_path)

        # STEP 5: freeze cross-asset IN-SAMPLE windows once for generalisation
        # pressure. Their lockbox tails are kept sealed.
        self.cross_paths = self._freeze_cross_assets()

        # STEP 3: real cumulative number of genome evaluations (the DSR trial count).
        self.dsr_min = float(self.evo_config.dsr_min)
        self.dsr_rejected = 0

        # Logging
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/evolution_{ts}.jsonl"
        self.report_path = f"reports/evolution_report_{ts}.json"
        logger.info(f"Evolution log: {self.log_path}")

    def _freeze_cross_assets(self) -> list:
        """STEP 5: write the cross-asset IN-SAMPLE feature frames to /tmp once.

        Loads each cross asset from the frozen benchmark snapshot via DataLockbox
        and keeps ONLY the in-sample portion (everything before that asset's
        lockbox-equivalent tail), so the cross-asset lockboxes stay sealed.
        Returns a list of pickle paths (picklable args for the spawn workers).
        Returns [] if disabled or the snapshot is unavailable.
        """
        if not self.evo_config.cross_asset_generalisation:
            logger.info("Cross-asset generalisation: DISABLED via config.")
            return []
        paths = []
        try:
            from benchmark.spec import BENCHMARK_SPEC
            from benchmark.data_lockbox import DataLockbox
            lb = DataLockbox(BENCHMARK_SPEC)
            cutoff_ms = BENCHMARK_SPEC.lockbox_days * 86_400_000
            wanted = list(self.evo_config.cross_asset_symbols)
            for sym in wanted:
                if sym not in lb.manifest.symbols:
                    continue
                cdf = lb._load(sym)
                # in-sample = drop the final lockbox_days tail
                tail_cutoff = int(cdf["timestamp"].iloc[-1]) - cutoff_ms
                in_sample = cdf[cdf["timestamp"] < tail_cutoff].reset_index(drop=True)
                if len(in_sample) < 200:
                    continue
                cpath = f"/tmp/crypto_evolve/cross_{sym.replace('/', '')}.pkl"
                in_sample.to_pickle(cpath)
                paths.append(cpath)
            logger.info(
                f"Cross-asset generalisation: ENABLED on {len(paths)} assets "
                f"(in-sample only; lockbox tails sealed) "
                f"[weight={self.evo_config.cross_asset_weight}]."
            )
        except Exception as e:
            logger.warning(
                f"Cross-asset generalisation unavailable ({e}); proceeding without "
                f"cross-asset pressure."
            )
            return []
        return paths

    def _log_event(self, event: dict):
        """Append event to JSONL log."""
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event["generation"] = self.generation
        event["total_evaluated"] = self.total_evaluated
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def _load_prior_trials(self) -> int:
        """Cumulative trial count carried from prior warm-started runs (0 if cold)."""
        if not getattr(self.evo_config, "warm_start", False):
            return 0
        path = "reports/evolution_checkpoint.json"
        if not os.path.exists(path):
            return 0
        try:
            with open(path) as f:
                ckpt = json.load(f)
            # Prefer an explicit cumulative count; fall back to last run's total.
            return int(ckpt.get("cumulative_trials", ckpt.get("total_evaluated", 0)) or 0)
        except Exception:
            return 0

    def _load_seed_genomes(self) -> list[Genome]:
        """
        Warm-start: reconstruct genomes from the previous run's Hall of Fame
        (reports/evolution_checkpoint.json) so successive 1h loops COMPOUND.
        Only strategies available in THIS environment are kept; fitness is reset
        to -999 so every seed is re-evaluated fresh on the in-sample data (we
        never trust a stale, possibly-leaky score).
        """
        if not getattr(self.evo_config, "warm_start", False):
            return []
        path = "reports/evolution_checkpoint.json"
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                ckpt = json.load(f)
        except Exception as e:
            logger.warning(f"Warm-start: could not read checkpoint ({e}); cold start.")
            return []
        # Prefer the per-strategy best regions (carried regardless of the DSR
        # gate) so the search compounds; fall back to the DSR-gated global HoF.
        candidates = []
        per_strat = ckpt.get("hall_of_fame_per_strategy", {})
        for strat, genomes in per_strat.items():
            for g in genomes:
                candidates.append(g)
        if not candidates:
            candidates = ckpt.get("hall_of_fame", [])

        seeds = []
        seen = set()
        for g in candidates:
            strat = g.get("strategy")
            params = g.get("params")
            if strat in self.strategies_available and isinstance(params, dict) and params:
                key = (strat, json.dumps(params, sort_keys=True, default=str))
                if key in seen:
                    continue
                seen.add(key)
                seeds.append(Genome(strategy=strat, params=dict(params), generation=0))
        cap = int(self.pop_size * getattr(self.evo_config, "warm_start_max_frac", 0.5))
        if len(seeds) > cap:
            seeds = seeds[:cap]
        if seeds:
            logger.info(f"Warm-start: seeded {len(seeds)} genomes from previous "
                        f"Hall of Fame (cap {cap}); rest random for diversity.")
        return seeds

    def generate_initial_population(self) -> list[Genome]:
        """Create diverse initial population across all strategies."""
        pop = self._load_seed_genomes()   # warm-start seeds (may be empty)

        per_strat = max(1, (self.pop_size - len(pop)) // len(self.strategies_available))
        for strat in self.strategies_available:
            for _ in range(per_strat):
                if len(pop) >= self.pop_size:
                    break
                pop.append(random_genome(strat, generation=0))

        # Fill remaining with random
        while len(pop) < self.pop_size:
            strat = random.choice(self.strategies_available)
            pop.append(random_genome(strat, generation=0))

        random.shuffle(pop)
        return pop[:self.pop_size]

    def evaluate_population(self, population: list[Genome]) -> list[Genome]:
        """Evaluate all genomes in parallel."""
        # STEP 3: n_trials = REAL cumulative number of genome evaluations so far.
        # Snapshot it at the start of the batch so every genome in this batch is
        # deflated by the same (monotonically growing) trial count. +1 so the
        # very first batch uses >= 1.
        # Cumulative across warm-started cycles: prior runs' trials + this run so far.
        n_trials_now = max(self.prior_trials + self.total_evaluated + 1, 1)
        evo_knobs = {
            "cross_asset_generalisation": bool(self.evo_config.cross_asset_generalisation),
            "cross_asset_sample_n": int(self.evo_config.cross_asset_sample_n),
            "cross_asset_weight": float(self.evo_config.cross_asset_weight),
            "cross_asset_xrp_only_penalty": float(self.evo_config.cross_asset_xrp_only_penalty),
            "dsr_min_trades": int(self.evo_config.dsr_min_trades),
            "n_trials": n_trials_now,
        }
        work_items = [
            (g.to_dict(), self.df_path, self.bt_config, self.risk_config,
             self.train_days, self.val_days, self.test_days,
             self.cross_paths, evo_knobs)
            for g in population
        ]

        results = []
        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=MP_CTX,
            initializer=_worker_init,
        ) as executor:
            futures = {executor.submit(evaluate_genome, item): i
                       for i, item in enumerate(work_items)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=300)
                    genome = population[idx]

                    if "error" not in result:
                        genome.fitness = result["fitness"]
                        genome.sharpe = result.get("sharpe", 0)
                        genome.pf = result.get("pf", 0)
                        genome.trades = result.get("trades", 0)
                        genome.win_rate = result.get("win_rate", 0)
                        genome.net_pnl = result.get("net_pnl", 0)
                        genome.max_dd = result.get("max_dd", 0)
                        genome.wf_sharpe = result.get("wf_sharpe", 0)
                        genome.wf_degradation = result.get("wf_degradation", 100)
                        genome.wf_robust = result.get("wf_robust", False)
                        genome.is_train_sharpe = result.get("is_train_sharpe", 0.0)
                        genome.is_val_sharpe = result.get("is_val_sharpe", 0.0)
                        genome.cross_asset_median_sharpe = result.get("cross_asset_median_sharpe", 0.0)
                        genome.cross_asset_profitable = result.get("cross_asset_profitable", 0)
                        genome.dsr = result.get("dsr", 0.0)
                        # STEP 3: DSR acceptance flag (in-sample trade returns,
                        # deflated by the real cumulative trial count).
                        genome.dsr_pass = bool(genome.dsr >= self.dsr_min)

                        self.learning.record(genome)
                        self.total_evaluated += 1
                        if genome.wf_robust:
                            self.total_robust += 1

                        # Update hall of fame with diversity enforcement +
                        # Deflated-Sharpe acceptance filter (STEP 3).
                        self._update_hall_of_fame(genome)

                        results.append(genome)
                    else:
                        genome.fitness = -999.0
                        results.append(genome)

                except Exception as e:
                    population[idx].fitness = -999.0
                    results.append(population[idx])

        return results

    @staticmethod
    def _genome_param_distance(g1: Genome, g2: Genome) -> float:
        """Compute normalized parameter distance between two genomes of the same strategy."""
        if g1.strategy != g2.strategy:
            return 1.0
        spec = STRATEGY_REGISTRY[g1.strategy]["param_space"]
        diffs = []
        weights = []
        for pname, pspec in spec.items():
            v1 = g1.params.get(pname)
            v2 = g2.params.get(pname)
            if v1 is None or v2 is None:
                continue
            ptype = pspec[0]
            if ptype in ("float", "int"):
                lo, hi = float(pspec[1]), float(pspec[2])
                rng = hi - lo if hi > lo else 1.0
                diffs.append(abs(float(v1) - float(v2)) / rng)
                weights.append(1.0)
            elif ptype in ("bool", "choice"):
                # Weight bool/choice 2x so they aren't diluted by many float params
                diffs.append(0.0 if v1 == v2 else 1.0)
                weights.append(2.0)
        if not diffs:
            return 0.0
        return float(np.average(diffs, weights=weights))

    def _update_hall_of_fame(self, genome: Genome):
        """Update hall of fame with diversity enforcement.

        Rules:
        - STEP 3: Deflated-Sharpe acceptance filter. A genome is only admitted to
          the global HoF if its in-sample-trade DSR clears self.dsr_min (0.95 by
          default), deflated by the REAL cumulative number of genome evaluations.
          Genomes failing DSR are EXCLUDED from the global HoF and LOGGED (never
          silently dropped). They remain tracked in the per-strategy list (flagged
          dsr_pass=False) so reporting still sees them.
        - Max 4 genomes per strategy in global HoF (ensures strategy diversity)
        - New genome must have param distance > 0.15 from existing HoF entries
          of the same strategy (prevents clones flooding HoF)
        - Also maintain per-strategy top 5 for best-per-strategy tracking
        """
        # Per-strategy tracking (no dedup needed) — keeps both pass/fail genomes
        per_strat = self.hall_of_fame_per_strategy[genome.strategy]
        per_strat.append(genome)
        per_strat.sort(key=lambda g: g.fitness, reverse=True)
        self.hall_of_fame_per_strategy[genome.strategy] = per_strat[:5]

        # STEP 3: Deflated-Sharpe acceptance gate for the GLOBAL Hall of Fame.
        if not genome.dsr_pass:
            self.dsr_rejected += 1
            logger.info(
                f"  DSR REJECT: [{genome.strategy}] {genome.genome_id} "
                f"fitness={genome.fitness:.3f} dsr={genome.dsr:.3f} "
                f"(< {self.dsr_min:.2f}, n_trials={self.prior_trials + self.total_evaluated}) "
                f"— excluded from global HoF"
            )
            self._log_event({
                "event": "dsr_reject",
                "genome_id": genome.genome_id,
                "strategy": genome.strategy,
                "fitness": genome.fitness,
                "dsr": genome.dsr,
                "dsr_min": self.dsr_min,
            })
            return

        # Global HoF: check if it's a clone of existing entries
        # Fix: check ALL same-strategy entries for clones, remove ALL that are
        # too similar AND worse. Then decide whether to add.
        same_strat_in_hof = [g for g in self.hall_of_fame if g.strategy == genome.strategy]

        is_clone = False
        to_remove = []
        for existing in same_strat_in_hof:
            dist = self._genome_param_distance(genome, existing)
            if dist < 0.15:
                is_clone = True
                if genome.fitness > existing.fitness:
                    to_remove.append(existing)
                else:
                    # Clone exists with equal/better fitness — reject new genome
                    return

        # Remove inferior clones
        for g in to_remove:
            self.hall_of_fame.remove(g)

        if is_clone and to_remove:
            # Replaced inferior clone(s) — add the new genome
            self.hall_of_fame.append(genome)
        elif not is_clone:
            # Not a clone — check strategy cap (max 4 per strategy in global HoF)
            MAX_PER_STRATEGY = 4
            current_count = len(same_strat_in_hof) - len(to_remove)
            if current_count >= MAX_PER_STRATEGY:
                remaining = [g for g in same_strat_in_hof if g not in to_remove]
                if remaining:
                    worst_of_strat = min(remaining, key=lambda g: g.fitness)
                    if genome.fitness > worst_of_strat.fitness:
                        self.hall_of_fame.remove(worst_of_strat)
                        self.hall_of_fame.append(genome)
            else:
                self.hall_of_fame.append(genome)

        self.hall_of_fame.sort(key=lambda g: g.fitness, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:20]

    def tournament_select(self, population: list[Genome], k: int = 3) -> Genome:
        """Tournament selection — pick the best of k random individuals."""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda g: g.fitness)

    def create_next_generation(self, population: list[Genome]) -> list[Genome]:
        """Create next generation via selection, crossover, mutation + injection.

        Key improvement: guarantees minimum exploration slots for each strategy
        to prevent premature convergence to a single strategy.
        """
        self.generation += 1
        next_gen = []

        # ── Phase A: Mandatory exploration (min_explore_pct of pop) ──
        # Ensure every strategy gets at least 1 fresh genome per generation
        # This prevents the "monoculture" problem where tournament selection
        # funnels all resources into the dominant strategy.
        explore_slots = max(len(self.strategies_available),
                            int(self.pop_size * self.min_explore_pct))
        strats_cycle = list(self.strategies_available)
        random.shuffle(strats_cycle)
        idx = 0
        for _ in range(explore_slots):
            strat = strats_cycle[idx % len(strats_cycle)]
            idx += 1
            # 50% chance smart random (if learned data exists), 50% pure random
            if random.random() < 0.5 and len(self.learning.successes.get(strat, [])) >= 3:
                child = self.learning.smart_random_genome(strat, self.generation)
            else:
                child = random_genome(strat, self.generation)
            next_gen.append(child)

        # ── Phase B: Elitism — keep top 10% (diverse) ──
        elite_count = max(2, self.pop_size // 10)
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        # Pick elites but try to include different strategies
        seen_strats_elite = set()
        for g in sorted_pop:
            if len(next_gen) >= explore_slots + elite_count:
                break
            # Allow max 2 elites per strategy
            if sum(1 for e in next_gen[explore_slots:] if e.strategy == g.strategy) < 2:
                elite = copy.deepcopy(g)
                elite.generation = self.generation
                next_gen.append(elite)
                seen_strats_elite.add(g.strategy)

        # ── Phase C: Fill remaining via crossover, mutation, smart random ──
        promising = self.learning.get_promising_strategies(top_n=5)

        while len(next_gen) < self.pop_size:
            r = random.random()

            if r < 0.30:
                # CROSSOVER: pick two parents of same strategy, combine
                parent1 = self.tournament_select(population)
                same_strat = [g for g in population if g.strategy == parent1.strategy and g is not parent1]
                if same_strat:
                    parent2 = max(random.sample(same_strat, min(3, len(same_strat))),
                                  key=lambda g: g.fitness)
                    child = crossover_genomes(parent1, parent2, self.generation)
                    child = mutate_genome(child, mutation_rate=0.15, generation=self.generation)
                else:
                    child = mutate_genome(parent1, mutation_rate=0.3, generation=self.generation)
                next_gen.append(child)

            elif r < 0.55:
                # MUTATION: pick a good parent, mutate
                parent = self.tournament_select(population)
                child = mutate_genome(parent, mutation_rate=0.3, generation=self.generation)
                next_gen.append(child)

            elif r < 0.75:
                # SMART RANDOM: use learned ranges for promising strategies
                strat = random.choice(promising) if promising else random.choice(self.strategies_available)
                child = self.learning.smart_random_genome(strat, self.generation)
                next_gen.append(child)

            else:
                # PURE RANDOM: maintain diversity, explore all strategies
                strat = random.choice(self.strategies_available)
                child = random_genome(strat, self.generation)
                next_gen.append(child)

        return next_gen[:self.pop_size]

    def print_generation_report(self, population: list[Genome]):
        """Print summary of current generation."""
        valid = [g for g in population if g.fitness > -999]
        if not valid:
            logger.info(f"  Gen {self.generation}: No valid results")
            return

        # Strategy breakdown
        strat_counts = Counter(g.strategy for g in valid)
        strat_best = {}
        for g in valid:
            if g.strategy not in strat_best or g.fitness > strat_best[g.strategy].fitness:
                strat_best[g.strategy] = g

        best = max(valid, key=lambda g: g.fitness)
        avg_fit = np.mean([g.fitness for g in valid])
        robust_count = sum(1 for g in valid if g.wf_robust)

        logger.info(f"\n{'─'*80}")
        logger.info(f"  GEN {self.generation} | Evaluated: {len(valid)}/{self.pop_size} | "
                     f"Robust: {robust_count} | Avg fitness: {avg_fit:.3f}")
        logger.info(f"  BEST: [{best.strategy}] fitness={best.fitness:.3f} "
                     f"IS_train_Sh={best.is_train_sharpe:.3f} IS_val_Sh={best.is_val_sharpe:.3f} "
                     f"WF_Sharpe(OOS,unweighted)={best.wf_sharpe:.3f} "
                     f"xa_med_Sh={best.cross_asset_median_sharpe:.3f} "
                     f"DSR={best.dsr:.3f}{'✓' if best.dsr_pass else '✗'} "
                     f"trades={best.trades} PnL=${best.net_pnl:.2f} "
                     f"degrad={best.wf_degradation:.0f}%")

        logger.info(f"  Strategy breakdown:")
        for strat in sorted(strat_counts.keys()):
            b = strat_best[strat]
            logger.info(f"    {strat:25s}: {strat_counts[strat]:2d} tested | "
                         f"best fit={b.fitness:+.3f} Sharpe={b.sharpe:.3f} "
                         f"WF={b.wf_sharpe:.3f} trades={b.trades}")

        if self.hall_of_fame:
            hof = self.hall_of_fame[0]
            logger.info(f"  HALL OF FAME #1: [{hof.strategy}] fitness={hof.fitness:.3f} "
                         f"gen={hof.generation}")

        # Log event
        self._log_event({
            "event": "generation_complete",
            "best_fitness": best.fitness,
            "best_strategy": best.strategy,
            "avg_fitness": avg_fit,
            "robust_count": robust_count,
            "strategy_breakdown": {s: c for s, c in strat_counts.items()},
        })

    def run(self):
        """Main evolution loop — runs for max_hours."""
        start_time = time.time()
        deadline = start_time + self.max_hours * 3600

        logger.info(f"\n{'═'*80}")
        logger.info(f"  AUTONOMOUS EVOLUTION ENGINE")
        logger.info(f"  Duration: {self.max_hours} hours")
        logger.info(f"  Population: {self.pop_size}")
        logger.info(f"  Workers: {self.max_workers}")
        logger.info(f"  Strategies: {len(self.strategies_available)}")
        logger.info(f"  Data: {len(self.df)} bars")
        logger.info(f"  Walk-forward: {self.train_days}d train / {self.val_days}d val / {self.test_days}d test")
        logger.info(f"{'═'*80}\n")

        # Phase 1: Initial population
        logger.info("Phase 1: Generating initial population...")
        population = self.generate_initial_population()
        logger.info(f"  {len(population)} genomes across {len(set(g.strategy for g in population))} strategies")

        # Evaluate
        logger.info("Phase 1: Evaluating initial population...")
        population = self.evaluate_population(population)
        self.print_generation_report(population)

        # Phase 2: Evolution loop
        while time.time() < deadline:
            elapsed_h = (time.time() - start_time) / 3600
            remaining_h = self.max_hours - elapsed_h

            logger.info(f"\n  [{elapsed_h:.1f}h elapsed, {remaining_h:.1f}h remaining, "
                         f"{self.total_evaluated} total evaluated, {self.total_robust} robust]")

            # Create and evaluate next generation
            population = self.create_next_generation(population)
            population = self.evaluate_population(population)
            self.print_generation_report(population)

            # Periodic summary
            if self.generation % 5 == 0:
                self._print_hall_of_fame()
                self._save_checkpoint()

        # Final report
        elapsed_h = (time.time() - start_time) / 3600
        logger.info(f"\n{'═'*80}")
        logger.info(f"  EVOLUTION COMPLETE — {elapsed_h:.1f} hours, "
                     f"{self.generation} generations, {self.total_evaluated} evaluations")
        logger.info(f"{'═'*80}")
        self._print_hall_of_fame()
        self._save_final_report()
        self._save_checkpoint()

    def _print_hall_of_fame(self):
        """Print the all-time best genomes (diverse)."""
        logger.info(f"\n{'='*80}")
        logger.info("HALL OF FAME — TOP 10 ALL-TIME (diversity enforced)")
        logger.info(f"{'='*80}")
        logger.info(f"{'Rank':>4} {'Strategy':>25} {'Fitness':>8} {'IS_Sh':>7} {'WF_Sh':>7} "
                     f"{'PF':>6} {'Trades':>7} {'WR%':>6} {'PnL$':>10} {'Deg%':>6} {'Robust':>6}")
        logger.info("-" * 100)

        for i, g in enumerate(self.hall_of_fame[:10]):
            robust_str = "✓" if g.wf_robust else "✗"
            logger.info(
                f"{i+1:>4} {g.strategy:>25} {g.fitness:>8.3f} {g.sharpe:>7.3f} "
                f"{g.wf_sharpe:>7.3f} {g.pf:>6.3f} {g.trades:>7} "
                f"{g.win_rate*100:>5.1f}% ${g.net_pnl:>9.2f} {g.wf_degradation:>5.0f}% "
                f"{robust_str:>6}"
            )

        # Per-strategy best
        if self.hall_of_fame_per_strategy:
            logger.info(f"\n  BEST PER STRATEGY:")
            for strat in sorted(self.hall_of_fame_per_strategy.keys()):
                bests = self.hall_of_fame_per_strategy[strat]
                if bests:
                    b = bests[0]
                    logger.info(f"    {strat:25s}: fit={b.fitness:+.3f} Sh={b.sharpe:.3f} "
                                f"WF={b.wf_sharpe:.3f} trades={b.trades}")

    def _save_checkpoint(self):
        """Save current state to disk."""
        checkpoint = {
            "generation": self.generation,
            "total_evaluated": self.total_evaluated,
            # Cumulative trials across all warm-started cycles (for honest DSR).
            "cumulative_trials": self.prior_trials + self.total_evaluated,
            "total_robust": self.total_robust,
            "dsr_rejected": self.dsr_rejected,
            "hall_of_fame": [g.to_dict() for g in self.hall_of_fame],
            # Per-strategy best regions — carried even when nothing passes the DSR
            # gate, so warm-start can COMPOUND the search across runs.
            "hall_of_fame_per_strategy": {
                strat: [g.to_dict() for g in genomes]
                for strat, genomes in self.hall_of_fame_per_strategy.items()
            },
            "promising_strategies": self.learning.get_promising_strategies(),
        }
        path = "reports/evolution_checkpoint.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def _save_final_report(self):
        """Save comprehensive final report."""
        report = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "generations": self.generation,
            "total_evaluated": self.total_evaluated,
            "total_robust": self.total_robust,
            "dsr_rejected": self.dsr_rejected,
            "hall_of_fame": [g.to_dict() for g in self.hall_of_fame],
            "promising_strategies": self.learning.get_promising_strategies(),
            "strategy_scores": {
                s: {
                    "count": len(scores),
                    "mean": float(np.mean(scores)),
                    "max": float(np.max(scores)),
                    "positive_pct": float(sum(1 for x in scores if x > 0) / len(scores) * 100),
                }
                for s, scores in self.learning.strategy_scores.items()
                if len(scores) > 0
            },
        }

        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\nFinal report saved: {self.report_path}")

        # Save best genome per strategy as ready-to-use configs.
        # STEP 3: prefer a DSR-passing genome; deprioritise DSR failures.
        for strat, bests in self.hall_of_fame_per_strategy.items():
            if not bests:
                continue
            dsr_ok = [g for g in bests if g.dsr_pass and g.fitness > 0]
            best = dsr_ok[0] if dsr_ok else (bests[0] if bests[0].fitness > 0 else None)
            if best is None:
                continue
            best_path = f"reports/best_genome_{strat}.json"
            with open(best_path, "w") as f:
                json.dump({
                    "strategy": best.strategy,
                    "params": best.params,
                    "fitness": best.fitness,
                    "sharpe": best.sharpe,
                    "wf_sharpe": best.wf_sharpe,
                    "is_train_sharpe": best.is_train_sharpe,
                    "is_val_sharpe": best.is_val_sharpe,
                    "cross_asset_median_sharpe": best.cross_asset_median_sharpe,
                    "dsr": best.dsr,
                    "dsr_pass": best.dsr_pass,
                    "trades": best.trades,
                    "net_pnl": best.net_pnl,
                    "wf_robust": best.wf_robust,
                }, f, indent=2)
            logger.info(f"Best genome saved: {best_path} (dsr_pass={best.dsr_pass})")

        # Also save global best (the global HoF is already DSR-gated).
        if self.hall_of_fame:
            best = self.hall_of_fame[0]
            best_path = f"reports/best_genome_overall.json"
            with open(best_path, "w") as f:
                json.dump({
                    "strategy": best.strategy,
                    "params": best.params,
                    "fitness": best.fitness,
                    "sharpe": best.sharpe,
                    "wf_sharpe": best.wf_sharpe,
                    "is_train_sharpe": best.is_train_sharpe,
                    "is_val_sharpe": best.is_val_sharpe,
                    "cross_asset_median_sharpe": best.cross_asset_median_sharpe,
                    "dsr": best.dsr,
                    "dsr_pass": best.dsr_pass,
                    "trades": best.trades,
                    "net_pnl": best.net_pnl,
                    "wf_robust": best.wf_robust,
                }, f, indent=2)
            logger.info(f"Overall best genome saved: {best_path}")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(history_days: int = 365, use_lockbox: bool = True) -> pd.DataFrame:
    """Load the price series the GA is allowed to optimise on.

    STEP 1 — Data lockbox wiring:
      When `use_lockbox` is True (default), evolution trains ONLY on the in-sample
      window of the frozen benchmark snapshot. The final `lockbox_days` of the
      primary series are sealed and NEVER enter any backtest the GA evaluates.
      If the frozen snapshot is missing, we log a clear warning and fall back to
      the previous full-series live load so the script still runs.
    """
    if use_lockbox:
        try:
            from benchmark.spec import BENCHMARK_SPEC
            from benchmark.data_lockbox import get_evolution_window, DataLockbox
            df = get_evolution_window(BENCHMARK_SPEC)
            cutoff = DataLockbox(BENCHMARK_SPEC).lockbox_cutoff_ts()
            # Hard assert: no in-sample bar may reach into the sealed lockbox.
            if "timestamp" in df.columns and len(df) > 0:
                assert int(df["timestamp"].max()) < int(cutoff), \
                    "Lockbox leakage: in-sample window crosses the lockbox cutoff!"
            logger.info(
                f"Data lockbox: loaded {len(df)} in-sample bars from frozen snapshot "
                f"(lockbox cutoff_ts={cutoff}, {BENCHMARK_SPEC.lockbox_days}d sealed)."
            )
            regime_counts = Counter(df["_regime"]) if "_regime" in df.columns else {}
            if regime_counts:
                total = len(df)
                dist = ", ".join(f"{r}: {c} ({c/total*100:.0f}%)"
                                 for r, c in sorted(regime_counts.items()))
                logger.info(f"Regime distribution: {dist}")
            logger.info(f"Ready: {len(df)} bars, {len(df.columns)} features")
            return df
        except Exception as e:
            logger.warning(
                f"Data lockbox unavailable ({e}); falling back to live full-series "
                f"load. WARNING: this load is NOT lockbox-sealed — for an honest run, "
                f"freeze the snapshot with `python -m benchmark snapshot`."
            )

    if not has_binance_keys():
        raise RuntimeError("No Binance API keys in .env")

    logger.info(f"Fetching XRP/USDT H1 data ({history_days} days)...")
    data_cfg = DataConfig(timeframes=[TimeFrame.H1], history_days=history_days, cross_exchanges=[])
    engine = DataIngestionEngine(data_cfg)
    raw = engine.fetch_ohlcv("XRP/USDT", TimeFrame.H1, since_days=history_days)

    if raw is None or (hasattr(raw, 'empty') and raw.empty) or len(raw) < 200:
        n = 0 if raw is None or (hasattr(raw, 'empty') and raw.empty) else len(raw)
        raise RuntimeError(f"Insufficient data: {n} candles")

    logger.info(f"Building features on {len(raw)} candles...")
    df = build_all_features(raw)

    # Per-bar regime labels
    from mirofish.scenario_engine import classify_regime_quantitative
    regime_labels = []
    for i in range(len(df)):
        if i < 50:
            regime_labels.append("unknown")
        else:
            rc = classify_regime_quantitative(df, bar_idx=i)
            regime_labels.append(rc.regime.value)
    df["_regime"] = regime_labels

    regime_counts = Counter(regime_labels)
    total = len(regime_labels)
    dist = ", ".join(f"{r}: {c} ({c/total*100:.0f}%)" for r, c in sorted(regime_counts.items()))
    logger.info(f"Regime distribution: {dist}")
    logger.info(f"Ready: {len(df)} bars, {len(df.columns)} features")
    return df


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Autonomous Evolutionary Strategy Optimizer")
    parser.add_argument("--hours", type=float, default=14.0, help="Hours to run (default: 14)")
    parser.add_argument("--pop-size", type=int, default=40, help="Population size (default: 40)")
    parser.add_argument("--cores", type=int, default=0, help="CPU cores (0=auto)")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    args = parser.parse_args()

    max_workers = args.cores if args.cores > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"Autonomous Evolution: {args.hours}h, pop={args.pop_size}, "
                f"cores={max_workers}, data={args.days}d")

    evo_cfg = EvolutionConfig()
    df = load_data(args.days, use_lockbox=evo_cfg.use_data_lockbox)

    engine = EvolutionEngine(
        df=df,
        max_hours=args.hours,
        pop_size=args.pop_size,
        max_workers=max_workers,
        train_days=90,
        val_days=15,
        test_days=45,
    )
    engine.run()


if __name__ == "__main__":
    main()
