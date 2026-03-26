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

from config import LabConfig, BacktestConfig, RiskConfig, DataConfig, TimeFrame
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
    # 10 strategies removed total. 8 active strategies remain.
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
}


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


def evaluate_genome(args_tuple):
    """Evaluate a genome via walk-forward validation. Runs in spawn process."""
    genome_dict, df_path, bt_cfg, risk_cfg, train_days, val_days, test_days = args_tuple

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

        # FULL-SAMPLE backtest
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

        # FITNESS: composite score that rewards robustness
        # = WF_Sharpe * 0.5 + IS_Sharpe * 0.3 + log(trades) * 0.2 - penalty_for_degradation
        # Minimum 30 trades required to avoid overfitting to micro-patterns
        MIN_TRADES_FOR_POSITIVE_FITNESS = 30
        wf_s = wf_agg.sharpe_ratio if wf_agg.total_trades >= 5 else -5.0
        is_s = metrics.sharpe_ratio if metrics.total_trades >= 5 else -5.0
        trade_bonus = math.log(max(metrics.total_trades, 1)) * 0.1
        degrad_penalty = max(0, wf_degrad - 30) * 0.02
        dd_penalty = max(0, metrics.max_drawdown_pct - 5) * 0.1
        # Low trade count penalty: strategies with <30 trades get heavily penalized
        low_trade_penalty = max(0, (MIN_TRADES_FOR_POSITIVE_FITNESS - metrics.total_trades) * 0.05) if metrics.total_trades < MIN_TRADES_FOR_POSITIVE_FITNESS else 0.0

        fitness = wf_s * 0.5 + is_s * 0.3 + trade_bonus - degrad_penalty - dd_penalty - low_trade_penalty

        # Monte Carlo simulation for promising genomes (fitness > 0.5)
        mc_result = None
        if fitness > 0.5 and metrics.total_trades >= 10:
            try:
                from engine.monte_carlo import monte_carlo_trades
                trade_pnls = np.array([t.pnl_net for t in trades_df]) if hasattr(trades_df, '__iter__') and len(trades_df) > 0 else None
                if trade_pnls is None and isinstance(trades_df, pd.DataFrame) and len(trades_df) > 0:
                    trade_pnls = trades_df["pnl_net"].values if "pnl_net" in trades_df.columns else None
                if trade_pnls is not None and len(trade_pnls) >= 10:
                    mc = monte_carlo_trades(
                        trade_pnls, initial_capital=200.0,
                        n_simulations=500, method="bootstrap", seed=42,
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

        self.learning = LearningEngine()
        self.hall_of_fame = []  # Top 20 all-time best genomes (diverse)
        self.hall_of_fame_per_strategy = defaultdict(list)  # per-strategy top 5
        self.generation = 0
        self.total_evaluated = 0
        self.total_robust = 0
        self.strategies_available = list(STRATEGY_REGISTRY.keys())
        # Minimum exploration: at least 15% of pop goes to underexplored strategies
        self.min_explore_pct = 0.15

        # Save df for workers
        os.makedirs("/tmp/crypto_evolve", exist_ok=True)
        self.df_path = "/tmp/crypto_evolve/df.pkl"
        df.to_pickle(self.df_path)

        # Logging
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/evolution_{ts}.jsonl"
        self.report_path = f"reports/evolution_report_{ts}.json"
        logger.info(f"Evolution log: {self.log_path}")

    def _log_event(self, event: dict):
        """Append event to JSONL log."""
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event["generation"] = self.generation
        event["total_evaluated"] = self.total_evaluated
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def generate_initial_population(self) -> list[Genome]:
        """Create diverse initial population across all strategies."""
        pop = []
        per_strat = max(2, self.pop_size // len(self.strategies_available))

        for strat in self.strategies_available:
            for _ in range(per_strat):
                pop.append(random_genome(strat, generation=0))

        # Fill remaining with random
        while len(pop) < self.pop_size:
            strat = random.choice(self.strategies_available)
            pop.append(random_genome(strat, generation=0))

        random.shuffle(pop)
        return pop[:self.pop_size]

    def evaluate_population(self, population: list[Genome]) -> list[Genome]:
        """Evaluate all genomes in parallel."""
        work_items = [
            (g.to_dict(), self.df_path, self.bt_config, self.risk_config,
             self.train_days, self.val_days, self.test_days)
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

                        self.learning.record(genome)
                        self.total_evaluated += 1
                        if genome.wf_robust:
                            self.total_robust += 1

                        # Update hall of fame with diversity enforcement
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
        - Max 4 genomes per strategy in global HoF (ensures strategy diversity)
        - New genome must have param distance > 0.15 from existing HoF entries
          of the same strategy (prevents clones flooding HoF)
        - Also maintain per-strategy top 5 for best-per-strategy tracking
        """
        # Per-strategy tracking (no dedup needed)
        per_strat = self.hall_of_fame_per_strategy[genome.strategy]
        per_strat.append(genome)
        per_strat.sort(key=lambda g: g.fitness, reverse=True)
        self.hall_of_fame_per_strategy[genome.strategy] = per_strat[:5]

        # Global HoF: check if it's a clone of existing entries
        same_strat_in_hof = [g for g in self.hall_of_fame if g.strategy == genome.strategy]
        for existing in same_strat_in_hof:
            dist = self._genome_param_distance(genome, existing)
            if dist < 0.15:
                # Too similar — only replace if strictly better fitness
                if genome.fitness > existing.fitness:
                    self.hall_of_fame.remove(existing)
                    self.hall_of_fame.append(genome)
                    self.hall_of_fame.sort(key=lambda g: g.fitness, reverse=True)
                    self.hall_of_fame = self.hall_of_fame[:20]
                return  # Skip adding a clone

        # Not a clone — check strategy cap (max 4 per strategy in global HoF)
        MAX_PER_STRATEGY = 4
        if len(same_strat_in_hof) >= MAX_PER_STRATEGY:
            worst_of_strat = min(same_strat_in_hof, key=lambda g: g.fitness)
            if genome.fitness > worst_of_strat.fitness:
                self.hall_of_fame.remove(worst_of_strat)
                self.hall_of_fame.append(genome)
            # else: don't add, strategy is already capped
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
                     f"IS_Sharpe={best.sharpe:.3f} WF_Sharpe={best.wf_sharpe:.3f} "
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
            "total_robust": self.total_robust,
            "hall_of_fame": [g.to_dict() for g in self.hall_of_fame],
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

        # Save best genome per strategy as ready-to-use configs
        for strat, bests in self.hall_of_fame_per_strategy.items():
            if bests and bests[0].fitness > 0:
                best = bests[0]
                best_path = f"reports/best_genome_{strat}.json"
                with open(best_path, "w") as f:
                    json.dump({
                        "strategy": best.strategy,
                        "params": best.params,
                        "fitness": best.fitness,
                        "sharpe": best.sharpe,
                        "wf_sharpe": best.wf_sharpe,
                        "trades": best.trades,
                        "net_pnl": best.net_pnl,
                        "wf_robust": best.wf_robust,
                    }, f, indent=2)
                logger.info(f"Best genome saved: {best_path}")

        # Also save global best
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
                    "trades": best.trades,
                    "net_pnl": best.net_pnl,
                    "wf_robust": best.wf_robust,
                }, f, indent=2)
            logger.info(f"Overall best genome saved: {best_path}")


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(history_days: int = 365) -> pd.DataFrame:
    """Load real data with features and regime labels."""
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

    df = load_data(args.days)

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
