"""
CryptoResearchLab — AutoResearch Experiment Runner
Autonomous research loop inspired by Karpathy's autoresearch.

Core cycle:
1. Generate or receive hypothesis (from MiroFish or self-generated)
2. Modify strategy parameters / rules
3. Run walk-forward backtest
4. Evaluate if improvement is real
5. Accept or reject changes
6. Log everything
7. Repeat

This is the agent's "research lab" — it NEVER executes real trades.
"""
import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from config import LabConfig, BacktestConfig, RiskConfig
from engine.backtester import Backtester, WalkForwardValidator, Signal
from engine.metrics import StrategyMetrics
from engine.features import build_all_features
from mirofish.scenario_engine import MiroFishReport

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT RECORD
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExperimentRecord:
    """Record of a single experiment."""
    experiment_id: int
    timestamp: str
    strategy_name: str
    hypothesis: str
    param_changes: dict
    baseline_metrics: dict
    experiment_metrics: dict
    accepted: bool
    reason: str
    improvement_pct: float = 0.0
    duration_seconds: float = 0.0
    regime_context: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.experiment_id,
            "timestamp": self.timestamp,
            "strategy": self.strategy_name,
            "hypothesis": self.hypothesis,
            "param_changes": self.param_changes,
            "baseline": self.baseline_metrics,
            "result": self.experiment_metrics,
            "accepted": self.accepted,
            "reason": self.reason,
            "improvement_%": round(self.improvement_pct, 2),
            "duration_s": round(self.duration_seconds, 1),
            "regime": self.regime_context,
        }


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT LOG
# ═══════════════════════════════════════════════════════════════

class ExperimentLog:
    """Persistent experiment log with full history."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records: list[ExperimentRecord] = []
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"experiment_log_{self.session_id}.jsonl")

    def add(self, record: ExperimentRecord):
        """Add a record and persist."""
        self.records.append(record)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def get_best(self, strategy_name: str, metric: str = "sharpe_ratio") -> Optional[ExperimentRecord]:
        """Get best accepted experiment for a strategy."""
        accepted = [
            r for r in self.records
            if r.accepted and r.strategy_name == strategy_name
        ]
        if not accepted:
            return None
        return max(accepted, key=lambda r: r.experiment_metrics.get(metric, 0))

    def summary(self) -> dict:
        """Return experiment session summary."""
        total = len(self.records)
        accepted = sum(1 for r in self.records if r.accepted)
        return {
            "session_id": self.session_id,
            "total_experiments": total,
            "accepted": accepted,
            "rejected": total - accepted,
            "acceptance_rate": round(accepted / total * 100, 1) if total > 0 else 0,
            "strategies_tested": list(set(r.strategy_name for r in self.records)),
        }


# ═══════════════════════════════════════════════════════════════
# PARAMETER MUTATOR
# ═══════════════════════════════════════════════════════════════

def mutate_params(
    current_params: dict,
    hypothesis: Optional[dict] = None,
) -> tuple[dict, str, dict]:
    """
    Mutate strategy parameters based on a hypothesis or random exploration.

    Returns: (new_params, hypothesis_statement, changes_dict)
    """
    new_params = copy.deepcopy(current_params)
    changes = {}
    statement = ""

    if hypothesis and "param_changes" in hypothesis:
        # Targeted mutation from MiroFish hypothesis
        statement = hypothesis.get("statement", "hypothesis-driven change")
        for param, values in hypothesis["param_changes"].items():
            if param in new_params and isinstance(values, list) and len(values) > 0:
                new_val = np.random.choice(values)
                changes[param] = {"from": new_params[param], "to": new_val}
                new_params[param] = new_val
    else:
        # Random exploration — pick a random numeric parameter to tweak
        numeric_params = {
            k: v for k, v in current_params.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if numeric_params:
            param_name = np.random.choice(list(numeric_params.keys()))
            current_val = numeric_params[param_name]

            # Random perturbation: ±10-30%
            perturbation = np.random.uniform(0.7, 1.3)
            if isinstance(current_val, int):
                new_val = max(1, int(current_val * perturbation))
            else:
                new_val = round(current_val * perturbation, 4)

            changes[param_name] = {"from": current_val, "to": new_val}
            new_params[param_name] = new_val
            statement = f"Random exploration: {param_name} {current_val} → {new_val}"

    return new_params, statement, changes


# ═══════════════════════════════════════════════════════════════
# AUTORESEARCH RUNNER
# ═══════════════════════════════════════════════════════════════

class AutoResearchRunner:
    """
    Autonomous research loop.
    The "agent" that iteratively improves strategies.
    """

    def __init__(self, config: LabConfig):
        self.config = config
        self.log = ExperimentLog(config.autoresearch.log_dir)
        self.experiment_counter = 0
        self.best_params: dict[str, dict] = {}  # strategy -> best params
        self.best_metrics: dict[str, StrategyMetrics] = {}

    def _make_strategy_fn(
        self,
        strategy_module,
        params: dict,
        regime: str = "unknown",
    ) -> Callable:
        """
        Create a strategy function with specific params.
        This wraps the strategy module's function with frozen params.
        """
        # Temporarily set params on the module
        original_params = copy.deepcopy(strategy_module.PARAMS)
        strategy_module.PARAMS = params

        # Get the strategy function name
        if hasattr(strategy_module, "volatility_breakout_strategy"):
            fn = strategy_module.volatility_breakout_strategy
        elif hasattr(strategy_module, "mean_reversion_strategy"):
            fn = strategy_module.mean_reversion_strategy
        elif hasattr(strategy_module, "trend_following_strategy"):
            fn = strategy_module.trend_following_strategy
        else:
            raise ValueError("Unknown strategy module")

        def wrapped(df, bar_idx, position):
            return fn(df, bar_idx, position, regime=regime)

        # Store original params for cleanup
        wrapped._original_params = original_params
        wrapped._module = strategy_module

        return wrapped

    def _cleanup_strategy_fn(self, wrapped_fn):
        """Restore original params after experiment."""
        if hasattr(wrapped_fn, "_original_params") and hasattr(wrapped_fn, "_module"):
            wrapped_fn._module.PARAMS = wrapped_fn._original_params

    def run_single_experiment(
        self,
        df: pd.DataFrame,
        strategy_module,
        strategy_name: str,
        hypothesis: Optional[dict] = None,
        regime: str = "unknown",
    ) -> ExperimentRecord:
        """
        Run one experiment cycle:
        1. Get current best params (baseline)
        2. Mutate params (from hypothesis or random)
        3. Run walk-forward backtest
        4. Compare to baseline
        5. Accept or reject
        """
        self.experiment_counter += 1
        start_time = time.time()

        # Get baseline params
        baseline_params = self.best_params.get(
            strategy_name, copy.deepcopy(strategy_module.PARAMS)
        )

        # Mutate
        new_params, hypothesis_statement, changes = mutate_params(
            baseline_params, hypothesis
        )

        if not changes:
            return ExperimentRecord(
                experiment_id=self.experiment_counter,
                timestamp=str(datetime.now(timezone.utc)),
                strategy_name=strategy_name,
                hypothesis="no_changes_generated",
                param_changes={},
                baseline_metrics={},
                experiment_metrics={},
                accepted=False,
                reason="no_parameter_changes",
            )

        # Run baseline
        validator = WalkForwardValidator(
            self.config.backtest, self.config.risk,
            train_days=self.config.data.train_window_days,
            val_days=self.config.data.val_window_days,
            test_days=self.config.data.test_window_days,
        )

        baseline_fn = self._make_strategy_fn(strategy_module, baseline_params, regime)
        baseline_result = validator.validate(df, baseline_fn, f"{strategy_name}_baseline")
        self._cleanup_strategy_fn(baseline_fn)
        baseline_metrics = baseline_result["aggregate"]

        # Run experiment
        experiment_fn = self._make_strategy_fn(strategy_module, new_params, regime)
        experiment_result = validator.validate(df, experiment_fn, f"{strategy_name}_experiment")
        self._cleanup_strategy_fn(experiment_fn)
        experiment_metrics = experiment_result["aggregate"]

        # Compare
        primary = self.config.autoresearch.primary_metric
        baseline_val = getattr(baseline_metrics, primary, 0)
        experiment_val = getattr(experiment_metrics, primary, 0)

        improvement = 0.0
        if baseline_val != 0:
            improvement = (experiment_val - baseline_val) / abs(baseline_val) * 100
        elif experiment_val > 0:
            improvement = 100.0

        # Acceptance criteria
        accepted = False
        reason = ""

        if improvement >= self.config.autoresearch.improvement_threshold * 100:
            # Check guard metrics don't degrade badly
            guard_ok = True
            for guard in self.config.autoresearch.guard_metrics:
                baseline_guard = getattr(baseline_metrics, guard, 0)
                exp_guard = getattr(experiment_metrics, guard, 0)
                if guard == "max_drawdown":
                    if exp_guard > baseline_guard * 1.5:
                        guard_ok = False
                        reason = f"guard_metric_{guard}_degraded"
                        break
                elif exp_guard < baseline_guard * 0.7:
                    guard_ok = False
                    reason = f"guard_metric_{guard}_degraded"
                    break

            if guard_ok and experiment_result["is_robust"]:
                accepted = True
                reason = f"improved_{primary}_by_{improvement:.1f}%"
                self.best_params[strategy_name] = new_params
                self.best_metrics[strategy_name] = experiment_metrics
            elif not guard_ok:
                pass  # reason already set
            else:
                reason = "not_robust_enough"
        else:
            reason = f"insufficient_improvement_{improvement:.1f}%"

        duration = time.time() - start_time

        record = ExperimentRecord(
            experiment_id=self.experiment_counter,
            timestamp=str(datetime.now(timezone.utc)),
            strategy_name=strategy_name,
            hypothesis=hypothesis_statement,
            param_changes=changes,
            baseline_metrics=baseline_metrics.summary_dict(),
            experiment_metrics=experiment_metrics.summary_dict(),
            accepted=accepted,
            reason=reason,
            improvement_pct=improvement,
            duration_seconds=duration,
            regime_context=regime,
        )

        self.log.add(record)

        status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
        logger.info(
            f"Experiment #{self.experiment_counter} [{strategy_name}] "
            f"{status}: {reason} "
            f"(Δ{primary}={improvement:+.1f}%, {duration:.0f}s)"
        )

        return record

    def run_research_session(
        self,
        df: pd.DataFrame,
        strategy_modules: dict,
        mirofish_report: Optional[MiroFishReport] = None,
        max_experiments: Optional[int] = None,
    ) -> dict:
        """
        Run a full autonomous research session.

        Args:
            df: OHLCV + features DataFrame
            strategy_modules: dict of name -> module with PARAMS and strategy function
            mirofish_report: Optional context from MiroFish
            max_experiments: Override max experiments

        Returns:
            Session summary with all results
        """
        max_exp = max_experiments or self.config.autoresearch.max_experiments
        regime = "unknown"
        hypotheses_queue = []

        # Get MiroFish context
        if mirofish_report:
            regime = mirofish_report.regime.regime.value
            hypotheses_queue = list(mirofish_report.hypotheses_for_autoresearch)
            logger.info(
                f"MiroFish regime: {regime} "
                f"({len(hypotheses_queue)} hypotheses queued)"
            )

        results = []
        for exp_num in range(max_exp):
            # Pick strategy to test
            strategy_names = list(strategy_modules.keys())
            strategy_name = strategy_names[exp_num % len(strategy_names)]
            module = strategy_modules[strategy_name]

            # Pick hypothesis (from MiroFish queue or random)
            hypothesis = None
            if hypotheses_queue:
                # Filter hypotheses relevant to this strategy
                relevant = [
                    h for h in hypotheses_queue
                    if h.get("strategy") in [strategy_name, "all"]
                ]
                if relevant:
                    hypothesis = relevant.pop(0)
                    hypotheses_queue = [h for h in hypotheses_queue if h != hypothesis]

            # Run experiment
            record = self.run_single_experiment(
                df, module, strategy_name, hypothesis, regime
            )
            results.append(record)

        # Session summary
        summary = self.log.summary()
        summary["best_params"] = {
            name: params for name, params in self.best_params.items()
        }
        summary["best_metrics"] = {
            name: m.summary_dict() for name, m in self.best_metrics.items()
        }

        return summary

    def generate_research_report(self) -> str:
        """Generate a human-readable research report."""
        lines = []
        lines.append("=" * 70)
        lines.append("AUTORESEARCH SESSION REPORT")
        lines.append("=" * 70)

        summary = self.log.summary()
        lines.append(f"\nSession: {summary['session_id']}")
        lines.append(f"Total experiments: {summary['total_experiments']}")
        lines.append(f"Accepted: {summary['accepted']}")
        lines.append(f"Rejected: {summary['rejected']}")
        lines.append(f"Acceptance rate: {summary['acceptance_rate']}%")

        lines.append("\n" + "-" * 40)
        lines.append("BEST PARAMETERS FOUND")
        lines.append("-" * 40)

        for name, params in self.best_params.items():
            lines.append(f"\n[{name}]")
            for k, v in sorted(params.items()):
                lines.append(f"  {k}: {v}")

        lines.append("\n" + "-" * 40)
        lines.append("BEST METRICS")
        lines.append("-" * 40)

        for name, metrics in self.best_metrics.items():
            lines.append(f"\n[{name}]")
            for k, v in metrics.summary_dict().items():
                lines.append(f"  {k}: {v}")

        lines.append("\n" + "-" * 40)
        lines.append("EXPERIMENT LOG (last 10)")
        lines.append("-" * 40)

        for record in self.log.records[-10:]:
            status = "✓" if record.accepted else "✗"
            lines.append(
                f"  {status} #{record.experiment_id} [{record.strategy_name}] "
                f"{record.reason} (Δ={record.improvement_pct:+.1f}%)"
            )

        return "\n".join(lines)
