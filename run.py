#!/usr/bin/env python3
"""
CryptoResearchLab — Main Orchestrator
Full pipeline: Data → MiroFish → AutoResearch → Report

Usage:
    python run.py                          # Full pipeline with real Binance data
    python run.py --experiment --strategy volatility_breakout
    python run.py --mirofish-only          # Just run scenario analysis
    python run.py --report                 # Generate report from logs
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ── Project imports ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    LabConfig, DataConfig, BacktestConfig, RiskConfig,
    AutoResearchConfig, MiroFishConfig, TimeFrame, MarketRegime
)
from engine.data_ingestion import DataIngestionEngine
from engine.env_loader import has_binance_keys
from engine.features import build_all_features
from engine.backtester import Backtester, WalkForwardValidator
from engine.metrics import StrategyMetrics, compute_metrics
from engine.risk_manager import RiskManager
from mirofish.scenario_engine import (
    run_mirofish_analysis, classify_regime_quantitative,
    MiroFishReport
)
from autoresearch.experiment_runner import AutoResearchRunner, ExperimentLog
from strategies import volatility_breakout, mean_reversion, trend_following

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CryptoResearchLab")


# ═════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ═════════════════════════════════════════════════════════════

def stage_1_data(config: LabConfig) -> dict:
    """Stage 1: Data ingestion and feature engineering (real Binance data only)."""
    logger.info("=" * 60)
    logger.info("STAGE 1 — DATA INGESTION (Binance)")
    logger.info("=" * 60)

    if not has_binance_keys():
        raise RuntimeError(
            "No Binance API keys found. Configure .env with BINANCE_API_KEY "
            "and BINANCE_API_SECRET before running the pipeline."
        )

    datasets = {}

    logger.info("Fetching real XRP/USDT data from Binance...")
    ingestion = DataIngestionEngine(config.data)
    primary = ingestion.fetch_all_primary(force_refresh=True)
    for tf_str, df in primary.items():
        if len(df) > 0:
            datasets[tf_str] = build_all_features(df)
            logger.info(f"  {tf_str}: {len(datasets[tf_str])} bars, {len(datasets[tf_str].columns)} features")

    if not datasets:
        raise RuntimeError(
            "Failed to fetch any data from Binance. Check your API keys and network connection."
        )

    # Cross-asset data for robustness (BTC, ETH correlation)
    cross = ingestion.fetch_cross_validation_data(TimeFrame.H1)
    for key, df in cross.items():
        if len(df) > 0:
            datasets[f"cross_{key}"] = build_all_features(df)
            logger.info(f"  Cross-asset {key}: {len(df)} bars")

    return datasets


def stage_2_mirofish(
    config: LabConfig, datasets: dict
) -> MiroFishReport:
    """Stage 2: MiroFish scenario analysis."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2 — MIROFISH SCENARIO ANALYSIS")
    logger.info("=" * 60)

    # Use 1H data for regime analysis
    df = datasets.get("1h", list(datasets.values())[0])
    report = run_mirofish_analysis(df, symbol="XRP/USDT")

    logger.info(f"  Regime: {report.regime.regime.value} "
                f"(confidence: {report.regime.confidence:.2f})")
    logger.info(f"  Signals: {report.regime.supporting_signals}")
    logger.info(f"  Scenarios generated: {len(report.scenarios)}")
    for s in report.scenarios:
        logger.info(f"    • {s.name} (p={s.probability:.0%}, "
                     f"dir={s.expected_direction}, "
                     f"mag={s.expected_magnitude_pct:.1f}%)")
    logger.info(f"  Recommended strategies: {report.recommended_strategies}")
    logger.info(f"  Avoid: {report.avoid_strategies}")
    logger.info(f"  Risk flags: {report.risk_flags}")
    logger.info(f"  Hypotheses for autoresearch: {len(report.hypotheses_for_autoresearch)}")
    for h in report.hypotheses_for_autoresearch:
        logger.info(f"    [{h['priority']}] {h['id']}: {h['statement']}")

    # Save report
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"reports/mirofish_report_{ts}.json"
    with open(path, "w") as f:
        f.write(report.to_json())
    logger.info(f"  Report saved: {path}")

    return report


def stage_3_autoresearch(
    config: LabConfig,
    datasets: dict,
    mirofish_report: MiroFishReport,
    max_experiments: int = 12,
) -> dict:
    """Stage 3: Autonomous research experiments."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3 — AUTORESEARCH EXPERIMENTS")
    logger.info("=" * 60)

    df = datasets.get("1h", list(datasets.values())[0])

    strategy_modules = {
        "volatility_breakout": volatility_breakout,
        "mean_reversion": mean_reversion,
        "trend_following": trend_following,
    }

    runner = AutoResearchRunner(config)

    logger.info(f"  Running {max_experiments} experiments across "
                f"{len(strategy_modules)} strategies...")
    logger.info(f"  Primary metric: {config.autoresearch.primary_metric}")
    logger.info(f"  Walk-forward: {config.data.train_window_days}d train / "
                f"{config.data.val_window_days}d val / "
                f"{config.data.test_window_days}d test")

    session_summary = runner.run_research_session(
        df, strategy_modules,
        mirofish_report=mirofish_report,
        max_experiments=max_experiments,
    )

    # Research report
    report_text = runner.generate_research_report()
    logger.info("\n" + report_text)

    os.makedirs("reports", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(f"reports/autoresearch_report_{ts}.txt", "w") as f:
        f.write(report_text)
    with open(f"reports/autoresearch_summary_{ts}.json", "w") as f:
        json.dump(session_summary, f, indent=2, default=str)

    return session_summary


def stage_4_baseline_backtest(
    config: LabConfig, datasets: dict, regime: str = "unknown"
) -> dict:
    """Stage 4: Run baseline backtests for all strategies.
    Uses regime='unknown' by default for backtesting to evaluate across all conditions.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4 — BASELINE BACKTESTS (all strategies)")
    logger.info(f"  Regime filter: '{regime}' (strategies run across all conditions)")
    logger.info("=" * 60)

    df = datasets.get("1h", list(datasets.values())[0])
    results = {}

    # Compute per-bar regime labels for regime-aware backtesting
    # This ensures strategies only fire in their intended market conditions
    logger.info("  Computing per-bar regime labels...")
    regime_labels = []
    for i in range(len(df)):
        if i < 50:
            regime_labels.append("unknown")
        else:
            rc = classify_regime_quantitative(df, bar_idx=i)
            regime_labels.append(rc.regime.value)
    df["_regime"] = regime_labels

    # Log regime distribution
    from collections import Counter
    regime_counts = Counter(regime_labels)
    total_bars = len(regime_labels)
    regime_dist = ", ".join(f"{r}: {c} ({c/total_bars*100:.0f}%)" for r, c in sorted(regime_counts.items()))
    logger.info(f"  Regime distribution: {regime_dist}")

    def _get_regime(d, i):
        """Get regime for bar i from precomputed labels."""
        return d.iloc[i].get("_regime", "unknown") if i < len(d) else "unknown"

    strategies = {
        "volatility_breakout": lambda d, i, p: volatility_breakout.volatility_breakout_strategy(d, i, p, regime=_get_regime(d, i)),
        "mean_reversion": lambda d, i, p: mean_reversion.mean_reversion_strategy(d, i, p, regime=_get_regime(d, i)),
        "trend_following": lambda d, i, p: trend_following.trend_following_strategy(d, i, p, regime=_get_regime(d, i)),
    }

    backtester = Backtester(config.backtest, config.risk)

    for name, fn in strategies.items():
        logger.info(f"\n  Running {name}...")
        trades_df, eq_series, metrics = backtester.run(df, fn, name)

        results[name] = {
            "trades": trades_df,
            "equity": eq_series,
            "metrics": metrics,
        }

        m = metrics
        logger.info(f"    Trades: {m.total_trades}")
        logger.info(f"    Sharpe: {m.sharpe_ratio:.3f}")
        logger.info(f"    Sortino: {m.sortino_ratio:.3f}")
        logger.info(f"    Max DD: {m.max_drawdown_pct:.2f}%")
        logger.info(f"    Profit Factor: {m.profit_factor:.3f}")
        logger.info(f"    Win Rate: {m.win_rate*100:.1f}%")
        logger.info(f"    Net PnL: ${m.net_pnl:.2f}")
        logger.info(f"    Expectancy: {m.expectancy:.4f}")
        logger.info(f"    Avg Holding: {m.avg_holding_hours:.1f}h")
        logger.info(f"    Commissions: ${m.total_commissions:.2f}")
        logger.info(f"    Slippage: ${m.total_slippage:.2f}")

    # Walk-forward for each
    logger.info("\n  Running walk-forward validation...")
    validator = WalkForwardValidator(
        config.backtest, config.risk,
        train_days=config.data.train_window_days,
        val_days=config.data.val_window_days,
        test_days=config.data.test_window_days,
    )

    for name, fn in strategies.items():
        wf_result = validator.validate(df, fn, name)
        agg = wf_result["aggregate"]
        results[name]["walk_forward"] = wf_result

        status = "✓ ROBUST" if wf_result["is_robust"] else "✗ NOT ROBUST"
        logger.info(f"    {name} WF: {status} | "
                     f"Sharpe={agg.sharpe_ratio:.3f} | "
                     f"DD={agg.max_drawdown_pct:.1f}% | "
                     f"OOS Degradation={wf_result['oos_degradation']:.1f}% | "
                     f"Trades={agg.total_trades}")

    return results


def stage_5_risk_check(
    config: LabConfig, backtest_results: dict
) -> dict:
    """Stage 5: Risk management gate."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 5 — RISK MANAGEMENT GATE")
    logger.info("=" * 60)

    risk_mgr = RiskManager(config.risk)
    approved = {}

    for name, result in backtest_results.items():
        m = result["metrics"]
        wf = result.get("walk_forward", {})
        is_robust = wf.get("is_robust", False)

        checks = {
            "sharpe_ok": m.sharpe_ratio >= config.backtest.min_sharpe,
            "drawdown_ok": m.max_drawdown_pct <= config.backtest.max_drawdown_pct,
            "trades_ok": m.total_trades >= config.backtest.min_trades,
            "profit_factor_ok": m.profit_factor > 1.0,
            "walk_forward_robust": is_robust,
        }

        all_pass = all(checks.values())
        status = "✓ APPROVED" if all_pass else "✗ BLOCKED"
        logger.info(f"  {name}: {status}")
        for check, passed in checks.items():
            icon = "✓" if passed else "✗"
            logger.info(f"    {icon} {check}")

        approved[name] = {
            "approved": all_pass,
            "checks": checks,
            "metrics_summary": m.summary_dict(),
        }

    return approved


# ═════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═════════════════════════════════════════════════════════════

def generate_final_report(
    mirofish_report: MiroFishReport,
    backtest_results: dict,
    risk_approval: dict,
    research_summary: dict,
) -> str:
    """Generate comprehensive human-readable final report."""
    lines = []
    lines.append("╔" + "═" * 68 + "╗")
    lines.append("║" + " CRYPTORESEARCHLAB — FULL PIPELINE REPORT ".center(68) + "║")
    lines.append("║" + f" {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ".center(68) + "║")
    lines.append("╚" + "═" * 68 + "╝")

    # MiroFish
    lines.append("\n┌─── MIROFISH: MARKET CONTEXT ─────────────────────────┐")
    lines.append(f"  Regime: {mirofish_report.regime.regime.value.upper()} "
                 f"(confidence: {mirofish_report.regime.confidence:.0%})")
    for sig in mirofish_report.regime.supporting_signals:
        lines.append(f"    • {sig}")
    lines.append(f"  Scenarios:")
    for s in mirofish_report.scenarios:
        lines.append(f"    {s.probability:.0%} — {s.name} "
                     f"({s.expected_direction}, ±{s.expected_magnitude_pct:.1f}%)")
    if mirofish_report.risk_flags:
        lines.append(f"  ⚠ Risk Flags:")
        for f in mirofish_report.risk_flags:
            lines.append(f"    • {f}")
    lines.append("└──────────────────────────────────────────────────────┘")

    # Backtests
    lines.append("\n┌─── STRATEGY BACKTESTS ────────────────────────────────┐")
    for name, result in backtest_results.items():
        m = result["metrics"]
        wf = result.get("walk_forward", {})
        agg = wf.get("aggregate", StrategyMetrics())
        robust = "ROBUST" if wf.get("is_robust", False) else "NOT ROBUST"
        approved = risk_approval.get(name, {}).get("approved", False)
        gate = "✓ APPROVED" if approved else "✗ BLOCKED"

        lines.append(f"\n  [{name.upper()}] — {gate}")
        lines.append(f"  ┄┄┄ Full-sample ┄┄┄")
        lines.append(f"    Sharpe: {m.sharpe_ratio:.3f}  |  Sortino: {m.sortino_ratio:.3f}")
        lines.append(f"    Max DD: {m.max_drawdown_pct:.2f}%  |  PF: {m.profit_factor:.3f}")
        lines.append(f"    Win Rate: {m.win_rate*100:.1f}%  |  Trades: {m.total_trades}")
        lines.append(f"    Net PnL: ${m.net_pnl:.2f}  |  Expectancy: {m.expectancy:.4f}")
        lines.append(f"    Costs: ${m.total_commissions + m.total_slippage:.2f} "
                     f"(comm ${m.total_commissions:.2f} + slip ${m.total_slippage:.2f})")
        lines.append(f"  ┄┄┄ Walk-Forward ({robust}) ┄┄┄")
        lines.append(f"    OOS Sharpe: {agg.sharpe_ratio:.3f}  |  "
                     f"Degradation: {wf.get('oos_degradation', 0):.1f}%")
        lines.append(f"    OOS Trades: {agg.total_trades}  |  "
                     f"OOS PF: {agg.profit_factor:.3f}")
    lines.append("\n└──────────────────────────────────────────────────────┘")

    # Research
    lines.append("\n┌─── AUTORESEARCH SESSION ──────────────────────────────┐")
    lines.append(f"  Experiments: {research_summary.get('total_experiments', 0)}")
    lines.append(f"  Accepted: {research_summary.get('accepted', 0)}")
    lines.append(f"  Rejected: {research_summary.get('rejected', 0)}")
    lines.append(f"  Acceptance Rate: {research_summary.get('acceptance_rate', 0)}%")

    best = research_summary.get("best_metrics", {})
    if best:
        lines.append(f"  Best params found:")
        for strat, metrics in best.items():
            lines.append(f"    [{strat}] Sharpe={metrics.get('sharpe', 0):.3f} "
                        f"PF={metrics.get('profit_factor', 0):.3f}")
    lines.append("└──────────────────────────────────────────────────────┘")

    # Recommendations
    lines.append("\n┌─── RECOMMENDATIONS ──────────────────────────────────┐")
    for name, approval in risk_approval.items():
        if approval["approved"]:
            lines.append(f"  ✓ {name}: READY for paper trading / controlled execution")
        else:
            failed = [k for k, v in approval["checks"].items() if not v]
            lines.append(f"  ✗ {name}: NEEDS WORK — failed: {', '.join(failed)}")
    lines.append("└──────────────────────────────────────────────────────┘")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CryptoResearchLab Pipeline")
    parser.add_argument("--experiments", type=int, default=12, help="Number of autoresearch experiments")
    parser.add_argument("--skip-research", action="store_true", help="Skip autoresearch stage")
    args = parser.parse_args()

    config = LabConfig()
    pipeline_start = time.time()

    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║     CryptoResearchLab — XRP/USDT Pipeline        ║")
    logger.info("║     Data source: Binance (real market data)       ║")
    logger.info("╚═══════════════════════════════════════════════════╝")

    # Stage 1: Data (always real Binance data)
    datasets = stage_1_data(config)

    # Stage 2: MiroFish
    mirofish_report = stage_2_mirofish(config, datasets)
    regime = mirofish_report.regime.regime.value

    # Stage 3: AutoResearch
    research_summary = {"total_experiments": 0, "accepted": 0, "rejected": 0}
    if not args.skip_research:
        research_summary = stage_3_autoresearch(
            config, datasets, mirofish_report, max_experiments=args.experiments
        )

    # Stage 4: Baseline backtests
    backtest_results = stage_4_baseline_backtest(config, datasets, regime=regime)

    # Stage 5: Risk gate
    risk_approval = stage_5_risk_check(config, backtest_results)

    # Final report
    report = generate_final_report(
        mirofish_report, backtest_results, risk_approval, research_summary
    )
    logger.info("\n" + report)

    # Save
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/pipeline_report_{ts}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    elapsed = time.time() - pipeline_start
    logger.info(f"\n✓ Pipeline completed in {elapsed:.1f}s")
    logger.info(f"  Report saved: {report_path}")

    # Save equity curves for visualization
    eq_data = {}
    for name, result in backtest_results.items():
        eq = result["equity"]
        if len(eq) > 0:
            eq_data[name] = {
                "timestamps": [str(t) for t in eq.index],
                "values": eq.values.tolist(),
            }
    with open(f"reports/equity_curves_{ts}.json", "w") as f:
        json.dump(eq_data, f, default=str)

    # Save trade logs
    for name, result in backtest_results.items():
        trades = result["trades"]
        if len(trades) > 0:
            trades.to_csv(f"reports/trades_{name}_{ts}.csv", index=False)

    return report, backtest_results, risk_approval


if __name__ == "__main__":
    main()
