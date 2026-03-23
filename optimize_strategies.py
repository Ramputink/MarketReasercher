#!/usr/bin/env python3
"""
CryptoResearchLab — Strategy Parameter Optimizer
Parallel grid search for optimal strategy parameters using real Binance data.

Uses multiprocessing to leverage all CPU cores.
Integrates LSTM model predictions as an additional strategy signal.

Usage:
    python optimize_strategies.py                 # Full optimization (all 3 strategies)
    python optimize_strategies.py --strategy vb   # Vol Breakout only
    python optimize_strategies.py --strategy mr   # Mean Reversion only
    python optimize_strategies.py --strategy tf   # Trend Following only
    python optimize_strategies.py --cores 8       # Use 8 cores
"""
import argparse
import copy
import itertools
import json
import logging
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# CRITICAL: Force 'spawn' context on macOS to avoid fork+TF GPU conflicts
# This must happen before any TF imports in child processes
MP_CTX = mp.get_context("spawn")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LabConfig, BacktestConfig, RiskConfig, DataConfig, TimeFrame
from engine.data_ingestion import DataIngestionEngine
from engine.env_loader import has_binance_keys
from engine.features import build_all_features
from engine.backtester import Backtester, Signal
from engine.metrics import StrategyMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("optimizer")


# ---------------------------------------------------------------
# DATA LOADING (shared across all workers)
# ---------------------------------------------------------------

def load_real_data(history_days: int = 365) -> pd.DataFrame:
    """Fetch real Binance data and compute features."""
    if not has_binance_keys():
        raise RuntimeError("No Binance API keys in .env")

    logger.info(f"Fetching XRP/USDT H1 data ({history_days} days)...")
    data_cfg = DataConfig(
        timeframes=[TimeFrame.H1],
        history_days=history_days,
        cross_exchanges=[],
    )
    engine = DataIngestionEngine(data_cfg)
    raw = engine.fetch_ohlcv("XRP/USDT", TimeFrame.H1, since_days=history_days)

    if raw is None or len(raw) < 100:
        raise RuntimeError(f"Insufficient data: {len(raw) if raw else 0} candles")

    logger.info(f"Building features on {len(raw)} candles...")
    df = build_all_features(raw)

    # Precompute per-bar regime labels for regime-aware optimization
    from mirofish.scenario_engine import classify_regime_quantitative
    regime_labels = []
    for i in range(len(df)):
        if i < 50:
            regime_labels.append("unknown")
        else:
            rc = classify_regime_quantitative(df, bar_idx=i)
            regime_labels.append(rc.regime.value)
    df["_regime"] = regime_labels

    from collections import Counter
    regime_counts = Counter(regime_labels)
    total_bars = len(regime_labels)
    regime_dist = ", ".join(f"{r}: {c} ({c/total_bars*100:.0f}%)" for r, c in sorted(regime_counts.items()))
    logger.info(f"Regime distribution: {regime_dist}")
    logger.info(f"Ready: {len(df)} bars, {len(df.columns)} features")
    return df


# ---------------------------------------------------------------
# LSTM PREDICTION INTEGRATION
# ---------------------------------------------------------------

def load_lstm_predictions(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Load LSTM model and generate predictions for each bar."""
    try:
        from engine.tf_model import load_model_for_inference, predict, FEATURE_COLUMNS
        model_dir = "models"
        model_name = "xrp_lstm_v1"

        if not os.path.exists(os.path.join(model_dir, f"{model_name}.keras")):
            logger.warning("No LSTM model found — running without LSTM signals")
            return None

        model, scaler, meta = load_model_for_inference(model_dir, model_name)
        seq_length = meta.get("sequence_length", 60)
        horizons = meta.get("horizons", [1, 6, 12, 24])
        features = meta.get("features", FEATURE_COLUMNS)

        logger.info(f"LSTM model loaded: {model_name} (seq={seq_length}, horizons={horizons})")

        # Generate predictions for each bar (from seq_length onward)
        predictions = []
        for i in range(seq_length, len(df)):
            window = df.iloc[i - seq_length:i]
            pred = predict(model, scaler, window, features, seq_length, horizons)
            if pred:
                row = {"bar_idx": i}
                for h in horizons:
                    key = f"{h}h"
                    if key in pred:
                        row[f"lstm_dir_{h}h"] = pred[key]["direction"]
                        row[f"lstm_conf_{h}h"] = pred[key]["confidence"]
                        row[f"lstm_mag_{h}h"] = pred[key].get("magnitude_pct", 0)
                        row[f"lstm_prob_up_{h}h"] = pred[key]["probabilities"]["up"]
                        row[f"lstm_prob_down_{h}h"] = pred[key]["probabilities"]["down"]
                predictions.append(row)

        if predictions:
            pred_df = pd.DataFrame(predictions).set_index("bar_idx")
            logger.info(f"LSTM predictions generated for {len(pred_df)} bars")
            return pred_df

    except Exception as e:
        logger.warning(f"LSTM integration failed: {e}")

    return None


# ---------------------------------------------------------------
# SINGLE BACKTEST WORKER (runs in separate process)
# ---------------------------------------------------------------

def _worker_init():
    """Initialize worker process — set path, avoid TF import."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Prevent TF from being imported in workers (backtester doesn't need it)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_single_backtest(args_tuple):
    """
    Worker function for parallel backtest.
    Runs in a SEPARATE process (spawn context) — imports are fresh.
    """
    strategy_name, params, df_pickle_path, bt_config_dict, risk_config_dict, combo_id, lstm_pred_path = args_tuple

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import pandas as pd
    import numpy as np
    from config import BacktestConfig, RiskConfig
    from engine.backtester import Backtester

    # Reconstruct objects in worker process
    df = pd.read_pickle(df_pickle_path)
    bt_config = BacktestConfig(**bt_config_dict)
    risk_config = RiskConfig(**risk_config_dict)

    lstm_preds = None
    if lstm_pred_path and os.path.exists(lstm_pred_path):
        lstm_preds = pd.read_pickle(lstm_pred_path)

    # Helper to get precomputed regime for each bar
    def _bar_regime(d, i):
        return d.iloc[i].get("_regime", "unknown") if i < len(d) else "unknown"

    # Inject params into strategy module (each spawn process has fresh module state)
    if strategy_name == "volatility_breakout":
        from strategies.volatility_breakout import volatility_breakout_strategy, PARAMS
        PARAMS.update(params)

        def strategy_fn(d, i, p):
            return volatility_breakout_strategy(d, i, p, regime=_bar_regime(d, i))

    elif strategy_name == "mean_reversion":
        from strategies.mean_reversion import mean_reversion_strategy, PARAMS
        PARAMS.update(params)

        def strategy_fn(d, i, p):
            return mean_reversion_strategy(d, i, p, regime=_bar_regime(d, i))

    elif strategy_name == "trend_following":
        from strategies.trend_following import trend_following_strategy, PARAMS
        PARAMS.update(params)

        def strategy_fn(d, i, p):
            return trend_following_strategy(d, i, p, regime=_bar_regime(d, i))

    else:
        return None

    # If LSTM predictions available, wrap strategy with LSTM filter
    if lstm_preds is not None and params.get("use_lstm_filter", False):
        base_fn = strategy_fn
        lstm_horizon = params.get("lstm_horizon", "6h")
        lstm_min_conf = params.get("lstm_min_confidence", 0.5)

        def strategy_fn_with_lstm(d, i, p):
            signal = base_fn(d, i, p)
            if signal is None:
                return None
            if i in lstm_preds.index:
                lstm_dir = lstm_preds.loc[i].get(f"lstm_dir_{lstm_horizon}", "flat")
                lstm_conf = lstm_preds.loc[i].get(f"lstm_conf_{lstm_horizon}", 0)
                if lstm_conf >= lstm_min_conf:
                    if signal.side == "long" and lstm_dir == "down":
                        return None
                    if signal.side == "short" and lstm_dir == "up":
                        return None
                if (signal.side == "long" and lstm_dir == "up") or \
                   (signal.side == "short" and lstm_dir == "down"):
                    signal.strength = min(signal.strength * 1.3, 1.0)
            return signal

        strategy_fn = strategy_fn_with_lstm

    # Run backtest
    backtester = Backtester(bt_config, risk_config)
    try:
        trades_df, eq_series, metrics = backtester.run(df, strategy_fn, strategy_name)
    except Exception as e:
        return {
            "combo_id": combo_id,
            "strategy": strategy_name,
            "params": params,
            "error": str(e),
        }

    return {
        "combo_id": combo_id,
        "strategy": strategy_name,
        "params": params,
        "trades": metrics.total_trades,
        "sharpe": metrics.sharpe_ratio,
        "sortino": metrics.sortino_ratio,
        "pf": metrics.profit_factor,
        "win_rate": metrics.win_rate,
        "max_dd": metrics.max_drawdown_pct,
        "net_pnl": metrics.net_pnl,
        "expectancy": metrics.expectancy,
        "avg_holding_h": metrics.avg_holding_hours,
    }


# ---------------------------------------------------------------
# VOLATILITY BREAKOUT: THRESHOLD SWEEP
# ---------------------------------------------------------------

def generate_vb_combos() -> list[dict]:
    """Generate parameter grid for Vol Breakout — focused sweep."""
    combos = []

    # Phase 1: Bandwidth threshold sweep (most impactful param)
    for bw_threshold in [10, 15, 20, 25, 30, 35, 40, 50, 60]:
        combos.append({
            "bandwidth_percentile_threshold": float(bw_threshold),
            "volume_surge_threshold": 1.5,
            "require_atr_expansion": False,
            "stop_loss_atr_mult": 2.0,
            "take_profit_atr_mult": 3.0,
            "use_pressure_imbalance": False,
            "min_compression_bars": 3,
        })

    # Phase 2: Best bandwidth x volume surge
    for bw in [20, 25, 30, 35, 40]:
        for vol_surge in [1.0, 1.3, 1.5, 1.8, 2.0, 2.5]:
            combos.append({
                "bandwidth_percentile_threshold": float(bw),
                "volume_surge_threshold": vol_surge,
                "require_atr_expansion": False,
                "stop_loss_atr_mult": 2.0,
                "take_profit_atr_mult": 3.0,
                "use_pressure_imbalance": False,
                "min_compression_bars": 3,
            })

    # Phase 3: Fine-tune with filters
    for bw in [25, 30, 35]:
        for vol_surge in [1.3, 1.5]:
            for atr_exp in [False, True]:
                for pressure in [False, True]:
                    for stop in [1.5, 2.0, 2.5]:
                        combos.append({
                            "bandwidth_percentile_threshold": float(bw),
                            "volume_surge_threshold": vol_surge,
                            "require_atr_expansion": atr_exp,
                            "atr_expansion_ratio": 1.15,
                            "stop_loss_atr_mult": stop,
                            "take_profit_atr_mult": 3.5,
                            "use_pressure_imbalance": pressure,
                            "min_compression_bars": 3,
                        })

    # Phase 4: LSTM filter on top of best params
    for bw in [25, 30]:
        for vol_surge in [1.3, 1.5]:
            for stop in [2.0, 2.5]:
                for horizon in ["1h", "6h", "12h", "24h"]:
                    for min_conf in [0.4, 0.5, 0.6]:
                        combos.append({
                            "bandwidth_percentile_threshold": float(bw),
                            "volume_surge_threshold": vol_surge,
                            "require_atr_expansion": True,
                            "atr_expansion_ratio": 1.15,
                            "stop_loss_atr_mult": stop,
                            "take_profit_atr_mult": 3.5,
                            "use_pressure_imbalance": False,
                            "min_compression_bars": 3,
                            "use_lstm_filter": True,
                            "lstm_horizon": horizon,
                            "lstm_min_confidence": min_conf,
                        })

    # Deduplicate
    seen = set()
    unique = []
    for c in combos:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ---------------------------------------------------------------
# MEAN REVERSION: 15 FILTER COMBINATIONS
# ---------------------------------------------------------------

def generate_mr_combos() -> list[dict]:
    """Generate 15 targeted filter combinations for Mean Reversion."""
    combos = [
        # 1. Minimal: just zscore + spike, no extra filters
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": False, "use_adx_filter": False,
         "require_volume_decline": False, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 2. RSI only
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": True, "rsi_oversold": 30.0, "rsi_overbought": 70.0,
         "use_adx_filter": False, "require_volume_decline": False,
         "require_wick_rejection": False, "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 3. ADX only (no trend)
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": False, "use_adx_filter": True, "max_adx_for_entry": 25.0,
         "require_volume_decline": False, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 4. RSI + ADX
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": True, "rsi_oversold": 30.0, "rsi_overbought": 70.0,
         "use_adx_filter": True, "max_adx_for_entry": 25.0,
         "require_volume_decline": False, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 5. Volume decline only
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": False, "use_adx_filter": False,
         "require_volume_decline": True, "volume_decline_threshold": 0.8,
         "require_wick_rejection": False, "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 6. Wick rejection only
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": False, "use_adx_filter": False,
         "require_volume_decline": False, "require_wick_rejection": True,
         "wick_rejection_ratio": 0.5, "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 7. Loose thresholds, no filters
        {"zscore_entry_threshold": 1.5, "spike_min_move_pct": 0.8,
         "use_rsi_filter": False, "use_adx_filter": False,
         "require_volume_decline": False, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.5, "bb_std": 2.0},

        # 8. Tight thresholds + RSI
        {"zscore_entry_threshold": 2.5, "spike_min_move_pct": 1.5,
         "use_rsi_filter": True, "rsi_oversold": 25.0, "rsi_overbought": 75.0,
         "use_adx_filter": False, "require_volume_decline": False,
         "require_wick_rejection": False, "stop_loss_atr_mult": 2.5, "bb_std": 2.5},

        # 9. All filters ON (current defaults)
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.5,
         "use_rsi_filter": True, "rsi_oversold": 25.0, "rsi_overbought": 75.0,
         "use_adx_filter": True, "max_adx_for_entry": 25.0,
         "require_volume_decline": True, "volume_decline_threshold": 0.7,
         "require_wick_rejection": True, "wick_rejection_ratio": 0.5,
         "stop_loss_atr_mult": 3.0, "bb_std": 2.5},

        # 10. RSI + volume decline (no ADX, no wick)
        {"zscore_entry_threshold": 1.8, "spike_min_move_pct": 1.0,
         "use_rsi_filter": True, "rsi_oversold": 30.0, "rsi_overbought": 70.0,
         "use_adx_filter": False, "require_volume_decline": True,
         "volume_decline_threshold": 0.8, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 11. ADX + wick (no RSI, no volume)
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.0,
         "use_rsi_filter": False, "use_adx_filter": True, "max_adx_for_entry": 30.0,
         "require_volume_decline": False, "require_wick_rejection": True,
         "wick_rejection_ratio": 0.4, "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 12. Wider stops, loose entry
        {"zscore_entry_threshold": 1.5, "spike_min_move_pct": 0.8,
         "use_rsi_filter": True, "rsi_oversold": 35.0, "rsi_overbought": 65.0,
         "use_adx_filter": True, "max_adx_for_entry": 30.0,
         "require_volume_decline": False, "require_wick_rejection": False,
         "stop_loss_atr_mult": 4.0, "bb_std": 2.0},

        # 13. Short spike window (3 candles)
        {"zscore_entry_threshold": 2.0, "spike_min_move_pct": 1.2,
         "spike_lookback_candles": 3, "use_rsi_filter": False,
         "use_adx_filter": True, "max_adx_for_entry": 25.0,
         "require_volume_decline": False, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.0, "bb_std": 2.0},

        # 14. Long spike window (8 candles), loose
        {"zscore_entry_threshold": 1.8, "spike_min_move_pct": 2.0,
         "spike_lookback_candles": 8, "use_rsi_filter": True,
         "rsi_oversold": 25.0, "rsi_overbought": 75.0,
         "use_adx_filter": False, "require_volume_decline": True,
         "volume_decline_threshold": 0.8, "require_wick_rejection": False,
         "stop_loss_atr_mult": 3.5, "bb_std": 2.5},

        # 15. Maximum signal quality (tight everything)
        {"zscore_entry_threshold": 2.5, "spike_min_move_pct": 2.0,
         "use_rsi_filter": True, "rsi_oversold": 20.0, "rsi_overbought": 80.0,
         "use_adx_filter": True, "max_adx_for_entry": 20.0,
         "require_volume_decline": True, "volume_decline_threshold": 0.7,
         "require_wick_rejection": True, "wick_rejection_ratio": 0.5,
         "stop_loss_atr_mult": 2.5, "bb_std": 3.0},
    ]
    return combos


# ---------------------------------------------------------------
# TREND FOLLOWING: ADX + LSTM FILTER SWEEP
# ---------------------------------------------------------------

def generate_tf_combos() -> list[dict]:
    """Generate parameter combos for Trend Following with ADX filter + LSTM."""
    combos = []
    # Base grid (no LSTM)
    for adx_thresh in [20, 25, 30, 35]:
        for stop_mult in [2.0, 2.5, 3.0]:
            for tp_mult in [3.0, 4.0, 5.0]:
                for fib_tol in [1.0, 1.5, 2.0]:
                    combo = {
                        "adx_threshold": float(adx_thresh),
                        "stop_loss_atr_mult": stop_mult,
                        "take_profit_atr_mult": tp_mult,
                        "fib_zone_tolerance_pct": fib_tol,
                    }
                    combos.append(combo)

    # LSTM filter on best TF params
    for adx_thresh in [20, 25]:
        for stop_mult in [2.5, 3.0]:
            for tp_mult in [4.0, 5.0]:
                for horizon in ["6h", "12h", "24h"]:
                    for min_conf in [0.4, 0.5, 0.6]:
                        combos.append({
                            "adx_threshold": float(adx_thresh),
                            "stop_loss_atr_mult": stop_mult,
                            "take_profit_atr_mult": tp_mult,
                            "fib_zone_tolerance_pct": 2.0,
                            "use_lstm_filter": True,
                            "lstm_horizon": horizon,
                            "lstm_min_confidence": min_conf,
                        })

    return combos


# ---------------------------------------------------------------
# MAIN OPTIMIZER
# ---------------------------------------------------------------

def run_optimization(
    strategy: str,
    df: pd.DataFrame,
    lstm_preds: Optional[pd.DataFrame],
    max_workers: int = None,
) -> list[dict]:
    """Run parallel parameter optimization for a strategy using spawn context."""
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    config = LabConfig()
    bt_config = asdict(config.backtest)
    risk_config = asdict(config.risk)

    # Save df to temp pickle for worker processes (spawn requires serializable data)
    os.makedirs("/tmp/crypto_opt", exist_ok=True)
    df_path = "/tmp/crypto_opt/df.pkl"
    df.to_pickle(df_path)

    lstm_path = None
    if lstm_preds is not None:
        lstm_path = "/tmp/crypto_opt/lstm_preds.pkl"
        lstm_preds.to_pickle(lstm_path)

    # Generate combos
    if strategy == "vb":
        combos = generate_vb_combos()
        strategy_name = "volatility_breakout"
    elif strategy == "mr":
        combos = generate_mr_combos()
        strategy_name = "mean_reversion"
    elif strategy == "tf":
        combos = generate_tf_combos()
        strategy_name = "trend_following"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.info(f"Optimizing {strategy_name}: {len(combos)} combinations, {max_workers} workers")
    logger.info(f"Using multiprocessing context: spawn (macOS safe)")

    # Build work items
    work_items = [
        (strategy_name, combo, df_path, bt_config, risk_config, i, lstm_path)
        for i, combo in enumerate(combos)
    ]

    results = []
    start = time.time()

    # Use spawn context to avoid fork+TF GPU deadlock on macOS
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=MP_CTX,
        initializer=_worker_init,
    ) as executor:
        futures = {executor.submit(run_single_backtest, item): item[5] for item in work_items}

        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                result = future.result(timeout=120)
                if result and "error" not in result:
                    results.append(result)
                elif result and "error" in result:
                    if done_count <= 3:
                        logger.warning(f"  Worker error: {result['error']}")
            except Exception as e:
                if done_count <= 3:
                    logger.warning(f"  Worker exception: {e}")

            if done_count % 20 == 0 or done_count == len(combos):
                elapsed = time.time() - start
                rate = done_count / elapsed if elapsed > 0 else 0
                valid = len(results)
                logger.info(
                    f"  Progress: {done_count}/{len(combos)} "
                    f"({elapsed:.0f}s, {rate:.1f}/sec, {valid} valid)"
                )

    elapsed = time.time() - start
    logger.info(f"Optimization complete: {len(results)} valid results in {elapsed:.1f}s")

    # Cleanup temp files
    try:
        os.remove(df_path)
        if lstm_path:
            os.remove(lstm_path)
    except OSError:
        pass

    return results


def print_top_results(results: list[dict], strategy_name: str, top_n: int = 10):
    """Print and save top results ranked by Sharpe ratio."""
    # Filter: must have trades
    valid = [r for r in results if r.get("trades", 0) >= 5]

    if not valid:
        logger.warning(f"No valid results for {strategy_name} (all had < 5 trades)")
        # Show best by trade count
        by_trades = sorted(results, key=lambda x: x.get("trades", 0), reverse=True)[:5]
        for r in by_trades:
            logger.info(f"  Max trades={r.get('trades', 0)}: {r.get('params', {})}")
        return []

    # Sort by robustness-adjusted score: Sharpe * sqrt(trades) * min(PF, 3) / 3
    # This penalizes low trade counts and extreme PF (likely overfitting)
    def _score(r):
        sharpe = r.get("sharpe", -99)
        trades = max(r.get("trades", 0), 1)
        pf = min(r.get("pf", 0), 3.0)  # Cap PF to avoid rewarding extreme values
        return sharpe * (trades ** 0.5) * (pf / 3.0)

    ranked = sorted(valid, key=_score, reverse=True)

    logger.info(f"\n{'='*80}")
    logger.info(f"TOP {min(top_n, len(ranked))} RESULTS: {strategy_name}")
    logger.info(f"{'='*80}")
    logger.info(f"{'Rank':>4} {'Score':>8} {'Sharpe':>8} {'PF':>8} {'WinR%':>7} {'Trades':>7} "
                f"{'MaxDD%':>8} {'PnL$':>10} {'Expect':>8}")
    logger.info("-" * 90)

    for i, r in enumerate(ranked[:top_n]):
        score = _score(r)
        logger.info(
            f"{i+1:>4} {score:>8.2f} {r['sharpe']:>8.3f} {r['pf']:>8.3f} "
            f"{r['win_rate']*100:>6.1f}% {r['trades']:>7} "
            f"{r['max_dd']:>7.2f}% ${r['net_pnl']:>9.2f} "
            f"{r['expectancy']:>8.4f}"
        )

    # Show winning params
    logger.info(f"\nBEST PARAMS ({strategy_name}):")
    best = ranked[0]
    for k, v in sorted(best["params"].items()):
        logger.info(f"  {k}: {v}")

    # Save results
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"reports/optimization_{strategy_name}_{ts}.json"
    with open(path, "w") as f:
        json.dump(ranked[:top_n], f, indent=2, default=str)
    logger.info(f"\nResults saved: {path}")

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer")
    parser.add_argument("--strategy", type=str, default="all",
                        choices=["all", "vb", "mr", "tf"],
                        help="Strategy to optimize (vb/mr/tf/all)")
    parser.add_argument("--cores", type=int, default=0,
                        help="Number of CPU cores (0 = auto)")
    parser.add_argument("--days", type=int, default=180,
                        help="Days of historical data to use")
    parser.add_argument("--no-lstm", action="store_true",
                        help="Skip LSTM prediction integration")
    args = parser.parse_args()

    max_workers = args.cores if args.cores > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"CPU cores available: {mp.cpu_count()}, using {max_workers} workers")

    # Load data
    df = load_real_data(args.days)

    # Load LSTM predictions
    lstm_preds = None
    if not args.no_lstm:
        lstm_preds = load_lstm_predictions(df)

    strategies_to_run = []
    if args.strategy == "all":
        strategies_to_run = ["vb", "mr", "tf"]
    else:
        strategies_to_run = [args.strategy]

    all_best = {}
    for strat in strategies_to_run:
        name_map = {"vb": "volatility_breakout", "mr": "mean_reversion", "tf": "trend_following"}
        logger.info(f"\n{'#'*80}")
        logger.info(f"# OPTIMIZING: {name_map[strat].upper()}")
        logger.info(f"{'#'*80}")

        results = run_optimization(strat, df, lstm_preds, max_workers)
        ranked = print_top_results(results, name_map[strat])
        if ranked:
            all_best[name_map[strat]] = ranked[0]

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION SUMMARY")
    logger.info(f"{'='*80}")
    for name, best in all_best.items():
        logger.info(f"  {name}: Sharpe={best['sharpe']:.3f} PF={best['pf']:.3f} "
                     f"Trades={best['trades']} PnL=${best['net_pnl']:.2f}")


if __name__ == "__main__":
    main()
