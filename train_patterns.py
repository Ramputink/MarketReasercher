#!/usr/bin/env python3
"""
CryptoResearchLab — Multi-Variant Pattern Clustering + LSTM Training Pipeline

Trains MULTIPLE clustering variants + LSTM models for 1H and 4H timeframes:
  1. Fetch historical XRP/USDT data (inception → 1yr ago for training)
  2. Build features
  3. Train 7 clustering variants per timeframe:
     - K-Means: k=8, 20, 50
     - Hierarchical 2-level: k=20, 50 (3 macro direction × N micro shape)
     - BisectingKMeans: k=20, 50 (divisive tree)
  4. LSTM training per variant on cluster transition sequences
  5. Save all models — evolution selects the best variant

Usage:
  python train_patterns.py --timeframe 1h                 # Train all variants for 1H
  python train_patterns.py --timeframe 4h                 # Train all variants for 4H
  python train_patterns.py --timeframe all                # Both timeframes
  python train_patterns.py --timeframe 1h --variant kmeans_50  # Single variant
  python train_patterns.py --timeframe all --lstm-epochs 15    # Quick LSTM training
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LabConfig, TimeFrame, DataConfig
from engine.data_ingestion import DataIngestionEngine
from engine.features import build_all_features
from engine.pattern_clustering import (
    train_pattern_clusters,
    extract_candle_windows,
    save_cluster_model,
    update_cluster_profiles,
)
from engine.multi_clustering import (
    CLUSTER_VARIANTS,
    ALL_VARIANT_IDS,
    train_variant,
    save_variant_model,
)
from engine.lstm_pattern_model import (
    prepare_lstm_sequences,
    train_lstm_pattern,
    save_lstm_pattern_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_patterns")


def fetch_full_history(
    timeframe: TimeFrame,
    test_days: int = 365,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch FULL XRP/USDT history and split into:
      - train_df: inception → (now - test_days)
      - test_df: last test_days (for comparison with evolution runs)

    XRP started trading ~2018. We fetch as much as Binance provides.
    """
    data_cfg = DataConfig(
        timeframes=[timeframe],
        history_days=2500,  # ~7 years, Binance will return what it has
        cross_exchanges=[],
    )
    engine = DataIngestionEngine(data_cfg)

    logger.info(f"Fetching XRP/USDT {timeframe.value} full history (~2500 days)...")
    df = engine.fetch_ohlcv(
        "XRP/USDT", timeframe,
        since_days=2500, force_refresh=True,
    )

    if df is None or len(df) < 500:
        raise RuntimeError(f"Insufficient data: {len(df) if df is not None else 0} candles")

    logger.info(f"Total candles: {len(df)}")

    # Find split point: now - test_days
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    cutoff_ms = now_ms - (test_days * 86_400_000)

    train_mask = df["timestamp"] < cutoff_ms
    test_mask = df["timestamp"] >= cutoff_ms

    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    ts_start = pd.Timestamp(train_df["timestamp"].iloc[0], unit="ms")
    ts_split = pd.Timestamp(train_df["timestamp"].iloc[-1], unit="ms")
    ts_end = pd.Timestamp(test_df["timestamp"].iloc[-1], unit="ms")

    logger.info(f"Train: {len(train_df)} candles ({ts_start.date()} → {ts_split.date()})")
    logger.info(f"Test:  {len(test_df)} candles ({ts_split.date()} → {ts_end.date()})")

    return train_df, test_df


def train_for_timeframe(
    timeframe: TimeFrame,
    window_size: int = 10,
    forward_bars: int = 5,
    seq_length: int = 20,
    direction_threshold: float = 0.005,
    lstm_epochs: int = 50,
    lstm_units: int = 48,
    verbose: int = 1,
    variants: list[str] | None = None,
):
    """
    Full multi-variant training pipeline for a single timeframe.

    Trains all clustering variants (or a subset) + one LSTM per variant.
    """
    tf_str = timeframe.value  # "1h" or "4h"
    target_variants = variants or ALL_VARIANT_IDS

    logger.info(f"\n{'='*60}")
    logger.info(f"  TRAINING PATTERN MODELS: {tf_str.upper()}")
    logger.info(f"  Variants: {', '.join(target_variants)}")
    logger.info(f"  Window={window_size}, Forward={forward_bars}, LSTM epochs={lstm_epochs}")
    logger.info(f"{'='*60}\n")

    # 1. Fetch data (once per timeframe — shared across all variants)
    train_raw, test_raw = fetch_full_history(timeframe, test_days=365)

    # 2. Build features on train + test sets
    logger.info("Building features on training data...")
    train_df = build_all_features(train_raw)
    logger.info(f"Train features: {len(train_df.columns)} columns")

    logger.info("Building features on test data...")
    test_df = build_all_features(test_raw)

    all_metas = []

    # 3. Train each variant
    for vid in target_variants:
        try:
            meta = _train_single_variant(
                vid, tf_str, train_df, test_df,
                window_size=window_size,
                forward_bars=forward_bars,
                seq_length=seq_length,
                direction_threshold=direction_threshold,
                lstm_epochs=lstm_epochs,
                lstm_units=lstm_units,
                verbose=verbose,
            )
            all_metas.append(meta)
        except Exception as e:
            logger.error(f"Failed to train variant {vid}: {e}", exc_info=True)
            continue

    # Also maintain backward-compatible "pattern_clusters_{tf}" model
    # by symlinking/copying the kmeans_50 variant
    _create_legacy_compat(tf_str)

    logger.info(f"\n{'='*60}")
    logger.info(f"  DONE: {tf_str.upper()} — {len(all_metas)}/{len(target_variants)} variants trained")
    logger.info(f"{'='*60}\n")

    return all_metas


def _train_single_variant(
    variant_id: str,
    tf_str: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_size: int = 10,
    forward_bars: int = 5,
    seq_length: int = 20,
    direction_threshold: float = 0.005,
    lstm_epochs: int = 50,
    lstm_units: int = 48,
    verbose: int = 1,
) -> dict:
    """Train one clustering variant + its LSTM."""

    logger.info(f"\n{'─'*50}")
    logger.info(f"  VARIANT: {variant_id} ({tf_str})")
    logger.info(f"{'─'*50}")

    # ── Clustering ──
    cluster_model, scaler, labels, indices, model_info, n_eff = train_variant(
        variant_id, train_df,
        window_size=window_size,
        forward_bars=forward_bars,
        direction_threshold=direction_threshold,
    )
    model_info.timeframe = tf_str
    save_variant_model(variant_id, cluster_model, scaler, model_info, output_dir="models")

    # ── LSTM sequences ──
    logger.info(f"  Preparing LSTM sequences (seq={seq_length})...")
    X, y_dir, y_mag = prepare_lstm_sequences(
        train_df,
        cluster_labels=labels,
        cluster_indices=indices,
        n_clusters=n_eff,
        seq_length=seq_length,
        forward_bars=forward_bars,
        direction_threshold=direction_threshold,
    )

    if len(X) < 100:
        logger.error(f"  Not enough LSTM sequences: {len(X)}. Skipping LSTM for {variant_id}.")
        return {"variant_id": variant_id, "timeframe": tf_str, "status": "no_lstm"}

    logger.info(f"  LSTM sequences: {X.shape} (samples, seq, features)")
    logger.info(f"  Direction: down={np.sum(y_dir==0)}, neutral={np.sum(y_dir==1)}, up={np.sum(y_dir==2)}")

    # ── Train LSTM ──
    logger.info(f"  Training LSTM ({lstm_epochs} epochs)...")
    model, history, val_metrics = train_lstm_pattern(
        X, y_dir, y_mag,
        epochs=lstm_epochs,
        lstm_units=lstm_units,
        verbose=verbose,
    )

    lstm_name = f"lstm_{variant_id}_{tf_str}"
    save_lstm_pattern_model(model, output_dir="models", name=lstm_name)

    # ── Test evaluation ──
    logger.info(f"  Evaluating on test set...")
    test_features, test_indices = extract_candle_windows(test_df, window_size=window_size)
    test_accuracy = None
    if len(test_features) > 0:
        test_scaled = scaler.transform(test_features)
        # Use appropriate predict method
        if hasattr(cluster_model, 'predict'):
            test_labels = cluster_model.predict(test_scaled)
        else:
            test_labels = cluster_model.predict(test_scaled)

        from collections import Counter
        train_used = len(set(labels))
        test_used = len(Counter(test_labels))
        logger.info(f"  Clusters used — train: {train_used}/{n_eff}, test: {test_used}/{n_eff}")

        X_test, y_dir_test, y_mag_test = prepare_lstm_sequences(
            test_df, test_labels, test_indices, n_eff,
            seq_length=seq_length, forward_bars=forward_bars,
            direction_threshold=direction_threshold,
        )
        if len(X_test) > 50:
            test_preds = model.predict(X_test, verbose=0)
            test_dir_preds = np.argmax(test_preds[0], axis=1)
            test_accuracy = float(np.mean(test_dir_preds == y_dir_test))
            logger.info(f"  Test accuracy: {test_accuracy:.3f} (n={len(X_test)})")

            for cls, name in [(0, "DOWN"), (1, "NEUTRAL"), (2, "UP")]:
                mask = y_dir_test == cls
                if mask.sum() > 0:
                    cls_acc = float(np.mean(test_dir_preds[mask] == cls))
                    logger.info(f"    {name}: {cls_acc:.3f} (n={mask.sum()})")

    # ── Save metadata ──
    meta = {
        "variant_id": variant_id,
        "timeframe": tf_str,
        "algo": CLUSTER_VARIANTS[variant_id]["algo"],
        "n_clusters": n_eff,
        "window_size": window_size,
        "forward_bars": forward_bars,
        "seq_length": seq_length,
        "direction_threshold": direction_threshold,
        "train_candles": len(train_df),
        "lstm_sequences": len(X),
        "val_metrics": val_metrics,
        "test_accuracy": test_accuracy,
        "lstm_name": lstm_name,
        "status": "ok",
        "trained_at": datetime.utcnow().isoformat(),
    }
    meta_path = f"models/{lstm_name}_training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"  ✓ {variant_id} complete (test_acc={test_accuracy})")
    return meta


def _create_legacy_compat(tf_str: str):
    """
    Create backward-compatible model files (pattern_clusters_{tf}_*)
    by copying kmeans_50 variant, so the old strategy code still works.
    """
    import shutil
    src_prefix = f"cluster_kmeans_50_{tf_str}"
    dst_prefix = f"pattern_clusters_{tf_str}"

    for suffix in ["_model.joblib", "_scaler.joblib", "_meta.json"]:
        src = f"models/{src_prefix}{suffix}"
        # Map model.joblib → kmeans.joblib for legacy compat
        if suffix == "_model.joblib":
            dst = f"models/{dst_prefix}_kmeans.joblib"
        elif suffix == "_scaler.joblib":
            dst = f"models/{dst_prefix}_scaler.joblib"
        else:
            dst = f"models/{dst_prefix}_meta.json"

        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Also copy LSTM
    src_lstm = f"models/lstm_kmeans_50_{tf_str}.keras"
    dst_lstm = f"models/lstm_pattern_{tf_str}.keras"
    if os.path.exists(src_lstm):
        shutil.copy2(src_lstm, dst_lstm)

    logger.info(f"  Created legacy-compatible models: {dst_prefix}_*")


def main():
    parser = argparse.ArgumentParser(description="Train multi-variant pattern clustering + LSTM models")
    parser.add_argument("--timeframe", type=str, default="1h",
                        choices=["1h", "4h", "all"],
                        help="Timeframe to train (default: 1h)")
    parser.add_argument("--variant", type=str, default=None,
                        choices=ALL_VARIANT_IDS + [None],
                        help="Train only this variant (default: all variants)")
    parser.add_argument("--window", type=int, default=10,
                        help="Candle window size (default: 10)")
    parser.add_argument("--forward", type=int, default=5,
                        help="Forward bars for return calc (default: 5)")
    parser.add_argument("--seq-length", type=int, default=20,
                        help="LSTM sequence length (default: 20)")
    parser.add_argument("--lstm-epochs", type=int, default=20,
                        help="LSTM training epochs per variant (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    tf_map = {"1h": TimeFrame.H1, "4h": TimeFrame.H4}
    verbose = 0 if args.quiet else 1
    variants = [args.variant] if args.variant else None

    os.makedirs("models", exist_ok=True)

    logger.info(f"Clustering variants to train: {variants or ALL_VARIANT_IDS}")
    logger.info(f"LSTM epochs per variant: {args.lstm_epochs}")

    if args.timeframe == "all":
        for tf_str, tf_enum in tf_map.items():
            train_for_timeframe(
                tf_enum, window_size=args.window,
                forward_bars=args.forward, seq_length=args.seq_length,
                lstm_epochs=args.lstm_epochs, verbose=verbose,
                variants=variants,
            )
    else:
        train_for_timeframe(
            tf_map[args.timeframe], window_size=args.window,
            forward_bars=args.forward, seq_length=args.seq_length,
            lstm_epochs=args.lstm_epochs, verbose=verbose,
            variants=variants,
        )

    logger.info("All pattern models trained!")


if __name__ == "__main__":
    main()
