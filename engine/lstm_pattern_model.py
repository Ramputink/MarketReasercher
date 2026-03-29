"""
CryptoResearchLab — LSTM Pattern Predictor
Predicts cluster transition probabilities using sequence of recent cluster IDs
+ technical features.

Architecture:
  Input: sequence of (cluster_id one-hot + technical features) over last S bars
  LSTM → Dense → Softmax over {LONG, NEUTRAL, SHORT}
  + regression head for expected magnitude

The LSTM sees temporal patterns in cluster transitions that K-Means alone
cannot capture (e.g., "after cluster 12 → 7 → 23, the market tends to
break out upward 68% of the time").
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger("lstm_pattern")

# Lazy TF import to avoid startup overhead in worker processes
_tf = None
_tf_keras = None


def _import_tf():
    global _tf, _tf_keras
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _tf_keras = tf.keras
    return _tf, _tf_keras


# ═══════════════════════════════════════════════════════════════
# SEQUENCE PREPARATION
# ═══════════════════════════════════════════════════════════════

def prepare_lstm_sequences(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_indices: np.ndarray,
    n_clusters: int,
    seq_length: int = 20,
    forward_bars: int = 5,
    direction_threshold: float = 0.005,
) -> tuple:
    """
    Prepare training sequences for the LSTM pattern predictor.

    Each sequence contains:
      - One-hot encoded cluster IDs for last seq_length windows
      - Technical features at each timestep (RSI, ADX, vol_ratio, ret_zscore)

    Target:
      - Direction: 0=down, 1=neutral, 2=up (based on forward return)
      - Magnitude: absolute forward return

    Returns:
        X: (n_samples, seq_length, n_features)
        y_dir: (n_samples,) int labels
        y_mag: (n_samples,) float magnitudes
    """
    # Build per-bar cluster assignment
    n_bars = len(df)
    bar_cluster = np.full(n_bars, -1, dtype=int)
    for label, idx in zip(cluster_labels, cluster_indices):
        if 0 <= idx < n_bars:
            bar_cluster[idx] = label

    # Forward-fill cluster assignments for bars between windows
    last_valid = -1
    for i in range(n_bars):
        if bar_cluster[i] >= 0:
            last_valid = bar_cluster[i]
        elif last_valid >= 0:
            bar_cluster[i] = last_valid

    # Technical features to include (from build_all_features)
    tech_cols = []
    available = df.columns.tolist()
    for col in ["rsi_14", "adx_14", "volume_ratio", "ret_zscore_20",
                 "bb_pct_b", "pressure_imbalance", "gk_vol", "vol_ratio"]:
        if col in available:
            tech_cols.append(col)

    n_tech = len(tech_cols)
    n_features = n_clusters + n_tech  # one-hot clusters + technical features

    closes = df["close"].values
    X_list = []
    y_dir_list = []
    y_mag_list = []

    start_idx = seq_length + 10  # Ensure enough lookback

    for i in range(start_idx, n_bars - forward_bars):
        # Check all bars in sequence have valid clusters
        seq_clusters = bar_cluster[i - seq_length:i]
        if np.any(seq_clusters < 0):
            continue

        # Build sequence features
        seq = np.zeros((seq_length, n_features), dtype=np.float32)
        for t in range(seq_length):
            bar_i = i - seq_length + t
            # One-hot cluster
            cid = seq_clusters[t]
            if 0 <= cid < n_clusters:
                seq[t, cid] = 1.0

            # Technical features
            for j, col in enumerate(tech_cols):
                val = df[col].iat[bar_i]
                if pd.notna(val) and np.isfinite(val):
                    seq[t, n_clusters + j] = float(val)

        # Forward return target
        if closes[i] <= 0:
            continue
        fwd_ret = (closes[i + forward_bars] - closes[i]) / closes[i]
        if not np.isfinite(fwd_ret):
            continue

        # Direction label
        if fwd_ret > direction_threshold:
            direction = 2  # up
        elif fwd_ret < -direction_threshold:
            direction = 0  # down
        else:
            direction = 1  # neutral

        X_list.append(seq)
        y_dir_list.append(direction)
        y_mag_list.append(abs(fwd_ret))

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    return np.array(X_list), np.array(y_dir_list), np.array(y_mag_list)


# ═══════════════════════════════════════════════════════════════
# MODEL BUILDING
# ═══════════════════════════════════════════════════════════════

def build_lstm_pattern_model(
    seq_length: int,
    n_features: int,
    lstm_units: int = 48,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
):
    """
    Build LSTM model for pattern-based direction prediction.

    Architecture:
      Input(seq_length, n_features) →
      LSTM(48, return_sequences=True) → Dropout →
      LSTM(32) → Dropout →
      Dense(32, relu) →
      [direction_head: Dense(3, softmax),
       magnitude_head: Dense(1, linear)]
    """
    tf, keras = _import_tf()

    inputs = keras.Input(shape=(seq_length, n_features), name="pattern_seq")

    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LSTM(lstm_units // 2, return_sequences=False, name="lstm_2")(x)
    x = keras.layers.Dropout(dropout)(x)

    trunk = keras.layers.Dense(32, activation="relu", name="trunk")(x)

    # Direction head: down(0) / neutral(1) / up(2)
    dir_out = keras.layers.Dense(3, activation="softmax", name="direction")(trunk)

    # Magnitude head: expected absolute return
    mag_out = keras.layers.Dense(1, activation="linear", name="magnitude")(trunk)

    model = keras.Model(inputs=inputs, outputs=[dir_out, mag_out])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "direction": "sparse_categorical_crossentropy",
            "magnitude": "mse",
        },
        loss_weights={"direction": 1.0, "magnitude": 0.3},
        metrics={"direction": "accuracy"},
    )

    return model


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_lstm_pattern(
    X: np.ndarray,
    y_dir: np.ndarray,
    y_mag: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.15,
    lstm_units: int = 48,
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    verbose: int = 1,
):
    """
    Train the LSTM pattern predictor.

    Uses temporal split (last 15% as validation) to prevent leakage.
    Returns (model, history, val_metrics).
    """
    tf, keras = _import_tf()

    if len(X) < 100:
        raise ValueError(f"Not enough training samples: {len(X)} (need >= 100)")

    n_val = int(len(X) * validation_split)
    n_train = len(X) - n_val

    X_train, X_val = X[:n_train], X[n_train:]
    y_dir_train, y_dir_val = y_dir[:n_train], y_dir[n_train:]
    y_mag_train, y_mag_val = y_mag[:n_train], y_mag[n_train:]

    seq_length = X.shape[1]
    n_features = X.shape[2]

    model = build_lstm_pattern_model(
        seq_length=seq_length,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    # Class weights → sample_weight (Keras doesn't support class_weight for multi-output)
    unique, counts = np.unique(y_dir_train, return_counts=True)
    total = len(y_dir_train)
    class_weight_map = {int(c): total / (len(unique) * n) for c, n in zip(unique, counts)}

    # Convert class weights to per-sample weights for the direction head
    sample_weights_train = np.array([class_weight_map.get(int(y), 1.0) for y in y_dir_train],
                                     dtype=np.float32)
    sample_weights_val = np.array([class_weight_map.get(int(y), 1.0) for y in y_dir_val],
                                   dtype=np.float32)

    logger.info(f"Class weights: {class_weight_map}")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_direction_accuracy", patience=10,
            restore_best_weights=True, mode="max",
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5,
        ),
    ]

    logger.info(f"Training LSTM pattern model: {n_train} train, {n_val} val, "
                f"seq={seq_length}, features={n_features}")

    # Outputs are a list [direction, magnitude], so sample_weight must also be a list
    # Index 0 = direction (weighted by class balance), index 1 = magnitude (uniform)
    history = model.fit(
        X_train, [y_dir_train, y_mag_train],
        validation_data=(X_val, [y_dir_val, y_mag_val]),
        epochs=epochs,
        batch_size=batch_size,
        sample_weight=[sample_weights_train,
                       np.ones(len(y_mag_train), dtype=np.float32)],
        callbacks=callbacks,
        verbose=verbose,
    )

    # Evaluate
    val_preds = model.predict(X_val, verbose=0)
    dir_preds = np.argmax(val_preds[0], axis=1)
    accuracy = float(np.mean(dir_preds == y_dir_val))

    # Per-class accuracy
    for cls, name in [(0, "DOWN"), (1, "NEUTRAL"), (2, "UP")]:
        mask = y_dir_val == cls
        if mask.sum() > 0:
            cls_acc = float(np.mean(dir_preds[mask] == cls))
            logger.info(f"  {name}: accuracy={cls_acc:.3f} (n={mask.sum()})")

    val_metrics = {
        "accuracy": accuracy,
        "n_train": n_train,
        "n_val": n_val,
        "best_epoch": int(np.argmax(history.history.get("val_direction_accuracy", [0]))) + 1,
    }

    logger.info(f"LSTM Pattern: val_accuracy={accuracy:.3f}")

    return model, history, val_metrics


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def predict_pattern_direction(
    model,
    df: pd.DataFrame,
    bar_idx: int,
    bar_cluster: np.ndarray,
    n_clusters: int,
    seq_length: int = 20,
) -> Optional[dict]:
    """
    Predict direction from current pattern sequence.

    Returns dict with:
      - prob_down, prob_neutral, prob_up
      - predicted_direction: "long", "short", or "neutral"
      - magnitude: expected absolute return
      - confidence: max(prob_up, prob_down)
    """
    if bar_idx < seq_length + 5:
        return None

    # Build sequence
    tech_cols = []
    available = df.columns.tolist()
    for col in ["rsi_14", "adx_14", "volume_ratio", "ret_zscore_20",
                 "bb_pct_b", "pressure_imbalance", "gk_vol", "vol_ratio"]:
        if col in available:
            tech_cols.append(col)

    n_features = n_clusters + len(tech_cols)
    seq = np.zeros((1, seq_length, n_features), dtype=np.float32)

    for t in range(seq_length):
        bi = bar_idx - seq_length + t
        if bi < 0 or bi >= len(df):
            return None

        cid = int(bar_cluster[bi]) if bi < len(bar_cluster) else -1
        if 0 <= cid < n_clusters:
            seq[0, t, cid] = 1.0

        for j, col in enumerate(tech_cols):
            val = df[col].iat[bi]
            if pd.notna(val) and np.isfinite(val):
                seq[0, t, n_clusters + j] = float(val)

    # Predict
    dir_probs, mag_pred = model.predict(seq, verbose=0)
    prob_down = float(dir_probs[0, 0])
    prob_neutral = float(dir_probs[0, 1])
    prob_up = float(dir_probs[0, 2])
    magnitude = float(abs(mag_pred[0, 0]))

    if prob_up > prob_down and prob_up > prob_neutral:
        direction = "long"
    elif prob_down > prob_up and prob_down > prob_neutral:
        direction = "short"
    else:
        direction = "neutral"

    return {
        "prob_down": prob_down,
        "prob_neutral": prob_neutral,
        "prob_up": prob_up,
        "predicted_direction": direction,
        "magnitude": magnitude,
        "confidence": max(prob_up, prob_down),
    }


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_lstm_pattern_model(
    model,
    output_dir: str = "models",
    name: str = "lstm_pattern",
):
    """Save LSTM model to disk."""
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f"{name}.keras"))
    logger.info(f"Saved LSTM pattern model to {output_dir}/{name}.keras")


def load_lstm_pattern_model(
    model_dir: str = "models",
    name: str = "lstm_pattern",
):
    """Load LSTM model from disk."""
    _, keras = _import_tf()
    path = os.path.join(model_dir, f"{name}.keras")
    model = keras.models.load_model(path)
    logger.info(f"Loaded LSTM pattern model from {path}")
    return model
