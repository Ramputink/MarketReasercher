"""
CryptoResearchLab -- TensorFlow Model for XRP Price Prediction
Optimized for Mac M2 Pro + Metal GPU (tensorflow-metal 1.2.0)

Architecture: LSTM + Attention + Dense heads
- Multi-horizon prediction (1h, 6h, 12h, 24h)
- Classification (up/down/flat) + Regression (magnitude)
- Walk-forward training with no future leakage
- Metal GPU acceleration via tf.device

This model is a COMPONENT of the research pipeline.
It does NOT make trading decisions alone -- its predictions feed
into the strategy layer alongside MiroFish context and risk management.
"""
import os
import json
import logging
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# GPU / METAL SETUP
# ---------------------------------------------------------------

def setup_metal_gpu():
    """Configure TensorFlow for Apple Metal GPU + multicore CPU ops."""
    import os

    # Configure CPU threading for data pipeline and non-GPU ops
    ncores = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(ncores)
    tf.config.threading.set_inter_op_parallelism_threads(ncores)
    logger.info(f"TF threading: intra={ncores}, inter={ncores}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        logger.info(f"Metal GPU detected: {len(gpus)} device(s)")
        for g in gpus:
            logger.info(f"  {g.name}")
        return True
    else:
        logger.warning("No Metal GPU detected -- falling back to CPU")
        return False


# ---------------------------------------------------------------
# DATA PREPARATION (no future leakage)
# ---------------------------------------------------------------

# Features to use from the feature pipeline (all backward-looking)
FEATURE_COLUMNS = [
    # Price-based
    "close", "high", "low", "open", "volume",
    # ATR
    "atr_14", "atr_7",
    # ADX
    "adx_14",
    # Bollinger Bands
    "bb_bandwidth", "bb_pct_b", "bb_bw_percentile",
    # BMSB
    "bmsb_spread",
    # Volatility
    "realized_vol_5", "realized_vol_10", "realized_vol_20",
    "vol_ratio", "gk_vol",
    # Volume
    "volume_ratio", "volume_zscore", "obv_divergence",
    # Momentum
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "ret_zscore_20", "rsi_14", "rsi_7", "roc_12",
    "dist_from_high_20", "dist_from_low_20",
    # Microstructure
    "close_location", "pressure_imbalance",
    "range_atr_ratio", "body_wick_ratio",
]


def create_labels(df: pd.DataFrame, horizons: list, threshold_pct: float = 0.3):
    """
    Create multi-horizon labels.
    For each horizon h:
      - direction: 0=down, 1=flat, 2=up  (classification)
      - magnitude: future return in %     (regression)

    Uses FUTURE returns -- only call this on training data,
    never on live prediction data.
    """
    labels = {}
    close = df["close"].values

    for h in horizons:
        future_ret = np.zeros(len(close))
        future_ret[:-h] = (close[h:] - close[:-h]) / close[:-h] * 100

        # Classification: 0=down, 1=flat, 2=up
        direction = np.ones(len(close), dtype=np.int32)  # default flat
        direction[future_ret > threshold_pct] = 2   # up
        direction[future_ret < -threshold_pct] = 0  # down

        # Last h bars have no valid label
        direction[-h:] = -1  # mask
        future_ret[-h:] = np.nan

        labels[f"dir_{h}h"] = direction
        labels[f"mag_{h}h"] = future_ret

    return labels


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_length: int = 60,
    horizons: list = None,
    threshold_pct: float = 0.3,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = True,
):
    """
    Convert DataFrame into (X, y) sequences for LSTM training.

    X shape: (n_samples, seq_length, n_features)
    y: dict of arrays per horizon

    Strict no-future-leakage:
    - Features are all backward-looking (computed by features.py)
    - Scaler is fit ONLY on training data
    - Labels use explicitly marked future returns
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    # Select available features
    available = [c for c in feature_cols if c in df.columns]
    if len(available) < 10:
        raise ValueError(
            f"Only {len(available)} features available. Need at least 10. "
            f"Missing: {set(feature_cols) - set(df.columns)}"
        )

    data = df[available].copy()

    # Forward-fill then drop remaining NaN rows
    data = data.ffill().bfill()

    # Scale features
    if scaler is None:
        scaler = RobustScaler()
    if fit_scaler:
        scaled = scaler.fit_transform(data.values)
    else:
        scaled = scaler.transform(data.values)

    # Create labels
    labels = create_labels(df, horizons, threshold_pct)

    # Build sequences
    max_horizon = max(horizons)
    n_valid = len(scaled) - seq_length - max_horizon

    if n_valid <= 0:
        raise ValueError(
            f"Not enough data: {len(scaled)} rows, need > {seq_length + max_horizon}"
        )

    X = np.zeros((n_valid, seq_length, len(available)), dtype=np.float32)
    y_dir = {h: np.zeros(n_valid, dtype=np.int32) for h in horizons}
    y_mag = {h: np.zeros(n_valid, dtype=np.float32) for h in horizons}

    for i in range(n_valid):
        X[i] = scaled[i:i + seq_length]
        idx = i + seq_length  # prediction point
        for h in horizons:
            y_dir[h][i] = labels[f"dir_{h}h"][idx]
            y_mag[h][i] = labels[f"mag_{h}h"][idx]

    # Remove samples with invalid labels (mask = -1)
    valid_mask = np.ones(n_valid, dtype=bool)
    for h in horizons:
        valid_mask &= y_dir[h] >= 0
        valid_mask &= ~np.isnan(y_mag[h])

    X = X[valid_mask]
    for h in horizons:
        y_dir[h] = y_dir[h][valid_mask]
        y_mag[h] = y_mag[h][valid_mask]

    return X, y_dir, y_mag, scaler, available


# ---------------------------------------------------------------
# IMBALANCE-AWARE CLASSIFICATION METRICS
# ---------------------------------------------------------------

# Class index -> name mapping for the 3-class direction heads.
# Matches create_labels(): 0=down, 1=flat/neutral, 2=up
CLASS_NAMES = ["down", "neutral", "up"]
NEUTRAL_CLASS = 1


def classification_report_imbalanced(y_true, y_pred, n_classes: int = 3):
    """
    Honest, imbalance-aware classification metrics for the direction heads.

    Raw accuracy is base-rate-inflated when one class (here "neutral") dominates,
    so this computes a fuller picture. Everything is derived by hand from the
    confusion matrix -- no new heavy dependencies (does not require sklearn).

    Returns a dict with:
      - accuracy: raw accuracy (base-rate INFLATED -- not the headline metric)
      - balanced_accuracy: mean per-class recall (the honest headline metric)
      - confusion_matrix: rows=true class, cols=predicted class (list of lists)
      - support: per-class true-count
      - per_class: {name: {precision, recall, f1, support}}
      - precision_macro / recall_macro / f1_macro
      - mcc: Matthews correlation coefficient (multi-class)
      - baseline: the trivial "always predict majority class (neutral)" classifier
                  reported side-by-side {majority_class, accuracy, balanced_accuracy}
      - beats_baseline: bool, model balanced_accuracy > baseline balanced_accuracy
      - n_samples
    """
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    n = int(y_true.shape[0])

    # Confusion matrix: cm[t][p] = count of true=t predicted=p
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    if n > 0:
        valid = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
        for t, p in zip(y_true[valid], y_pred[valid]):
            cm[t, p] += 1

    support = cm.sum(axis=1)              # true count per class
    pred_count = cm.sum(axis=0)           # predicted count per class
    tp = np.diag(cm).astype(np.float64)

    total = float(cm.sum())
    accuracy = float(tp.sum() / total) if total > 0 else 0.0

    # Per-class precision / recall / f1
    per_class = {}
    recalls = []
    precisions = []
    f1s = []
    for c in range(n_classes):
        rec = float(tp[c] / support[c]) if support[c] > 0 else 0.0
        prec = float(tp[c] / pred_count[c]) if pred_count[c] > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        per_class[name] = {
            "precision": prec,
            "recall": rec,
            "f1": float(f1),
            "support": int(support[c]),
        }
        # Only count classes that actually appear in y_true for the macro averages
        if support[c] > 0:
            recalls.append(rec)
            precisions.append(prec)
            f1s.append(float(f1))

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    precision_macro = float(np.mean(precisions)) if precisions else 0.0
    recall_macro = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0

    # Matthews correlation coefficient (multi-class generalization)
    # MCC = (c*s - sum_k p_k*t_k) / sqrt((s^2 - sum p_k^2)(s^2 - sum t_k^2))
    c_correct = float(tp.sum())
    s = total
    p_k = pred_count.astype(np.float64)
    t_k = support.astype(np.float64)
    cov_ytyp = c_correct * s - float(np.dot(p_k, t_k))
    denom_sq = (s * s - float(np.dot(p_k, p_k))) * (s * s - float(np.dot(t_k, t_k)))
    mcc = float(cov_ytyp / np.sqrt(denom_sq)) if denom_sq > 0 else 0.0

    # Baseline: "always predict the majority class".
    # Majority class is taken as the neutral class if it is in fact the largest
    # (it is, by construction of the 0.8% threshold); otherwise the true argmax.
    majority_class = int(np.argmax(support)) if total > 0 else NEUTRAL_CLASS
    base_acc = float(support[majority_class] / total) if total > 0 else 0.0
    # A constant classifier gets recall=1 on the majority class, 0 on every other
    # present class -> balanced accuracy = 1 / (#classes present).
    n_present = int(np.sum(support > 0))
    base_bal_acc = float(1.0 / n_present) if n_present > 0 else 0.0

    return {
        "n_samples": n,
        # Raw accuracy retained for backward-compat but explicitly de-emphasized.
        "accuracy": accuracy,
        "accuracy_note": "RAW accuracy -- base-rate INFLATED by the dominant "
                         "'neutral' class; NOT the headline metric. Use "
                         "balanced_accuracy / macro_f1 / mcc instead.",
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": f1_macro,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "mcc": mcc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES[:n_classes]),
        "support": support.tolist(),
        "per_class": per_class,
        "baseline_majority": {
            "majority_class": majority_class,
            "majority_class_name": CLASS_NAMES[majority_class]
                                   if majority_class < len(CLASS_NAMES) else str(majority_class),
            "accuracy": base_acc,
            "balanced_accuracy": base_bal_acc,
            "note": "Trivial 'always predict majority (neutral)' classifier, "
                    "shown side-by-side so the model is judged against the base rate.",
        },
        "beats_baseline": bool(balanced_accuracy > base_bal_acc),
        "accuracy_vs_baseline_delta": float(accuracy - base_acc),
        "balanced_accuracy_vs_baseline_delta": float(balanced_accuracy - base_bal_acc),
    }


def _log_imbalanced_report(rep: dict, label: str, log=logger):
    """Pretty-log an imbalance-aware report block."""
    log.info(f"  [{label}] imbalance-aware metrics (n={rep['n_samples']}):")
    log.info(f"    raw accuracy (INFLATED): {rep['accuracy']:.4f}  "
             f"| majority-baseline acc: {rep['baseline_majority']['accuracy']:.4f}")
    log.info(f"    balanced accuracy:       {rep['balanced_accuracy']:.4f}  "
             f"| baseline bal-acc:      {rep['baseline_majority']['balanced_accuracy']:.4f}  "
             f"| beats_baseline={rep['beats_baseline']}")
    log.info(f"    macro-F1: {rep['macro_f1']:.4f}  |  MCC: {rep['mcc']:.4f}")
    for name in CLASS_NAMES:
        if name in rep["per_class"]:
            pc = rep["per_class"][name]
            log.info(f"    {name:>7}: P={pc['precision']:.3f} R={pc['recall']:.3f} "
                     f"F1={pc['f1']:.3f} (support={pc['support']})")
    log.info(f"    confusion matrix (rows=true {CLASS_NAMES}, cols=pred): {rep['confusion_matrix']}")


# ---------------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------------

class AttentionLayer(tf.keras.layers.Layer):
    """Simple temporal attention for LSTM outputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        # x shape: (batch, seq_len, features)
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(x * a, axis=1)
        return output


def build_model(
    seq_length: int,
    n_features: int,
    lstm_units: int = 64,
    hidden_units: list = None,
    dropout_rate: float = 0.3,
    horizons: list = None,
    learning_rate: float = 1e-3,
):
    """
    Build multi-output LSTM + Attention model.

    Outputs per horizon:
    - direction: softmax (3 classes: down/flat/up)
    - magnitude: linear regression

    Optimized for Metal GPU:
    - Uses standard LSTM (not CuDNN-only ops)
    - Avoids ops not supported on Metal
    - Batch norm instead of layer norm for GPU efficiency
    """
    if hidden_units is None:
        hidden_units = [128, 64, 32]
    if horizons is None:
        horizons = [1, 6, 12, 24]

    # Input
    inputs = tf.keras.Input(shape=(seq_length, n_features), name="price_sequence")

    # LSTM backbone
    x = tf.keras.layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm_1",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_lstm1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="drop_lstm1")(x)

    x = tf.keras.layers.LSTM(
        lstm_units // 2,
        return_sequences=True,
        name="lstm_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_lstm2")(x)

    # Attention pooling
    attended = AttentionLayer(name="attention")(x)

    # Shared dense trunk
    trunk = attended
    for i, units in enumerate(hidden_units):
        trunk = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(trunk)
        trunk = tf.keras.layers.BatchNormalization(name=f"bn_dense_{i}")(trunk)
        trunk = tf.keras.layers.Dropout(dropout_rate * 0.5, name=f"drop_dense_{i}")(trunk)

    # Multi-horizon output heads
    outputs = {}
    losses = {}
    loss_weights = {}
    metrics_dict = {}

    for h in horizons:
        # Direction head (classification)
        dir_head = tf.keras.layers.Dense(32, activation="relu", name=f"dir_hidden_{h}h")(trunk)
        dir_out = tf.keras.layers.Dense(3, activation="softmax", name=f"dir_{h}h")(dir_head)
        outputs[f"dir_{h}h"] = dir_out
        losses[f"dir_{h}h"] = "sparse_categorical_crossentropy"
        loss_weights[f"dir_{h}h"] = 1.0
        metrics_dict[f"dir_{h}h"] = "accuracy"

        # Magnitude head (regression)
        mag_head = tf.keras.layers.Dense(32, activation="relu", name=f"mag_hidden_{h}h")(trunk)
        mag_out = tf.keras.layers.Dense(1, activation="linear", name=f"mag_{h}h")(mag_head)
        outputs[f"mag_{h}h"] = mag_out
        losses[f"mag_{h}h"] = "huber"
        loss_weights[f"mag_{h}h"] = 0.5

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CryptoLSTM")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={k: v for k, v in metrics_dict.items()},
    )

    return model


# ---------------------------------------------------------------
# TRAINING PIPELINE
# ---------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    config=None,
    model_name: str = "xrp_lstm_v1",
    verbose: int = 1,
):
    """
    Full training pipeline:
    1. Prepare data (temporal split, no future leakage)
    2. Build model
    3. Train with callbacks (early stopping, LR reduction)
    4. Evaluate on held-out test set
    5. Save model + scaler + metadata

    Returns: (model, history, test_metrics, scaler)
    """
    from config import TFModelConfig
    if config is None:
        config = TFModelConfig()

    has_gpu = setup_metal_gpu()
    device = "/GPU:0" if has_gpu else "/CPU:0"

    logger.info(f"Training on device: {device}")
    logger.info(f"Model: {model_name}")

    # -- Prepare sequences --
    horizons = config.prediction_horizons
    X, y_dir, y_mag, scaler, used_features = prepare_sequences(
        df,
        FEATURE_COLUMNS,
        seq_length=config.sequence_length,
        horizons=horizons,
        threshold_pct=config.label_threshold_pct,
        fit_scaler=True,
    )

    n_samples, seq_len, n_features = X.shape
    logger.info(f"Data: {n_samples} samples, {seq_len} timesteps, {n_features} features")

    # -- Temporal split (respecting time order) --
    test_size = int(n_samples * config.test_split)
    val_size = int(n_samples * config.val_split)
    train_size = n_samples - val_size - test_size

    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]

    y_train = {}
    y_val = {}
    y_test = {}
    for h in horizons:
        y_train[f"dir_{h}h"] = y_dir[h][:train_size]
        y_train[f"mag_{h}h"] = y_mag[h][:train_size]
        y_val[f"dir_{h}h"] = y_dir[h][train_size:train_size + val_size]
        y_val[f"mag_{h}h"] = y_mag[h][train_size:train_size + val_size]
        y_test[f"dir_{h}h"] = y_dir[h][train_size + val_size:]
        y_test[f"mag_{h}h"] = y_mag[h][train_size + val_size:]

    logger.info(f"Split: train={train_size}, val={val_size}, test={test_size}")

    # Label distribution + class weights
    class_weight_per_horizon = {}
    for h in horizons:
        counts = np.bincount(y_train[f"dir_{h}h"], minlength=3)
        total = counts.sum()
        logger.info(
            f"  {h}h labels: down={counts[0]} ({counts[0]/total:.0%}), "
            f"flat={counts[1]} ({counts[1]/total:.0%}), "
            f"up={counts[2]} ({counts[2]/total:.0%})"
        )
        # Compute balanced class weights: n_samples / (n_classes * count)
        n_classes = 3
        weights = {}
        for cls_id in range(n_classes):
            if counts[cls_id] > 0:
                weights[cls_id] = float(total / (n_classes * counts[cls_id]))
            else:
                weights[cls_id] = 1.0
        class_weight_per_horizon[h] = weights
        logger.info(f"  {h}h class weights: {weights}")

    # Build sample weights from the shortest-horizon direction labels
    # TF class_weight doesn't work well with multi-output, so use sample_weight
    primary_h = horizons[0]  # 1h
    primary_weights = class_weight_per_horizon[primary_h]
    sample_weights = np.array(
        [primary_weights.get(int(lbl), 1.0) for lbl in y_train[f"dir_{primary_h}h"]],
        dtype=np.float32,
    )

    # -- Build TF Dataset (optimized for Metal + multicore) --
    # Use tf.data.AUTOTUNE for automatic parallelism tuning
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights))
    train_ds = train_ds.cache()  # Cache in memory for faster epochs 2+
    train_ds = train_ds.shuffle(buffer_size=min(10000, train_size))
    train_ds = train_ds.batch(config.batch_size)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(config.batch_size)
    val_ds = val_ds.prefetch(AUTOTUNE)

    # -- Build model --
    with tf.device(device):
        model = build_model(
            seq_length=seq_len,
            n_features=n_features,
            lstm_units=config.lstm_units,
            hidden_units=config.hidden_units,
            dropout_rate=config.dropout_rate,
            horizons=horizons,
            learning_rate=config.learning_rate,
        )

    model.summary(print_fn=logger.info)
    logger.info(f"Total params: {model.count_params():,}")

    # -- Callbacks --
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_lr,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, f"{model_name}_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.model_dir, f"{model_name}_training_log.csv"),
        ),
    ]

    # -- Train --
    logger.info(f"Starting training: {config.epochs} epochs, batch_size={config.batch_size}")
    start_time = time.time()

    with tf.device(device):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # -- Evaluate on test set --
    logger.info("Evaluating on test set...")
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(config.batch_size)

    test_results = model.evaluate(test_ds, return_dict=True, verbose=0)

    test_metrics = {
        "test_loss": float(test_results.get("loss", 0)),
        "training_time_seconds": train_time,
        "epochs_trained": len(history.history.get("loss", [])),
        "n_params": model.count_params(),
        "device": device,
        "features_used": used_features,
        "n_features": n_features,
    }

    # -- Imbalance-aware classification report per horizon --
    # The 0.8% move threshold makes "neutral" dominate, so raw accuracy is
    # base-rate-inflated. Compute honest metrics from the confusion matrix.
    test_predictions = model.predict(test_ds, verbose=0)
    test_metrics["classification_report"] = {}

    for h in horizons:
        dir_key = f"dir_{h}h_accuracy"
        if dir_key in test_results:
            # Kept for backward-compat, but flagged as inflated below.
            test_metrics[f"test_accuracy_{h}h"] = float(test_results[dir_key])

        mag_key = f"mag_{h}h_loss"
        if mag_key in test_results:
            test_metrics[f"test_mag_loss_{h}h"] = float(test_results[mag_key])

        out_key = f"dir_{h}h"
        if out_key in test_predictions:
            y_pred_h = np.argmax(np.asarray(test_predictions[out_key]), axis=1)
            y_true_h = np.asarray(y_test[out_key])
            rep = classification_report_imbalanced(y_true_h, y_pred_h, n_classes=3)
            test_metrics["classification_report"][f"{h}h"] = rep
            # Surface the honest headline metrics as flat fields too.
            test_metrics[f"test_balanced_accuracy_{h}h"] = rep["balanced_accuracy"]
            test_metrics[f"test_macro_f1_{h}h"] = rep["macro_f1"]
            test_metrics[f"test_mcc_{h}h"] = rep["mcc"]
            test_metrics[f"baseline_accuracy_{h}h"] = rep["baseline_majority"]["accuracy"]
            _log_imbalanced_report(rep, f"test {h}h")

    # -- Save model + artifacts --
    model_path = os.path.join(config.model_dir, f"{model_name}.keras")
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")

    # Save scaler
    import joblib
    scaler_path = os.path.join(config.model_dir, f"{model_name}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save metadata
    meta = {
        "model_name": model_name,
        "architecture": "LSTM_Attention_MultiHead",
        "sequence_length": seq_len,
        "n_features": n_features,
        "features": used_features,
        "horizons": horizons,
        "threshold_pct": config.label_threshold_pct,
        "test_metrics": test_metrics,
        "training_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "lstm_units": config.lstm_units,
            "hidden_units": config.hidden_units,
            "dropout_rate": config.dropout_rate,
        },
    }
    meta_path = os.path.join(config.model_dir, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info(f"Artifacts saved to: {config.model_dir}/")

    return model, history, test_metrics, scaler


# ---------------------------------------------------------------
# INFERENCE (for live prediction)
# ---------------------------------------------------------------

def load_model_for_inference(model_dir: str, model_name: str = "xrp_lstm_v1"):
    """Load a trained model + scaler for live prediction."""
    import joblib

    model_path = os.path.join(model_dir, f"{model_name}.keras")
    scaler_path = os.path.join(model_dir, f"{model_name}_scaler.joblib")
    meta_path = os.path.join(model_dir, f"{model_name}_meta.json")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"AttentionLayer": AttentionLayer},
    )
    scaler = joblib.load(scaler_path)
    with open(meta_path) as f:
        meta = json.load(f)

    return model, scaler, meta


def predict(
    model,
    scaler,
    df: pd.DataFrame,
    feature_cols: list,
    seq_length: int = 60,
    horizons: list = None,
):
    """
    Run inference on the latest data.
    Returns prediction dict with direction probabilities and magnitude.

    IMPORTANT: Only uses the last seq_length bars -- no future data.
    """
    if horizons is None:
        horizons = [1, 6, 12, 24]

    available = [c for c in feature_cols if c in df.columns]
    data = df[available].tail(seq_length).copy()
    data = data.ffill().bfill()

    if len(data) < seq_length:
        logger.warning(f"Not enough data for prediction: {len(data)} < {seq_length}")
        return None

    scaled = scaler.transform(data.values)
    X = scaled.reshape(1, seq_length, len(available))

    predictions = model.predict(X, verbose=0)

    result = {}
    for h in horizons:
        dir_key = f"dir_{h}h"
        mag_key = f"mag_{h}h"

        if dir_key in predictions:
            probs = predictions[dir_key][0]
            direction = ["down", "flat", "up"][np.argmax(probs)]
            result[f"{h}h"] = {
                "direction": direction,
                "confidence": float(np.max(probs)),
                "probabilities": {
                    "down": float(probs[0]),
                    "flat": float(probs[1]),
                    "up": float(probs[2]),
                },
            }

        if mag_key in predictions:
            result[f"{h}h"]["magnitude_pct"] = float(predictions[mag_key][0][0])

    return result


# ---------------------------------------------------------------
# WALK-FORWARD TRAINING (for robust evaluation)
# ---------------------------------------------------------------

def walk_forward_train(
    df: pd.DataFrame,
    config=None,
    train_days: int = 60,
    val_days: int = 15,
    test_days: int = 15,
    model_name_prefix: str = "xrp_wf",
):
    """
    Walk-forward model training and evaluation.
    Trains separate models on rolling windows, evaluates on out-of-sample.
    Returns aggregated metrics across all folds.
    """
    from config import TFModelConfig
    if config is None:
        config = TFModelConfig()

    # Estimate candles per day (from timestamp diffs)
    if "timestamp" in df.columns:
        tf_ms = df["timestamp"].diff().median()
        cpd = int(86_400_000 / tf_ms)
    else:
        cpd = 24  # assume hourly

    train_bars = train_days * cpd
    val_bars = val_days * cpd
    test_bars = test_days * cpd
    fold_size = train_bars + val_bars + test_bars
    step_size = test_bars

    fold_results = []
    fold_idx = 0
    start = 0

    while start + fold_size <= len(df):
        fold_idx += 1
        logger.info(f"\n--- Walk-Forward Fold {fold_idx} ---")

        train_df = df.iloc[start:start + train_bars].copy()
        val_df = df.iloc[start + train_bars:start + train_bars + val_bars].copy()
        test_df = df.iloc[start + train_bars + val_bars:start + fold_size].copy()

        # Train on this fold's data
        combined_train = pd.concat([train_df, val_df], ignore_index=True)

        try:
            model, history, test_metrics, fold_scaler = train_model(
                combined_train,
                config=config,
                model_name=f"{model_name_prefix}_fold{fold_idx}",
                verbose=0,
            )

            # Evaluate on true OOS
            X_test, y_dir_test, y_mag_test, _, _ = prepare_sequences(
                test_df,
                FEATURE_COLUMNS,
                seq_length=config.sequence_length,
                horizons=config.prediction_horizons,
                threshold_pct=config.label_threshold_pct,
                scaler=fold_scaler,
                fit_scaler=False,
            )

            if len(X_test) > 0:
                y_test_dict = {}
                for h in config.prediction_horizons:
                    y_test_dict[f"dir_{h}h"] = y_dir_test[h]
                    y_test_dict[f"mag_{h}h"] = y_mag_test[h]

                test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test_dict))
                test_ds = test_ds.batch(config.batch_size)
                oos_results = model.evaluate(test_ds, return_dict=True, verbose=0)
                oos_predictions = model.predict(test_ds, verbose=0)

                fold_result = {"fold": fold_idx, "oos_loss": float(oos_results.get("loss", 0))}
                for h in config.prediction_horizons:
                    acc_key = f"dir_{h}h_accuracy"
                    if acc_key in oos_results:
                        # Raw accuracy kept for backward-compat (base-rate inflated).
                        fold_result[f"oos_accuracy_{h}h"] = float(oos_results[acc_key])

                    out_key = f"dir_{h}h"
                    if out_key in oos_predictions:
                        y_pred_h = np.argmax(np.asarray(oos_predictions[out_key]), axis=1)
                        rep = classification_report_imbalanced(
                            y_dir_test[h], y_pred_h, n_classes=3,
                        )
                        # Honest, imbalance-aware OOS headline metrics + baseline.
                        fold_result[f"oos_balanced_accuracy_{h}h"] = rep["balanced_accuracy"]
                        fold_result[f"oos_macro_f1_{h}h"] = rep["macro_f1"]
                        fold_result[f"oos_mcc_{h}h"] = rep["mcc"]
                        fold_result[f"oos_baseline_accuracy_{h}h"] = \
                            rep["baseline_majority"]["accuracy"]
                        fold_result[f"oos_beats_baseline_{h}h"] = rep["beats_baseline"]
                        _log_imbalanced_report(rep, f"fold{fold_idx} OOS {h}h")

                fold_results.append(fold_result)
                logger.info(f"  Fold {fold_idx} OOS: {fold_result}")

        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")

        start += step_size

        # Clean up GPU memory between folds
        tf.keras.backend.clear_session()

    # Aggregate
    if fold_results:
        agg = {"n_folds": len(fold_results)}
        for key in fold_results[0]:
            if key != "fold":
                vals = [r[key] for r in fold_results if key in r]
                agg[f"mean_{key}"] = float(np.mean(vals))
                agg[f"std_{key}"] = float(np.std(vals))
        logger.info(f"\nWalk-Forward Aggregate: {agg}")
        return fold_results, agg

    return [], {}
