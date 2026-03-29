"""
CryptoResearchLab — Pattern Clustering Engine
K-Means clustering on normalized candlestick windows for pattern recognition.

Architecture:
  1. Extract sliding windows of N candles (OHLCV)
  2. Normalize each window (scale-invariant via ATR or range normalization)
  3. Flatten into feature vectors
  4. K-Means clustering → discover recurring patterns (analogous to classic chart
     patterns: double top, HCH, channels, wedges, etc.)
  5. For each cluster: measure forward return distribution (next M bars)
  6. Assign directional probabilities per cluster

Training data: historical data from XRP inception → 1 year ago
Test data: last 1 year (matches evolution backtest period)
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field, asdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger("pattern_clustering")


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class ClusterProfile:
    """Statistical profile of a single pattern cluster."""
    cluster_id: int
    n_samples: int
    mean_forward_return: float      # Mean return over forward_bars
    std_forward_return: float
    prob_up: float                  # P(return > threshold)
    prob_down: float                # P(return < -threshold)
    prob_neutral: float             # P(|return| <= threshold)
    median_forward_return: float
    sharpe_forward: float           # mean/std of forward returns
    win_rate: float                 # % of forward returns > 0
    avg_up_return: float            # Mean of positive forward returns
    avg_down_return: float          # Mean of negative forward returns


@dataclass
class PatternClusterModel:
    """Complete clustering model with profiles."""
    n_clusters: int
    window_size: int
    forward_bars: int
    direction_threshold: float
    timeframe: str
    feature_names: list
    cluster_profiles: list          # List[ClusterProfile]
    training_samples: int
    training_date_range: str


# ═══════════════════════════════════════════════════════════════
# WINDOW EXTRACTION & NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def extract_candle_windows(
    df: pd.DataFrame,
    window_size: int = 10,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows of OHLCV candles, normalized per-window.

    Normalization: each window is divided by its first candle's close price
    and scaled by ATR to be scale-invariant. This makes a $2 XRP pattern
    comparable to a $0.30 XRP pattern from 2018.

    Returns:
        features: (n_windows, window_size * n_features) flattened feature matrix
        indices: (n_windows,) bar indices of window ends (for forward return calc)
    """
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    n = len(df)
    if n < window_size + 1:
        return np.array([]), np.array([])

    features_list = []
    indices_list = []

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    v = df["volume"].values

    for start in range(0, n - window_size, step):
        end = start + window_size

        # Reference price: first close of window
        ref_price = c[start]
        if ref_price <= 0 or np.isnan(ref_price):
            continue

        # Window range for normalization (avoid div by 0)
        window_range = np.max(h[start:end]) - np.min(l[start:end])
        if window_range <= 0 or np.isnan(window_range):
            continue

        # Normalize OHLC by reference price (relative position)
        norm_o = (o[start:end] - ref_price) / window_range
        norm_h = (h[start:end] - ref_price) / window_range
        norm_l = (l[start:end] - ref_price) / window_range
        norm_c = (c[start:end] - ref_price) / window_range

        # Volume: normalize to mean of window
        v_window = v[start:end]
        v_mean = np.mean(v_window)
        norm_v = v_window / v_mean if v_mean > 0 else np.ones(window_size)

        # Derived features per candle
        body = norm_c - norm_o                    # Body direction & size
        upper_wick = norm_h - np.maximum(norm_c, norm_o)   # Upper shadow
        lower_wick = np.minimum(norm_c, norm_o) - norm_l   # Lower shadow
        bar_range = norm_h - norm_l                # Total candle range

        # Close-to-close returns within window
        returns = np.diff(c[start:end]) / c[start:end - 1]
        returns = np.concatenate([[0.0], returns])  # Pad first

        # Stack: [norm_o, norm_h, norm_l, norm_c, norm_v, body, upper_wick,
        #         lower_wick, bar_range, returns] = 10 features per candle
        window_features = np.column_stack([
            norm_o, norm_h, norm_l, norm_c, norm_v,
            body, upper_wick, lower_wick, bar_range, returns,
        ])

        # Check for NaN/Inf
        if not np.all(np.isfinite(window_features)):
            continue

        features_list.append(window_features.flatten())
        indices_list.append(end - 1)  # Last bar index of window

    if not features_list:
        return np.array([]), np.array([])

    return np.array(features_list), np.array(indices_list)


def compute_forward_returns(
    df: pd.DataFrame,
    indices: np.ndarray,
    forward_bars: int = 5,
) -> np.ndarray:
    """
    Compute forward returns for each window endpoint.
    Return = (close[i + forward_bars] - close[i]) / close[i]
    """
    closes = df["close"].values
    n = len(closes)
    returns = np.full(len(indices), np.nan)

    for j, idx in enumerate(indices):
        future_idx = idx + forward_bars
        if future_idx < n:
            if closes[idx] > 0:
                returns[j] = (closes[future_idx] - closes[idx]) / closes[idx]

    return returns


# ═══════════════════════════════════════════════════════════════
# CLUSTERING TRAINING
# ═══════════════════════════════════════════════════════════════

def train_pattern_clusters(
    df: pd.DataFrame,
    n_clusters: int = 50,
    window_size: int = 10,
    forward_bars: int = 5,
    direction_threshold: float = 0.005,  # 0.5% threshold for up/down
    step: int = 1,
    random_state: int = 42,
) -> tuple[MiniBatchKMeans, StandardScaler, PatternClusterModel]:
    """
    Train K-Means clustering on candle pattern windows.

    Args:
        df: OHLCV DataFrame (training data: inception → 1yr ago)
        n_clusters: Number of pattern clusters
        window_size: Candles per window
        forward_bars: Bars ahead for forward return measurement
        direction_threshold: Return threshold for up/down classification
        step: Sliding window step size
        random_state: Seed for reproducibility

    Returns:
        kmeans: Trained MiniBatchKMeans model
        scaler: Fitted StandardScaler for feature normalization
        model_info: PatternClusterModel with cluster profiles
    """
    logger.info(f"Extracting candle windows (size={window_size}, step={step})...")
    features, indices = extract_candle_windows(df, window_size=window_size, step=step)

    if len(features) < n_clusters * 10:
        raise ValueError(
            f"Not enough windows ({len(features)}) for {n_clusters} clusters. "
            f"Need at least {n_clusters * 10}."
        )

    logger.info(f"Extracted {len(features)} windows, feature dim={features.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means clustering
    logger.info(f"Training MiniBatchKMeans with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(1024, len(features_scaled)),
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(features_scaled)

    # Compute forward returns
    logger.info(f"Computing forward returns ({forward_bars} bars ahead)...")
    forward_returns = compute_forward_returns(df, indices, forward_bars=forward_bars)

    # Build cluster profiles
    logger.info("Building cluster profiles...")
    profiles = []
    for cid in range(n_clusters):
        mask = (labels == cid) & np.isfinite(forward_returns)
        rets = forward_returns[mask]

        if len(rets) < 5:
            profiles.append(ClusterProfile(
                cluster_id=cid, n_samples=len(rets),
                mean_forward_return=0.0, std_forward_return=1.0,
                prob_up=0.33, prob_down=0.33, prob_neutral=0.34,
                median_forward_return=0.0, sharpe_forward=0.0,
                win_rate=0.5, avg_up_return=0.0, avg_down_return=0.0,
            ))
            continue

        mean_ret = float(np.mean(rets))
        std_ret = float(np.std(rets)) if np.std(rets) > 0 else 1e-6
        prob_up = float(np.mean(rets > direction_threshold))
        prob_down = float(np.mean(rets < -direction_threshold))
        prob_neutral = 1.0 - prob_up - prob_down
        up_rets = rets[rets > 0]
        down_rets = rets[rets < 0]

        profiles.append(ClusterProfile(
            cluster_id=cid,
            n_samples=int(mask.sum()),
            mean_forward_return=mean_ret,
            std_forward_return=std_ret,
            prob_up=prob_up,
            prob_down=prob_down,
            prob_neutral=prob_neutral,
            median_forward_return=float(np.median(rets)),
            sharpe_forward=mean_ret / std_ret,
            win_rate=float(np.mean(rets > 0)),
            avg_up_return=float(np.mean(up_rets)) if len(up_rets) > 0 else 0.0,
            avg_down_return=float(np.mean(down_rets)) if len(down_rets) > 0 else 0.0,
        ))

    # Feature names for reference
    feature_names = [
        f"{feat}_bar{i}" for i in range(window_size)
        for feat in ["norm_o", "norm_h", "norm_l", "norm_c", "norm_v",
                      "body", "upper_wick", "lower_wick", "bar_range", "returns"]
    ]

    ts_start = pd.Timestamp(df["timestamp"].iloc[0], unit="ms")
    ts_end = pd.Timestamp(df["timestamp"].iloc[-1], unit="ms")

    model_info = PatternClusterModel(
        n_clusters=n_clusters,
        window_size=window_size,
        forward_bars=forward_bars,
        direction_threshold=direction_threshold,
        timeframe="unknown",
        feature_names=feature_names,
        cluster_profiles=profiles,
        training_samples=len(features),
        training_date_range=f"{ts_start.date()} to {ts_end.date()}",
    )

    # Log top patterns
    actionable = [p for p in profiles if abs(p.sharpe_forward) > 0.3 and p.n_samples >= 20]
    actionable.sort(key=lambda p: abs(p.sharpe_forward), reverse=True)
    logger.info(f"Found {len(actionable)} actionable clusters (|Sharpe| > 0.3, n >= 20):")
    for p in actionable[:10]:
        direction = "LONG" if p.mean_forward_return > 0 else "SHORT"
        logger.info(
            f"  Cluster {p.cluster_id}: {direction} Sharpe={p.sharpe_forward:.3f}, "
            f"P(up)={p.prob_up:.2f}, P(down)={p.prob_down:.2f}, "
            f"n={p.n_samples}, mean_ret={p.mean_forward_return*100:.3f}%"
        )

    return kmeans, scaler, model_info


# ═══════════════════════════════════════════════════════════════
# INFERENCE — classify current bar's pattern
# ═══════════════════════════════════════════════════════════════

def classify_current_pattern(
    df: pd.DataFrame,
    bar_idx: int,
    window_size: int,
    kmeans: MiniBatchKMeans,
    scaler: StandardScaler,
) -> Optional[int]:
    """
    Classify the current window ending at bar_idx into a cluster.
    Returns cluster_id or None if insufficient data.
    """
    if bar_idx < window_size:
        return None

    start = bar_idx - window_size + 1
    end = bar_idx + 1

    o = df["open"].values[start:end]
    h = df["high"].values[start:end]
    l = df["low"].values[start:end]
    c = df["close"].values[start:end]
    v = df["volume"].values[start:end]

    ref_price = c[0]
    if ref_price <= 0 or np.isnan(ref_price):
        return None

    window_range = np.max(h) - np.min(l)
    if window_range <= 0 or np.isnan(window_range):
        return None

    norm_o = (o - ref_price) / window_range
    norm_h = (h - ref_price) / window_range
    norm_l = (l - ref_price) / window_range
    norm_c = (c - ref_price) / window_range

    v_mean = np.mean(v)
    norm_v = v / v_mean if v_mean > 0 else np.ones(window_size)

    body = norm_c - norm_o
    upper_wick = norm_h - np.maximum(norm_c, norm_o)
    lower_wick = np.minimum(norm_c, norm_o) - norm_l
    bar_range = norm_h - norm_l

    returns = np.diff(c) / c[:-1]
    returns = np.concatenate([[0.0], returns])

    features = np.column_stack([
        norm_o, norm_h, norm_l, norm_c, norm_v,
        body, upper_wick, lower_wick, bar_range, returns,
    ]).flatten()

    if not np.all(np.isfinite(features)):
        return None

    features_scaled = scaler.transform(features.reshape(1, -1))
    cluster_id = int(kmeans.predict(features_scaled)[0])

    return cluster_id


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def save_cluster_model(
    kmeans: MiniBatchKMeans,
    scaler: StandardScaler,
    model_info: PatternClusterModel,
    output_dir: str = "models",
    name_prefix: str = "pattern_clusters",
):
    """Save clustering model, scaler, and profiles to disk."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(kmeans, os.path.join(output_dir, f"{name_prefix}_kmeans.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, f"{name_prefix}_scaler.joblib"))

    # Convert profiles to dicts for JSON serialization
    info_dict = {
        "n_clusters": model_info.n_clusters,
        "window_size": model_info.window_size,
        "forward_bars": model_info.forward_bars,
        "direction_threshold": model_info.direction_threshold,
        "timeframe": model_info.timeframe,
        "training_samples": model_info.training_samples,
        "training_date_range": model_info.training_date_range,
        "cluster_profiles": [asdict(p) for p in model_info.cluster_profiles],
    }
    with open(os.path.join(output_dir, f"{name_prefix}_meta.json"), "w") as f:
        json.dump(info_dict, f, indent=2)

    logger.info(f"Saved cluster model to {output_dir}/{name_prefix}_*")


def load_cluster_model(
    model_dir: str = "models",
    name_prefix: str = "pattern_clusters",
) -> tuple[MiniBatchKMeans, StandardScaler, PatternClusterModel]:
    """Load clustering model from disk."""
    kmeans = joblib.load(os.path.join(model_dir, f"{name_prefix}_kmeans.joblib"))
    scaler = joblib.load(os.path.join(model_dir, f"{name_prefix}_scaler.joblib"))

    with open(os.path.join(model_dir, f"{name_prefix}_meta.json")) as f:
        info_dict = json.load(f)

    profiles = [ClusterProfile(**p) for p in info_dict["cluster_profiles"]]
    model_info = PatternClusterModel(
        n_clusters=info_dict["n_clusters"],
        window_size=info_dict["window_size"],
        forward_bars=info_dict["forward_bars"],
        direction_threshold=info_dict["direction_threshold"],
        timeframe=info_dict["timeframe"],
        feature_names=[],
        cluster_profiles=profiles,
        training_samples=info_dict["training_samples"],
        training_date_range=info_dict["training_date_range"],
    )

    logger.info(
        f"Loaded cluster model: {model_info.n_clusters} clusters, "
        f"window={model_info.window_size}, trained on {model_info.training_samples} samples"
    )
    return kmeans, scaler, model_info


# ═══════════════════════════════════════════════════════════════
# INCREMENTAL CLUSTER PROFILE UPDATE
# ═══════════════════════════════════════════════════════════════

def update_cluster_profiles(
    df_new: pd.DataFrame,
    kmeans: MiniBatchKMeans,
    scaler: StandardScaler,
    model_info: PatternClusterModel,
    decay: float = 0.85,
) -> PatternClusterModel:
    """
    Update cluster profiles with new data (test set or live deploy data).

    The K-Means centroids stay FROZEN — we don't retrain clustering.
    Instead, we:
      1. Classify new windows into existing clusters
      2. Compute forward returns on new data
      3. Blend old profiles with new observations using exponential decay:
         updated_stat = decay * old_stat + (1 - decay) * new_stat

    This keeps cluster behavior adaptive: if a pattern that was bullish
    in training becomes bearish in recent data, the profile gradually shifts.

    Args:
        df_new: New OHLCV data with features already built
        kmeans: Frozen K-Means model (centroids unchanged)
        scaler: Frozen StandardScaler
        model_info: Current PatternClusterModel with existing profiles
        decay: Blending weight for old profiles (0.85 = 85% old, 15% new)

    Returns:
        Updated PatternClusterModel with blended profiles
    """
    window_size = model_info.window_size
    forward_bars = model_info.forward_bars
    direction_threshold = model_info.direction_threshold

    # Extract & classify windows from new data
    features, indices = extract_candle_windows(df_new, window_size=window_size, step=1)
    if len(features) < 50:
        logger.warning(f"Too few windows in new data ({len(features)}), skipping profile update")
        return model_info

    features_scaled = scaler.transform(features)
    labels = kmeans.predict(features_scaled)

    # Compute forward returns on new data
    forward_returns = compute_forward_returns(df_new, indices, forward_bars=forward_bars)

    # Build new per-cluster stats
    old_profiles = {p.cluster_id: p for p in model_info.cluster_profiles}
    updated_profiles = []

    for cid in range(model_info.n_clusters):
        old_p = old_profiles.get(cid)
        mask = (labels == cid) & np.isfinite(forward_returns)
        rets = forward_returns[mask]

        if len(rets) < 3 or old_p is None:
            # Not enough new data for this cluster — keep old profile
            updated_profiles.append(old_p if old_p else ClusterProfile(
                cluster_id=cid, n_samples=0,
                mean_forward_return=0.0, std_forward_return=1.0,
                prob_up=0.33, prob_down=0.33, prob_neutral=0.34,
                median_forward_return=0.0, sharpe_forward=0.0,
                win_rate=0.5, avg_up_return=0.0, avg_down_return=0.0,
            ))
            continue

        # Compute new stats
        new_mean = float(np.mean(rets))
        new_std = float(np.std(rets)) if np.std(rets) > 0 else 1e-6
        new_prob_up = float(np.mean(rets > direction_threshold))
        new_prob_down = float(np.mean(rets < -direction_threshold))
        new_prob_neutral = 1.0 - new_prob_up - new_prob_down
        new_median = float(np.median(rets))
        new_win_rate = float(np.mean(rets > 0))
        up_rets = rets[rets > 0]
        down_rets = rets[rets < 0]
        new_avg_up = float(np.mean(up_rets)) if len(up_rets) > 0 else 0.0
        new_avg_down = float(np.mean(down_rets)) if len(down_rets) > 0 else 0.0

        # Exponential blend: updated = decay * old + (1-decay) * new
        d = decay
        nd = 1.0 - decay
        blended_mean = d * old_p.mean_forward_return + nd * new_mean
        blended_std = d * old_p.std_forward_return + nd * new_std
        blended_sharpe = blended_mean / blended_std if blended_std > 0 else 0.0

        updated_profiles.append(ClusterProfile(
            cluster_id=cid,
            n_samples=old_p.n_samples + int(mask.sum()),
            mean_forward_return=blended_mean,
            std_forward_return=blended_std,
            prob_up=d * old_p.prob_up + nd * new_prob_up,
            prob_down=d * old_p.prob_down + nd * new_prob_down,
            prob_neutral=d * old_p.prob_neutral + nd * new_prob_neutral,
            median_forward_return=d * old_p.median_forward_return + nd * new_median,
            sharpe_forward=blended_sharpe,
            win_rate=d * old_p.win_rate + nd * new_win_rate,
            avg_up_return=d * old_p.avg_up_return + nd * new_avg_up,
            avg_down_return=d * old_p.avg_down_return + nd * new_avg_down,
        ))

    n_updated = sum(1 for p in updated_profiles if p.n_samples > old_profiles.get(p.cluster_id, ClusterProfile(
        cluster_id=0, n_samples=0, mean_forward_return=0, std_forward_return=1,
        prob_up=0.33, prob_down=0.33, prob_neutral=0.34, median_forward_return=0,
        sharpe_forward=0, win_rate=0.5, avg_up_return=0, avg_down_return=0
    )).n_samples)

    logger.info(f"Updated {n_updated}/{model_info.n_clusters} cluster profiles "
                f"with {len(features)} new windows (decay={decay})")

    ts_end = pd.Timestamp(df_new["timestamp"].iloc[-1], unit="ms")
    return PatternClusterModel(
        n_clusters=model_info.n_clusters,
        window_size=model_info.window_size,
        forward_bars=model_info.forward_bars,
        direction_threshold=model_info.direction_threshold,
        timeframe=model_info.timeframe,
        feature_names=model_info.feature_names,
        cluster_profiles=updated_profiles,
        training_samples=model_info.training_samples + len(features),
        training_date_range=f"{model_info.training_date_range.split(' to ')[0]} to {ts_end.date()}",
    )
