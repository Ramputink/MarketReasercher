"""
CryptoResearchLab — Multi-Algorithm Clustering Engine

Trains and manages multiple clustering variants that compete in evolution:
  1. K-Means (flat): k=8, 20, 50 — increasing granularity
  2. Hierarchical 2-level (tree): macro direction (3) × micro shape (N)
     → produces a genealogical tree where effective sub-branches propagate
  3. BisectingKMeans (divisive hierarchy): top-down splits, naturally creates
     a tree where each cluster is a refinement of its parent

Each variant produces:
  - Cluster labels per window
  - ClusterProfile per cluster (forward return statistics)
  - A unique prefix for model persistence

The evolution system treats `cluster_variant` as an evolvable parameter,
so the genetic algorithm discovers which granularity + algorithm works best.

Variants naming convention:
  "kmeans_8"      → MiniBatchKMeans with 8 clusters
  "kmeans_20"     → MiniBatchKMeans with 20 clusters
  "kmeans_50"     → MiniBatchKMeans with 50 clusters (baseline)
  "hier_20"       → 2-level hierarchical: 3 macro × ~7 micro
  "hier_50"       → 2-level hierarchical: 3 macro × ~17 micro
  "bisect_20"     → BisectingKMeans with 20 clusters (divisive tree)
  "bisect_50"     → BisectingKMeans with 50 clusters (divisive tree)
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import asdict

from sklearn.cluster import MiniBatchKMeans, BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
import joblib

from engine.pattern_clustering import (
    extract_candle_windows,
    compute_forward_returns,
    ClusterProfile,
    PatternClusterModel,
    save_cluster_model,
)

logger = logging.getLogger("multi_clustering")

# ═══════════════════════════════════════════════════════════════
# VARIANT REGISTRY
# ═══════════════════════════════════════════════════════════════

CLUSTER_VARIANTS = {
    "kmeans_8":   {"algo": "kmeans",   "n_clusters": 8},
    "kmeans_20":  {"algo": "kmeans",   "n_clusters": 20},
    "kmeans_50":  {"algo": "kmeans",   "n_clusters": 50},
    "hier_20":    {"algo": "hier",     "n_clusters": 20, "n_macro": 3},
    "hier_50":    {"algo": "hier",     "n_clusters": 50, "n_macro": 3},
    "bisect_20":  {"algo": "bisect",   "n_clusters": 20},
    "bisect_50":  {"algo": "bisect",   "n_clusters": 50},
}

ALL_VARIANT_IDS = list(CLUSTER_VARIANTS.keys())


# ═══════════════════════════════════════════════════════════════
# FLAT K-MEANS (existing approach, generalized)
# ═══════════════════════════════════════════════════════════════

def _train_flat_kmeans(
    features_scaled: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> MiniBatchKMeans:
    """Train MiniBatchKMeans and return (model, labels)."""
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(1024, len(features_scaled)),
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(features_scaled)
    return kmeans, labels


# ═══════════════════════════════════════════════════════════════
# BISECTING K-MEANS (divisive hierarchy — top-down tree)
# ═══════════════════════════════════════════════════════════════

def _train_bisecting_kmeans(
    features_scaled: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple:
    """
    BisectingKMeans: starts with ALL data as one cluster, repeatedly
    splits the biggest cluster into two. Produces a natural binary tree.
    Effective clusters = leaf nodes; parent structure is preserved internally.
    """
    bkmeans = BisectingKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=5,
        max_iter=300,
        bisecting_strategy="biggest_inertia",
    )
    labels = bkmeans.fit_predict(features_scaled)
    return bkmeans, labels


# ═══════════════════════════════════════════════════════════════
# HIERARCHICAL 2-LEVEL (macro direction → micro pattern shape)
# ═══════════════════════════════════════════════════════════════

class HierarchicalClusterModel:
    """
    2-level clustering tree:
      Level 1 (macro): Direction-based (bullish / neutral / bearish) by forward returns
      Level 2 (micro): Shape-based sub-clustering within each macro group

    This creates a genealogical tree where:
      - Root splits into 3 direction branches
      - Each branch sub-splits into N/3 pattern-shape clusters
      - Total ~N clusters, but with hierarchical structure

    A "long slow uptrend" and "sharp bullish breakout" both land in the
    bullish macro group but get distinct micro clusters.
    """

    def __init__(self):
        self.macro_thresholds = None  # (low_thresh, high_thresh) for direction
        self.micro_models = {}        # {macro_id: MiniBatchKMeans}
        self.micro_scalers = {}       # {macro_id: StandardScaler} (reuse global scaler for now)
        self.n_macro = 3
        self.n_micro_per_macro = 0
        self.total_clusters = 0

    def fit_predict(self, features_scaled, forward_returns, n_clusters=20, n_macro=3):
        """
        Train the 2-level hierarchy.

        Args:
            features_scaled: (n_windows, n_features) normalized feature matrix
            forward_returns: (n_windows,) forward return per window
            n_clusters: Total desired clusters (split across macro groups)
            n_macro: Number of macro direction groups (always 3: down/neutral/up)
        """
        self.n_macro = n_macro
        valid = np.isfinite(forward_returns)
        fwd_valid = forward_returns[valid]
        feat_valid = features_scaled[valid]

        # Level 1: macro direction by forward return percentiles
        # Use percentile-based thresholds for balanced groups
        low_pct = np.percentile(fwd_valid, 33)
        high_pct = np.percentile(fwd_valid, 67)
        self.macro_thresholds = (float(low_pct), float(high_pct))

        macro_labels = np.full(len(forward_returns), -1, dtype=int)
        macro_labels[valid] = np.where(
            fwd_valid < low_pct, 0,                    # Bearish
            np.where(fwd_valid > high_pct, 2, 1)       # Bullish / Neutral
        )

        # Level 2: micro shape clustering within each macro group
        n_micro = max(2, n_clusters // n_macro)
        self.n_micro_per_macro = n_micro
        self.total_clusters = n_macro * n_micro

        composite_labels = np.full(len(forward_returns), -1, dtype=int)

        for macro_id in range(n_macro):
            mask = macro_labels == macro_id
            if mask.sum() < n_micro * 5:
                # Too few samples for this macro group — assign single cluster
                composite_labels[mask] = macro_id * n_micro
                self.micro_models[macro_id] = None
                continue

            micro_features = features_scaled[mask]
            micro_kmeans = MiniBatchKMeans(
                n_clusters=n_micro,
                random_state=42 + macro_id,
                batch_size=min(512, len(micro_features)),
                n_init=5,
                max_iter=200,
            )
            micro_labels = micro_kmeans.fit_predict(micro_features)
            self.micro_models[macro_id] = micro_kmeans

            # Composite label: macro_id * n_micro + micro_id
            composite_labels[mask] = macro_id * n_micro + micro_labels

        logger.info(
            f"Hierarchical clustering: {n_macro} macro × {n_micro} micro = "
            f"{self.total_clusters} total clusters"
        )

        # Log macro distribution
        for mid in range(n_macro):
            direction = ["BEARISH", "NEUTRAL", "BULLISH"][mid]
            count = (macro_labels == mid).sum()
            logger.info(f"  Macro {mid} ({direction}): {count} windows")

        return composite_labels

    def predict(self, features_scaled, forward_returns=None):
        """
        Classify new windows into the existing hierarchy.

        If forward_returns are provided, use them for macro assignment.
        If not (live/backtest mode), use NearestCentroid to infer macro group.
        """
        if forward_returns is not None and np.any(np.isfinite(forward_returns)):
            low_t, high_t = self.macro_thresholds
            valid = np.isfinite(forward_returns)
            macro_labels = np.full(len(features_scaled), 1, dtype=int)  # default neutral
            macro_labels[valid] = np.where(
                forward_returns[valid] < low_t, 0,
                np.where(forward_returns[valid] > high_t, 2, 1)
            )
        else:
            # Inference mode: assign to nearest macro centroid by feature distance
            macro_labels = self._infer_macro_from_features(features_scaled)

        composite_labels = np.full(len(features_scaled), -1, dtype=int)

        for macro_id in range(self.n_macro):
            mask = macro_labels == macro_id
            if mask.sum() == 0:
                continue

            micro_km = self.micro_models.get(macro_id)
            if micro_km is None:
                composite_labels[mask] = macro_id * self.n_micro_per_macro
            else:
                micro_labels = micro_km.predict(features_scaled[mask])
                composite_labels[mask] = macro_id * self.n_micro_per_macro + micro_labels

        return composite_labels

    def _infer_macro_from_features(self, features_scaled):
        """
        For inference without forward returns: build macro centroids from
        micro model centroids and assign by nearest centroid distance.
        """
        macro_centroids = []
        for macro_id in range(self.n_macro):
            micro_km = self.micro_models.get(macro_id)
            if micro_km is not None:
                # Average of all micro centroids = macro centroid
                macro_centroids.append(micro_km.cluster_centers_.mean(axis=0))
            else:
                # Fallback: zero vector (will be distant from everything)
                macro_centroids.append(np.zeros(features_scaled.shape[1]))

        macro_centroids = np.array(macro_centroids)
        # Assign each window to nearest macro centroid
        dists = np.linalg.norm(
            features_scaled[:, np.newaxis, :] - macro_centroids[np.newaxis, :, :],
            axis=2,
        )
        return np.argmin(dists, axis=1)


# ═══════════════════════════════════════════════════════════════
# UNIFIED VARIANT TRAINER
# ═══════════════════════════════════════════════════════════════

def train_variant(
    variant_id: str,
    df: pd.DataFrame,
    window_size: int = 10,
    forward_bars: int = 5,
    direction_threshold: float = 0.005,
    random_state: int = 42,
) -> tuple:
    """
    Train a single clustering variant.

    Returns:
        (cluster_model, scaler, labels, model_info, n_effective_clusters)

    The cluster_model can be MiniBatchKMeans, BisectingKMeans, or
    HierarchicalClusterModel — all have .predict() for inference.
    """
    if variant_id not in CLUSTER_VARIANTS:
        raise ValueError(f"Unknown variant: {variant_id}. Options: {ALL_VARIANT_IDS}")

    cfg = CLUSTER_VARIANTS[variant_id]
    algo = cfg["algo"]
    n_clusters = cfg["n_clusters"]

    logger.info(f"\n--- Training variant: {variant_id} (algo={algo}, k={n_clusters}) ---")

    # Extract windows
    features, indices = extract_candle_windows(df, window_size=window_size, step=1)
    if len(features) < n_clusters * 10:
        raise ValueError(f"Not enough windows ({len(features)}) for {n_clusters} clusters")

    # Scale
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Forward returns (needed for profiles and hierarchical macro labels)
    forward_returns = compute_forward_returns(df, indices, forward_bars=forward_bars)

    # ── Train clustering ──
    if algo == "kmeans":
        cluster_model, labels = _train_flat_kmeans(features_scaled, n_clusters, random_state)
        n_effective = n_clusters

    elif algo == "bisect":
        cluster_model, labels = _train_bisecting_kmeans(features_scaled, n_clusters, random_state)
        n_effective = n_clusters

    elif algo == "hier":
        n_macro = cfg.get("n_macro", 3)
        cluster_model = HierarchicalClusterModel()
        labels = cluster_model.fit_predict(features_scaled, forward_returns, n_clusters, n_macro)
        n_effective = cluster_model.total_clusters
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # ── Build cluster profiles ──
    profiles = _build_profiles(labels, forward_returns, n_effective, direction_threshold)

    # Log actionable clusters
    actionable = [p for p in profiles if abs(p.sharpe_forward) > 0.3 and p.n_samples >= 20]
    actionable.sort(key=lambda p: abs(p.sharpe_forward), reverse=True)
    logger.info(f"  {variant_id}: {len(actionable)} actionable clusters (|Sharpe| > 0.3)")
    for p in actionable[:5]:
        direction = "LONG" if p.mean_forward_return > 0 else "SHORT"
        logger.info(
            f"    Cluster {p.cluster_id}: {direction} Sharpe={p.sharpe_forward:.3f}, "
            f"P(up)={p.prob_up:.2f}, P(dn)={p.prob_down:.2f}, n={p.n_samples}"
        )

    ts_start = pd.Timestamp(df["timestamp"].iloc[0], unit="ms")
    ts_end = pd.Timestamp(df["timestamp"].iloc[-1], unit="ms")

    model_info = PatternClusterModel(
        n_clusters=n_effective,
        window_size=window_size,
        forward_bars=forward_bars,
        direction_threshold=direction_threshold,
        timeframe="unknown",
        feature_names=[],
        cluster_profiles=profiles,
        training_samples=len(features),
        training_date_range=f"{ts_start.date()} to {ts_end.date()}",
    )

    return cluster_model, scaler, labels, indices, model_info, n_effective


def _build_profiles(labels, forward_returns, n_clusters, direction_threshold):
    """Build ClusterProfile for each cluster."""
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

    return profiles


# ═══════════════════════════════════════════════════════════════
# PERSISTENCE (variant-aware)
# ═══════════════════════════════════════════════════════════════

def save_variant_model(
    variant_id: str,
    cluster_model,
    scaler: StandardScaler,
    model_info: PatternClusterModel,
    output_dir: str = "models",
):
    """Save a clustering variant to disk."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"cluster_{variant_id}_{model_info.timeframe}"

    # Save cluster model (type-specific)
    model_path = os.path.join(output_dir, f"{prefix}_model.joblib")
    joblib.dump(cluster_model, model_path)

    # Save scaler
    scaler_path = os.path.join(output_dir, f"{prefix}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save metadata + profiles
    info_dict = {
        "variant_id": variant_id,
        "algo": CLUSTER_VARIANTS[variant_id]["algo"],
        "n_clusters": model_info.n_clusters,
        "window_size": model_info.window_size,
        "forward_bars": model_info.forward_bars,
        "direction_threshold": model_info.direction_threshold,
        "timeframe": model_info.timeframe,
        "training_samples": model_info.training_samples,
        "training_date_range": model_info.training_date_range,
        "cluster_profiles": [asdict(p) for p in model_info.cluster_profiles],
    }
    meta_path = os.path.join(output_dir, f"{prefix}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(info_dict, f, indent=2)

    logger.info(f"Saved variant {variant_id} to {output_dir}/{prefix}_*")
    return prefix


def load_variant_model(
    variant_id: str,
    timeframe: str,
    model_dir: str = "models",
) -> tuple:
    """
    Load a clustering variant from disk.

    Returns:
        (cluster_model, scaler, model_info)
    """
    prefix = f"cluster_{variant_id}_{timeframe}"

    model_path = os.path.join(model_dir, f"{prefix}_model.joblib")
    scaler_path = os.path.join(model_dir, f"{prefix}_scaler.joblib")
    meta_path = os.path.join(model_dir, f"{prefix}_meta.json")

    cluster_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(meta_path) as f:
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
        f"Loaded variant {variant_id}: {model_info.n_clusters} clusters, "
        f"tf={timeframe}, algo={info_dict['algo']}"
    )
    return cluster_model, scaler, model_info


def get_available_variants(timeframe: str, model_dir: str = "models") -> list[str]:
    """Return list of variant IDs that have trained models for this timeframe."""
    available = []
    for vid in ALL_VARIANT_IDS:
        meta_path = os.path.join(model_dir, f"cluster_{vid}_{timeframe}_meta.json")
        if os.path.exists(meta_path):
            available.append(vid)
    return available
