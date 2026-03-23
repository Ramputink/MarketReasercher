"""
CryptoResearchLab — Feature Engineering Pipeline
Computes technical indicators, microstructure features, and regime signals.
All features are computed WITHOUT future data leakage.
"""
import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# CORE TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — no future leakage."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength."""
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr_vals = atr(df, period)
    plus_di = 100 * ema(plus_dm, period) / atr_vals.replace(0, np.nan)
    minus_di = 100 * ema(minus_dm, period) / atr_vals.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return ema(dx, period)


# ═══════════════════════════════════════════════════════════════
# BOLLINGER BANDS
# ═══════════════════════════════════════════════════════════════

def bollinger_bands(
    series: pd.Series, period: int = 20, std_mult: float = 2.0
) -> dict[str, pd.Series]:
    """Returns dict with middle, upper, lower, bandwidth, pct_b."""
    middle = sma(series, period)
    rolling_std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_mult * rolling_std
    lower = middle - std_mult * rolling_std
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)

    return {
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_pct_b": pct_b,
    }


def bb_bandwidth_percentile(
    bandwidth: pd.Series, lookback: int = 100
) -> pd.Series:
    """Rolling percentile rank of BB bandwidth — low = compression."""
    def pctile(window):
        if len(window) < 2:
            return np.nan
        return (window.values[-1] > window.values[:-1]).mean() * 100
    return bandwidth.rolling(window=lookback, min_periods=20).apply(pctile, raw=False)


# ═══════════════════════════════════════════════════════════════
# FIBONACCI RETRACEMENT
# ═══════════════════════════════════════════════════════════════

def fibonacci_levels(
    df: pd.DataFrame, lookback: int = 50
) -> dict[str, pd.Series]:
    """
    Rolling Fibonacci retracement levels based on recent swing high/low.
    No future data used.
    """
    rolling_high = df["high"].rolling(window=lookback, min_periods=lookback).max()
    rolling_low = df["low"].rolling(window=lookback, min_periods=lookback).min()
    diff = rolling_high - rolling_low

    levels = {}
    for fib in [0.236, 0.382, 0.5, 0.618, 0.786]:
        # In uptrend: retracement from high
        levels[f"fib_{fib:.3f}_up"] = rolling_high - fib * diff
        # In downtrend: retracement from low
        levels[f"fib_{fib:.3f}_dn"] = rolling_low + fib * diff

    levels["fib_range"] = diff
    return levels


# ═══════════════════════════════════════════════════════════════
# BULL MARKET SUPPORT BAND
# ═══════════════════════════════════════════════════════════════

def bull_market_support_band(
    series: pd.Series, sma_period: int = 20, ema_period: int = 21
) -> dict[str, pd.Series]:
    """
    BMSB: 20-week SMA and 21-week EMA (adapted to hourly: ~3360h / ~3528h).
    For hourly data: sma_period and ema_period are in the timeframe's units.
    """
    bmsb_sma = sma(series, sma_period)
    bmsb_ema = ema(series, ema_period)

    return {
        "bmsb_sma": bmsb_sma,
        "bmsb_ema": bmsb_ema,
        "bmsb_bullish": (series > bmsb_sma) & (series > bmsb_ema),
        "bmsb_spread": (bmsb_sma - bmsb_ema) / bmsb_sma.replace(0, np.nan),
    }


# ═══════════════════════════════════════════════════════════════
# VOLATILITY FEATURES
# ═══════════════════════════════════════════════════════════════

def volatility_features(df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
    """Compute various volatility measures."""
    if periods is None:
        periods = [5, 10, 20, 50]

    features = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for p in periods:
        features[f"realized_vol_{p}"] = log_ret.rolling(p, min_periods=p).std() * np.sqrt(252 * 24)
        features[f"range_vol_{p}"] = (
            (np.log(df["high"] / df["low"])).rolling(p, min_periods=p).mean()
        )

    # Volatility ratio: short-term vs long-term
    if len(periods) >= 2:
        features["vol_ratio"] = (
            features[f"realized_vol_{periods[0]}"] /
            features[f"realized_vol_{periods[-1]}"].replace(0, np.nan)
        )

    # Garman-Klass volatility estimator
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    features["gk_vol"] = np.sqrt(
        (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(20, min_periods=20).mean()
        * 252 * 24
    )

    return features


# ═══════════════════════════════════════════════════════════════
# VOLUME FEATURES
# ═══════════════════════════════════════════════════════════════

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features."""
    features = pd.DataFrame(index=df.index)

    vol = df["volume"]
    features["volume_sma_20"] = sma(vol, 20)
    features["volume_ratio"] = vol / features["volume_sma_20"].replace(0, np.nan)

    # Volume-weighted price
    features["vwap_20"] = (
        (df["close"] * vol).rolling(20, min_periods=20).sum() /
        vol.rolling(20, min_periods=20).sum().replace(0, np.nan)
    )

    # On-Balance Volume
    obv_sign = np.sign(df["close"].diff())
    features["obv"] = (vol * obv_sign).cumsum()
    features["obv_sma"] = sma(features["obv"], 20)
    features["obv_divergence"] = features["obv"] - features["obv_sma"]

    # Volume spike detector
    features["volume_zscore"] = (
        (vol - vol.rolling(50, min_periods=20).mean()) /
        vol.rolling(50, min_periods=20).std().replace(0, np.nan)
    )

    return features


# ═══════════════════════════════════════════════════════════════
# MOMENTUM & RETURN FEATURES
# ═══════════════════════════════════════════════════════════════

def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum and return-based features."""
    features = pd.DataFrame(index=df.index)
    close = df["close"]

    # Returns at various horizons
    for h in [1, 3, 6, 12, 24]:
        features[f"ret_{h}"] = close.pct_change(h)

    # Z-score of returns
    ret_1 = features["ret_1"]
    features["ret_zscore_20"] = (
        (ret_1 - ret_1.rolling(20, min_periods=20).mean()) /
        ret_1.rolling(20, min_periods=20).std().replace(0, np.nan)
    )

    # RSI
    features["rsi_14"] = rsi(close, 14)
    features["rsi_7"] = rsi(close, 7)

    # Rate of change
    features["roc_12"] = (close / close.shift(12) - 1) * 100

    # Distance from recent high/low
    features["dist_from_high_20"] = close / close.rolling(20).max() - 1
    features["dist_from_low_20"] = close / close.rolling(20).min() - 1

    return features


# ═══════════════════════════════════════════════════════════════
# MICROSTRUCTURE FEATURES
# ═══════════════════════════════════════════════════════════════

def microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Microstructure features derived from OHLCV
    (approximations without tick-level data).
    """
    features = pd.DataFrame(index=df.index)

    # Close location value (where in the bar did it close)
    bar_range = df["high"] - df["low"]
    features["close_location"] = (
        (df["close"] - df["low"]) / bar_range.replace(0, np.nan)
    )

    # Buying/selling pressure proxy
    features["buying_pressure"] = (df["close"] - df["low"]) / bar_range.replace(0, np.nan)
    features["selling_pressure"] = (df["high"] - df["close"]) / bar_range.replace(0, np.nan)

    # Imbalance proxy
    features["pressure_imbalance"] = features["buying_pressure"] - features["selling_pressure"]
    features["pressure_imbalance_ma"] = sma(features["pressure_imbalance"], 10)

    # Bar range normalized by ATR
    atr_14 = atr(df, 14)
    features["range_atr_ratio"] = bar_range / atr_14.replace(0, np.nan)

    # Acceleration of price movement
    features["price_acceleration"] = df["close"].diff().diff()

    # Body to wick ratio
    body = (df["close"] - df["open"]).abs()
    features["body_wick_ratio"] = body / bar_range.replace(0, np.nan)

    return features


# ═══════════════════════════════════════════════════════════════
# MASTER FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_all_features(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    fib_lookback: int = 50,
    bmsb_sma: int = 20,
    bmsb_ema: int = 21,
) -> pd.DataFrame:
    """
    Build all features from OHLCV data.
    Returns a copy with all feature columns appended.
    No future data leakage.
    """
    result = df.copy()

    # ATR
    result["atr_14"] = atr(df, 14)
    result["atr_7"] = atr(df, 7)

    # ADX
    result["adx_14"] = adx(df, 14)

    # Bollinger Bands
    bb = bollinger_bands(df["close"], bb_period, bb_std)
    for k, v in bb.items():
        result[k] = v
    result["bb_bw_percentile"] = bb_bandwidth_percentile(result["bb_bandwidth"])

    # Fibonacci
    fib = fibonacci_levels(df, fib_lookback)
    for k, v in fib.items():
        result[k] = v

    # BMSB
    bmsb = bull_market_support_band(df["close"], bmsb_sma, bmsb_ema)
    for k, v in bmsb.items():
        result[k] = v

    # Volatility
    vol_feats = volatility_features(df)
    for col in vol_feats.columns:
        result[col] = vol_feats[col]

    # Volume
    vol_f = volume_features(df)
    for col in vol_f.columns:
        result[col] = vol_f[col]

    # Momentum
    mom = momentum_features(df)
    for col in mom.columns:
        result[col] = mom[col]

    # Microstructure
    micro = microstructure_features(df)
    for col in micro.columns:
        result[col] = micro[col]

    return result
