"""
phase2.signals — cross-sectional signals (strictly no look-ahead)
================================================================

Every signal returns, for a rebalance date t, a cross-sectional SCORE per asset
computed only from information available at the close of t. Higher score ⇒ more
attractive to be LONG; lower ⇒ more attractive to be SHORT. The portfolio engine
ranks these scores; it never sees prices beyond t.

Eligibility: an asset gets a score only if it has the full lookback window of
finite data ending at t. Newly-listed assets (NaN history) score NaN and are
excluded from that date's book — this is what keeps the universe survivorship-
free and look-ahead-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(row: pd.Series) -> pd.Series:
    x = row.astype(float)
    mu, sd = x.mean(), x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return x * 0.0
    return (x - mu) / sd


def momentum_score(closes: pd.DataFrame, lookback: int, skip: int = 0,
                   risk_adj: bool = False) -> pd.DataFrame:
    """
    Cross-sectional momentum: trailing log-return over `lookback` days, optionally
    skipping the most recent `skip` days (classic 12-1 style: skip short-term
    reversal). Score at t uses closes[t-skip] / closes[t-skip-lookback].

    risk_adj=True divides the trailing return by its trailing realized vol over
    the same window (Sharpe-momentum) — the documented improvement that ranks by
    *risk-adjusted* relative strength instead of raw return, damping the high-vol
    names that dominate a raw-return ranking.
    """
    logc = np.log(closes)
    # return from (t-skip-lookback) to (t-skip), known at close t
    mom = logc.shift(skip) - logc.shift(skip + lookback)
    if risk_adj:
        dret = logc.diff()
        vol = dret.shift(skip).rolling(lookback, min_periods=max(2, lookback // 2)).std(ddof=0)
        mom = mom / (vol * np.sqrt(lookback)).replace(0.0, np.nan)
    # require full window finite
    valid = closes.shift(skip + lookback).notna() & closes.shift(skip).notna()
    return mom.where(valid)


def idio_momentum_score(closes: pd.DataFrame, lookback: int, skip: int = 0,
                        risk_adj: bool = False) -> pd.DataFrame:
    """
    Idiosyncratic (market-neutralised) momentum. Strips the common crypto factor
    BEFORE ranking: each day's per-asset log-return has the cross-sectional mean
    (the 'market') removed, and momentum is the trailing sum of those RESIDUAL
    returns. In a highly-correlated universe this isolates true relative strength
    from beta — what a dollar-neutral book is actually trying to harvest.

    risk_adj divides the residual momentum by the trailing residual vol.
    """
    logc = np.log(closes)
    dret = logc.diff()
    market = dret.mean(axis=1)                 # equal-weight market factor return
    resid = dret.sub(market, axis=0)           # idiosyncratic daily return
    cum = resid.cumsum()
    # residual momentum from (t-skip-lookback) to (t-skip)
    mom = cum.shift(skip) - cum.shift(skip + lookback)
    if risk_adj:
        rvol = resid.shift(skip).rolling(lookback, min_periods=max(2, lookback // 2)).std(ddof=0)
        mom = mom / (rvol * np.sqrt(lookback)).replace(0.0, np.nan)
    valid = closes.shift(skip + lookback).notna() & closes.shift(skip).notna()
    return mom.where(valid)


def carry_score(funding: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Funding carry: average daily funding over `lookback` days. Positive funding is
    paid by longs to shorts, so high funding ⇒ attractive to SHORT ⇒ LOW score.
    Score = -mean(funding) so the ranking convention (high=long) holds.
    """
    avg = funding.rolling(lookback, min_periods=lookback).mean()
    return -avg


def volatility(ret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing realized vol per asset (for vol-scaling / risk parity weights)."""
    return ret.rolling(lookback, min_periods=max(2, lookback // 2)).std(ddof=0)


def combine_scores(parts: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """
    Blend several score frames by cross-sectional z-score then weighted sum.
    Each part is (score_frame, weight). Z-scoring per-date puts momentum and
    carry on a comparable scale before blending.
    """
    total = None
    for frame, w in parts:
        if w == 0:
            continue
        z = frame.apply(_zscore, axis=1)
        total = z * w if total is None else total.add(z * w, fill_value=np.nan)
    return total
