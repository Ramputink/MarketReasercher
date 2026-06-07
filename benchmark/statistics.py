"""
benchmark.statistics — Multiple-testing-aware statistics
========================================================

Self-contained (only numpy + math) implementations of:

    - Probabilistic Sharpe Ratio (PSR)         Bailey & Lopez de Prado (2012)
    - Deflated Sharpe Ratio (DSR)              Bailey & Lopez de Prado (2014)
    - Expected maximum Sharpe over N trials    (Gumbel / extreme-value approx)
    - One-sided t-stat & p-value of a mean
    - Inverse standard-normal CDF (Acklam)     — no scipy dependency

These are the tools that separate "we found a 3.5 Sharpe" from "we found a 3.5
Sharpe that is statistically distinguishable from the best of 30,000 random
draws". The second statement is the only one worth money.

All Sharpe inputs here are *per-observation* (e.g. per-trade) — DO NOT pass an
annualised Sharpe. Annualisation is a scaling that destroys the link between the
Sharpe and the sample size T, which is exactly what these formulas exploit.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


# ─────────────────────────────────────────────────────────────────────────────
# Normal distribution helpers (no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (quantile). Acklam's rational approximation,
    |error| < 1.15e-9. Deterministic, dependency-free.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Sharpe statistics
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_per_obs(returns: np.ndarray) -> float:
    """Per-observation Sharpe (mean/std). NOT annualised."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(r.mean() / sd)


def skew_kurt(returns: np.ndarray) -> tuple[float, float]:
    """
    Sample skewness and (non-excess, Pearson) kurtosis. Normal => (0, 3).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return 0.0, 3.0
    m = r.mean()
    sd = r.std(ddof=0)
    if sd <= 0:
        return 0.0, 3.0
    z = (r - m) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    return skew, kurt


def probabilistic_sharpe_ratio(
    sr_hat: float,
    n_obs: int,
    skew: float,
    kurt: float,
    sr_benchmark: float = 0.0,
) -> float:
    """
    PSR(sr_benchmark) = P( true SR > sr_benchmark ), correcting for the
    non-normality (skew/kurtosis) of the return distribution and the sample
    size n_obs. All Sharpes per-observation.

    Returns a probability in [0, 1].
    """
    if n_obs < 2:
        return 0.0
    denom = 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2)
    # Guard against pathological negative variance estimates.
    denom = max(denom, 1e-9)
    z = (sr_hat - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(norm_cdf(z))


def expected_max_sharpe(n_trials: int, trials_sharpe_std: float) -> float:
    """
    Expected maximum of N i.i.d. Sharpe ratios drawn from a strategy family with
    cross-sectional std `trials_sharpe_std`, under the false-strategy null
    (true SR = 0). Gumbel/extreme-value approximation (Bailey & LdP 2014):

        E[max] ≈ σ * [ (1-γ)·Z(1 - 1/N) + γ·Z(1 - 1/(N·e)) ]

    This is the bar a candidate must clear: the best of N noise strategies.
    """
    n = max(int(n_trials), 2)
    sigma = max(float(trials_sharpe_std), 1e-9)
    z1 = norm_ppf(1.0 - 1.0 / n)
    z2 = norm_ppf(1.0 - 1.0 / (n * math.e))
    return float(sigma * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trials_sharpe_std: Optional[float] = None,
) -> dict:
    """
    Deflated Sharpe Ratio: PSR evaluated against the *expected maximum* Sharpe of
    n_trials random strategies, instead of against zero. This is the single most
    important defence against picking the luckiest genome out of tens of
    thousands of evaluations.

    Returns a dict with sr_hat, sr_star (the deflation benchmark), dsr (prob),
    plus the moments used.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n_obs = len(r)
    sr_hat = sharpe_per_obs(r)
    skew, kurt = skew_kurt(r)

    if trials_sharpe_std is None or trials_sharpe_std <= 0:
        # Principled default: under the no-skill null, a per-trade Sharpe estimated
        # from T trades has standard error ≈ 1/sqrt(T) (Lo, 2002). That is the
        # natural dispersion of the trial Sharpes when no genome has real edge, and
        # it keeps sr_hat and sr_star in the SAME (per-trade) units. Supplying the
        # *measured* cross-trial Sharpe std from the evolution run is stronger; this
        # is the honest fallback when it is unavailable.
        trials_sharpe_std = 1.0 / math.sqrt(max(n_obs, 2))

    sr_star = expected_max_sharpe(n_trials, trials_sharpe_std)
    dsr = probabilistic_sharpe_ratio(sr_hat, n_obs, skew, kurt, sr_benchmark=sr_star)

    return {
        "dsr": dsr,
        "sr_hat": sr_hat,
        "sr_star": sr_star,
        "n_obs": n_obs,
        "n_trials": int(n_trials),
        "skew": skew,
        "kurt": kurt,
        "trials_sharpe_std": float(trials_sharpe_std),
    }


def tstat_pvalue(returns: np.ndarray) -> dict:
    """
    One-sided t-stat and p-value for H0: mean return <= 0.
    Uses the normal approximation for the p-value (n is typically >= 30).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {"tstat": 0.0, "pvalue": 1.0, "n": n}
    sd = r.std(ddof=1)
    if sd <= 0:
        return {"tstat": 0.0, "pvalue": 1.0, "n": n}
    t = r.mean() / (sd / math.sqrt(n))
    pvalue = 1.0 - norm_cdf(t)
    return {"tstat": float(t), "pvalue": float(pvalue), "n": n}
