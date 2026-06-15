"""
phase2 — Cross-sectional momentum + funding carry (the structural pivot)
========================================================================

Phase 1 (sweep/) asked: does a single-asset directional technical edge survive
the honest gauntlet? Answer on XRP/USDT-H1 and BTC/ETH·USDC: no. Those "edges"
were fitting one asset's own noise.

Phase 2 asks a *structurally different* question. Instead of timing one asset
with retail indicators, it harvests two cross-sectional, economically-grounded
return sources on a basket of liquid perps:

  1. CROSS-SECTIONAL MOMENTUM — long the strongest, short the weakest. The edge
     is relative (winners keep winning vs losers), not absolute timing. Survives
     in equities, futures, FX and crypto in the academic record.

  2. FUNDING CARRY — perp funding is paid longs→shorts when positive. Shorting
     persistently-high-funding perps (and/or longing negative-funding ones)
     harvests a structural cash flow, separate from price direction.

Both are scored on the SAME honest referee as Phase 1: a sealed lockbox the
search never sees, Deflated-Sharpe-deflated by the real number of trials. No
threshold is relaxed. If nothing certifies, that is the honest answer.

Universe: top-15 liquid USDⓈ-M perps. Assets enter the tradeable universe only
once they have enough history at each rebalance date — no survivorship / no
look-ahead from a fixed modern basket.
"""
from __future__ import annotations

import os

# ── Universe (USDⓈ-M perpetual, vs USDT) ─────────────────────────────────────
# MATIC was renamed POL on Binance; POL perp only lists 2024-09 — handled by the
# dynamic-eligibility rule (an asset is rankable only after lookback bars exist).
UNIVERSE = [
    # original 15
    "BTC", "ETH", "SOL", "XRP", "ADA",
    "AVAX", "LINK", "DOGE", "BNB", "LTC",
    "DOT", "POL", "ATOM", "NEAR", "APT",
    # +15 (all listed 2019-2020 → deep overlapping history for cross-section)
    "BCH", "TRX", "ETC", "XLM", "DASH",
    "ZEC", "VET", "THETA", "ALGO", "COMP",
    "CRV", "RUNE", "SUSHI", "EGLD", "UNI",
]

def perp(symbol: str) -> str:
    """Base ticker -> ccxt USDⓈ-M perp symbol."""
    return f"{symbol}/USDT:USDT"

# ── Timeframe / rebalance ────────────────────────────────────────────────────
BASE_TF = "1d"                 # daily bars: the natural XS-momentum / carry cadence
FUNDING_INTERVAL_HOURS = 8     # Binance perp funding cadence

# ── Honest split ─────────────────────────────────────────────────────────────
LOCKBOX_DAYS = 730             # final TWO years sealed (~104 weekly OOS periods):
                               # 1yr (~53 periods) made the t-stat≥2 bar require an
                               # unrealistic per-period Sharpe; 2yr lowers it and
                               # spans more than one regime.
MIN_UNIVERSE = 5               # need >= this many eligible assets to form a book

# ── Paths ────────────────────────────────────────────────────────────────────
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(PKG_DIR, "snapshots")
RESULTS_DIR = os.path.join(PKG_DIR, "results")
MANIFEST = os.path.join(SNAP_DIR, "manifest.json")

# ── Exchange ─────────────────────────────────────────────────────────────────
EXCHANGE_ID = "binanceusdm"
