"""
forward_test — pre-registered TRUE out-of-sample (paper) testing
================================================================

Every result in this project so far is a single read of a sealed lockbox that
was carved out of HISTORY. That controls for in-sample fitting, but it cannot
rule out that the lockbox itself was (consciously or not) part of the discovery
loop. The only thing that can is data that did not exist when the config was
frozen: a genuine FORWARD test.

This package freezes the exact configs to track (with a content hash and a
"forward starts here" timestamp) and provides a runner that, on each invocation,
scores ONLY the data after that timestamp and appends to an append-only log. No
re-fitting, ever — the configs are immutable once registered. Run it on a
schedule (weekly) and the forward track record accumulates honestly.
"""
from __future__ import annotations

import hashlib
import json
import os

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PREREG_PATH = os.path.join(PKG_DIR, "preregistration.json")
LOG_PATH = os.path.join(PKG_DIR, "forward_log.jsonl")
LIVE_SNAP = os.path.join(PKG_DIR, "live_snapshot")

# Date the configs below were frozen and committed. All data with timestamp
# strictly AFTER FORWARD_START is genuine out-of-sample — it did not exist (in
# the frozen panels) when these configs were registered.
REGISTERED_DATE = "2026-06-15"
FORWARD_START = "2026-06-07"     # last date in the frozen research panels
EXCHANGE = "binanceusdm"

# ── the frozen, pre-registered configs (immutable) ────────────────────────────
PREREGISTERED = {
    # Phase-2 cross-sectional funding carry — the one edge with a real claim.
    # NB: P4 showed it does NOT replicate on Bybit's lockbox; the forward test is
    # the tiebreaker — does Binance-forward confirm the Binance-lockbox positive,
    # or side with Bybit's negative?
    "carry_mn": {
        "type": "carry_market_neutral",
        "family": "carry", "carry_lookback": 7, "k": 4, "rebalance_every": 7,
        "long_only": False, "dollar_neutral": True, "weight_mode": "equal",
        "gross": 1.0, "vol_lookback": 20,
    },
    # Phase-3 spot long-only 3-sleeve cash-gating ensemble WITH the de-risking
    # overlay (the operable version from P1).
    "phase3_ensemble_overlay": {
        "type": "spot_longonly_ensemble_overlay",
        "sleeves": ["trend_ts", "breakout", "capitulation"], "max_positions": 10,
        "overlay": {"target_vol_ann": 0.15, "vol_lookback": 20,
                    "dd_trigger": -0.20, "dd_reentry": -0.10, "risk_floor": 0.25},
    },
}


def config_hash() -> str:
    blob = json.dumps({"registered": REGISTERED_DATE, "forward_start": FORWARD_START,
                       "exchange": EXCHANGE, "configs": PREREGISTERED},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()
