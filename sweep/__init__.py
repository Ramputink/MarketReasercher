"""
Timeframe Sweep — honest evaluation of which (timeframe + multi-TF confirmation)
combination, if any, carries a certifiable directional edge on BTC/ETH.

Built on the same honesty machinery as the XRP work:
  - frozen, hash-verified data snapshots (sweep/snapshots/)
  - data lockbox (final 120d sealed from the search)
  - the SAME benchmark gauntlet as the referee (spec stays XRP-independent via params)

Phase 1 (this package): directional strategy families + a no-look-ahead higher-
timeframe CONFIRMATION filter, swept across base TFs {15m, 1h, 4h, 1d} on BTC+ETH.
Phase 2 (later): cross-sectional momentum + funding carry (different paradigm).
"""

# Trade BTC and ETH each independently vs the USDC stablecoin (not against each
# other). The cross-asset figure in the sweep is a GENERALISATION check only —
# a BTC-fit strategy tested on ETH — never a BTC-vs-ETH pairs trade.
SWEEP_SYMBOLS = ["BTC/USDC", "ETH/USDC"]

# base timeframe -> higher timeframes used to CONFIRM signals (no look-ahead).
# A signal on the base TF is only taken if the higher TF(s) agree on direction.
SWEEP_TIMEFRAMES = {
    "15m": ["1h", "4h"],
    "1h":  ["4h", "1d"],
    "4h":  ["1d"],
    "1d":  [],          # no higher TF in the set; trades standalone
}

LOCKBOX_DAYS = 120
HISTORY_DAYS = 365
