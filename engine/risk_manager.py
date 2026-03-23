"""
CryptoResearchLab — Risk Management Engine
Independent risk layer that filters and controls strategy execution.
Neither autoresearch nor MiroFish can bypass this layer.
"""
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config import RiskConfig
from engine.backtester import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    """Current risk management state."""
    equity: float = 10_000.0
    peak_equity: float = 10_000.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    open_positions: int = 0
    last_trade_time: Optional[int] = None
    circuit_breaker_active: bool = False
    daily_reset_timestamp: int = 0
    consecutive_losses: int = 0
    session_trades: list = field(default_factory=list)


class RiskManager:
    """
    Independent risk management layer.
    Acts as a gate between validated strategies and execution.

    Rules enforced:
    1. Max concurrent positions
    2. Max daily loss
    3. Circuit breaker on drawdown
    4. Min time between trades
    5. Position sizing caps
    6. No trading if circuit breaker is active
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.state = RiskState()

    def update_equity(self, new_equity: float):
        """Update equity after trade or mark-to-market."""
        self.state.equity = new_equity
        self.state.peak_equity = max(self.state.peak_equity, new_equity)

    def record_trade_result(self, pnl: float, timestamp: int):
        """Record a completed trade."""
        self.state.daily_pnl += pnl
        self.state.daily_trades += 1
        self.state.last_trade_time = timestamp
        self.state.session_trades.append({"pnl": pnl, "timestamp": timestamp})

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

    def _check_daily_reset(self, current_timestamp: int):
        """Reset daily counters at UTC midnight."""
        current_day = current_timestamp // 86_400_000
        stored_day = self.state.daily_reset_timestamp // 86_400_000
        if current_day > stored_day:
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_reset_timestamp = current_timestamp

    def can_trade(self, signal: Signal, current_timestamp: int) -> tuple[bool, str]:
        """
        Check if a trade is allowed given current risk state.
        Returns (allowed, reason).
        """
        self._check_daily_reset(current_timestamp)

        # 1. Circuit breaker
        if self.state.circuit_breaker_active:
            return False, "circuit_breaker_active"

        current_dd_pct = (
            (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
            if self.state.peak_equity > 0 else 0
        )
        if current_dd_pct >= self.config.circuit_breaker_drawdown_pct:
            self.state.circuit_breaker_active = True
            logger.warning(
                f"CIRCUIT BREAKER TRIGGERED: drawdown {current_dd_pct:.1f}% "
                f">= {self.config.circuit_breaker_drawdown_pct}%"
            )
            return False, f"circuit_breaker_triggered_dd_{current_dd_pct:.1f}%"

        # 2. Max concurrent positions
        if self.state.open_positions >= self.config.max_positions:
            return False, f"max_positions_{self.config.max_positions}"

        # 3. Daily loss limit
        daily_loss_pct = abs(self.state.daily_pnl) / self.state.equity * 100 if self.state.daily_pnl < 0 else 0
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            return False, f"daily_loss_limit_{daily_loss_pct:.1f}%"

        # 4. Min time between trades
        if self.state.last_trade_time is not None:
            minutes_since = (current_timestamp - self.state.last_trade_time) / 60_000
            if minutes_since < self.config.min_trade_interval_minutes:
                return False, f"min_interval_{minutes_since:.0f}m"

        # 5. Excessive consecutive losses (optional safety)
        if self.state.consecutive_losses >= 5:
            logger.warning(f"5 consecutive losses — proceed with caution")

        return True, "approved"

    def compute_allowed_size(
        self,
        signal: Signal,
        price: float,
        atr_value: float,
    ) -> float:
        """
        Compute risk-adjusted position size, capping at max allowed.
        """
        equity = self.state.equity

        if self.config.sizing_method == "volatility_scaled":
            if atr_value <= 0 or price <= 0:
                return equity * 0.05
            daily_vol = atr_value / price
            annual_vol = daily_vol * np.sqrt(365)
            target_size = equity * (self.config.vol_target_annualized / max(annual_vol, 0.01))
        elif self.config.sizing_method == "kelly":
            target_size = equity * self.config.kelly_fraction
        else:
            target_size = equity * 0.10

        # Apply signal strength
        target_size *= signal.strength

        # Cap
        max_size = equity * 0.15  # Never more than 15% of equity
        remaining_capacity = equity * 0.30 - (self.state.open_positions * equity * 0.10)
        return min(target_size, max_size, max(remaining_capacity, 0))

    def reset_circuit_breaker(self):
        """Manual reset of circuit breaker (requires human decision)."""
        self.state.circuit_breaker_active = False
        self.state.consecutive_losses = 0
        logger.info("Circuit breaker reset manually")

    def get_status_report(self) -> dict:
        """Return current risk status."""
        dd_pct = (
            (self.state.peak_equity - self.state.equity) / self.state.peak_equity * 100
            if self.state.peak_equity > 0 else 0
        )
        return {
            "equity": round(self.state.equity, 2),
            "peak_equity": round(self.state.peak_equity, 2),
            "drawdown_pct": round(dd_pct, 2),
            "daily_pnl": round(self.state.daily_pnl, 2),
            "open_positions": self.state.open_positions,
            "circuit_breaker": self.state.circuit_breaker_active,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_trades": self.state.daily_trades,
        }
