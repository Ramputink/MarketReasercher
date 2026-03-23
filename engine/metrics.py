"""
CryptoResearchLab — Evaluation Metrics
All quantitative metrics for strategy assessment.
Includes cost-adjusted performance, robustness, and cross-validation metrics.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyMetrics:
    """Complete metrics snapshot for a strategy evaluation."""
    # Identification
    strategy_name: str = ""
    dataset_label: str = ""  # "train", "val", "test", "oos"
    period_start: str = ""
    period_end: str = ""

    # Performance (net of costs)
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    volatility_annualized: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    avg_holding_hours: float = 0.0

    # Cost analysis
    total_commissions: float = 0.0
    total_slippage: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Robustness
    oos_sharpe: float = 0.0
    oos_degradation_pct: float = 0.0
    cross_asset_sharpe_mean: float = 0.0
    cross_asset_sharpe_std: float = 0.0
    regime_stability: float = 0.0

    def is_acceptable(
        self,
        min_sharpe: float = 1.0,
        max_drawdown: float = 15.0,
        min_trades: int = 30,
        max_oos_degradation: float = 30.0,
    ) -> bool:
        """Check if strategy meets minimum acceptance criteria."""
        return (
            self.sharpe_ratio >= min_sharpe
            and self.max_drawdown_pct <= max_drawdown
            and self.total_trades >= min_trades
            and self.profit_factor > 1.0
            and self.oos_degradation_pct <= max_oos_degradation
        )

    def summary_dict(self) -> dict:
        """Return key metrics as a flat dictionary."""
        return {
            "strategy": self.strategy_name,
            "dataset": self.dataset_label,
            "sharpe": round(self.sharpe_ratio, 3),
            "sortino": round(self.sortino_ratio, 3),
            "max_dd_%": round(self.max_drawdown_pct, 2),
            "profit_factor": round(self.profit_factor, 3),
            "win_rate_%": round(self.win_rate * 100, 1),
            "expectancy": round(self.expectancy, 4),
            "trades": self.total_trades,
            "net_pnl": round(self.net_pnl, 2),
            "oos_sharpe": round(self.oos_sharpe, 3),
            "oos_degrad_%": round(self.oos_degradation_pct, 1),
        }


# ═══════════════════════════════════════════════════════════════
# METRIC COMPUTATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    initial_capital: float = 10_000.0,
    strategy_name: str = "",
    dataset_label: str = "",
    annualization_factor: float = 365.25 * 24,  # hourly data default
) -> StrategyMetrics:
    """
    Compute all metrics from a list of trades and equity curve.

    trades DataFrame expected columns:
        entry_time, exit_time, side, entry_price, exit_price,
        size, pnl_gross, commission, slippage, pnl_net

    equity_curve: Series indexed by timestamp with portfolio value.
    """
    m = StrategyMetrics(strategy_name=strategy_name, dataset_label=dataset_label)

    if len(trades) == 0 or len(equity_curve) < 2:
        return m

    # Period
    m.period_start = str(equity_curve.index[0])
    m.period_end = str(equity_curve.index[-1])

    # Returns
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return m

    m.total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    periods = len(returns)
    m.annualized_return_pct = (
        ((1 + m.total_return_pct / 100) ** (annualization_factor / periods) - 1) * 100
        if periods > 0 else 0.0
    )

    # Volatility
    m.volatility_annualized = returns.std() * np.sqrt(annualization_factor)

    # Sharpe
    excess_return = returns.mean()
    m.sharpe_ratio = (
        (excess_return / returns.std()) * np.sqrt(annualization_factor)
        if returns.std() > 0 else 0.0
    )

    # Sortino
    downside = returns[returns < 0]
    m.downside_deviation = downside.std() * np.sqrt(annualization_factor) if len(downside) > 0 else 0.0
    m.sortino_ratio = (
        (excess_return / downside.std()) * np.sqrt(annualization_factor)
        if len(downside) > 0 and downside.std() > 0 else 0.0
    )

    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    m.max_drawdown_pct = abs(drawdown.min()) * 100

    # Drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        dd_groups = (~in_drawdown).cumsum()
        dd_durations = in_drawdown.groupby(dd_groups).sum()
        m.max_drawdown_duration_hours = float(dd_durations.max())

    # Calmar
    m.calmar_ratio = (
        m.annualized_return_pct / m.max_drawdown_pct
        if m.max_drawdown_pct > 0 else 0.0
    )

    # VaR / CVaR
    m.var_95 = float(np.percentile(returns, 5))
    m.cvar_95 = float(returns[returns <= m.var_95].mean()) if (returns <= m.var_95).any() else m.var_95

    # Trade statistics
    m.total_trades = len(trades)
    winners = trades[trades["pnl_net"] > 0]
    losers = trades[trades["pnl_net"] <= 0]
    m.win_rate = len(winners) / len(trades) if len(trades) > 0 else 0.0

    # P&L
    m.gross_pnl = float(trades["pnl_gross"].sum())
    m.total_commissions = float(trades["commission"].sum())
    m.total_slippage = float(trades["slippage"].sum())
    m.net_pnl = float(trades["pnl_net"].sum())

    # Profit factor
    gross_profit = float(winners["pnl_net"].sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers["pnl_net"].sum())) if len(losers) > 0 else 0.0
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win/loss averages
    m.avg_win = float(winners["pnl_net"].mean()) if len(winners) > 0 else 0.0
    m.avg_loss = float(losers["pnl_net"].mean()) if len(losers) > 0 else 0.0

    # Expectancy
    m.expectancy = (m.win_rate * m.avg_win + (1 - m.win_rate) * m.avg_loss) / initial_capital

    # Max consecutive losses
    pnl_signs = (trades["pnl_net"] > 0).astype(int)
    if len(pnl_signs) > 0:
        groups = pnl_signs.ne(pnl_signs.shift()).cumsum()
        loss_streaks = pnl_signs.groupby(groups).apply(
            lambda x: len(x) if x.iloc[0] == 0 else 0
        )
        m.max_consecutive_losses = int(loss_streaks.max()) if len(loss_streaks) > 0 else 0

    # Holding time
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        holding = (
            pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
        ).dt.total_seconds() / 3600
        m.avg_holding_hours = float(holding.mean())

    return m


def compute_walk_forward_metrics(
    fold_metrics: list[StrategyMetrics],
) -> StrategyMetrics:
    """
    Aggregate metrics across walk-forward folds.
    Returns averaged metrics with OOS assessment.
    """
    if not fold_metrics:
        return StrategyMetrics()

    agg = StrategyMetrics(
        strategy_name=fold_metrics[0].strategy_name,
        dataset_label="walk_forward_aggregate",
    )

    sharpes = [m.sharpe_ratio for m in fold_metrics]
    sortinos = [m.sortino_ratio for m in fold_metrics]
    drawdowns = [m.max_drawdown_pct for m in fold_metrics]
    pfs = [m.profit_factor for m in fold_metrics]

    agg.sharpe_ratio = float(np.mean(sharpes))
    agg.sortino_ratio = float(np.mean(sortinos))
    agg.max_drawdown_pct = float(np.max(drawdowns))
    agg.profit_factor = float(np.mean(pfs))
    agg.total_trades = sum(m.total_trades for m in fold_metrics)
    agg.win_rate = float(np.mean([m.win_rate for m in fold_metrics]))
    agg.net_pnl = sum(m.net_pnl for m in fold_metrics)
    agg.expectancy = float(np.mean([m.expectancy for m in fold_metrics]))

    # Stability: std of sharpe across folds
    agg.regime_stability = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0

    return agg


def compute_oos_degradation(
    in_sample: StrategyMetrics,
    out_of_sample: StrategyMetrics,
) -> float:
    """
    Compute % degradation from in-sample to out-of-sample Sharpe.
    Returns 0 if OOS is better (no degradation).
    """
    if in_sample.sharpe_ratio <= 0:
        return 100.0
    degradation = (
        (in_sample.sharpe_ratio - out_of_sample.sharpe_ratio)
        / abs(in_sample.sharpe_ratio)
    ) * 100
    return max(0.0, degradation)
