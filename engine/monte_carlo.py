"""
CryptoResearchLab — Monte Carlo Simulation Engine
==================================================
Generates probabilistic performance estimates by resampling trade sequences.
Three simulation methods:
1. Trade Shuffling: randomly reorder actual trades
2. Return Bootstrap: resample trade returns with replacement
3. Equity Path Simulation: bootstrap equity returns at bar level

Outputs confidence intervals for: final PnL, max drawdown, Sharpe ratio,
win rate stability, and probability of ruin.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation run."""
    n_simulations: int = 0
    method: str = ""

    # PnL distribution
    pnl_mean: float = 0.0
    pnl_median: float = 0.0
    pnl_std: float = 0.0
    pnl_p5: float = 0.0     # 5th percentile (worst case)
    pnl_p25: float = 0.0
    pnl_p75: float = 0.0
    pnl_p95: float = 0.0    # 95th percentile (best case)

    # Drawdown distribution
    max_dd_mean: float = 0.0
    max_dd_median: float = 0.0
    max_dd_p95: float = 0.0  # worst-case drawdown (95th percentile)

    # Sharpe distribution
    sharpe_mean: float = 0.0
    sharpe_median: float = 0.0
    sharpe_p5: float = 0.0
    sharpe_p95: float = 0.0

    # Risk metrics
    prob_profit: float = 0.0    # P(final PnL > 0)
    prob_ruin: float = 0.0      # P(drawdown > 50%)
    prob_target: float = 0.0    # P(PnL > target)
    expected_cagr: float = 0.0

    # Win rate stability
    win_rate_mean: float = 0.0
    win_rate_std: float = 0.0

    # Raw distributions for plotting
    pnl_distribution: list = field(default_factory=list)
    dd_distribution: list = field(default_factory=list)
    sharpe_distribution: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "method": self.method,
            "pnl": {
                "mean": self.pnl_mean, "median": self.pnl_median,
                "std": self.pnl_std,
                "p5": self.pnl_p5, "p25": self.pnl_p25,
                "p75": self.pnl_p75, "p95": self.pnl_p95,
            },
            "max_drawdown": {
                "mean": self.max_dd_mean, "median": self.max_dd_median,
                "p95_worst": self.max_dd_p95,
            },
            "sharpe": {
                "mean": self.sharpe_mean, "median": self.sharpe_median,
                "p5": self.sharpe_p5, "p95": self.sharpe_p95,
            },
            "prob_profit": self.prob_profit,
            "prob_ruin": self.prob_ruin,
            "prob_target": self.prob_target,
            "win_rate_mean": self.win_rate_mean,
            "win_rate_std": self.win_rate_std,
        }

    def summary(self) -> str:
        lines = [
            f"Monte Carlo ({self.method}, {self.n_simulations} sims)",
            f"  PnL:    mean=${self.pnl_mean:.2f}  median=${self.pnl_median:.2f}  "
            f"[5%=${self.pnl_p5:.2f}, 95%=${self.pnl_p95:.2f}]",
            f"  MaxDD:  mean={self.max_dd_mean:.1f}%  worst95%={self.max_dd_p95:.1f}%",
            f"  Sharpe: mean={self.sharpe_mean:.3f}  "
            f"[5%={self.sharpe_p5:.3f}, 95%={self.sharpe_p95:.3f}]",
            f"  P(profit)={self.prob_profit*100:.1f}%  P(ruin)={self.prob_ruin*100:.1f}%  "
            f"WR={self.win_rate_mean*100:.1f}%±{self.win_rate_std*100:.1f}%",
        ]
        return "\n".join(lines)


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    """Compute max drawdown % from an equity curve array."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0) * 100
    return float(np.max(dd))


def _sharpe_from_returns(returns: np.ndarray, periods_per_year: float = 8760 / 1) -> float:
    """Compute annualized Sharpe from an array of period returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))


def monte_carlo_trades(
    trade_pnls: np.ndarray,
    initial_capital: float = 200.0,
    n_simulations: int = 1000,
    target_pnl: float = 100.0,
    ruin_dd_pct: float = 50.0,
    method: str = "shuffle",
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation on a sequence of trade PnLs.

    Methods:
    - "shuffle": randomly reorder trade sequence (preserves distribution)
    - "bootstrap": resample trades with replacement (allows repeated trades)
    - "block_bootstrap": resample blocks of consecutive trades

    Args:
        trade_pnls: Array of net PnL per trade (positive = win, negative = loss)
        initial_capital: Starting capital
        n_simulations: Number of simulation runs
        target_pnl: Target PnL for P(target) calculation
        ruin_dd_pct: Drawdown % threshold for ruin probability
        method: "shuffle", "bootstrap", or "block_bootstrap"
        seed: Random seed for reproducibility
    """
    if len(trade_pnls) < 5:
        logger.warning(f"Monte Carlo: only {len(trade_pnls)} trades, need at least 5")
        return MonteCarloResult(n_simulations=0, method=method)

    rng = np.random.RandomState(seed)
    n_trades = len(trade_pnls)

    final_pnls = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)
    win_rates = np.zeros(n_simulations)

    block_size = max(3, n_trades // 10)

    for sim in range(n_simulations):
        if method == "shuffle":
            idx = rng.permutation(n_trades)
            sim_trades = trade_pnls[idx]
        elif method == "bootstrap":
            idx = rng.randint(0, n_trades, size=n_trades)
            sim_trades = trade_pnls[idx]
        elif method == "block_bootstrap":
            blocks = []
            total = 0
            while total < n_trades:
                start = rng.randint(0, max(1, n_trades - block_size))
                block = trade_pnls[start:start + block_size]
                blocks.append(block)
                total += len(block)
            sim_trades = np.concatenate(blocks)[:n_trades]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Build equity curve
        equity = np.zeros(n_trades + 1)
        equity[0] = initial_capital
        for i, pnl in enumerate(sim_trades):
            equity[i + 1] = equity[i] + pnl

        final_pnls[sim] = equity[-1] - initial_capital
        max_dds[sim] = _max_drawdown_from_equity(equity)

        # Sharpe from trade returns
        returns = sim_trades / initial_capital
        sharpes[sim] = _sharpe_from_returns(returns, periods_per_year=n_trades * 2)
        win_rates[sim] = np.sum(sim_trades > 0) / len(sim_trades)

    result = MonteCarloResult(
        n_simulations=n_simulations,
        method=method,

        pnl_mean=float(np.mean(final_pnls)),
        pnl_median=float(np.median(final_pnls)),
        pnl_std=float(np.std(final_pnls)),
        pnl_p5=float(np.percentile(final_pnls, 5)),
        pnl_p25=float(np.percentile(final_pnls, 25)),
        pnl_p75=float(np.percentile(final_pnls, 75)),
        pnl_p95=float(np.percentile(final_pnls, 95)),

        max_dd_mean=float(np.mean(max_dds)),
        max_dd_median=float(np.median(max_dds)),
        max_dd_p95=float(np.percentile(max_dds, 95)),

        sharpe_mean=float(np.mean(sharpes)),
        sharpe_median=float(np.median(sharpes)),
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p95=float(np.percentile(sharpes, 95)),

        prob_profit=float(np.mean(final_pnls > 0)),
        prob_ruin=float(np.mean(max_dds > ruin_dd_pct)),
        prob_target=float(np.mean(final_pnls > target_pnl)),

        win_rate_mean=float(np.mean(win_rates)),
        win_rate_std=float(np.std(win_rates)),

        pnl_distribution=final_pnls.tolist(),
        dd_distribution=max_dds.tolist(),
        sharpe_distribution=sharpes.tolist(),
    )

    return result


def monte_carlo_equity(
    equity_returns: np.ndarray,
    initial_capital: float = 200.0,
    n_simulations: int = 1000,
    n_periods: Optional[int] = None,
    target_pnl: float = 100.0,
    ruin_dd_pct: float = 50.0,
    seed: Optional[int] = 42,
) -> MonteCarloResult:
    """
    Run Monte Carlo on equity curve returns (bar-level bootstrap).
    Useful when you have the per-bar returns from a backtest equity curve.

    Args:
        equity_returns: Array of per-bar % returns from the equity curve
        initial_capital: Starting capital
        n_simulations: Number of runs
        n_periods: Length of simulated path (default: same as input)
        target_pnl: Target for P(target)
        ruin_dd_pct: DD threshold for ruin
    """
    if len(equity_returns) < 20:
        return MonteCarloResult(n_simulations=0, method="equity_bootstrap")

    rng = np.random.RandomState(seed)
    n = n_periods or len(equity_returns)

    final_pnls = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    sharpes = np.zeros(n_simulations)

    for sim in range(n_simulations):
        # Bootstrap returns
        idx = rng.randint(0, len(equity_returns), size=n)
        sim_returns = equity_returns[idx]

        # Build equity
        equity = np.zeros(n + 1)
        equity[0] = initial_capital
        for i in range(n):
            equity[i + 1] = equity[i] * (1 + sim_returns[i])

        final_pnls[sim] = equity[-1] - initial_capital
        max_dds[sim] = _max_drawdown_from_equity(equity)
        sharpes[sim] = _sharpe_from_returns(sim_returns, periods_per_year=8760)

    return MonteCarloResult(
        n_simulations=n_simulations,
        method="equity_bootstrap",

        pnl_mean=float(np.mean(final_pnls)),
        pnl_median=float(np.median(final_pnls)),
        pnl_std=float(np.std(final_pnls)),
        pnl_p5=float(np.percentile(final_pnls, 5)),
        pnl_p25=float(np.percentile(final_pnls, 25)),
        pnl_p75=float(np.percentile(final_pnls, 75)),
        pnl_p95=float(np.percentile(final_pnls, 95)),

        max_dd_mean=float(np.mean(max_dds)),
        max_dd_median=float(np.median(max_dds)),
        max_dd_p95=float(np.percentile(max_dds, 95)),

        sharpe_mean=float(np.mean(sharpes)),
        sharpe_median=float(np.median(sharpes)),
        sharpe_p5=float(np.percentile(sharpes, 5)),
        sharpe_p95=float(np.percentile(sharpes, 95)),

        prob_profit=float(np.mean(final_pnls > 0)),
        prob_ruin=float(np.mean(max_dds > ruin_dd_pct)),
        prob_target=float(np.mean(final_pnls > target_pnl)),

        win_rate_mean=0.0,
        win_rate_std=0.0,

        pnl_distribution=final_pnls.tolist(),
        dd_distribution=max_dds.tolist(),
        sharpe_distribution=sharpes.tolist(),
    )
