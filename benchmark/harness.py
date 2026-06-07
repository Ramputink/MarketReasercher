"""
benchmark.harness — Conservative execution harness (the referee)
================================================================

A deliberately *pessimistic*, leak-free backtester used ONLY by the benchmark.
It is intentionally NOT the research backtester (`engine/backtester.py`), because
a benchmark must be independent of the thing it is judging. Key differences from
the research engine, each closing a specific way of cheating:

    1. NEXT-BAR-OPEN FILLS (entry_lag_bars). A signal computed from bar i's close
       cannot be filled at bar i's close. It fills at bar i+1's open. This single
       change removes the most common silent look-ahead.

    2. INTRABAR SL/TP WITH WORST-CASE ORDERING. If a bar's range touches both the
       stop and the target, we assume the STOP filled first. No optimistic
       "it must have hit TP" assumptions.

    3. EXPLICIT FUNDING COST. Perp funding is modelled as an always-on cost
       accrued per bar to any open position. Crypto strategies that ignore
       funding are lying about their edge.

    4. HARD RUIN FLOOR. If equity ever crosses `ruin_equity_floor`, the account
       is flagged ruined and trading stops. A backtest cannot "recover" from a
       blown account — that is how you get the fictional 135% drawdowns.

    5. FIXED, COMPARABLE SIZING. Every candidate uses the same notional fraction,
       so drawdowns and Sharpes are comparable across strategies and runs.

The harness consumes the existing strategy interface unchanged:

    strategy_fn(df, bar_idx, position_or_None, regime=...) -> Optional[Signal]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Cost / execution model (overridable per-gate, e.g. for cost-stress tests)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecModel:
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    funding_bps_per_8h: float = 1.0
    entry_lag_bars: int = 1
    position_pct: float = 0.10
    initial_capital: float = 10_000.0
    ruin_equity_floor: float = 0.0
    warmup_bars: int = 60
    bar_hours: float = 1.0          # H1 default; funding scales with this
    default_time_stop_hours: int = 48
    default_sl_atr_mult: float = 2.0
    default_tp_atr_mult: float = 3.0


@dataclass
class _Pos:
    entry_time: int
    entry_idx: int
    side: str
    entry_price: float
    notional: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    time_stop_hours: float
    entry_commission: float = 0.0     # attributed to the trade at close (books balance)
    accrued_funding: float = 0.0      # accumulates per bar held


@dataclass
class HarnessResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    trade_returns: np.ndarray            # per-trade return on notional (for stats)
    ruined: bool
    n_bars: int
    # Convenience scalar metrics (computed leak-free, by this harness)
    net_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe_annualised: float = 0.0       # bar-level, correctly annualised
    sharpe_per_trade: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    n_trades: int = 0
    avg_holding_hours: float = 0.0
    regime_pnl: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "net_pnl": round(self.net_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe_ann": round(self.sharpe_annualised, 3),
            "sharpe_per_trade": round(self.sharpe_per_trade, 4),
            "profit_factor": round(self.profit_factor, 3),
            "win_rate": round(self.win_rate, 3),
            "max_dd_pct": round(self.max_drawdown_pct, 2),
            "ruined": self.ruined,
        }


class ConservativeBacktester:
    """Leak-free, cost-aware, ruin-aware backtester. See module docstring."""

    def __init__(self, exec_model: ExecModel):
        self.x = exec_model

    # ── cost helpers ────────────────────────────────────────────────────────
    def _slip(self, price: float, side: str, is_entry: bool) -> float:
        s = self.x.slippage_bps / 10_000.0
        adverse_up = (side == "long" and is_entry) or (side == "short" and not is_entry)
        return price * (1 + s) if adverse_up else price * (1 - s)

    def _commission(self, notional: float) -> float:
        return notional * self.x.commission_rate

    def _funding_per_bar(self, notional: float) -> float:
        # funding_bps_per_8h spread over the number of bars in 8h. Always a cost.
        bars_per_interval = max(8.0 / self.x.bar_hours, 1e-9)
        return notional * (abs(self.x.funding_bps_per_8h) / 10_000.0) / bars_per_interval

    # ── main loop ────────────────────────────────────────────────────────────
    def run(
        self,
        df: pd.DataFrame,
        strategy_fn: Callable,
        strategy_name: str = "unnamed",
    ) -> HarnessResult:
        x = self.x
        n = len(df)
        o = df["open"].to_numpy(dtype=float)
        h = df["high"].to_numpy(dtype=float)
        l = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        ts = df["timestamp"].to_numpy()
        atr = df["atr_14"].to_numpy(dtype=float) if "atr_14" in df.columns else np.full(n, np.nan)
        regimes = (df["_regime"].astype(str).to_numpy()
                   if "_regime" in df.columns else np.full(n, "unknown", dtype=object))

        equity = x.initial_capital
        eq_ts = []
        eq_val = []
        trades = []
        ruined = False

        pos: Optional[_Pos] = None
        pending_entry = None          # (Signal) -> fill at next bar open
        pending_exit = False          # signal-based exit -> fill at next bar open

        start = max(x.warmup_bars, x.entry_lag_bars)

        for i in range(start, n):
            price_close = c[i]

            # ── A) execute pending fills at THIS bar's open ──────────────────
            if pos is None and pending_entry is not None:
                sig = pending_entry
                pending_entry = None
                entry_price = self._slip(o[i], sig.side, is_entry=True)
                notional = equity * x.position_pct
                atr_i = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else entry_price * 0.02
                sl = sig.stop_loss
                tp = sig.take_profit
                if sl is None:
                    sl = (entry_price - x.default_sl_atr_mult * atr_i if sig.side == "long"
                          else entry_price + x.default_sl_atr_mult * atr_i)
                if tp is None:
                    tp = (entry_price + x.default_tp_atr_mult * atr_i if sig.side == "long"
                          else entry_price - x.default_tp_atr_mult * atr_i)
                pos = _Pos(
                    entry_time=int(ts[i]), entry_idx=i, side=sig.side,
                    entry_price=entry_price, notional=notional,
                    stop_loss=sl, take_profit=tp,
                    time_stop_hours=float(sig.time_stop_hours or x.default_time_stop_hours),
                    entry_commission=self._commission(notional),   # booked at close
                )

            elif pos is not None and pending_exit:
                pending_exit = False
                exit_price = self._slip(o[i], pos.side, is_entry=False)
                equity = self._close(pos, exit_price, int(ts[i]), i, "signal_exit",
                                     equity, trades, regimes)
                pos = None

            # ── B) manage open position intrabar on bar i ────────────────────
            if pos is not None:
                # funding accrual (booked into the trade at close, not equity here)
                pos.accrued_funding += self._funding_per_bar(pos.notional)

                exit_price = None
                exit_reason = ""
                # SL/TP — worst-case ordering: stop checked first.
                if pos.side == "long":
                    if pos.stop_loss is not None and l[i] <= pos.stop_loss:
                        exit_price, exit_reason = pos.stop_loss, "stop_loss"
                    elif pos.take_profit is not None and h[i] >= pos.take_profit:
                        exit_price, exit_reason = pos.take_profit, "take_profit"
                else:  # short
                    if pos.stop_loss is not None and h[i] >= pos.stop_loss:
                        exit_price, exit_reason = pos.stop_loss, "stop_loss"
                    elif pos.take_profit is not None and l[i] <= pos.take_profit:
                        exit_price, exit_reason = pos.take_profit, "take_profit"

                # time stop
                if exit_price is None:
                    hours_held = (int(ts[i]) - pos.entry_time) / 3_600_000
                    if hours_held >= pos.time_stop_hours:
                        exit_price, exit_reason = price_close, "time_stop"

                if exit_price is not None:
                    exit_price = self._slip(exit_price, pos.side, is_entry=False)
                    equity = self._close(pos, exit_price, int(ts[i]), i, exit_reason,
                                         equity, trades, regimes)
                    pos = None

            # ── C) ruin check (hard floor) ───────────────────────────────────
            # MTM reflects realised equity + unrealised pnl minus costs already
            # incurred but not yet booked (entry commission + accrued funding).
            mtm = equity
            if pos is not None:
                mtm += (self._unrealised(pos, price_close)
                        - pos.entry_commission - pos.accrued_funding)
            if mtm <= x.ruin_equity_floor:
                ruined = True
                eq_ts.append(int(ts[i]))
                eq_val.append(max(mtm, 0.0))
                break

            # ── D) signal generation (strict 1-bar lag via pending_*) ────────
            if pos is None and pending_entry is None:
                sig = strategy_fn(df, i, None)
                if sig is not None:
                    pending_entry = sig
            elif pos is not None and not pending_exit:
                sig = strategy_fn(df, i, pos)
                if sig is not None and sig.side != pos.side:
                    pending_exit = True

            eq_ts.append(int(ts[i]))
            eq_val.append(mtm)

        # close any residual position at the last available close
        if pos is not None and not ruined:
            last = c[n - 1]
            exit_price = self._slip(last, pos.side, is_entry=False)
            equity = self._close(pos, exit_price, int(ts[n - 1]), n - 1, "end_of_data",
                                 equity, trades, regimes)
            pos = None

        return self._finalise(trades, eq_ts, eq_val, ruined, n - start)

    # ── trade close + pnl ─────────────────────────────────────────────────────
    def _unrealised(self, pos: _Pos, price: float) -> float:
        if pos.side == "long":
            return (price - pos.entry_price) / pos.entry_price * pos.notional
        return (pos.entry_price - price) / pos.entry_price * pos.notional

    def _close(self, pos, exit_price, exit_time, exit_idx, reason, equity, trades, regimes):
        exit_commission = self._commission(pos.notional)
        if pos.side == "long":
            pnl_gross = (exit_price - pos.entry_price) / pos.entry_price * pos.notional
        else:
            pnl_gross = (pos.entry_price - exit_price) / pos.entry_price * pos.notional
        # ALL costs booked into the trade so sum(pnl_net) == equity change exactly.
        commission = pos.entry_commission + exit_commission
        pnl_net = pnl_gross - commission - pos.accrued_funding
        equity += pnl_net
        trades.append({
            "entry_time": pos.entry_time, "exit_time": exit_time,
            "side": pos.side, "entry_price": pos.entry_price, "exit_price": exit_price,
            "notional": pos.notional, "pnl_gross": pnl_gross, "commission": commission,
            "funding": pos.accrued_funding,
            "pnl_net": pnl_net, "ret_on_notional": pnl_net / pos.notional if pos.notional else 0.0,
            "exit_reason": reason,
            "entry_regime": str(regimes[pos.entry_idx]) if pos.entry_idx < len(regimes) else "unknown",
            "holding_hours": (exit_time - pos.entry_time) / 3_600_000,
        })
        return equity

    # ── metrics (computed by the referee, not the research engine) ────────────
    def _finalise(self, trades, eq_ts, eq_val, ruined, n_bars) -> HarnessResult:
        x = self.x
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        if len(eq_val) >= 2:
            eq = pd.Series(eq_val, index=pd.to_datetime(eq_ts, unit="ms"))
        else:
            eq = pd.Series(dtype=float)

        res = HarnessResult(
            trades=trades_df,
            equity_curve=eq,
            trade_returns=(trades_df["ret_on_notional"].to_numpy()
                           if len(trades_df) else np.array([])),
            ruined=ruined,
            n_bars=n_bars,
            n_trades=len(trades_df),
        )
        if len(trades_df) == 0 or len(eq) < 2:
            return res

        res.net_pnl = float(trades_df["pnl_net"].sum())
        # Return is taken from the realised ledger (sum of booked pnl_net), which
        # is the source of truth. The equity curve's final point is a per-bar
        # mark-to-market and may differ by the residual position's exit costs;
        # using the ledger keeps total_return reconciled with net_pnl exactly.
        res.total_return_pct = float(res.net_pnl / x.initial_capital * 100)

        # bar-level returns, correctly annualised (bars per year = 8760 / bar_hours)
        bar_rets = eq.pct_change().dropna()
        ann_factor = (365.0 * 24.0) / x.bar_hours
        if len(bar_rets) > 1 and bar_rets.std() > 0:
            res.sharpe_annualised = float(bar_rets.mean() / bar_rets.std() * np.sqrt(ann_factor))

        tr = trades_df["ret_on_notional"].to_numpy()
        if len(tr) > 1 and tr.std(ddof=1) > 0:
            res.sharpe_per_trade = float(tr.mean() / tr.std(ddof=1))

        wins = trades_df[trades_df["pnl_net"] > 0]["pnl_net"]
        losses = trades_df[trades_df["pnl_net"] <= 0]["pnl_net"]
        gp = float(wins.sum())
        gl = abs(float(losses.sum()))
        res.profit_factor = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
        res.win_rate = float(len(wins) / len(trades_df))

        peak = eq.cummax()
        dd = (eq - peak) / peak
        res.max_drawdown_pct = float(abs(dd.min()) * 100)
        res.avg_holding_hours = float(trades_df["holding_hours"].mean())

        # per-regime pnl attribution
        res.regime_pnl = {
            str(k): float(v) for k, v in
            trades_df.groupby("entry_regime")["pnl_net"].sum().items()
        }
        return res
