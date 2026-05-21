import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .signals import generate_signals_series, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    position_size_pct: float = 0.1,
    fee_pct: float = 0.001,
) -> Dict:
    df = df.copy().dropna(subset=["rsi", "macd", "ema_50"])
    df = generate_signals_series(df)

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades: List[Dict] = []
    equity_curve = []

    for i, (ts, row) in enumerate(df.iterrows()):
        price = row["close"]
        signal = row["signal"]
        equity = capital + position * price
        equity_curve.append({"timestamp": ts, "equity": equity, "price": price})

        if position > 0:
            if price <= stop_loss:
                pnl = (price - entry_price) * position
                fee = price * position * fee_pct
                capital += position * price - fee
                trades.append({
                    "entry": entry_price,
                    "exit": price,
                    "pnl": pnl - fee,
                    "pnl_pct": (price - entry_price) / entry_price,
                    "exit_reason": "stop_loss",
                    "timestamp": ts,
                })
                position = 0.0
                continue

            if price >= take_profit:
                pnl = (price - entry_price) * position
                fee = price * position * fee_pct
                capital += position * price - fee
                trades.append({
                    "entry": entry_price,
                    "exit": price,
                    "pnl": pnl - fee,
                    "pnl_pct": (price - entry_price) / entry_price,
                    "exit_reason": "take_profit",
                    "timestamp": ts,
                })
                position = 0.0
                continue

            if signal == SIGNAL_SELL:
                pnl = (price - entry_price) * position
                fee = price * position * fee_pct
                capital += position * price - fee
                trades.append({
                    "entry": entry_price,
                    "exit": price,
                    "pnl": pnl - fee,
                    "pnl_pct": (price - entry_price) / entry_price,
                    "exit_reason": "signal",
                    "timestamp": ts,
                })
                position = 0.0

        elif position == 0 and signal == SIGNAL_BUY:
            invest = capital * position_size_pct
            fee = invest * fee_pct
            position = (invest - fee) / price
            entry_price = price
            capital -= invest
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)

    if position > 0:
        price = df["close"].iloc[-1]
        pnl = (price - entry_price) * position
        fee = price * position * fee_pct
        capital += position * price - fee
        trades.append({
            "entry": entry_price,
            "exit": price,
            "pnl": pnl - fee,
            "pnl_pct": (price - entry_price) / entry_price,
            "exit_reason": "end_of_data",
            "timestamp": df.index[-1],
        })

    equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
    metrics = _compute_metrics(trades, equity_df, initial_capital, capital)
    return {
        "metrics": metrics,
        "trades": trades,
        "equity_curve": equity_df,
        "df": df,
    }


def _compute_metrics(
    trades: List[Dict],
    equity_df: pd.DataFrame,
    initial_capital: float,
    final_capital: float,
) -> Dict:
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "final_capital": initial_capital,
        }

    pnls = [t["pnl"] for t in trades]
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p <= 0]

    win_rate = len(winning) / total_trades if total_trades > 0 else 0
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    equity = equity_df["equity"].values
    returns = np.diff(equity) / equity[:-1]
    sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24))
        if np.std(returns) > 0
        else 0.0
    )

    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = float(np.min(drawdowns)) * 100

    avg_win = float(np.mean(winning)) if winning else 0.0
    avg_loss = float(np.mean(losing)) if losing else 0.0
    gross_profit = sum(winning) if winning else 0.0
    gross_loss = abs(sum(losing)) if losing else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "win_rate": round(win_rate * 100, 2),
        "total_trades": total_trades,
        "winning_trades": len(winning),
        "losing_trades": len(losing),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 3),
        "final_capital": round(final_capital, 2),
    }
