"""Rolling window metrics over ordered trades."""

from __future__ import annotations

import math
from typing import Any, Optional


def compute_rolling_metrics(trades: list[dict[str, Any]], window: int = 20) -> dict[str, Any]:
    """Compute rolling window metrics and equity curve smoothness.

    Returns
    -------
    dict with keys:
        ``window``  — the requested window size
        ``rolling`` — list of per-trade dicts (grows from trade 0)
        ``equity_r_squared`` — R² of linear regression on cumulative P&L,
                                or None if fewer than 2 trades
    """
    n = len(trades)
    rolling: list[dict[str, Any]] = []

    for i in range(n):
        w = min(i + 1, window)
        window_trades = trades[i - w + 1 : i + 1]
        pnls = [t["gross_pnl"] for t in window_trades]

        expectancy = sum(pnls) / w
        win_rate = sum(1 for p in pnls if p > 0) / w

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        if losers and sum(winners) > 0:
            profit_factor: Optional[float] = round(sum(winners) / -sum(losers), 4)
        else:
            profit_factor = None

        sharpe: Optional[float] = None
        if w >= 2:
            mean = expectancy
            variance = sum((p - mean) ** 2 for p in pnls) / (w - 1)
            std = math.sqrt(variance) if variance > 0 else 0.0
            if std > 0:
                sharpe = round(mean / std, 4)

        rolling.append({
            "trade_id": trades[i].get("trade_id", i + 1),
            "trade_index": i,
            "rolling_expectancy": round(expectancy, 4),
            "rolling_win_rate": round(win_rate, 4),
            "rolling_profit_factor": profit_factor,
            "rolling_sharpe": sharpe,
            "window_size": w,
        })

    return {
        "window": window,
        "rolling": rolling,
        "equity_r_squared": _equity_r_squared(trades),
    }


def _equity_r_squared(trades: list[dict[str, Any]]) -> Optional[float]:
    """R² of a linear regression on the cumulative gross P&L curve."""
    if len(trades) < 2:
        return None

    cumulative = 0.0
    y: list[float] = []
    for t in trades:
        cumulative += t["gross_pnl"]
        y.append(cumulative)

    n = len(y)
    x = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n

    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_yy = sum((yi - y_mean) ** 2 for yi in y)

    if ss_xx == 0 or ss_yy == 0:
        return None

    r = ss_xy / math.sqrt(ss_xx * ss_yy)
    return round(r * r, 6)
