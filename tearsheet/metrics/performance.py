"""Performance metrics computation."""

from __future__ import annotations

import math
from typing import Any, Optional


def _safe_div(a: float, b: float) -> Optional[float]:
    return a / b if b != 0 else None


def compute_metrics(trades: list[dict[str, Any]], equity_curve: list[dict[str, Any]], be_threshold: float = 0.0) -> dict[str, Any]:
    """Return a metrics dict from a list of trade dicts and the equity curve.

    Parameters
    ----------
    trades:
        List of trade dicts as returned by :func:`tearsheet.recon.trades.reconstruct_trades`.
    equity_curve:
        List of ``{DateTime, balance}`` dicts from
        :func:`tearsheet.recon.equity.build_equity_curve`.
    """
    if not trades:
        return _empty_metrics()

    n = len(trades)
    pnls = [t["gross_pnl"] for t in trades]
    net_pnls = [t["net_pnl"] for t in trades]
    fees = [t["fees"] for t in trades]

    total_gross = sum(pnls)
    total_fees = sum(fees)
    total_net = sum(net_pnls)

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / n if n else 0.0
    win_rate_be = sum(1 for p in pnls if p > -be_threshold) / n if n else 0.0
    avg_win = sum(winners) / len(winners) if winners else 0.0
    avg_loss = sum(losers) / len(losers) if losers else 0.0
    profit_factor = _safe_div(sum(winners), -sum(losers)) if losers else None

    expectancy = total_gross / n if n else 0.0
    std_pnl = _std(pnls)

    sqn = None
    if std_pnl and std_pnl > 0:
        sqn = (expectancy / std_pnl) * math.sqrt(n)

    # Drawdown from equity curve
    max_dd, max_dd_pct, ulcer = _drawdown_stats(equity_curve)

    # Daily returns for Sharpe / Sortino
    daily = _daily_pnl(trades)
    sharpe = _sharpe(daily)
    sortino = _sortino(daily)
    calmar = _safe_div(total_net, -max_dd) if max_dd else None
    recovery_factor = _safe_div(total_net, abs(max_dd)) if max_dd else None

    # Streak
    max_consec_win, max_consec_loss = _streaks(pnls)

    # --- Phase 2 extended metrics ---
    biggest_winner = max(pnls) if pnls else None
    biggest_loser = min(pnls) if pnls else None
    payoff_ratio = _safe_div(abs(avg_win), abs(avg_loss)) if avg_loss != 0 else None
    gain_to_pain = _safe_div(total_gross, sum(-p for p in pnls if p < 0)) if losers else None

    # Duration stats (seconds)
    durations = _durations_s(trades)
    avg_duration_s = sum(durations) / len(durations) if durations else None
    win_durs = [d for t, d in zip(trades, durations) if t["gross_pnl"] > 0]
    loss_durs = [d for t, d in zip(trades, durations) if t["gross_pnl"] < 0]
    avg_duration_winners_s = sum(win_durs) / len(win_durs) if win_durs else None
    avg_duration_losers_s = sum(loss_durs) / len(loss_durs) if loss_durs else None
    shortest_trade_s = min(durations) if durations else None
    longest_trade_s = max(durations) if durations else None

    # MFE / MAE averages
    mfes = [t["mfe"] for t in trades]
    maes = [t["mae"] for t in trades]
    avg_mfe = sum(mfes) / n if n else 0.0
    avg_mae = sum(maes) / n if n else 0.0

    # MFE capture: how much of the MFE was captured as gross PnL
    captures = [t["gross_pnl"] / t["mfe"] for t in trades if t["mfe"] and t["mfe"] > 0]
    mfe_capture_pct = sum(captures) / len(captures) if captures else None

    # Long / short breakdown
    longs = [t for t in trades if t.get("direction") == "long"]
    shorts = [t for t in trades if t.get("direction") == "short"]
    long_count = len(longs)
    short_count = len(shorts)
    long_winners = [t for t in longs if t["gross_pnl"] > 0]
    short_winners = [t for t in shorts if t["gross_pnl"] > 0]
    long_win_rate = len(long_winners) / long_count if long_count else None
    short_win_rate = len(short_winners) / short_count if short_count else None

    # Phase 3: R-multiple stats
    import pandas as pd
    r_multiples = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    r_avg = sum(r_multiples) / len(r_multiples) if r_multiples else None
    r_median = _median(r_multiples) if r_multiples else None
    r_pct_1r = sum(1 for r in r_multiples if r >= 1) / len(r_multiples) if r_multiples else None
    r_pct_2r = sum(1 for r in r_multiples if r >= 2) / len(r_multiples) if r_multiples else None
    total_r = sum(r_multiples) if r_multiples else None
    avg_r_multiple = r_avg  # alias kept for backward compat
    positive_r_count = sum(1 for r in r_multiples if r > 0)
    negative_r_count = sum(1 for r in r_multiples if r < 0)

    # Phase 3: Calendar stats
    exit_dates = set()
    for t in trades:
        if t.get("exit_time") is not None:
            exit_dates.add(pd.Timestamp(t["exit_time"]).date())
    total_trading_days = len(exit_dates)

    daily_pnls_by_date: dict = {}
    for t in trades:
        if t.get("exit_time") is not None:
            d = pd.Timestamp(t["exit_time"]).date()
            daily_pnls_by_date[d] = daily_pnls_by_date.get(d, 0.0) + t["gross_pnl"]
    profitable_days = sum(1 for v in daily_pnls_by_date.values() if v > 0)
    pct_profitable_days = profitable_days / total_trading_days if total_trading_days else None
    avg_trades_per_day = round(n / total_trading_days, 2) if total_trading_days else None
    avg_daily_net_pnl = round(total_net / total_trading_days, 2) if total_trading_days else None

    return {
        "n_trades": n,
        "total_gross_pnl": round(total_gross, 2),
        "total_fees": round(total_fees, 2),
        "total_net_pnl": round(total_net, 2),
        "win_rate": round(win_rate, 4),
        "win_rate_be": round(win_rate_be, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "expectancy": round(expectancy, 2),
        "sqn": round(sqn, 4) if sqn is not None else None,
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "ulcer_index": round(ulcer, 4),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "calmar_ratio": round(calmar, 4) if calmar is not None else None,
        "recovery_factor": round(recovery_factor, 4) if recovery_factor is not None else None,
        "max_consec_wins": max_consec_win,
        "max_consec_losses": max_consec_loss,
        # Phase 2 extended
        "biggest_winner": round(biggest_winner, 2) if biggest_winner is not None else None,
        "biggest_loser": round(biggest_loser, 2) if biggest_loser is not None else None,
        "payoff_ratio": round(payoff_ratio, 4) if payoff_ratio is not None else None,
        "gain_to_pain": round(gain_to_pain, 4) if gain_to_pain is not None else None,
        "avg_duration_s": round(avg_duration_s, 1) if avg_duration_s is not None else None,
        "avg_duration_winners_s": round(avg_duration_winners_s, 1) if avg_duration_winners_s is not None else None,
        "avg_duration_losers_s": round(avg_duration_losers_s, 1) if avg_duration_losers_s is not None else None,
        "shortest_trade_s": round(shortest_trade_s, 1) if shortest_trade_s is not None else None,
        "longest_trade_s": round(longest_trade_s, 1) if longest_trade_s is not None else None,
        "avg_mfe": round(avg_mfe, 2),
        "avg_mae": round(avg_mae, 2),
        "mfe_capture_pct": round(mfe_capture_pct, 4) if mfe_capture_pct is not None else None,
        "long_win_rate": round(long_win_rate, 4) if long_win_rate is not None else None,
        "short_win_rate": round(short_win_rate, 4) if short_win_rate is not None else None,
        "long_count": long_count,
        "short_count": short_count,
        # Phase 3
        "avg_r_multiple": round(avg_r_multiple, 4) if avg_r_multiple is not None else None,
        "r_avg": round(r_avg, 4) if r_avg is not None else None,
        "r_median": round(r_median, 4) if r_median is not None else None,
        "r_pct_1r": round(r_pct_1r, 4) if r_pct_1r is not None else None,
        "r_pct_2r": round(r_pct_2r, 4) if r_pct_2r is not None else None,
        "total_r": round(total_r, 4) if total_r is not None else None,
        "positive_r_count": positive_r_count,
        "negative_r_count": negative_r_count,
        "trading_days": total_trading_days,
        "total_trading_days": total_trading_days,
        "pct_profitable_days": round(pct_profitable_days, 4) if pct_profitable_days is not None else None,
        "avg_trades_per_day": avg_trades_per_day,
        "avg_daily_net_pnl": avg_daily_net_pnl,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_metrics() -> dict[str, Any]:
    return {k: None for k in [
        "n_trades", "total_gross_pnl", "total_fees", "total_net_pnl",
        "win_rate", "avg_win", "avg_loss", "profit_factor", "expectancy",
        "sqn", "max_drawdown", "max_drawdown_pct", "ulcer_index",
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_consec_wins", "max_consec_losses",
        # Phase 2 extended
        "biggest_winner", "biggest_loser", "payoff_ratio", "gain_to_pain",
        "avg_duration_s", "avg_duration_winners_s", "avg_duration_losers_s",
        "shortest_trade_s", "longest_trade_s",
        "avg_mfe", "avg_mae", "mfe_capture_pct",
        "long_win_rate", "short_win_rate", "long_count", "short_count",
        # Phase 3
        "avg_r_multiple", "r_avg", "r_median", "r_pct_1r", "r_pct_2r", "total_r",
        "positive_r_count", "negative_r_count",
        "recovery_factor", "win_rate_be",
        "trading_days", "total_trading_days", "pct_profitable_days", "avg_trades_per_day",
        "avg_daily_net_pnl",
    ]}


def _median(values: list[float]) -> float:
    sorted_v = sorted(values)
    n = len(sorted_v)
    mid = n // 2
    return (sorted_v[mid] + sorted_v[~mid]) / 2.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _drawdown_stats(equity_curve: list[dict]) -> tuple[float, float, float]:
    """Return ``(max_dd_abs, max_dd_pct, ulcer_index)``."""
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0.0

    balances = [p["balance"] for p in equity_curve]
    peak = balances[0]
    max_dd = 0.0
    max_dd_pct = 0.0
    dd_sq_sum = 0.0

    for bal in balances:
        if bal > peak:
            peak = bal
        dd = peak - bal
        if dd > max_dd:
            max_dd = dd
        dd_pct = dd / peak if peak else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
        dd_sq_sum += dd_pct ** 2

    ulcer = math.sqrt(dd_sq_sum / len(balances))
    return max_dd, max_dd_pct, ulcer


def _daily_pnl(trades: list[dict]) -> list[float]:
    """Aggregate gross P&L by date."""
    by_date: dict = {}
    for t in trades:
        if t["exit_time"] is None:
            continue
        import pandas as pd
        date = pd.Timestamp(t["exit_time"]).date()
        by_date[date] = by_date.get(date, 0.0) + t["gross_pnl"]
    return list(by_date.values())


def _sharpe(daily: list[float], risk_free: float = 0.0) -> Optional[float]:
    if len(daily) < 2:
        return None
    mean = sum(daily) / len(daily)
    std = _std(daily)
    if std == 0:
        return None
    return (mean - risk_free) / std * math.sqrt(252)


def _sortino(daily: list[float], risk_free: float = 0.0) -> Optional[float]:
    if len(daily) < 2:
        return None
    mean = sum(daily) / len(daily)
    neg = [d for d in daily if d < risk_free]
    if not neg:
        return None
    downside_std = _std(neg)
    if downside_std == 0:
        return None
    return (mean - risk_free) / downside_std * math.sqrt(252)


def _streaks(pnls: list[float]) -> tuple[int, int]:
    max_win = max_loss = cur_win = cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1
            cur_loss = 0
        elif p < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = cur_loss = 0
        if cur_win > max_win:
            max_win = cur_win
        if cur_loss > max_loss:
            max_loss = cur_loss
    return max_win, max_loss


def _durations_s(trades: list[dict]) -> list[float]:
    """Return list of trade durations in seconds (only for closed trades)."""
    import pandas as pd
    durations = []
    for t in trades:
        if t.get("entry_time") is not None and t.get("exit_time") is not None:
            try:
                dur = (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds()
                if dur >= 0:
                    durations.append(dur)
            except Exception:
                pass
    return durations
