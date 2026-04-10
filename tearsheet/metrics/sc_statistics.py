"""Sierra-Chart-style FlatToFlat trade statistics.

Mimics the layout and metrics produced by Sierra Chart's Trade Statistics window.
All P&L figures use net_pnl (after commissions).  MFE/MAE are scaled by total_qty
to approximate position-level open-profit/loss (note: underestimates scale-in trades
where the maximum position was held only briefly).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b and b != 0 else None


def _round2(v) -> Optional[float]:
    return round(v, 2) if v is not None else None


def _fmt_duration(seconds: float) -> str:
    """Format seconds → 'M:SS' (< 1 hr) or 'H:MM:SS' (≥ 1 hr)."""
    s = int(round(seconds))
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}:{sec:02d}"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _sort_trades(trades: list[dict]) -> list[dict]:
    """Sort trades by exit_time then trade_id for deterministic ordering."""
    def _key(t):
        et = t.get("exit_time")
        ts = pd.Timestamp(et) if et is not None else pd.Timestamp.max
        return (ts, t.get("trade_id", 0))
    return sorted(trades, key=_key)


def _cum_stats(pnls: list[float]) -> dict:
    """Compute cumulative P&L stats: highest cumulative profit/loss, max runup, max drawdown."""
    cum = 0.0
    max_cum = 0.0
    min_cum = 0.0
    peak = 0.0      # for drawdown tracking (peak → later trough)
    trough = 0.0    # for runup tracking (trough → later peak)
    max_dd = 0.0
    max_ru = 0.0

    for p in pnls:
        cum += p
        if cum > max_cum:
            max_cum = cum
        if cum < min_cum:
            min_cum = cum
        # drawdown: track peak, measure drop below it
        if cum > peak:
            peak = cum
        else:
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        # runup: track trough, measure rise above it
        if cum < trough:
            trough = cum
        else:
            ru = cum - trough
            if ru > max_ru:
                max_ru = ru

    return {
        "highest_cum_profit": _round2(max_cum),
        "lowest_cum_loss": _round2(min_cum),
        "max_runup": _round2(max_ru),
        "max_drawdown": _round2(-max_dd),  # stored as negative per SC convention
    }


def _streaks(pnls: list[float]) -> tuple[int, int]:
    """Return (max_consec_wins, max_consec_losses). Zero-P&L breaks both streaks."""
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


def _duration_s(t: dict) -> Optional[float]:
    """Return trade duration in seconds, or None if timestamps unavailable."""
    et, xt = t.get("entry_time"), t.get("exit_time")
    if et is None or xt is None:
        return None
    try:
        d = (pd.Timestamp(xt) - pd.Timestamp(et)).total_seconds()
        return d if d >= 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Streak distribution tables
# ---------------------------------------------------------------------------

def _streak_tables(trades: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (winners_table, losers_table) streak distribution rows.

    Each row: {"streak": int, "freq": int, "avg_pnl": float, "avg_next": float|None}
    avg_pnl is the average *total* P&L per streak occurrence (sum of trades in streak).
    avg_next is the average net_pnl of the trade immediately following the streak
    (averaged only over occurrences where a next trade exists).
    """
    pnls = [t["net_pnl"] for t in trades]

    streaks_w: dict[int, list[float]] = {}   # length → list of streak-total pnls
    next_w:    dict[int, list[float]] = {}   # length → list of next-trade pnls
    streaks_l: dict[int, list[float]] = {}
    next_l:    dict[int, list[float]] = {}

    i, n = 0, len(pnls)
    while i < n:
        if pnls[i] > 0:
            j = i + 1
            while j < n and pnls[j] > 0:
                j += 1
            length = j - i
            streak_total = sum(pnls[i:j])
            streaks_w.setdefault(length, []).append(streak_total)
            if j < n:
                next_w.setdefault(length, []).append(pnls[j])
            i = j
        elif pnls[i] < 0:
            j = i + 1
            while j < n and pnls[j] < 0:
                j += 1
            length = j - i
            streak_total = sum(pnls[i:j])
            streaks_l.setdefault(length, []).append(streak_total)
            if j < n:
                next_l.setdefault(length, []).append(pnls[j])
            i = j
        else:
            i += 1  # zero-P&L: breaks streak

    def _build(streaks, nexts):
        rows = []
        for length in sorted(streaks):
            freq = len(streaks[length])
            avg_pnl = sum(streaks[length]) / freq
            nx = nexts.get(length, [])
            avg_next = (sum(nx) / len(nx)) if nx else None
            rows.append({
                "streak": length,
                "freq": freq,
                "avg_pnl": _round2(avg_pnl),
                "avg_next": _round2(avg_next),
            })
        return rows

    return _build(streaks_w, next_w), _build(streaks_l, next_l)


# ---------------------------------------------------------------------------
# Per-subset stats
# ---------------------------------------------------------------------------

def _empty_stats() -> dict:
    return {k: None for k in [
        "closed_pnl", "total_profit", "total_loss", "profit_factor",
        "highest_cum_profit", "lowest_cum_loss", "max_runup", "max_drawdown",
        "max_open_profit", "max_open_loss",
        "avg_open_profit", "avg_open_loss",
        "avg_win_open_profit", "avg_win_open_loss",
        "avg_loss_open_profit", "avg_loss_open_loss",
        "total_commissions",
        "n_ftf", "pct_profitable",
        "n_winners", "n_losers", "n_longs", "n_shorts",
        "avg_ftf_pnl", "avg_winning", "avg_losing",
        "avg_profit_factor",
        "largest_winner", "largest_loser",
        "largest_winner_pct", "largest_loser_pct",
        "max_consec_wins", "max_consec_losses",
        "avg_time", "avg_time_win", "avg_time_loss",
        "longest_win", "longest_loss",
        "total_qty", "winning_qty", "losing_qty",
        "avg_qty_ftf", "avg_qty_win", "avg_qty_loss",
        "largest_trade_qty",
        "last_ftf_pnl", "expectancy",
    ]}


def _compute_subset(trades: list[dict]) -> dict:
    """Compute all SC-style stats for a given ordered subset of FTF trades."""
    if not trades:
        return _empty_stats()

    net_pnls = [t["net_pnl"] for t in trades]
    n = len(trades)

    winners = [t for t in trades if t["net_pnl"] > 0]
    losers  = [t for t in trades if t["net_pnl"] < 0]
    longs   = [t for t in trades if t.get("direction") == "long"]
    shorts  = [t for t in trades if t.get("direction") == "short"]

    total_profit = sum(t["net_pnl"] for t in winners)
    total_loss   = sum(t["net_pnl"] for t in losers)
    closed_pnl   = total_profit + total_loss
    profit_factor = _round2(total_profit / abs(total_loss)) if total_loss != 0 else None

    cum = _cum_stats(net_pnls)

    # Position-level MFE/MAE (scale by entry qty to approximate full-position excursion)
    pos_mfe = [t["mfe"] * t["total_qty"] for t in trades]
    pos_mae = [t["mae"] * t["total_qty"] for t in trades]  # mae is negative
    win_pos_mfe = [t["mfe"] * t["total_qty"] for t in winners]
    win_pos_mae = [t["mae"] * t["total_qty"] for t in winners]
    loss_pos_mfe = [t["mfe"] * t["total_qty"] for t in losers]
    loss_pos_mae = [t["mae"] * t["total_qty"] for t in losers]

    avg_win  = sum(t["net_pnl"] for t in winners) / len(winners) if winners else None
    avg_loss = sum(t["net_pnl"] for t in losers)  / len(losers)  if losers  else None
    avg_profit_factor = _round2(abs(avg_win) / abs(avg_loss)) if (avg_win and avg_loss) else None

    max_consec_wins, max_consec_losses = _streaks(net_pnls)

    # Duration stats
    all_durs  = [d for t in trades   if (d := _duration_s(t)) is not None]
    win_durs  = [d for t in winners  if (d := _duration_s(t)) is not None]
    loss_durs = [d for t in losers   if (d := _duration_s(t)) is not None]

    def _avg_dur(durs): return _fmt_duration(sum(durs) / len(durs)) if durs else None
    def _max_dur(durs): return _fmt_duration(max(durs)) if durs else None

    # Quantity stats
    total_qty   = sum(t["total_qty"] for t in trades)
    winning_qty = sum(t["total_qty"] for t in winners)
    losing_qty  = sum(t["total_qty"] for t in losers)

    largest_winner = _round2(max(t["net_pnl"] for t in winners)) if winners else None
    largest_loser  = _round2(min(t["net_pnl"] for t in losers))  if losers  else None

    lw_pct = _round2(largest_winner / total_profit * 100)  if (largest_winner and total_profit) else None
    ll_pct = _round2(abs(largest_loser) / abs(total_loss) * 100) if (largest_loser and total_loss) else None

    return {
        # P&L
        "closed_pnl":    _round2(closed_pnl),
        "total_profit":  _round2(total_profit),
        "total_loss":    _round2(total_loss),
        "profit_factor": profit_factor,
        # Equity
        **cum,
        # Open profit
        "max_open_profit":      _round2(max(pos_mfe)) if pos_mfe else None,
        "max_open_loss":        _round2(min(pos_mae)) if pos_mae else None,
        "avg_open_profit":      _round2(sum(pos_mfe) / n),
        "avg_open_loss":        _round2(sum(pos_mae) / n),
        "avg_win_open_profit":  _round2(sum(win_pos_mfe) / len(winners)) if winners else None,
        "avg_win_open_loss":    _round2(sum(win_pos_mae) / len(winners)) if winners else None,
        "avg_loss_open_profit": _round2(sum(loss_pos_mfe) / len(losers)) if losers else None,
        "avg_loss_open_loss":   _round2(sum(loss_pos_mae) / len(losers)) if losers else None,
        # Commissions
        "total_commissions": _round2(sum(t["fees"] for t in trades)),
        # Counts
        "n_ftf":          n,
        "pct_profitable": _round2(len(winners) / n * 100) if n else None,
        "n_winners": len(winners),
        "n_losers":  len(losers),
        "n_longs":   len(longs),
        "n_shorts":  len(shorts),
        # Averages
        "avg_ftf_pnl":      _round2(closed_pnl / n),
        "avg_winning":      _round2(avg_win),
        "avg_losing":       _round2(avg_loss),
        "avg_profit_factor": avg_profit_factor,
        # Extremes
        "largest_winner":     largest_winner,
        "largest_loser":      largest_loser,
        "largest_winner_pct": lw_pct,
        "largest_loser_pct":  ll_pct,
        # Streaks
        "max_consec_wins":   max_consec_wins,
        "max_consec_losses": max_consec_losses,
        # Time
        "avg_time":      _avg_dur(all_durs),
        "avg_time_win":  _avg_dur(win_durs),
        "avg_time_loss": _avg_dur(loss_durs),
        "longest_win":   _max_dur(win_durs),
        "longest_loss":  _max_dur(loss_durs),
        # Quantity
        "total_qty":   total_qty,
        "winning_qty": winning_qty,
        "losing_qty":  losing_qty,
        "avg_qty_ftf": _round2(total_qty / n) if n else None,
        "avg_qty_win": _round2(winning_qty / len(winners)) if winners else None,
        "avg_qty_loss": _round2(losing_qty / len(losers)) if losers else None,
        "largest_trade_qty": max(t["total_qty"] for t in trades) if trades else None,
        # Last trade / expectancy
        "last_ftf_pnl": _round2(trades[-1]["net_pnl"]) if trades else None,
        "expectancy":   _round2(closed_pnl / n) if n else None,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_sc_statistics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return SC-style statistics for all/long/short/daily subsets plus streak tables.

    Parameters
    ----------
    trades:
        Enriched FlatToFlat trade dicts as produced by
        :func:`tearsheet.recon.trades.enrich_trades`.

    Returns
    -------
    dict with keys:
        ``all``, ``long``, ``short``, ``daily`` — per-subset stats dicts
        ``consec_winners``, ``consec_losers`` — streak distribution tables (all trades)
    """
    if not trades:
        return {
            "all": _empty_stats(), "long": _empty_stats(),
            "short": _empty_stats(), "daily": _empty_stats(),
            "consec_winners": [], "consec_losers": [],
        }

    sorted_all = _sort_trades(trades)

    longs  = [t for t in sorted_all if t.get("direction") == "long"]
    shorts = [t for t in sorted_all if t.get("direction") == "short"]

    # Daily: trades whose exit date equals the most recent exit date in the dataset
    exit_dates = [
        pd.Timestamp(t["exit_time"]).date()
        for t in sorted_all if t.get("exit_time") is not None
    ]
    daily_trades: list[dict] = []
    if exit_dates:
        max_date = max(exit_dates)
        daily_trades = [
            t for t in sorted_all
            if t.get("exit_time") is not None
            and pd.Timestamp(t["exit_time"]).date() == max_date
        ]

    consec_winners, consec_losers = _streak_tables(sorted_all)

    return {
        "all":   _compute_subset(sorted_all),
        "long":  _compute_subset(longs),
        "short": _compute_subset(shorts),
        "daily": _compute_subset(daily_trades),
        "consec_winners": consec_winners,
        "consec_losers":  consec_losers,
    }
