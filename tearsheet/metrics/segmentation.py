"""Trade segmentation helpers — direction, note/tag, and session breakdowns."""

from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------
# Shared stats builder
# ---------------------------------------------------------------------------

def _segment_stats(seg_trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-segment statistics for a slice of trades."""
    n = len(seg_trades)
    if n == 0:
        return {
            "n_trades": 0,
            "total_gross_pnl": 0.0,
            "total_net_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": None,
            "expectancy": 0.0,
            "biggest_winner": 0.0,
            "biggest_loser": 0.0,
        }

    pnls = [t["gross_pnl"] for t in seg_trades]
    net_pnls = [t["net_pnl"] for t in seg_trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    win_rate = len(winners) / n
    avg_win = sum(winners) / len(winners) if winners else 0.0
    avg_loss = sum(losers) / len(losers) if losers else 0.0
    profit_factor: Optional[float] = (
        sum(winners) / -sum(losers) if losers and sum(winners) > 0 else None
    )
    expectancy = sum(pnls) / n

    return {
        "n_trades": n,
        "total_gross_pnl": round(sum(pnls), 2),
        "total_net_pnl": round(sum(net_pnls), 2),
        "avg_pnl": round(expectancy, 2),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "expectancy": round(expectancy, 2),
        "biggest_winner": round(max(pnls), 2) if pnls else 0.0,
        "biggest_loser": round(min(pnls), 2) if pnls else 0.0,
    }


# ---------------------------------------------------------------------------
# Direction segmentation
# ---------------------------------------------------------------------------

def segment_by_direction(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return ``{'long': {...stats}, 'short': {...stats}}``."""
    long_trades = [t for t in trades if t.get("direction") == "long"]
    short_trades = [t for t in trades if t.get("direction") == "short"]
    return {
        "long": _segment_stats(long_trades),
        "short": _segment_stats(short_trades),
    }


# ---------------------------------------------------------------------------
# Note / tag segmentation
# ---------------------------------------------------------------------------

def segment_by_note(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a dict keyed by note/tag with per-note stat dicts."""
    by_note: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        tag = t.get("note", "") or ""
        by_note.setdefault(tag, []).append(t)

    return {note: _segment_stats(note_trades) for note, note_trades in by_note.items()}


# ---------------------------------------------------------------------------
# Session segmentation  (times in local CT as stored in the file)
# ---------------------------------------------------------------------------
# open   = entry before 10:30 CT  (hour < 10, or hour==10 and minute < 30)
# midday = 10:30 – 13:59 CT
# close  = 14:00 CT and later

def _session_bucket(entry_time) -> str:
    """Classify an entry_time timestamp into a session bucket."""
    try:
        import pandas as pd
        ts = pd.Timestamp(entry_time)
        h = ts.hour
        m = ts.minute
        minutes = h * 60 + m
        if minutes < 10 * 60 + 30:       # before 10:30
            return "open"
        elif minutes < 14 * 60:           # 10:30 – 13:59
            return "midday"
        else:
            return "close"
    except Exception:
        return "unknown"


def segment_by_session(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return ``{'open': {...}, 'midday': {...}, 'close': {...}}``."""
    buckets: dict[str, list[dict[str, Any]]] = {
        "open": [],
        "midday": [],
        "close": [],
    }
    for t in trades:
        bucket = _session_bucket(t.get("entry_time"))
        if bucket in buckets:
            buckets[bucket].append(t)

    return {k: _segment_stats(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Date-based segmentations (Phase 3)
# ---------------------------------------------------------------------------

def segment_by_date(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by exit date. Returns a list of stat dicts sorted chronologically.

    Each item includes all ``_segment_stats`` fields plus:
    * ``date`` — ISO date string (``"YYYY-MM-DD"``)
    * ``is_profitable`` — True if total_gross_pnl > 0
    """
    by_date: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        et = t.get("exit_time")
        if et is None:
            continue
        try:
            d = et.date()
        except AttributeError:
            import pandas as pd
            d = pd.Timestamp(et).date()
        key = str(d)
        by_date.setdefault(key, []).append(t)

    result: list[dict[str, Any]] = []
    for key in sorted(by_date.keys()):
        stats = _segment_stats(by_date[key])
        stats["date"] = key
        stats["is_profitable"] = stats["total_gross_pnl"] > 0
        result.append(stats)
    return result


def segment_by_day_of_week(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by day-of-week of exit date. Always returns all 5 weekday keys."""
    import pandas as pd

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    by_dow: dict[str, list[dict[str, Any]]] = {d: [] for d in day_names}

    for t in trades:
        if t.get("exit_time") is None:
            continue
        ts = pd.Timestamp(t["exit_time"])
        day_name = ts.day_name()
        if day_name in by_dow:
            by_dow[day_name].append(t)

    return {d: _segment_stats(by_dow[d]) for d in day_names}


def segment_by_week(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by ISO week of exit date. Keys: "YYYY-Www". Sorted chronologically."""
    import pandas as pd
    from collections import OrderedDict

    by_week: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        if t.get("exit_time") is None:
            continue
        ts = pd.Timestamp(t["exit_time"])
        iso = ts.isocalendar()
        key = f"{iso[0]}-W{iso[1]:02d}"
        by_week.setdefault(key, []).append(t)

    return OrderedDict(
        (k, _segment_stats(by_week[k])) for k in sorted(by_week.keys())
    )


def segment_by_month(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Group by month of exit date. Keys: "YYYY-MM". Sorted chronologically."""
    import pandas as pd
    from collections import OrderedDict

    by_month: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        if t.get("exit_time") is None:
            continue
        ts = pd.Timestamp(t["exit_time"])
        key = ts.strftime("%Y-%m")
        by_month.setdefault(key, []).append(t)

    return OrderedDict(
        (k, _segment_stats(by_month[k])) for k in sorted(by_month.keys())
    )


def pct_profitable_periods(segment_data) -> float:
    """Return fraction of segments with positive total_gross_pnl.

    Accepts either a ``list[dict]`` (from ``segment_by_date``) or a ``dict``
    (from ``segment_by_day_of_week`` / ``segment_by_week`` / ``segment_by_month``).
    """
    if isinstance(segment_data, dict):
        periods = list(segment_data.values())
    else:
        periods = list(segment_data)
    if not periods:
        return 0.0
    positive = sum(1 for s in periods if s.get("total_gross_pnl", s.get("gross_pnl", 0)) > 0)
    return positive / len(periods)


# ---------------------------------------------------------------------------
# Weekday / hour segmentation (Phase 3)  — no pandas import at module level
# ---------------------------------------------------------------------------

_WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}


def segment_by_weekday(trades: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Group by weekday of exit date.

    Returns ``dict[int, dict]`` keyed 0–6 (Monday=0). Only weekdays with trades
    are included.  Each stats dict has extra keys ``weekday`` (int) and
    ``weekday_name`` (3-letter abbreviation).
    """
    by_wd: dict[int, list[dict[str, Any]]] = {}
    for t in trades:
        et = t.get("exit_time")
        if et is None:
            continue
        try:
            wd = et.weekday()
        except AttributeError:
            import pandas as pd
            wd = pd.Timestamp(et).weekday()
        by_wd.setdefault(wd, []).append(t)

    result: dict[int, dict[str, Any]] = {}
    for wd in sorted(by_wd.keys()):
        stats = _segment_stats(by_wd[wd])
        stats["weekday"] = wd
        stats["weekday_name"] = _WEEKDAY_NAMES.get(wd, str(wd))
        result[wd] = stats
    return result


def segment_by_hour(trades: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Group by entry hour (0–23). Only hours with at least one trade are included.

    Each stats dict has an extra ``hour`` key (int).
    """
    by_hour: dict[int, list[dict[str, Any]]] = {}
    for t in trades:
        et = t.get("entry_time")
        if et is None:
            continue
        try:
            h = et.hour
        except AttributeError:
            import pandas as pd
            h = pd.Timestamp(et).hour
        by_hour.setdefault(h, []).append(t)

    result: dict[int, dict[str, Any]] = {}
    for h in sorted(by_hour.keys()):
        stats = _segment_stats(by_hour[h])
        stats["hour"] = h
        result[h] = stats
    return result
