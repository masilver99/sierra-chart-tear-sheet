"""Hierarchical year → quarter → month → week → day summary rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

FUTURES_LONG_TERM_SHARE = 0.60
FUTURES_SHORT_TERM_SHARE = 0.40
FUTURES_LONG_TERM_RATE = 0.15
DEFAULT_TAX_RATE = 0.24


def _round2(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _sort_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(trade: dict[str, Any]) -> tuple[pd.Timestamp, Any]:
        exit_time = trade.get("exit_time")
        ts = pd.Timestamp(exit_time) if exit_time is not None else pd.Timestamp.max
        return ts, trade.get("trade_id", 0)

    return sorted(trades, key=_key)


def _duration_s(trade: dict[str, Any]) -> float | None:
    entry_time = trade.get("entry_time")
    exit_time = trade.get("exit_time")
    if entry_time is None or exit_time is None:
        return None
    try:
        duration = (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).total_seconds()
    except Exception:
        return None
    return duration if duration >= 0 else None


def _date_label(date_value) -> str:
    ts = pd.Timestamp(date_value)
    return f"{ts.strftime('%a')} {ts.strftime('%Y-%m-%d')}"


def _short_date_label(date_value) -> str:
    ts = pd.Timestamp(date_value)
    return f"{ts.strftime('%b')} {ts.day}"


def _week_label(week_key: str, start_date, end_date) -> str:
    week_num = int(week_key.split("-W", 1)[1])
    start_label = _short_date_label(start_date)
    end_label = _short_date_label(end_date)
    return f"W{week_num:02d} ({start_label} - {end_label})"


def _quarter_label(year: int, quarter: int) -> str:
    return f"Q{quarter} {year}"


def estimate_futures_taxes(net_pnl: float | None, tax_rate: float = DEFAULT_TAX_RATE) -> float:
    taxable_pnl = max(float(net_pnl or 0.0), 0.0)
    effective_rate = FUTURES_LONG_TERM_SHARE * FUTURES_LONG_TERM_RATE + FUTURES_SHORT_TERM_SHARE * tax_rate
    return round(taxable_pnl * effective_rate, 2)


def _summary_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "net_pnl": 0.0,
            "n_trades": 0,
            "n_days": 0,
            "avg_pnl_per_day": None,
            "avg_pnl_per_trade": None,
            "win_rate": None,
            "profit_factor": None,
            "best_trade": None,
            "worst_trade": None,
            "total_fees": None,
            "avg_hold_s": None,
            "estimated_taxes": 0.0,
        }

    ordered = _sort_trades(trades)
    net_pnls = [float(trade.get("net_pnl", 0.0) or 0.0) for trade in ordered]
    fees = [float(trade.get("fees", 0.0) or 0.0) for trade in ordered]
    win_pnls = [pnl for pnl in net_pnls if pnl > 0]
    loss_pnls = [pnl for pnl in net_pnls if pnl < 0]
    exit_dates = {
        pd.Timestamp(trade["exit_time"]).date()
        for trade in ordered
        if trade.get("exit_time") is not None
    }
    durations = [duration for trade in ordered if (duration := _duration_s(trade)) is not None]

    total_net = sum(net_pnls)
    total_loss = sum(loss_pnls)
    n_trades = len(ordered)
    n_days = len(exit_dates)

    return {
        "net_pnl": _round2(total_net),
        "n_trades": n_trades,
        "n_days": n_days,
        "avg_pnl_per_day": _round2(total_net / n_days) if n_days else None,
        "avg_pnl_per_trade": _round2(total_net / n_trades) if n_trades else None,
        "win_rate": round(len(win_pnls) / n_trades, 4) if n_trades else None,
        "profit_factor": _round2(sum(win_pnls) / abs(total_loss)) if total_loss else None,
        "best_trade": _round2(max(net_pnls)) if net_pnls else None,
        "worst_trade": _round2(min(net_pnls)) if net_pnls else None,
        "total_fees": _round2(sum(fees)),
        "avg_hold_s": round(sum(durations) / len(durations)) if durations else None,
        "estimated_taxes": estimate_futures_taxes(total_net),
    }


def _row(
    *,
    row_id: str,
    parent_id: str | None,
    level: str,
    label: str,
    start_date,
    end_date,
    has_children: bool,
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    stats = _summary_stats(trades)
    return {
        "id": row_id,
        "parent_id": parent_id,
        "level": level,
        "label": label,
        "start_date": str(pd.Timestamp(start_date).date()),
        "end_date": str(pd.Timestamp(end_date).date()),
        "has_children": has_children,
        **stats,
    }


def compute_monthly_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Return flattened hierarchical summary rows for year → quarter → month → week → day."""
    ordered = [
        trade for trade in _sort_trades(trades)
        if trade.get("exit_time") is not None
    ]
    if not ordered:
        return {
            "rows": [],
            "year_count": 0,
            "quarter_count": 0,
            "month_count": 0,
            "default_tax_rate": DEFAULT_TAX_RATE,
            "long_term_rate": FUTURES_LONG_TERM_RATE,
            "long_term_share": FUTURES_LONG_TERM_SHARE,
            "short_term_share": FUTURES_SHORT_TERM_SHARE,
        }

    years: dict[str, dict[str, Any]] = {}
    for trade in ordered:
        ts = pd.Timestamp(trade["exit_time"])
        year_key = str(ts.year)
        quarter_num = ((ts.month - 1) // 3) + 1
        quarter_key = f"{ts.year}-Q{quarter_num}"
        month_key = ts.strftime("%Y-%m")
        iso = ts.isocalendar()
        week_key = f"{iso.year}-W{iso.week:02d}"
        day_key = ts.strftime("%Y-%m-%d")

        year_bucket = years.setdefault(year_key, {"trades": [], "quarters": {}})
        year_bucket["trades"].append(trade)

        quarter_bucket = year_bucket["quarters"].setdefault(quarter_key, {"trades": [], "months": {}})
        quarter_bucket["trades"].append(trade)

        month_bucket = quarter_bucket["months"].setdefault(month_key, {"trades": [], "weeks": {}})
        month_bucket["trades"].append(trade)

        week_bucket = month_bucket["weeks"].setdefault(week_key, {"trades": [], "days": {}})
        week_bucket["trades"].append(trade)

        day_bucket = week_bucket["days"].setdefault(day_key, {"trades": []})
        day_bucket["trades"].append(trade)

    rows: list[dict[str, Any]] = []
    month_count = 0
    quarter_count = 0
    for year_key in sorted(years.keys()):
        year_bucket = years[year_key]
        year_trades = _sort_trades(year_bucket["trades"])
        year_row_id = f"year-{year_key}"
        year_start = year_trades[0]["exit_time"]
        year_end = year_trades[-1]["exit_time"]
        rows.append(_row(
            row_id=year_row_id,
            parent_id=None,
            level="year",
            label=year_key,
            start_date=year_start,
            end_date=year_end,
            has_children=bool(year_bucket["quarters"]),
            trades=year_trades,
        ))

        for quarter_key in sorted(year_bucket["quarters"].keys()):
            quarter_bucket = year_bucket["quarters"][quarter_key]
            quarter_trades = _sort_trades(quarter_bucket["trades"])
            quarter_row_id = f"quarter-{quarter_key}"
            quarter_start = quarter_trades[0]["exit_time"]
            quarter_end = quarter_trades[-1]["exit_time"]
            quarter_count += 1
            rows.append(_row(
                row_id=quarter_row_id,
                parent_id=year_row_id,
                level="quarter",
                label=_quarter_label(pd.Timestamp(quarter_start).year, ((pd.Timestamp(quarter_start).month - 1) // 3) + 1),
                start_date=quarter_start,
                end_date=quarter_end,
                has_children=bool(quarter_bucket["months"]),
                trades=quarter_trades,
            ))

            for month_key in sorted(quarter_bucket["months"].keys()):
                month_bucket = quarter_bucket["months"][month_key]
                month_trades = _sort_trades(month_bucket["trades"])
                month_row_id = f"month-{month_key}"
                month_start = month_trades[0]["exit_time"]
                month_end = month_trades[-1]["exit_time"]
                month_count += 1
                rows.append(_row(
                    row_id=month_row_id,
                    parent_id=quarter_row_id,
                    level="month",
                    label=pd.Timestamp(month_start).strftime("%b %Y"),
                    start_date=month_start,
                    end_date=month_end,
                    has_children=bool(month_bucket["weeks"]),
                    trades=month_trades,
                ))

                for week_key in sorted(month_bucket["weeks"].keys()):
                    week_bucket = month_bucket["weeks"][week_key]
                    week_trades = _sort_trades(week_bucket["trades"])
                    week_row_id = f"week-{month_key}-{week_key}"
                    week_start = week_trades[0]["exit_time"]
                    week_end = week_trades[-1]["exit_time"]
                    rows.append(_row(
                        row_id=week_row_id,
                        parent_id=month_row_id,
                        level="week",
                        label=_week_label(week_key, week_start, week_end),
                        start_date=week_start,
                        end_date=week_end,
                        has_children=bool(week_bucket["days"]),
                        trades=week_trades,
                    ))

                    for day_key in sorted(week_bucket["days"].keys()):
                        day_bucket = week_bucket["days"][day_key]
                        day_trades = _sort_trades(day_bucket["trades"])
                        day_time = day_trades[0]["exit_time"]
                        rows.append(_row(
                            row_id=f"day-{day_key}",
                            parent_id=week_row_id,
                            level="day",
                            label=_date_label(day_time),
                            start_date=day_time,
                            end_date=day_time,
                            has_children=False,
                            trades=day_trades,
                        ))

    return {
        "rows": rows,
        "year_count": len(years),
        "quarter_count": quarter_count,
        "month_count": month_count,
        "default_tax_rate": DEFAULT_TAX_RATE,
        "long_term_rate": FUTURES_LONG_TERM_RATE,
        "long_term_share": FUTURES_LONG_TERM_SHARE,
        "short_term_share": FUTURES_SHORT_TERM_SHARE,
    }
