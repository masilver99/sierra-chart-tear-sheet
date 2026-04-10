from __future__ import annotations

import pandas as pd

from tearsheet.metrics.monthly_summary import DEFAULT_TAX_RATE, compute_monthly_summary, estimate_futures_taxes


def _trade(trade_id: int, exit_time: str, net_pnl: float, *, fees: float = 1.25, minutes: int = 5) -> dict:
    exit_ts = pd.Timestamp(exit_time)
    return {
        "trade_id": trade_id,
        "entry_time": exit_ts - pd.Timedelta(minutes=minutes),
        "exit_time": exit_ts,
        "net_pnl": net_pnl,
        "fees": fees,
    }


def test_monthly_summary_builds_year_quarter_month_week_day_rows():
    trades = [
        _trade(1, "2026-04-01 10:00:00", 100.0, minutes=10),
        _trade(2, "2026-04-02 11:00:00", -50.0, minutes=20),
        _trade(3, "2026-04-10 09:45:00", 75.0, minutes=30),
        _trade(4, "2026-05-04 10:15:00", 40.0, minutes=15),
        _trade(5, "2026-05-05 14:20:00", -20.0, minutes=25),
    ]

    summary = compute_monthly_summary(trades)
    rows = summary["rows"]

    assert summary["year_count"] == 1
    assert summary["quarter_count"] == 1
    assert summary["month_count"] == 2
    assert summary["default_tax_rate"] == DEFAULT_TAX_RATE
    assert [row["level"] for row in rows] == [
        "year", "quarter", "month", "week", "day", "day", "week", "day", "month", "week", "day", "day",
    ]

    year = rows[0]
    assert year["id"] == "year-2026"
    assert year["label"] == "2026"
    assert year["n_trades"] == 5
    assert year["n_days"] == 5
    assert year["net_pnl"] == 145.0
    assert year["estimated_taxes"] == 26.97

    quarter = rows[1]
    assert quarter["id"] == "quarter-2026-Q2"
    assert quarter["parent_id"] == year["id"]
    assert quarter["label"] == "Q2 2026"
    assert quarter["n_trades"] == 5
    assert quarter["n_days"] == 5

    april = rows[2]
    assert april["id"] == "month-2026-04"
    assert april["parent_id"] == quarter["id"]
    assert april["label"] == "Apr 2026"
    assert april["n_trades"] == 3
    assert april["n_days"] == 3
    assert april["net_pnl"] == 125.0
    assert april["avg_pnl_per_day"] == 41.67
    assert april["avg_pnl_per_trade"] == 41.67
    assert april["win_rate"] == 0.6667
    assert april["profit_factor"] == 3.5
    assert april["best_trade"] == 100.0
    assert april["worst_trade"] == -50.0
    assert april["total_fees"] == 3.75
    assert april["avg_hold_s"] == 1200
    assert april["estimated_taxes"] == 23.25

    first_week = rows[3]
    assert first_week["parent_id"] == "month-2026-04"
    assert first_week["label"] == "W14 (Apr 1 - Apr 2)"
    assert first_week["n_trades"] == 2
    assert first_week["net_pnl"] == 50.0

    first_day = rows[4]
    assert first_day["parent_id"] == first_week["id"]
    assert first_day["label"] == "Wed 2026-04-01"
    assert first_day["n_trades"] == 1
    assert first_day["n_days"] == 1
    assert first_day["net_pnl"] == 100.0
    assert first_day["avg_pnl_per_day"] == 100.0
    assert first_day["avg_hold_s"] == 600
    assert first_day["estimated_taxes"] == 18.6

    losing_day = rows[5]
    assert losing_day["net_pnl"] == -50.0
    assert losing_day["estimated_taxes"] == 0.0


def test_monthly_summary_skips_trades_without_exit_time():
    summary = compute_monthly_summary([
        {"trade_id": 1, "entry_time": pd.Timestamp("2026-04-01 10:00:00"), "exit_time": None, "net_pnl": 10.0, "fees": 1.0},
    ])

    assert summary == {
        "rows": [],
        "year_count": 0,
        "quarter_count": 0,
        "month_count": 0,
        "default_tax_rate": DEFAULT_TAX_RATE,
        "long_term_rate": 0.15,
        "long_term_share": 0.60,
        "short_term_share": 0.40,
    }


def test_estimate_futures_taxes_uses_6040_and_clamps_losses():
    assert estimate_futures_taxes(100.0) == 18.6
    assert estimate_futures_taxes(-25.0) == 0.0
    assert estimate_futures_taxes(100.0, tax_rate=0.30) == 21.0
