"""Tests for extended segmentation functions (Phase 3)."""

from __future__ import annotations

import pytest
from pathlib import Path

REAL_FILE = Path("TradeActivityLog_2026-04-09.txt")


def _get_trades():
    from tearsheet.dataio.loader import load_file
    from tearsheet.normalize.events import split_events
    from tearsheet.normalize.orders import normalize_orders
    from tearsheet.recon.trades import reconstruct_trades, enrich_trades
    df = load_file(REAL_FILE)
    fills, cash = split_events(df)
    orders = normalize_orders(df)
    trades = reconstruct_trades(fills, cash)
    return enrich_trades(trades, orders)


def test_by_date_has_date_keys():
    if not REAL_FILE.exists():
        pytest.skip()
    from tearsheet.metrics.segmentation import segment_by_date
    trades = _get_trades()
    result = segment_by_date(trades)
    assert len(result) >= 1
    for item in result:
        k = item["date"]
        assert len(k) == 10 and k[4] == '-' and k[7] == '-'


def test_by_dow_has_all_days():
    if not REAL_FILE.exists():
        pytest.skip()
    from tearsheet.metrics.segmentation import segment_by_day_of_week
    trades = _get_trades()
    result = segment_by_day_of_week(trades)
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        assert day in result


def test_by_week_keys_format():
    if not REAL_FILE.exists():
        pytest.skip()
    from tearsheet.metrics.segmentation import segment_by_week
    trades = _get_trades()
    result = segment_by_week(trades)
    assert len(result) >= 1
    for k in result:
        assert "W" in k


def test_by_month_keys_format():
    if not REAL_FILE.exists():
        pytest.skip()
    from tearsheet.metrics.segmentation import segment_by_month
    trades = _get_trades()
    result = segment_by_month(trades)
    assert len(result) >= 1
    for k in result:
        assert len(k) == 7 and k[4] == '-'


def test_by_date_counts_match_total():
    if not REAL_FILE.exists():
        pytest.skip()
    from tearsheet.metrics.segmentation import segment_by_date
    trades = _get_trades()
    result = segment_by_date(trades)
    total = sum(s["n_trades"] for s in result)
    assert total == len(trades)


def test_pct_profitable_single_day():
    if not REAL_FILE.exists():
        pytest.skip()
    from tearsheet.metrics.segmentation import segment_by_date, pct_profitable_periods
    trades = _get_trades()
    by_date = segment_by_date(trades)
    pct = pct_profitable_periods(by_date)
    assert 0.0 <= pct <= 1.0
