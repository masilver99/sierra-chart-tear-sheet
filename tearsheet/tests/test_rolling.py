"""Tests for rolling window metrics (Phase 3 spec -- dict API with equity_r_squared)."""
from __future__ import annotations
import pytest
from tearsheet.metrics.rolling import compute_rolling_metrics


def test_rolling_returns_dict(three_trades):
    result = compute_rolling_metrics(three_trades)
    assert isinstance(result, dict)
    assert "window" in result
    assert "rolling" in result
    assert "equity_r_squared" in result


def test_rolling_returns_one_per_trade(three_trades):
    result = compute_rolling_metrics(three_trades)
    assert len(result["rolling"]) == len(three_trades)


def test_rolling_trade_id_matches(three_trades):
    result = compute_rolling_metrics(three_trades)
    for i, r in enumerate(result["rolling"]):
        assert r["trade_id"] == three_trades[i]["trade_id"]


def test_rolling_window_clamps_to_available(three_trades):
    result = compute_rolling_metrics(three_trades, window=20)
    rolling = result["rolling"]
    assert all(r["window_size"] <= 3 for r in rolling)
    assert rolling[0]["window_size"] == 1
    assert rolling[2]["window_size"] == 3


def test_rolling_expectancy_first_trade(three_trades):
    result = compute_rolling_metrics(three_trades)
    assert abs(result["rolling"][0]["rolling_expectancy"] - three_trades[0]["gross_pnl"]) < 0.01


def test_rolling_all_fields_present(three_trades):
    result = compute_rolling_metrics(three_trades)
    required = {
        "trade_id", "trade_index", "rolling_expectancy", "rolling_win_rate",
        "rolling_profit_factor", "rolling_sharpe", "window_size",
    }
    for r in result["rolling"]:
        assert required.issubset(r.keys())


def test_equity_r_squared_in_range(three_trades):
    result = compute_rolling_metrics(three_trades)
    r2 = result["equity_r_squared"]
    assert r2 is not None
    assert 0.0 <= r2 <= 1.0


def test_equity_r_squared_none_single_trade():
    trades = [{"trade_id": 1, "gross_pnl": 100.0}]
    result = compute_rolling_metrics(trades)
    assert result["equity_r_squared"] is None


def test_rolling_real_file_count():
    from pathlib import Path
    p = Path("TradeActivityLog_2026-04-09.txt")
    if not p.exists():
        pytest.skip("sample file not available")
    from tearsheet.app.main import run
    result = run(str(p), "test_report_p3.html")
    rolling = result["rolling_metrics"]
    assert isinstance(rolling, dict)
    assert len(rolling["rolling"]) == 8
    assert rolling["equity_r_squared"] is not None
    Path("test_report_p3.html").unlink(missing_ok=True)
