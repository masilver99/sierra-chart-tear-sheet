"""Tests for benchmark comparison helpers."""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd


def test_compute_benchmark_metrics_includes_alpha():
    from tearsheet.dataio.benchmark import compute_benchmark_metrics

    metrics = compute_benchmark_metrics(
        {
            dt.date(2026, 4, 9): 100.0,
            dt.date(2026, 4, 10): 50.0,
        },
        10_000.0,
        {
            "ticker": "SPY",
            "dates": ["2026-04-08", "2026-04-09", "2026-04-10"],
            "normalized": [100.0, 100.6, 101.2],
            "total_return_pct": 1.2,
        },
    )

    assert metrics["strategy_total_return_pct"] == 1.5
    assert metrics["benchmark_total_return_pct"] == 1.2
    assert metrics["alpha"] == 0.3
    assert metrics["ticker"] == "SPY"


def test_fetch_benchmark_supports_single_day(monkeypatch):
    from tearsheet.dataio.benchmark import fetch_benchmark

    class _FakeYFinance:
        @staticmethod
        def download(*args, **kwargs):
            return pd.DataFrame(
                [[100.0], [101.25]],
                columns=pd.MultiIndex.from_tuples([("Close", "SPY")]),
                index=pd.to_datetime(["2026-04-08", "2026-04-09"]),
            )

    monkeypatch.setitem(sys.modules, "yfinance", _FakeYFinance)

    result = fetch_benchmark(dt.date(2026, 4, 9), dt.date(2026, 4, 9))

    assert result is not None
    assert result["dates"] == ["2026-04-08", "2026-04-09"]
    assert result["closes"] == [100.0, 101.25]
    assert result["normalized"] == [100.0, 101.25]
    assert result["total_return_pct"] == 1.25
