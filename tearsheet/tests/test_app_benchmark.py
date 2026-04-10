"""Tests for benchmark integration in the app pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_run_single_day_can_render_benchmark(monkeypatch, tmp_path):
    from tearsheet.app import main as app_main

    sample_file = Path(__file__).parents[2] / "TradeActivityLog_2026-04-09.txt"
    if not sample_file.exists():
        pytest.skip("sample file not available")

    benchmark_data = {
        "ticker": "SPY",
        "dates": ["2026-04-08", "2026-04-09"],
        "normalized": [100.0, 101.2],
        "total_return_pct": 1.2,
    }
    benchmark_metrics = {
        "strategy_total_return_pct": 0.35,
        "benchmark_total_return_pct": 1.2,
        "alpha": -0.85,
        "beta": None,
        "alpha_annualized": None,
        "correlation": None,
        "ticker": "SPY",
    }

    monkeypatch.setattr(app_main, "fetch_benchmark", lambda start, end: benchmark_data)
    monkeypatch.setattr(app_main, "compute_benchmark_metrics", lambda pnl, start_balance, data: benchmark_metrics)

    out = tmp_path / "benchmark-report.html"
    result = app_main.run(sample_file, out)
    html = out.read_text(encoding="utf-8")

    assert result["benchmark_data"] == benchmark_data
    assert result["benchmark_metrics"] == benchmark_metrics
    assert "Benchmark Comparison" in html
    assert "Alpha" in html
    assert "Returns" in html
