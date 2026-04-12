"""Tests for report rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def minimal_report_inputs(three_trades):
    """Return (trades, equity_curve, metrics) suitable for render_report."""
    from tearsheet.recon.equity import build_equity_curve
    from tearsheet.metrics.performance import compute_metrics
    import pandas as pd

    # Build a tiny equity curve
    eq = [
        {"DateTime": pd.Timestamp("2026-04-09 10:00:00"), "balance": 18000.0},
        {"DateTime": pd.Timestamp("2026-04-09 11:00:00"), "balance": 18050.0},
        {"DateTime": pd.Timestamp("2026-04-09 12:00:00"), "balance": 17970.0},
    ]
    metrics = compute_metrics(three_trades, eq)
    return three_trades, eq, metrics


def test_render_produces_html_file(minimal_report_inputs, tmp_path):
    from tearsheet.report.render import render_report
    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(trades, eq, metrics, out, source_file="test.txt")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<html" in html.lower()


def test_render_html_contains_trade_count(minimal_report_inputs, tmp_path):
    from tearsheet.report.render import render_report
    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(trades, eq, metrics, out, source_file="test.txt")
    html = out.read_text(encoding="utf-8")
    # Should mention 3 trades in the header
    assert "3 completed trades" in html


def test_render_html_contains_kpi_values(minimal_report_inputs, tmp_path):
    from tearsheet.report.render import render_report
    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(trades, eq, metrics, out, source_file="test.txt")
    html = out.read_text(encoding="utf-8")
    # Gross P&L = -62.50 should appear somewhere
    assert "-62.50" in html or "62.50" in html


def test_render_html_contains_chart_divs(minimal_report_inputs, tmp_path):
    from tearsheet.report.render import render_report
    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(trades, eq, metrics, out, source_file="test.txt")
    html = out.read_text(encoding="utf-8")
    # Plotly renders divs with id attributes
    assert 'id="' in html


def test_render_html_contains_new_visual_sections(minimal_report_inputs, tmp_path):
    from tearsheet.report.render import render_report

    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(trades, eq, metrics, out, source_file="test.txt")
    html = out.read_text(encoding="utf-8")

    assert "Trade P&amp;L Distribution" in html
    assert "Timing Heatmap" in html
    assert "Trade Mix" in html
    assert "Direction Mix" in html
    assert "Session Mix" in html
    assert "Outcome Mix" in html
    assert "Drawdown Recovery Profile" in html
    assert "Expectancy by Time Bucket" in html
    assert "Excursion Percentiles" in html
    assert "Holding-Time Efficiency" in html
    assert "Streak-State Analysis" in html
    assert "Exit Efficiency" in html
    assert "Profit Concentration" in html
    assert "Position Size Sensitivity" in html
    assert "Monthly Return Heatmap" in html


def test_render_html_contains_instrument_breakdown(minimal_report_inputs, tmp_path):
    from tearsheet.metrics.segmentation import segment_by_instrument
    from tearsheet.report.render import render_report

    trades, eq, metrics = minimal_report_inputs
    instrument_trades = [dict(t) for t in trades]
    instrument_trades[0]["symbol"] = "MESM26_FUT_CME"
    instrument_trades[1]["symbol"] = "MNQM26_FUT_CME"
    instrument_trades[2]["symbol"] = "MNQM26_FUT_CME"

    out = tmp_path / "test_report.html"
    render_report(
        instrument_trades,
        eq,
        metrics,
        out,
        source_file="test.txt",
        segmentation={"by_instrument": segment_by_instrument(instrument_trades)},
    )
    html = out.read_text(encoding="utf-8")

    assert "By Instrument" in html
    assert "MESM26_FUT_CME" in html
    assert "MNQM26_FUT_CME" in html


def test_render_html_contains_monthly_summary(minimal_report_inputs, tmp_path):
    from tearsheet.metrics.monthly_summary import compute_monthly_summary
    from tearsheet.report.render import render_report

    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(
        trades,
        eq,
        metrics,
        out,
        source_file="test.txt",
        monthly_summary=compute_monthly_summary(trades),
    )
    html = out.read_text(encoding="utf-8")
    assert "Period Summary" in html
    assert "Tax Rate (%)" in html
    assert "Estimated Taxes" in html
    assert 'value="24"' in html
    assert "toggleSummaryRow" in html
    assert "period-summary-effective-rate" in html
    assert "summary-row-year" in html
    assert "summary-row-quarter" in html
    assert "summary-row-month" in html


def test_render_html_contains_benchmark_alpha_and_equity_overlay(minimal_report_inputs, tmp_path):
    from tearsheet.report.render import render_report

    trades, eq, metrics = minimal_report_inputs
    out = tmp_path / "test_report.html"
    render_report(
        trades,
        eq,
        metrics,
        out,
        source_file="test.txt",
        benchmark_data={
            "ticker": "SPY",
            "dates": ["2026-04-09", "2026-04-10"],
            "normalized": [100.0, 101.2],
            "total_return_pct": 1.2,
        },
        benchmark_metrics={
            "strategy_total_return_pct": 1.5,
            "benchmark_total_return_pct": 1.2,
            "alpha": 0.3,
            "beta": 0.8,
            "alpha_annualized": 0.12,
            "correlation": 0.65,
            "ticker": "SPY",
        },
    )
    html = out.read_text(encoding="utf-8")

    assert "Benchmark Comparison" in html
    assert "S&amp;P 500 Return (SPY)" in html
    assert "Returns" in html
    assert "0.30%" in html
    assert "1.50%" in html
    assert "1.20%" in html


def test_metrics_keys_present(minimal_report_inputs):
    _, _, metrics = minimal_report_inputs
    required = {
        "n_trades", "total_gross_pnl", "total_fees", "total_net_pnl",
        "win_rate", "profit_factor", "expectancy", "max_drawdown",
        "drawdown_episode_count", "pct_time_at_highs", "top1_profit_share",
        "mae_percentiles", "mfe_percentiles",
    }
    assert required.issubset(metrics.keys())


def test_metrics_n_trades(minimal_report_inputs):
    trades, _, metrics = minimal_report_inputs
    assert metrics["n_trades"] == len(trades)


def test_render_html_contains_new_chart_sections(minimal_report_inputs, tmp_path):
    """New charts from the 'Add additional charts' issue should appear in the HTML."""
    from tearsheet.report.render import render_report
    import pandas as pd

    trades, _, metrics = minimal_report_inputs
    # Build a multi-day equity curve so rolling windows have something to chew on
    eq = [
        {"DateTime": pd.Timestamp(f"2026-0{m}-{d:02d} 16:00:00"), "balance": 10000.0 + i * 50.0}
        for i, (m, d) in enumerate(
            [(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
             (1, 13), (1, 14), (1, 15), (1, 16), (1, 17)]
        )
    ]
    out = tmp_path / "test_new_charts.html"
    render_report(trades, eq, metrics, out, source_file="test.txt")
    html = out.read_text(encoding="utf-8")

    assert "Worst 5 Drawdown Periods" in html
    assert "Rolling Volatility (6-Months)" in html
    assert "Rolling Sharpe (6-Months)" in html
    assert "Rolling Sortino (6-Months)" in html
    assert "EOY Returns  vs Benchmark" in html
    assert "Distribution of Monthly Returns" in html


def test_render_html_new_charts_with_benchmark(minimal_report_inputs, tmp_path):
    """New charts that use benchmark data render correctly when benchmark is supplied."""
    from tearsheet.report.render import render_report
    import pandas as pd

    trades, _, metrics = minimal_report_inputs
    eq = [
        {"DateTime": pd.Timestamp(f"2026-0{m}-{d:02d} 16:00:00"), "balance": 10000.0 + i * 50.0}
        for i, (m, d) in enumerate(
            [(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
             (1, 13), (1, 14), (1, 15), (1, 16), (1, 17)]
        )
    ]
    bench_dates = [f"2026-01-{d:02d}" for d in [2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17]]
    bench_norm = [100.0 + i * 0.3 for i in range(len(bench_dates))]
    benchmark_data = {
        "ticker": "SPY",
        "dates": bench_dates,
        "normalized": bench_norm,
        "total_return_pct": (bench_norm[-1] / bench_norm[0] - 1) * 100,
    }
    benchmark_metrics = {
        "strategy_total_return_pct": 5.5,
        "benchmark_total_return_pct": 3.3,
        "alpha": 2.2,
        "beta": 0.7,
        "alpha_annualized": 0.08,
        "correlation": 0.6,
        "ticker": "SPY",
    }
    out = tmp_path / "test_new_charts_benchmark.html"
    render_report(
        trades, eq, metrics, out, source_file="test.txt",
        benchmark_data=benchmark_data,
        benchmark_metrics=benchmark_metrics,
    )
    html = out.read_text(encoding="utf-8")

    assert "Daily Active Returns" in html
    assert "EOY Returns  vs Benchmark" in html
    assert "Distribution of Monthly Returns" in html
    assert "SPY" in html
