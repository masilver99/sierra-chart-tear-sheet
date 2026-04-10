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


def test_metrics_keys_present(minimal_report_inputs):
    _, _, metrics = minimal_report_inputs
    required = {
        "n_trades", "total_gross_pnl", "total_fees", "total_net_pnl",
        "win_rate", "profit_factor", "expectancy", "max_drawdown",
    }
    assert required.issubset(metrics.keys())


def test_metrics_n_trades(minimal_report_inputs):
    trades, _, metrics = minimal_report_inputs
    assert metrics["n_trades"] == len(trades)
