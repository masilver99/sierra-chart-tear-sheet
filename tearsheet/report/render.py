"""Build Plotly charts and render the Jinja2 HTML template."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_DIR = Path(__file__).parent
_DARK = "plotly_dark"

_CHART_LAYOUT = dict(
    template=_DARK,
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    margin=dict(l=48, r=16, t=16, b=48),
    height=280,
    font=dict(color="#c9d1d9", size=11),
)


def format_duration(duration_s) -> str:
    """Format a duration in seconds to a human-readable string (e.g. '4m 32s')."""
    if duration_s is None or duration_s == "":
        return "—"
    try:
        secs = int(float(duration_s))
    except (TypeError, ValueError):
        return "—"
    if secs < 0:
        return "—"
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        m, s = divmod(secs, 60)
        return f"{m}m {s}s"
    h, rem = divmod(secs, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m}m" if m else f"{h}h"


def _safe_json(obj) -> str:
    """Serialize *obj* to a JSON string safe for embedding inside ``<script>`` tags.

    Escapes ``</`` → ``<\\/`` to prevent premature ``</script>`` tag closure.
    """
    import json
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s.replace("</", "<\\/")



def _div(fig: go.Figure) -> str:
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"responsive": True})


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _equity_chart(equity_curve: list[dict], cash_flows: list[dict] | None = None) -> str:
    if not equity_curve:
        return "<p style='padding:16px;color:#8b949e'>No equity data.</p>"

    import pandas as pd

    # Daily smoothing: use last balance per calendar day for a clean chart
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    has_adjusted = "adjusted_balance" in df.columns

    agg: dict = {"balance": "last"}
    if has_adjusted:
        agg["adjusted_balance"] = "last"
    daily = df.groupby("date").agg(agg).reset_index()
    dts = [str(d) for d in daily["date"]]

    fig = go.Figure()

    if has_adjusted:
        fig.add_trace(go.Scatter(
            x=dts, y=daily["adjusted_balance"].tolist(),
            mode="lines", name="Adjusted Balance",
            line=dict(color="#58a6ff", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=dts, y=daily["balance"].tolist(),
            mode="lines", name="Raw Balance",
            line=dict(color="#8b949e", width=1, dash="dot"),
            opacity=0.45,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=dts, y=daily["balance"].tolist(),
            mode="lines", name="Balance",
            line=dict(color="#58a6ff", width=1.5),
        ))

    if cash_flows:
        date_to_bal = {str(r["date"]): r.get("adjusted_balance", r["balance"])
                       for _, r in daily.iterrows()}
        cf_dates, cf_bals, cf_texts, cf_colors = [], [], [], []
        for cf in cash_flows:
            dt_str = str(pd.Timestamp(cf["DateTime"]).date())
            amount = cf["amount"]
            cf_dates.append(dt_str)
            cf_bals.append(date_to_bal.get(dt_str))
            cf_texts.append(f"+${amount:,.0f}" if amount > 0 else f"−${abs(amount):,.0f}")
            cf_colors.append("#3fb950" if amount > 0 else "#f85149")
        fig.add_trace(go.Scatter(
            x=cf_dates, y=cf_bals, mode="markers+text",
            name="Cash Flow",
            marker=dict(symbol="triangle-up", size=12, color=cf_colors),
            text=cf_texts, textposition="top center",
            textfont=dict(size=10, color=cf_colors),
            hovertemplate="%{text}<br>Date: %{x}<extra></extra>",
        ))

    fig.update_layout(**_CHART_LAYOUT, yaxis_title="Balance ($)", showlegend=has_adjusted)
    return _div(fig)


def _drawdown_chart(equity_curve: list[dict]) -> str:
    if not equity_curve:
        return "<p style='padding:16px;color:#8b949e'>No equity data.</p>"
    balance_key = "adjusted_balance" if "adjusted_balance" in equity_curve[0] else "balance"
    balances = [p[balance_key] for p in equity_curve]
    dts = [p["DateTime"] for p in equity_curve]
    peak = balances[0]
    dds = []
    for b in balances:
        if b > peak:
            peak = b
        dds.append(b - peak)
    fig = go.Figure(go.Scatter(x=dts, y=dds, mode="lines", fill="tozeroy",
                               line=dict(color="#f85149", width=1), fillcolor="rgba(248,81,73,.15)"))
    fig.update_layout(**_CHART_LAYOUT, yaxis_title="Drawdown ($)")
    return _div(fig)


def _daily_pnl_chart(trades: list[dict]) -> str:
    if not trades:
        return "<p style='padding:16px;color:#8b949e'>No trade data.</p>"
    import pandas as pd
    by_date: dict = {}
    for t in trades:
        if t["exit_time"] is None:
            continue
        date = pd.Timestamp(t["exit_time"]).date()
        by_date[date] = by_date.get(date, 0.0) + t["gross_pnl"]
    if not by_date:
        return "<p style='padding:16px;color:#8b949e'>No daily data.</p>"
    dates = sorted(by_date)
    pnls = [by_date[d] for d in dates]
    colors = ["#3fb950" if p >= 0 else "#f85149" for p in pnls]
    fig = go.Figure(go.Bar(x=[str(d) for d in dates], y=pnls, marker_color=colors))
    fig.update_layout(**_CHART_LAYOUT, yaxis_title="P&L ($)", bargap=0.3)
    return _div(fig)


def _mfe_chart(trades: list[dict]) -> str:
    if not trades:
        return "<p style='padding:16px;color:#8b949e'>No trade data.</p>"
    mfes = [t["mfe"] for t in trades]
    maes = [t["mae"] for t in trades]
    pnls = [t["gross_pnl"] for t in trades]
    colors = ["#3fb950" if p >= 0 else "#f85149" for p in pnls]
    fig = go.Figure(go.Scatter(
        x=maes, y=mfes, mode="markers",
        marker=dict(color=colors, size=8, opacity=0.8),
        text=[f"T{t['trade_id']}: {t['gross_pnl']:.2f}" for t in trades],
        hovertemplate="MAE: %{x:.2f}<br>MFE: %{y:.2f}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(**_CHART_LAYOUT, xaxis_title="MAE ($)", yaxis_title="MFE ($)")
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.add_vline(x=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _duration_chart(trades: list[dict]) -> str:
    """Horizontal bar chart: one bar per trade, colored by P&L sign, length = duration (seconds)."""
    if not trades:
        return "<p style='padding:16px;color:#8b949e'>No trade data.</p>"
    import pandas as pd
    labels, durations, colors = [], [], []
    for t in trades:
        if t.get("entry_time") is None or t.get("exit_time") is None:
            continue
        try:
            dur = (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds()
        except Exception:
            continue
        labels.append(f"T{t['trade_id']}")
        durations.append(max(dur, 0))
        colors.append("#3fb950" if t.get("gross_pnl", 0) >= 0 else "#f85149")

    if not labels:
        return "<p style='padding:16px;color:#8b949e'>No duration data.</p>"

    fig = go.Figure(go.Bar(
        x=durations, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{d:.0f}s" for d in durations],
        textposition="outside",
        hovertemplate="Trade %{y}: %{x:.0f}s<extra></extra>",
    ))
    layout = dict(_CHART_LAYOUT)
    layout["height"] = max(200, 40 * len(labels))
    fig.update_layout(**layout, xaxis_title="Duration (seconds)")
    return _div(fig)


def _rolling_expectancy_chart(rolling_metrics: dict | None) -> str:
    """Dual-axis line chart: rolling expectancy (left) and rolling win rate (right)."""
    if not rolling_metrics:
        return ""
    rolling = rolling_metrics.get("rolling", []) if isinstance(rolling_metrics, dict) else rolling_metrics
    if not rolling:
        return "<p style='padding:16px;color:#8b949e'>Insufficient data for rolling metrics.</p>"
    return _rolling_chart(rolling)


def _rolling_chart(rolling: list[dict]) -> str:
    """Dual-axis line chart: rolling expectancy (left) and rolling win rate (right)."""
    if not rolling:
        return "<p style='padding:16px;color:#8b949e'>Insufficient data for rolling metrics.</p>"

    trade_labels = [f"T{r['trade_id']}" for r in rolling]
    expectancies = [r["rolling_expectancy"] for r in rolling]
    win_rates = [r["rolling_win_rate"] * 100 for r in rolling]

    exp_colors = ["#3fb950" if e >= 0 else "#f85149" for e in expectancies]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_labels, y=expectancies, name="Expectancy ($)",
        line=dict(color="#3fb950", width=2), mode="lines+markers",
        marker=dict(color=exp_colors, size=6),
    ))
    fig.add_trace(go.Scatter(
        x=trade_labels, y=win_rates, name="Win Rate (%)",
        line=dict(color="#58a6ff", width=1.5, dash="dot"),
        yaxis="y2",
    ))
    layout = dict(_CHART_LAYOUT)
    layout["yaxis"] = {"title": "Rolling Expectancy ($)", "color": "#c9d1d9"}
    layout["yaxis2"] = {"title": "Win Rate (%)", "overlaying": "y", "side": "right",
                        "color": "#58a6ff", "range": [0, 100]}
    layout["legend"] = {"x": 0.02, "y": 0.98}
    fig.update_layout(**layout)
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _r_multiple_chart(trades: list[dict]) -> str:
    """Histogram of R-multiples. Red bars for R < 0, green for R >= 0."""
    r_values = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    if not r_values:
        return "<p style='padding:16px;color:#8b949e'>No R-multiple data (stop orders not found).</p>"

    r_trades = [t for t in trades if t.get("r_multiple") is not None]
    colors = ["#3fb950" if r >= 0 else "#f85149" for r in r_values]
    labels = [f"T{t['trade_id']}: {t['r_multiple']:.2f}R" for t in r_trades]

    fig = go.Figure(go.Bar(
        x=labels, y=r_values,
        marker_color=colors,
        text=[f"{r:.2f}R" for r in r_values],
        textposition="outside",
    ))
    fig.update_layout(**_CHART_LAYOUT, yaxis_title="R-Multiple")
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _benchmark_chart(equity_curve: list[dict], benchmark_data: dict | None) -> str:
    """Compare strategy equity curve vs benchmark indexed to same start."""
    if not benchmark_data or not equity_curve:
        return ""

    import pandas as pd

    start_bal = equity_curve[0]["balance"]
    strat_x = [p["DateTime"] for p in equity_curve]
    strat_y = [p["balance"] / start_bal * 100 for p in equity_curve]

    ticker = benchmark_data.get("ticker", "Benchmark")
    bench_dates = [pd.Timestamp(d) for d in benchmark_data.get("dates", [])]
    bench_normalized = benchmark_data.get("normalized", [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strat_x, y=strat_y, name="Strategy",
                             line=dict(color="#58a6ff", width=2)))
    if bench_dates and bench_normalized:
        fig.add_trace(go.Scatter(x=bench_dates, y=bench_normalized, name=ticker,
                                 line=dict(color="#d29922", width=1.5, dash="dot")))
    fig.update_layout(**_CHART_LAYOUT, yaxis_title="Indexed Return (100 = start)")
    fig.add_hline(y=100, line_color="#30363d", line_width=1)
    return _div(fig)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_report(
    trades: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_path: str | Path,
    source_file: str = "",
    exec_metrics: dict[str, Any] | None = None,
    segmentation: dict[str, Any] | None = None,
    rolling_metrics: dict[str, Any] | None = None,
    benchmark_data: list[dict[str, Any]] | None = None,
    benchmark_metrics: dict[str, Any] | None = None,
    calendar_data: dict[str, Any] | None = None,
    cash_flows: list[dict[str, Any]] | None = None,
) -> None:
    """Render the tear sheet HTML to *output_path*."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    env.filters["format_duration"] = format_duration
    env.filters["safe_json"] = _safe_json
    tmpl = env.get_template("template.html")

    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    has_r_multiples = any(t.get("r_multiple") is not None for t in trades)

    html = tmpl.render(
        report_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_file=source_file,
        trades=trades,
        metrics=metrics,
        exec_metrics=exec_metrics or {},
        segmentation=segmentation or {},
        equity_chart=_equity_chart(equity_curve, cash_flows),
        drawdown_chart=_drawdown_chart(equity_curve),
        daily_pnl_chart=_daily_pnl_chart(trades),
        mfe_chart=_mfe_chart(trades),
        duration_chart=_duration_chart(trades),
        rolling_chart=_rolling_expectancy_chart(rolling_metrics),
        r_multiple_chart=_r_multiple_chart(trades),
        benchmark_chart=_benchmark_chart(equity_curve, benchmark_data),
        rolling_window=20,
        rolling_metrics=rolling_metrics,
        has_r_multiples=has_r_multiples,
        benchmark_metrics=benchmark_metrics,
        calendar_data=calendar_data,
        plotly_js=plotly_js,
    )

    Path(output_path).write_text(html, encoding="utf-8")
