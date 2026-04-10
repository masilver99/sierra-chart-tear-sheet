"""Build Plotly charts and render the Jinja2 HTML template."""

from __future__ import annotations

import datetime
import math
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


def _empty_chart(message: str) -> str:
    return f"<p style='padding:16px;color:#8b949e'>{message}</p>"


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def _equity_chart(
    equity_curve: list[dict],
    cash_flows: list[dict] | None = None,
) -> str:
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


def _returns_chart(equity_curve: list[dict], benchmark_data: dict | None) -> str:
    """Cumulative % return for strategy and benchmark, both starting at 0%."""
    if not equity_curve:
        return ""

    import pandas as pd

    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    balance_key = "adjusted_balance" if "adjusted_balance" in df.columns else "balance"
    daily = df.groupby("date").agg({balance_key: "last"}).reset_index()
    start_bal = float(daily[balance_key].iloc[0])
    dts = [str(d) for d in daily["date"]]
    strat_returns = [round((float(v) / start_bal - 1) * 100, 4) for v in daily[balance_key]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dts, y=strat_returns,
        mode="lines", name="Strategy Return",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="Strategy: %{y:.2f}%<br>Date: %{x}<extra></extra>",
    ))

    if benchmark_data:
        ticker = benchmark_data.get("ticker", "Benchmark")
        bench_dates = [str(d) for d in benchmark_data.get("dates", [])]
        bench_normalized = benchmark_data.get("normalized", [])
        if bench_dates and bench_normalized and len(bench_dates) == len(bench_normalized):
            bench_returns = [round(float(v) - 100.0, 4) for v in bench_normalized]
            fig.add_trace(go.Scatter(
                x=bench_dates, y=bench_returns,
                mode="lines", name=f"{ticker} Return",
                line=dict(color="#d29922", width=1.5, dash="dot"),
                hovertemplate=f"{ticker}: %{{y:.2f}}%<br>Date: %{{x}}<extra></extra>",
            ))

    fig.update_layout(
        **_CHART_LAYOUT,
        yaxis_title="Cumulative Return (%)",
        yaxis_ticksuffix="%",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _trade_pnl_distribution_chart(trades: list[dict]) -> str:
    if not trades:
        return _empty_chart("No trade data.")

    pnls = [t["gross_pnl"] for t in trades if t.get("gross_pnl") is not None]
    if not pnls:
        return _empty_chart("No trade P&L data.")

    mean = sum(pnls) / len(pnls)
    median = sorted(pnls)[len(pnls) // 2] if len(pnls) % 2 == 1 else (
        sorted(pnls)[len(pnls) // 2 - 1] + sorted(pnls)[len(pnls) // 2]
    ) / 2.0
    std_dev = 0.0
    if len(pnls) > 1:
        variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std_dev = math.sqrt(variance)

    fig = go.Figure(go.Histogram(
        x=pnls,
        nbinsx=min(40, max(12, int(math.sqrt(len(pnls)) * 2))),
        marker=dict(color="#58a6ff"),
        opacity=0.85,
        hovertemplate="P&L: %{x:.2f}<br>Trades: %{y}<extra></extra>",
    ))
    fig.update_layout(**_CHART_LAYOUT, xaxis_title="Gross P&L ($)", yaxis_title="Trades", bargap=0.05)
    fig.add_vline(x=0, line_color="#30363d", line_width=1)
    fig.add_vline(
        x=mean,
        line_color="#3fb950" if mean >= 0 else "#f85149",
        line_width=2,
    )
    fig.add_vline(x=median, line_color="#d29922", line_width=2, line_dash="dot")
    if std_dev > 0:
        fig.add_vrect(
            x0=mean - std_dev,
            x1=mean + std_dev,
            fillcolor="rgba(88,166,255,0.12)",
            line_width=0,
            layer="below",
        )
    fig.add_annotation(
        x=0.01,
        y=0.98,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        bgcolor="rgba(13,17,23,0.75)",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=10, color="#c9d1d9"),
        text=(
            f"Mean: {mean:,.2f}<br>"
            f"Median: {median:,.2f}<br>"
            f"Std Dev: {std_dev:,.2f}"
        ),
    )
    return _div(fig)


def _timing_heatmap(trades: list[dict]) -> str:
    if not trades:
        return _empty_chart("No trade data.")

    import pandas as pd

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    by_slot: dict[int, dict[int, list[float]]] = {d: {} for d in range(5)}
    active_hours: set[int] = set()

    for trade in trades:
        entry_time = trade.get("entry_time")
        if entry_time is None or trade.get("gross_pnl") is None:
            continue
        ts = pd.Timestamp(entry_time)
        weekday = ts.weekday()
        if weekday > 4:
            continue
        hour = ts.hour
        by_slot.setdefault(weekday, {}).setdefault(hour, []).append(trade["gross_pnl"])
        active_hours.add(hour)

    if not active_hours:
        return _empty_chart("No timing data.")

    hours = sorted(active_hours)
    z: list[list[float | None]] = []
    customdata: list[list[list[float | int]]] = []

    for weekday in range(5):
        row: list[float | None] = []
        custom_row: list[list[float | int]] = []
        for hour in hours:
            pnls = by_slot.get(weekday, {}).get(hour, [])
            if pnls:
                avg_pnl = sum(pnls) / len(pnls)
                row.append(round(avg_pnl, 2))
                custom_row.append([len(pnls), round(sum(pnls), 2)])
            else:
                row.append(None)
                custom_row.append([0, 0.0])
        z.append(row)
        customdata.append(custom_row)

    fig = go.Figure(go.Heatmap(
        x=[f"{hour:02d}:00" for hour in hours],
        y=day_names,
        z=z,
        customdata=customdata,
        colorscale=[
            [0.0, "#f85149"],
            [0.5, "#30363d"],
            [1.0, "#3fb950"],
        ],
        zmid=0,
        colorbar=dict(title="Avg P&L"),
        hovertemplate=(
            "Day: %{y}<br>"
            "Hour: %{x}<br>"
            "Avg P&L: %{z:.2f}<br>"
            "Trades: %{customdata[0]}<br>"
            "Total P&L: %{customdata[1]:.2f}<extra></extra>"
        ),
    ))
    layout = dict(_CHART_LAYOUT)
    layout["height"] = 320
    fig.update_layout(**layout, xaxis_title="Entry hour", yaxis_title="Entry weekday")
    return _div(fig)


def _donut_chart(values: dict[str, int], colors: list[str], empty_message: str) -> str:
    filtered = [(label, value) for label, value in values.items() if value > 0]
    if not filtered:
        return _empty_chart(empty_message)

    labels = [label for label, _ in filtered]
    counts = [value for _, value in filtered]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts,
        hole=0.58,
        marker=dict(colors=colors[:len(labels)]),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value} trades (%{percent})<extra></extra>",
        sort=False,
    ))
    layout = dict(_CHART_LAYOUT)
    layout.update(margin=dict(l=12, r=12, t=8, b=8), height=260, showlegend=False)
    fig.update_layout(**layout)
    return _div(fig)


def _direction_mix_chart(trades: list[dict]) -> str:
    counts = {"Long": 0, "Short": 0}
    for trade in trades:
        direction = (trade.get("direction") or "").lower()
        if direction == "long":
            counts["Long"] += 1
        elif direction == "short":
            counts["Short"] += 1
    return _donut_chart(counts, ["#58a6ff", "#d29922"], "No direction data.")


def _session_mix_chart(trades: list[dict]) -> str:
    import pandas as pd

    counts = {"Open": 0, "Midday": 0, "Close": 0}
    for trade in trades:
        entry_time = trade.get("entry_time")
        if entry_time is None:
            continue
        ts = pd.Timestamp(entry_time)
        minutes = ts.hour * 60 + ts.minute
        if minutes < 10 * 60 + 30:
            counts["Open"] += 1
        elif minutes < 14 * 60:
            counts["Midday"] += 1
        else:
            counts["Close"] += 1
    return _donut_chart(counts, ["#58a6ff", "#3fb950", "#f85149"], "No session data.")


def _outcome_mix_chart(trades: list[dict]) -> str:
    counts = {"Winners": 0, "Losers": 0, "Breakeven": 0}
    for trade in trades:
        pnl = trade.get("gross_pnl")
        if pnl is None:
            continue
        if pnl > 0:
            counts["Winners"] += 1
        elif pnl < 0:
            counts["Losers"] += 1
        else:
            counts["Breakeven"] += 1
    return _donut_chart(counts, ["#3fb950", "#f85149", "#8b949e"], "No outcome data.")


def _duration_profit_scatter_chart(trades: list[dict]) -> str:
    """Scatter plot of time-in-trade (minutes) vs gross P&L with ±1σ / ±2σ bands on both axes."""
    import pandas as pd

    points = []
    for t in trades:
        if t.get("entry_time") is None or t.get("exit_time") is None:
            continue
        if t.get("gross_pnl") is None:
            continue
        try:
            dur_min = (
                pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])
            ).total_seconds() / 60.0
        except Exception:
            continue
        if dur_min < 0:
            continue
        points.append({"dur": dur_min, "pnl": float(t["gross_pnl"]), "tid": t.get("trade_id", "")})

    if not points:
        return _empty_chart("No duration / P&L data available.")

    durs = [p["dur"] for p in points]
    pnls = [p["pnl"] for p in points]
    n = len(points)

    def _stats(vals):
        mu = sum(vals) / len(vals)
        sigma = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1))
        return mu, sigma

    mu_d, sig_d = _stats(durs)
    mu_p, sig_p = _stats(pnls)

    colors = ["#3fb950" if p >= 0 else "#f85149" for p in pnls]

    fig = go.Figure()

    # ±1σ / ±2σ shaded bands for P&L (horizontal)
    for mult, alpha in ((2, 0.07), (1, 0.13)):
        fig.add_hrect(
            y0=mu_p - mult * sig_p,
            y1=mu_p + mult * sig_p,
            fillcolor=f"rgba(88,166,255,{alpha})",
            line_width=0,
            layer="below",
        )

    # ±1σ / ±2σ shaded bands for duration (vertical)
    for mult, alpha in ((2, 0.07), (1, 0.13)):
        fig.add_vrect(
            x0=max(0, mu_d - mult * sig_d),
            x1=mu_d + mult * sig_d,
            fillcolor=f"rgba(210,153,34,{alpha})",
            line_width=0,
            layer="below",
        )

    # Mean reference lines
    fig.add_hline(y=mu_p, line_color="#58a6ff", line_width=1, line_dash="dash")
    fig.add_vline(x=mu_d, line_color="#d29922", line_width=1, line_dash="dash")
    fig.add_hline(y=0, line_color="#30363d", line_width=1)

    fig.add_trace(go.Scatter(
        x=durs,
        y=pnls,
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.8, line=dict(width=0.5, color="#0d1117")),
        text=[f"T{p['tid']}: {p['dur']:.1f}m / ${p['pnl']:,.2f}" for p in points],
        hovertemplate="%{text}<extra></extra>",
        name="Trades",
    ))

    annotation_text = (
        f"<b>Time in Trade</b><br>"
        f"μ={mu_d:.1f}m  σ={sig_d:.1f}m<br>"
        f"<b>P&amp;L</b><br>"
        f"μ=${mu_p:,.2f}  σ=${sig_p:,.2f}<br>"
        f"n={n}"
    )
    fig.add_annotation(
        x=0.99, y=0.98,
        xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        showarrow=False,
        align="right",
        bgcolor="rgba(13,17,23,0.80)",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=10, color="#c9d1d9"),
        text=annotation_text,
    )

    fig.update_layout(
        **_CHART_LAYOUT,
        xaxis_title="Time in Trade (minutes)",
        yaxis_title="Gross P&L ($)",
        showlegend=False,
    )
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
    benchmark_data: dict[str, Any] | None = None,
    benchmark_metrics: dict[str, Any] | None = None,
    calendar_data: dict[str, Any] | None = None,
    cash_flows: list[dict[str, Any]] | None = None,
    monthly_summary: dict[str, Any] | None = None,
    sc_statistics: dict[str, Any] | None = None,
) -> None:
    """Render the tear sheet HTML to *output_path*."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    env.filters["format_duration"] = format_duration
    env.filters["safe_json"] = _safe_json
    tmpl = env.get_template("template.html")

    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    has_r_multiples = any(t.get("r_multiple") is not None for t in trades)
    segmentation_data = {
        "by_direction": {},
        "by_instrument": {},
        "by_note": {},
        "by_session": {},
        "by_date": [],
        "by_day_of_week": {},
        "by_week": {},
        "by_month": {},
        "by_hour": {},
        "pct_profitable_days": None,
        "pct_profitable_weeks": None,
        "pct_profitable_months": None,
        **(segmentation or {}),
    }

    html = tmpl.render(
        report_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        source_file=source_file,
        trades=trades,
        metrics=metrics,
        exec_metrics=exec_metrics or {},
        segmentation=segmentation_data,
        equity_chart=_equity_chart(equity_curve, cash_flows),
        drawdown_chart=_drawdown_chart(equity_curve),
        daily_pnl_chart=_daily_pnl_chart(trades),
        mfe_chart=_mfe_chart(trades),
        duration_chart=_duration_chart(trades),
        rolling_chart=_rolling_expectancy_chart(rolling_metrics),
        r_multiple_chart=_r_multiple_chart(trades),
        returns_chart=_returns_chart(equity_curve, benchmark_data),
        pnl_distribution_chart=_trade_pnl_distribution_chart(trades),
        duration_profit_scatter_chart=_duration_profit_scatter_chart(trades),
        timing_heatmap_chart=_timing_heatmap(trades),
        direction_mix_chart=_direction_mix_chart(trades),
        session_mix_chart=_session_mix_chart(trades),
        outcome_mix_chart=_outcome_mix_chart(trades),
        rolling_window=20,
        rolling_metrics=rolling_metrics,
        has_r_multiples=has_r_multiples,
        benchmark_metrics=benchmark_metrics,
        calendar_data=calendar_data,
        plotly_js=plotly_js,
        monthly_summary=monthly_summary,
        sc_statistics=sc_statistics,
    )

    Path(output_path).write_text(html, encoding="utf-8")
