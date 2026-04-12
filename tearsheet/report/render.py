"""Build Plotly charts and render the Jinja2 HTML template."""

from __future__ import annotations

import datetime
import math
from pathlib import Path
from typing import Any

import sys

from tearsheet.metrics.montecarlo import run_monte_carlo

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots

# When frozen by PyInstaller, __file__ is unreliable; use _MEIPASS instead.
if getattr(sys, "frozen", False):
    _TEMPLATE_DIR = Path(sys._MEIPASS) / "tearsheet" / "report"
else:
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


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    sorted_v = sorted(values)
    if len(sorted_v) == 1:
        return sorted_v[0]
    pos = (len(sorted_v) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_v[lo]
    frac = pos - lo
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * frac


def _safe_pct(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator else None


def _daily_equity_series(equity_curve: list[dict]) -> list[dict[str, Any]]:
    if not equity_curve:
        return []

    import pandas as pd

    balance_key = "adjusted_balance" if "adjusted_balance" in equity_curve[0] else "balance"
    by_date: dict[Any, float] = {}
    for point in equity_curve:
        dt = pd.Timestamp(point["DateTime"]).date()
        by_date[dt] = float(point[balance_key])
    return [{"date": d, "balance": by_date[d]} for d in sorted(by_date)]


def _drawdown_episodes(equity_curve: list[dict]) -> list[dict[str, Any]]:
    series = _daily_equity_series(equity_curve)
    if len(series) < 2:
        return []

    peak_balance = series[0]["balance"]
    active: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] = []

    for point in series:
        date = point["date"]
        balance = point["balance"]

        if balance >= peak_balance:
            if active is not None:
                active["recovery_date"] = date
                active["duration_days"] = (date - active["start_date"]).days
                active["recovery_days"] = (date - active["trough_date"]).days
                episodes.append(active)
                active = None
            peak_balance = balance
            continue

        if active is None:
            active = {
                "start_date": date,
                "trough_date": date,
                "trough_balance": balance,
                "max_depth": balance - peak_balance,
            }
        elif balance < active["trough_balance"]:
            active["trough_date"] = date
            active["trough_balance"] = balance
            active["max_depth"] = balance - peak_balance

    if active is not None:
        end_date = series[-1]["date"]
        active["duration_days"] = (end_date - active["start_date"]).days
        active["recovery_days"] = None
        episodes.append(active)

    return episodes


def _duration_bucket_label(duration_s: float) -> str:
    if duration_s < 60:
        return "<1m"
    if duration_s < 5 * 60:
        return "1-5m"
    if duration_s < 15 * 60:
        return "5-15m"
    if duration_s < 30 * 60:
        return "15-30m"
    return "30m+"


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
    """Triple-axis line chart: rolling expectancy (left), win rate (right), Sharpe (right2)."""
    if not rolling:
        return "<p style='padding:16px;color:#8b949e'>Insufficient data for rolling metrics.</p>"

    trade_labels = [f"T{r['trade_id']}" for r in rolling]
    expectancies = [r["rolling_expectancy"] for r in rolling]
    win_rates = [r["rolling_win_rate"] * 100 for r in rolling]
    sharpes = [r.get("rolling_sharpe") for r in rolling]
    has_sharpe = any(s is not None for s in sharpes)

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
    if has_sharpe:
        fig.add_trace(go.Scatter(
            x=trade_labels,
            y=[s if s is not None else None for s in sharpes],
            name="Sharpe",
            line=dict(color="#d29922", width=1.5, dash="dash"),
            yaxis="y3",
        ))
    layout = dict(_CHART_LAYOUT)
    layout["yaxis"] = {"title": "Rolling Expectancy ($)", "color": "#c9d1d9"}
    layout["yaxis2"] = {"title": "Win Rate (%)", "overlaying": "y", "side": "right",
                        "color": "#58a6ff", "range": [0, 100]}
    if has_sharpe:
        layout["yaxis3"] = {
            "title": "Sharpe", "overlaying": "y", "side": "right",
            "anchor": "free", "position": 1.0, "color": "#d29922",
        }
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


# ---------------------------------------------------------------------------
# Rolling time-based analytics (6-month window)
# ---------------------------------------------------------------------------

def _rolling_period_days() -> int:
    """Approximate trading days in 6 months."""
    return 126


def _daily_pct_returns(equity_curve: list[dict]) -> "list[tuple]":
    """Return [(date, pct_return), ...] from equity curve daily series."""
    import pandas as pd
    series = _daily_equity_series(equity_curve)
    if len(series) < 2:
        return []
    result = []
    for i in range(1, len(series)):
        prev = series[i - 1]["balance"]
        curr = series[i]["balance"]
        if prev and prev != 0:
            result.append((series[i]["date"], (curr - prev) / prev))
    return result


def _benchmark_daily_pct_returns(benchmark_data: dict) -> "dict":
    """Return {date: pct_return} from benchmark data."""
    import datetime as _dt
    dates = benchmark_data.get("dates", [])
    normalized = benchmark_data.get("normalized", [])
    result = {}
    for i in range(1, len(normalized)):
        if normalized[i - 1] and normalized[i - 1] != 0:
            d = dates[i]
            if isinstance(d, str):
                d = _dt.date.fromisoformat(d)
            result[d] = (normalized[i] - normalized[i - 1]) / normalized[i - 1]
    return result


def _rolling_volatility_chart(equity_curve: list[dict], benchmark_data: dict | None) -> str:
    """Rolling 6-month annualised volatility for strategy and benchmark."""
    import math
    daily_rets = _daily_pct_returns(equity_curve)
    if len(daily_rets) < 10:
        return _empty_chart("Insufficient data for rolling volatility.")

    window = _rolling_period_days()
    dates, vols = [], []
    for i in range(window - 1, len(daily_rets)):
        window_rets = [r for _, r in daily_rets[i - window + 1:i + 1]]
        if len(window_rets) < 2:
            continue
        mean = sum(window_rets) / len(window_rets)
        variance = sum((r - mean) ** 2 for r in window_rets) / (len(window_rets) - 1)
        vol = math.sqrt(variance) * math.sqrt(252)
        dates.append(str(daily_rets[i][0]))
        vols.append(round(vol, 6))

    if not vols:
        return _empty_chart("Insufficient data for rolling volatility.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=vols, mode="lines", name="Strategy",
        line=dict(color="#58a6ff", width=1.5),
        hovertemplate="Strategy Vol: %{y:.2%}<br>Date: %{x}<extra></extra>",
    ))

    if benchmark_data:
        bench_map = _benchmark_daily_pct_returns(benchmark_data)
        bench_dates_all = sorted(bench_map.keys())
        b_dates, b_vols = [], []
        for i in range(window - 1, len(bench_dates_all)):
            w_rets = [bench_map[bench_dates_all[j]] for j in range(i - window + 1, i + 1)]
            if len(w_rets) < 2:
                continue
            mean = sum(w_rets) / len(w_rets)
            variance = sum((r - mean) ** 2 for r in w_rets) / (len(w_rets) - 1)
            b_vols.append(round(math.sqrt(variance) * math.sqrt(252), 6))
            b_dates.append(str(bench_dates_all[i]))
        if b_dates:
            ticker = benchmark_data.get("ticker", "SPY")
            fig.add_trace(go.Scatter(
                x=b_dates, y=b_vols, mode="lines", name=ticker,
                line=dict(color="#d29922", width=1.5),
                hovertemplate=f"{ticker} Vol: %{{y:.2%}}<br>Date: %{{x}}<extra></extra>",
            ))

    mean_vol = sum(vols) / len(vols)
    fig.add_hline(y=mean_vol, line_color="#f85149", line_width=1.5, line_dash="dash")
    fig.add_hline(y=0, line_color="#30363d", line_width=1, line_dash="dash")

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["yaxis"] = {"title": "Annualised Volatility", "tickformat": ".0%"}
    layout["showlegend"] = True
    layout["legend"] = dict(x=0.78, y=0.98)
    layout["title"] = dict(text="Rolling Volatility (6-Months)", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
    return _div(fig)


def _rolling_sharpe_chart(equity_curve: list[dict]) -> str:
    """Rolling 6-month annualised Sharpe ratio (risk-free rate = 0)."""
    import math
    daily_rets = _daily_pct_returns(equity_curve)
    if len(daily_rets) < 10:
        return _empty_chart("Insufficient data for rolling Sharpe.")

    window = _rolling_period_days()
    dates, sharpes = [], []
    for i in range(window - 1, len(daily_rets)):
        window_rets = [r for _, r in daily_rets[i - window + 1:i + 1]]
        if len(window_rets) < 2:
            continue
        mean = sum(window_rets) / len(window_rets)
        variance = sum((r - mean) ** 2 for r in window_rets) / (len(window_rets) - 1)
        std = math.sqrt(variance)
        sharpe = (mean / std * math.sqrt(252)) if std > 0 else 0.0
        dates.append(str(daily_rets[i][0]))
        sharpes.append(round(sharpe, 4))

    if not sharpes:
        return _empty_chart("Insufficient data for rolling Sharpe.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=sharpes, mode="lines", name="Sharpe",
        line=dict(color="#58a6ff", width=1.5),
        hovertemplate="Sharpe: %{y:.2f}<br>Date: %{x}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_color="#f85149", line_width=1.5, line_dash="dash")
    fig.add_hline(y=0, line_color="#30363d", line_width=1, line_dash="dash")

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["yaxis"] = {"title": "Rolling Sharpe Ratio"}
    layout["title"] = dict(text="Rolling Sharpe (6-Months)", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
    return _div(fig)


def _rolling_sortino_chart(equity_curve: list[dict]) -> str:
    """Rolling 6-month annualised Sortino ratio (risk-free rate = 0)."""
    import math
    daily_rets = _daily_pct_returns(equity_curve)
    if len(daily_rets) < 10:
        return _empty_chart("Insufficient data for rolling Sortino.")

    window = _rolling_period_days()
    dates, sortinos = [], []
    for i in range(window - 1, len(daily_rets)):
        window_rets = [r for _, r in daily_rets[i - window + 1:i + 1]]
        if len(window_rets) < 2:
            continue
        mean = sum(window_rets) / len(window_rets)
        neg_rets = [r for r in window_rets if r < 0]
        if neg_rets:
            downside_var = sum(r ** 2 for r in neg_rets) / len(window_rets)
            downside_std = math.sqrt(downside_var)
        else:
            downside_std = 0.0
        sortino = (mean / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0
        dates.append(str(daily_rets[i][0]))
        sortinos.append(round(sortino, 4))

    if not sortinos:
        return _empty_chart("Insufficient data for rolling Sortino.")

    mean_sortino = sum(sortinos) / len(sortinos)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=sortinos, mode="lines", name="Sortino",
        line=dict(color="#58a6ff", width=1.5),
        hovertemplate="Sortino: %{y:.2f}<br>Date: %{x}<extra></extra>",
    ))
    fig.add_hline(y=mean_sortino, line_color="#f85149", line_width=1.5, line_dash="dash")
    fig.add_hline(y=0, line_color="#30363d", line_width=1, line_dash="dash")

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["yaxis"] = {"title": "Rolling Sortino Ratio"}
    layout["title"] = dict(text="Rolling Sortino (6-Months)", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
    return _div(fig)


def _worst_drawdown_periods_chart(equity_curve: list[dict]) -> str:
    """Cumulative % returns chart with the 5 worst drawdown periods shaded."""
    if not equity_curve:
        return _empty_chart("No equity data.")

    import pandas as pd

    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    balance_key = "adjusted_balance" if "adjusted_balance" in df.columns else "balance"
    daily = df.groupby("date").agg({balance_key: "last"}).reset_index()
    start_bal = float(daily[balance_key].iloc[0])
    dts = [str(d) for d in daily["date"]]
    cum_returns = [round((float(v) / start_bal - 1) * 100, 4) for v in daily[balance_key]]

    episodes = _drawdown_episodes(equity_curve)
    if episodes:
        worst5 = sorted(episodes, key=lambda e: e["max_depth"])[:5]
    else:
        worst5 = []

    fig = go.Figure()

    # Shade worst drawdown periods
    for ep in worst5:
        x0 = str(ep["start_date"])
        x1 = str(ep.get("recovery_date", daily["date"].iloc[-1]))
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(248,81,73,0.15)",
            layer="below", line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=dts, y=cum_returns, mode="lines", name="Strategy",
        line=dict(color="#58a6ff", width=1.5),
        hovertemplate="Return: %{y:.1f}%<br>Date: %{x}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#30363d", line_width=1, line_dash="dash")

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 320
    layout["yaxis"] = {"title": "Cumulative Return", "ticksuffix": "%"}
    layout["title"] = dict(text="Strategy - Worst 5 Drawdown Periods", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
    return _div(fig)


def _eoy_returns_chart(equity_curve: list[dict], benchmark_data: dict | None) -> str:
    """Side-by-side bar chart of annual returns: strategy vs benchmark."""
    if not equity_curve:
        return _empty_chart("No equity data.")

    import pandas as pd

    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    balance_key = "adjusted_balance" if "adjusted_balance" in df.columns else "balance"
    daily = df.groupby("date").agg({balance_key: "last"}).reset_index()
    daily["year"] = [d.year for d in daily["date"]]

    # Strategy annual returns
    strat_by_year: dict[int, float] = {}
    for yr, grp in daily.groupby("year"):
        vals = grp[balance_key].tolist()
        # Find starting balance: last balance of previous year or first balance ever
        prev = daily[daily["year"] < yr][balance_key]
        start = float(prev.iloc[-1]) if not prev.empty else float(vals[0])
        end = float(vals[-1])
        if start and start != 0:
            strat_by_year[int(yr)] = round((end / start - 1) * 100, 4)

    # Benchmark annual returns
    bench_by_year: dict[int, float] = {}
    if benchmark_data:
        import datetime as _dt
        b_dates = benchmark_data.get("dates", [])
        b_norm = benchmark_data.get("normalized", [])
        b_by_yr: dict[int, list[float]] = {}
        for d_str, n in zip(b_dates, b_norm):
            d = _dt.date.fromisoformat(d_str) if isinstance(d_str, str) else d_str
            b_by_yr.setdefault(d.year, []).append((d, n))
        for yr, entries in b_by_yr.items():
            entries.sort()
            first_n = entries[0][1]
            last_n = entries[-1][1]
            if first_n and first_n != 0:
                # use start of year: last value of prior year if available
                prev_yr_entries = b_by_yr.get(yr - 1)
                if prev_yr_entries:
                    start_n = sorted(prev_yr_entries)[-1][1]
                else:
                    start_n = first_n
                if start_n and start_n != 0:
                    bench_by_year[yr] = round((last_n / start_n - 1) * 100, 4)

    all_years = sorted(set(list(strat_by_year.keys()) + list(bench_by_year.keys())))
    strat_vals = [strat_by_year.get(yr) for yr in all_years]
    bench_vals = [bench_by_year.get(yr) for yr in all_years]
    year_labels = [str(yr) for yr in all_years]

    fig = go.Figure()

    if any(v is not None for v in bench_vals):
        ticker = benchmark_data.get("ticker", "SPY") if benchmark_data else "SPY"
        fig.add_trace(go.Bar(
            x=year_labels,
            y=[v if v is not None else 0 for v in bench_vals],
            name=ticker,
            marker_color="#d29922",
            opacity=0.8,
            hovertemplate=f"{ticker}: %{{y:.1f}}%<br>Year: %{{x}}<extra></extra>",
        ))

    strat_colors = ["#58a6ff" if (v or 0) >= 0 else "#f85149" for v in strat_vals]
    fig.add_trace(go.Bar(
        x=year_labels,
        y=[v if v is not None else 0 for v in strat_vals],
        name="Strategy",
        marker_color=strat_colors,
        hovertemplate="Strategy: %{y:.1f}%<br>Year: %{x}<extra></extra>",
    ))

    # Mean strategy annual return as red dashed line
    valid_strat = [v for v in strat_vals if v is not None]
    if valid_strat:
        mean_ann = sum(valid_strat) / len(valid_strat)
        fig.add_hline(y=mean_ann, line_color="#f85149", line_width=1.5, line_dash="dash")

    fig.add_hline(y=0, line_color="#30363d", line_width=1, line_dash="dash")

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 320
    layout["yaxis"] = {"title": "Annual Return", "ticksuffix": "%"}
    layout["barmode"] = "group"
    layout["showlegend"] = True
    layout["legend"] = dict(x=0.02, y=0.98)
    layout["title"] = dict(text="EOY Returns vs Benchmark", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
    return _div(fig)


def _monthly_returns_dist_chart(equity_curve: list[dict], benchmark_data: dict | None) -> str:
    """Overlapping histogram of monthly returns: strategy and benchmark."""
    if not equity_curve:
        return _empty_chart("No equity data.")

    import pandas as pd
    import datetime as _dt

    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["DateTime"]).dt.date
    balance_key = "adjusted_balance" if "adjusted_balance" in df.columns else "balance"
    daily = df.groupby("date").agg({balance_key: "last"}).reset_index()
    daily["ym"] = [_dt.date(d.year, d.month, 1) for d in daily["date"]]

    strat_monthly: list[float] = []
    for ym, grp in daily.groupby("ym"):
        vals = grp[balance_key].tolist()
        all_prev = daily[daily["ym"] < ym][balance_key]
        start = float(all_prev.iloc[-1]) if not all_prev.empty else float(vals[0])
        end = float(vals[-1])
        if start and start != 0:
            strat_monthly.append(round((end / start - 1) * 100, 4))

    bench_monthly: list[float] = []
    if benchmark_data:
        b_dates = benchmark_data.get("dates", [])
        b_norm = benchmark_data.get("normalized", [])
        b_by_ym: dict = {}
        for d_str, n in zip(b_dates, b_norm):
            d = _dt.date.fromisoformat(d_str) if isinstance(d_str, str) else d_str
            ym = _dt.date(d.year, d.month, 1)
            b_by_ym.setdefault(ym, []).append((d, n))
        sorted_yms = sorted(b_by_ym.keys())
        for i, ym in enumerate(sorted_yms):
            entries = sorted(b_by_ym[ym])
            last_n = entries[-1][1]
            if i > 0:
                prev_entries = sorted(b_by_ym[sorted_yms[i - 1]])
                start_n = prev_entries[-1][1]
            else:
                start_n = entries[0][1]
            if start_n and start_n != 0:
                bench_monthly.append(round((last_n / start_n - 1) * 100, 4))

    if not strat_monthly:
        return _empty_chart("Insufficient monthly data.")

    fig = go.Figure()
    nbins = min(30, max(8, int(len(strat_monthly) ** 0.5 * 2)))
    fig.add_trace(go.Histogram(
        x=strat_monthly, name="Strategy",
        nbinsx=nbins,
        marker_color="#58a6ff", opacity=0.7,
        hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra>Strategy</extra>",
    ))
    if bench_monthly:
        ticker = benchmark_data.get("ticker", "SPY") if benchmark_data else "SPY"
        fig.add_trace(go.Histogram(
            x=bench_monthly, name=ticker,
            nbinsx=nbins,
            marker_color="#d29922", opacity=0.7,
            hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra>" + ticker + "</extra>",
        ))

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["barmode"] = "overlay"
    layout["xaxis"] = {"title": "Monthly Return (%)", "ticksuffix": "%"}
    layout["yaxis"] = {"title": "Count"}
    layout["showlegend"] = True
    layout["legend"] = dict(x=0.78, y=0.98)
    layout["title"] = dict(text="Distribution of Monthly Returns", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
    fig.add_vline(x=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _daily_active_returns_chart(equity_curve: list[dict], benchmark_data: dict | None) -> str:
    """Bar chart of daily active returns (strategy daily return minus benchmark daily return)."""
    if not equity_curve:
        return _empty_chart("No equity data.")

    daily_rets = _daily_pct_returns(equity_curve)
    if not daily_rets:
        return _empty_chart("Insufficient data for active returns.")

    if not benchmark_data:
        return _empty_chart("Benchmark data required for active returns.")

    bench_map = _benchmark_daily_pct_returns(benchmark_data)

    dates, active = [], []
    for d, r in daily_rets:
        bench_r = bench_map.get(d)
        if bench_r is not None:
            dates.append(str(d))
            active.append(round((r - bench_r) * 100, 4))

    if not dates:
        return _empty_chart("No overlapping dates for active returns.")

    colors = ["#3fb950" if a >= 0 else "#f85149" for a in active]
    fig = go.Figure(go.Bar(
        x=dates, y=active,
        marker_color=colors,
        hovertemplate="Active Return: %{y:.2f}%<br>Date: %{x}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#30363d", line_width=1)

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["yaxis"] = {"title": "Active Return (%)", "ticksuffix": "%"}
    layout["bargap"] = 0.1
    layout["title"] = dict(text="Daily Active Returns", x=0.5, font=dict(size=14))
    layout["margin"] = dict(l=60, r=16, t=48, b=48)
    fig.update_layout(**layout)
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


def _mae_winners_scatter_chart(trades: list[dict]) -> str:
    """Scatter plot of |MAE| vs gross P&L for winning trades with ±1σ / ±2σ bands.

    Answers the question: "How much heat did my winners take before closing in profit?"
    A tight cluster near the origin means winners rarely experienced much adversity.
    A wide spread on the x-axis suggests winners often required significant drawdown
    tolerance before turning profitable.
    """
    winners = [
        t for t in trades
        if t.get("gross_pnl", 0.0) > 0
        and t.get("mae") is not None
    ]

    if len(winners) < 3:
        return _empty_chart("Not enough winning trade data for MAE chart.")

    abs_maes = [abs(float(t["mae"])) for t in winners]
    pnls = [float(t["gross_pnl"]) for t in winners]
    n = len(winners)

    def _stats(vals):
        mu = sum(vals) / len(vals)
        sigma = math.sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1))
        return mu, sigma

    mu_m, sig_m = _stats(abs_maes)
    mu_p, sig_p = _stats(pnls)

    fig = go.Figure()

    # ±1σ / ±2σ shaded bands for P&L (horizontal)
    for mult, alpha in ((2, 0.07), (1, 0.13)):
        fig.add_hrect(
            y0=max(0, mu_p - mult * sig_p),
            y1=mu_p + mult * sig_p,
            fillcolor=f"rgba(88,166,255,{alpha})",
            line_width=0,
            layer="below",
        )

    # ±1σ / ±2σ shaded bands for MAE (vertical)
    for mult, alpha in ((2, 0.07), (1, 0.13)):
        fig.add_vrect(
            x0=max(0, mu_m - mult * sig_m),
            x1=mu_m + mult * sig_m,
            fillcolor=f"rgba(248,81,73,{alpha})",
            line_width=0,
            layer="below",
        )

    # Mean reference lines
    fig.add_hline(y=mu_p, line_color="#58a6ff", line_width=1, line_dash="dash")
    fig.add_vline(x=mu_m, line_color="#f85149", line_width=1, line_dash="dash")
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.add_vline(x=0, line_color="#30363d", line_width=1)

    fig.add_trace(go.Scatter(
        x=abs_maes,
        y=pnls,
        mode="markers",
        marker=dict(color="#3fb950", size=7, opacity=0.8, line=dict(width=0.5, color="#0d1117")),
        text=[f"T{t.get('trade_id', '')}: MAE=${abs(float(t['mae'])):,.2f} P&L=${float(t['gross_pnl']):,.2f}" for t in winners],
        hovertemplate="%{text}<extra></extra>",
        name="Winning Trades",
    ))

    annotation_text = (
        f"<b>MAE (heat taken)</b><br>"
        f"μ=${mu_m:,.2f}  σ=${sig_m:,.2f}<br>"
        f"<b>P&amp;L</b><br>"
        f"μ=${mu_p:,.2f}  σ=${sig_p:,.2f}<br>"
        f"n={n} winners"
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
        xaxis_title="|MAE| — Heat Taken ($)",
        yaxis_title="Gross P&L ($)",
        showlegend=False,
    )
    return _div(fig)


def _drawdown_recovery_chart(equity_curve: list[dict]) -> str:
    episodes = _drawdown_episodes(equity_curve)
    if not episodes:
        return _empty_chart("No completed drawdown episodes.")

    labels = [f"DD{idx + 1}" for idx in range(len(episodes))]
    durations = [ep["duration_days"] for ep in episodes]
    recoveries = [ep["recovery_days"] for ep in episodes]
    depths = [abs(ep["max_depth"]) for ep in episodes]
    recovered = [ep["recovery_days"] is not None for ep in episodes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=durations,
        name="Duration (days)",
        marker_color=["#f85149" if not ok else "#d29922" for ok in recovered],
        text=[f"{d}d" for d in durations],
        textposition="outside",
        customdata=[
            [
                str(ep["start_date"]),
                str(ep["trough_date"]),
                str(ep.get("recovery_date")) if ep.get("recovery_date") is not None else "Open",
                abs(ep["max_depth"]),
                ep["recovery_days"],
            ]
            for ep in episodes
        ],
        hovertemplate=(
            "Episode %{x}<br>"
            "Start: %{customdata[0]}<br>"
            "Trough: %{customdata[1]}<br>"
            "Recovery: %{customdata[2]}<br>"
            "Duration: %{y} days<br>"
            "Recovery after trough: %{customdata[4]} days<br>"
            "Max depth: -$%{customdata[3]:,.2f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=[r if r is not None else None for r in recoveries],
        mode="markers+lines",
        name="Recovery Days",
        marker=dict(color="#58a6ff", size=8, symbol="diamond"),
        line=dict(color="#58a6ff", width=1.5),
        connectgaps=False,
        hovertemplate="Recovery days: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **dict(_CHART_LAYOUT, height=320),
        yaxis_title="Days",
        barmode="overlay",
        legend=dict(x=0.02, y=0.98),
    )
    return _div(fig)


def _time_bucket_expectancy_chart(segmentation: dict | None) -> str:
    if not segmentation:
        return _empty_chart("No segmentation data.")

    datasets = [
        (
            "By Session",
            [(label.title(), segmentation.get("by_session", {}).get(label)) for label in ("open", "midday", "close")],
        ),
        (
            "By Day of Week",
            [(label[:3], segmentation.get("by_day_of_week", {}).get(label)) for label in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")],
        ),
        (
            "By Entry Hour",
            [
                (f"{hour:02d}:00", stats)
                for hour, stats in sorted((segmentation.get("by_hour") or {}).items())
            ],
        ),
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[title for title, _ in datasets],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    has_data = False
    for row, (_, rows_data) in enumerate(datasets, start=1):
        labels = []
        expectancies = []
        counts = []
        for label, stats in rows_data:
            if not stats or stats.get("n_trades", 0) == 0:
                continue
            labels.append(label)
            expectancies.append(stats.get("expectancy", 0.0))
            counts.append(stats.get("n_trades", 0))

        if not labels:
            continue
        has_data = True
        colors = ["#3fb950" if value >= 0 else "#f85149" for value in expectancies]
        fig.add_trace(go.Bar(
            x=labels,
            y=expectancies,
            marker_color=colors,
            text=[f"n={count}" for count in counts],
            textposition="outside",
            name="Expectancy",
            hovertemplate="%{x}<br>Expectancy: %{y:$,.2f}<br>%{text}<extra></extra>",
            showlegend=row == 1,
        ), row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=labels,
            y=counts,
            mode="lines+markers",
            name="Trades",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=7),
            hovertemplate="%{x}<br>Trades: %{y}<extra></extra>",
            showlegend=row == 1,
        ), row=row, col=1, secondary_y=True)
        fig.add_hline(y=0, line_color="#30363d", line_width=1, row=row, col=1)
        fig.update_yaxes(title_text="Expectancy ($)", row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Trades", row=row, col=1, secondary_y=True)

    if not has_data:
        return _empty_chart("No time-bucket data.")

    fig.update_layout(**dict(_CHART_LAYOUT, height=760), legend=dict(x=0.02, y=1.05, orientation="h"))
    return _div(fig)


def _excursion_percentile_chart(trades: list[dict]) -> str:
    percentiles = [0.50, 0.75, 0.90, 0.95]
    labels = ["P50", "P75", "P90", "P95"]

    mae_values = [abs(float(t["mae"])) for t in trades if t.get("mae") is not None]
    mfe_values = [max(float(t["mfe"]), 0.0) for t in trades if t.get("mfe") is not None]
    if not mae_values and not mfe_values:
        return _empty_chart("No excursion data.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    mae_curve = [_quantile(mae_values, q) for q in percentiles] if mae_values else []
    mfe_curve = [_quantile(mfe_values, q) for q in percentiles] if mfe_values else []
    if mae_curve:
        fig.add_trace(go.Scatter(
            x=labels,
            y=mae_curve,
            mode="lines+markers",
            name="|MAE| ($)",
            line=dict(color="#f85149", width=2),
            hovertemplate="%{x}<br>|MAE|: %{y:$,.2f}<extra></extra>",
        ), row=1, col=1, secondary_y=False)
    if mfe_curve:
        fig.add_trace(go.Scatter(
            x=labels,
            y=mfe_curve,
            mode="lines+markers",
            name="MFE ($)",
            line=dict(color="#3fb950", width=2),
            hovertemplate="%{x}<br>MFE: %{y:$,.2f}<extra></extra>",
        ), row=1, col=1, secondary_y=False)

    mae_r = [
        abs(float(t["mae"])) / float(t["initial_risk"])
        for t in trades
        if t.get("mae") is not None and t.get("initial_risk") is not None and float(t["initial_risk"]) > 0
    ]
    mfe_r = [
        max(float(t["mfe"]), 0.0) / float(t["initial_risk"])
        for t in trades
        if t.get("mfe") is not None and t.get("initial_risk") is not None and float(t["initial_risk"]) > 0
    ]
    if mae_r:
        fig.add_trace(go.Scatter(
            x=labels,
            y=[_quantile(mae_r, q) for q in percentiles],
            mode="lines+markers",
            name="|MAE| (R)",
            line=dict(color="#ff7b72", width=1.5, dash="dash"),
            hovertemplate="%{x}<br>|MAE|: %{y:.2f}R<extra></extra>",
        ), row=1, col=1, secondary_y=True)
    if mfe_r:
        fig.add_trace(go.Scatter(
            x=labels,
            y=[_quantile(mfe_r, q) for q in percentiles],
            mode="lines+markers",
            name="MFE (R)",
            line=dict(color="#56d364", width=1.5, dash="dash"),
            hovertemplate="%{x}<br>MFE: %{y:.2f}R<extra></extra>",
        ), row=1, col=1, secondary_y=True)

    fig.update_layout(**dict(_CHART_LAYOUT, height=320), legend=dict(x=0.02, y=1.02, orientation="h"))
    fig.update_yaxes(title_text="Excursion ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Excursion (R)", row=1, col=1, secondary_y=True)
    return _div(fig)


def _holding_time_efficiency_chart(trades: list[dict]) -> str:
    import pandas as pd

    buckets = {label: [] for label in ("<1m", "1-5m", "5-15m", "15-30m", "30m+")}
    for trade in trades:
        if trade.get("entry_time") is None or trade.get("exit_time") is None:
            continue
        try:
            duration_s = (pd.Timestamp(trade["exit_time"]) - pd.Timestamp(trade["entry_time"])).total_seconds()
        except Exception:
            continue
        if duration_s < 0:
            continue
        buckets[_duration_bucket_label(duration_s)].append(trade)

    labels, expectancy, win_rate, capture = [], [], [], []
    for label, bucket_trades in buckets.items():
        if not bucket_trades:
            continue
        labels.append(label)
        pnls = [float(t.get("gross_pnl", 0.0) or 0.0) for t in bucket_trades]
        expectancy.append(sum(pnls) / len(pnls))
        win_rate.append(sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100)
        captures = [
            (float(t.get("gross_pnl", 0.0) or 0.0) / float(t["mfe"])) * 100
            for t in bucket_trades
            if t.get("mfe") is not None and float(t["mfe"]) > 0
        ]
        capture.append(sum(captures) / len(captures) if captures else None)

    if not labels:
        return _empty_chart("No duration data.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels,
        y=expectancy,
        marker_color=["#3fb950" if value >= 0 else "#f85149" for value in expectancy],
        text=[f"n={len(buckets[label])}" for label in labels],
        textposition="outside",
        name="Expectancy",
        hovertemplate="%{x}<br>Expectancy: %{y:$,.2f}<br>%{text}<extra></extra>",
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels,
        y=win_rate,
        mode="lines+markers",
        name="Win Rate",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="%{x}<br>Win rate: %{y:.1f}%<extra></extra>",
    ), row=1, col=1, secondary_y=True)
    if any(value is not None for value in capture):
        fig.add_trace(go.Scatter(
            x=labels,
            y=[value if value is not None else None for value in capture],
            mode="lines+markers",
            name="MFE Capture",
            line=dict(color="#d29922", width=2, dash="dash"),
            hovertemplate="%{x}<br>MFE capture: %{y:.1f}%<extra></extra>",
        ), row=1, col=1, secondary_y=True)

    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.update_layout(**dict(_CHART_LAYOUT, height=320), legend=dict(x=0.02, y=1.02, orientation="h"))
    fig.update_yaxes(title_text="Expectancy ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Percent", row=1, col=1, secondary_y=True)
    return _div(fig)


def _streak_state_chart(trades: list[dict]) -> str:
    states = {label: [] for label in ("After 1L", "After 2L", "After 3L+", "After 1W", "After 2W", "After 3W+")}
    prev_sign = None
    prev_run = 0

    for trade in trades:
        pnl = float(trade.get("gross_pnl", 0.0) or 0.0)
        if prev_sign == "W":
            label = f"After {min(prev_run, 3)}W" + ("+" if prev_run >= 3 else "")
            states[label].append(trade)
        elif prev_sign == "L":
            label = f"After {min(prev_run, 3)}L" + ("+" if prev_run >= 3 else "")
            states[label].append(trade)

        if pnl > 0:
            prev_run = prev_run + 1 if prev_sign == "W" else 1
            prev_sign = "W"
        elif pnl < 0:
            prev_run = prev_run + 1 if prev_sign == "L" else 1
            prev_sign = "L"
        else:
            prev_sign = None
            prev_run = 0

    labels, expectancy, win_rates, counts = [], [], [], []
    for label, bucket_trades in states.items():
        if not bucket_trades:
            continue
        labels.append(label)
        pnls = [float(t.get("gross_pnl", 0.0) or 0.0) for t in bucket_trades]
        expectancy.append(sum(pnls) / len(pnls))
        win_rates.append(sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100)
        counts.append(len(bucket_trades))

    if not labels:
        return _empty_chart("Not enough trade history for streak-state analysis.")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels,
        y=expectancy,
        marker_color=["#3fb950" if value >= 0 else "#f85149" for value in expectancy],
        text=[f"n={count}" for count in counts],
        textposition="outside",
        name="Expectancy",
        hovertemplate="%{x}<br>Expectancy: %{y:$,.2f}<br>%{text}<extra></extra>",
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels,
        y=win_rates,
        mode="lines+markers",
        name="Win Rate",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="%{x}<br>Win rate: %{y:.1f}%<extra></extra>",
    ), row=1, col=1, secondary_y=True)
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.update_layout(**dict(_CHART_LAYOUT, height=320), legend=dict(x=0.02, y=1.02, orientation="h"))
    fig.update_yaxes(title_text="Next-Trade Expectancy ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Next-Trade Win Rate (%)", row=1, col=1, secondary_y=True)
    return _div(fig)


def _exit_efficiency_chart(trades: list[dict]) -> str:
    by_exit: dict[str, dict[str, Any]] = {}
    for trade in trades:
        label = (trade.get("exit_type") or "").strip() or "untagged"
        bucket = by_exit.setdefault(label, {"capture": [], "giveback": [], "count": 0})
        bucket["count"] += 1
        mfe = trade.get("mfe")
        pnl = float(trade.get("gross_pnl", 0.0) or 0.0)
        if mfe is None or float(mfe) <= 0:
            continue
        mfe_val = float(mfe)
        bucket["capture"].append((pnl / mfe_val) * 100)
        bucket["giveback"].append(max(mfe_val - pnl, 0.0))

    labels = [label for label, stats in by_exit.items() if stats["count"] > 0]
    if not labels:
        return _empty_chart("No exit-type data.")

    avg_capture = [
        sum(by_exit[label]["capture"]) / len(by_exit[label]["capture"]) if by_exit[label]["capture"] else None
        for label in labels
    ]
    avg_giveback = [
        sum(by_exit[label]["giveback"]) / len(by_exit[label]["giveback"]) if by_exit[label]["giveback"] else None
        for label in labels
    ]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average MFE Capture", "Average Profit Left on Table"))
    fig.add_trace(go.Bar(
        x=labels,
        y=[value if value is not None else 0 for value in avg_capture],
        marker_color="#58a6ff",
        text=[f"n={by_exit[label]['count']}" for label in labels],
        textposition="outside",
        hovertemplate="%{x}<br>Capture: %{y:.1f}%<extra></extra>",
        name="Capture",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=labels,
        y=[value if value is not None else 0 for value in avg_giveback],
        marker_color="#d29922",
        text=[f"n={by_exit[label]['count']}" for label in labels],
        textposition="outside",
        hovertemplate="%{x}<br>Left on table: %{y:$,.2f}<extra></extra>",
        name="Giveback",
    ), row=1, col=2)
    fig.update_yaxes(title_text="Capture (%)", row=1, col=1)
    fig.update_yaxes(title_text="Giveback ($)", row=1, col=2)
    fig.update_layout(**dict(_CHART_LAYOUT, height=320), showlegend=False)
    return _div(fig)


def _lorenz_curve_chart(trades: list[dict]) -> str:
    winners = sorted(float(t.get("gross_pnl", 0.0) or 0.0) for t in trades if (t.get("gross_pnl") or 0) > 0)
    if len(winners) < 2:
        return _empty_chart("Not enough winning trades for concentration analysis.")

    total = sum(winners)
    cumulative = [0.0]
    running = 0.0
    for value in winners:
        running += value
        cumulative.append(running / total if total else 0.0)
    trade_share = [idx / len(winners) for idx in range(len(winners) + 1)]
    weighted = sum((idx + 1) * value for idx, value in enumerate(winners))
    gini = (2.0 * weighted) / (len(winners) * total) - (len(winners) + 1) / len(winners) if total else 0.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trade_share,
        y=cumulative,
        mode="lines",
        name="Winner Lorenz Curve",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="Trade share: %{x:.0%}<br>Profit share: %{y:.0%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Equality",
        line=dict(color="#8b949e", width=1, dash="dash"),
        hovertemplate="Perfect equality<extra></extra>",
    ))
    fig.add_annotation(
        x=0.98,
        y=0.08,
        xref="paper",
        yref="paper",
        xanchor="right",
        showarrow=False,
        bgcolor="rgba(13,17,23,0.80)",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=10, color="#c9d1d9"),
        text=f"Gini: {gini:.2f}",
    )
    fig.update_layout(**dict(_CHART_LAYOUT, height=320), xaxis_title="Share of Winning Trades", yaxis_title="Share of Gross Profit")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _div(fig)


def _position_size_sensitivity_chart(trades: list[dict]) -> str:
    by_size: dict[int, list[dict]] = {}
    for trade in trades:
        qty = trade.get("total_qty")
        if qty is None:
            continue
        try:
            key = int(qty)
        except (TypeError, ValueError):
            continue
        if key <= 0:
            continue
        by_size.setdefault(key, []).append(trade)

    if not by_size:
        return _empty_chart("No position-size data.")

    sizes = sorted(by_size)
    labels = [str(size) for size in sizes]
    expectancy = []
    win_rates = []
    counts = []
    for size in sizes:
        bucket_trades = by_size[size]
        pnls = [float(t.get("gross_pnl", 0.0) or 0.0) for t in bucket_trades]
        expectancy.append(sum(pnls) / len(pnls))
        win_rates.append(sum(1 for pnl in pnls if pnl > 0) / len(pnls) * 100)
        counts.append(len(bucket_trades))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels,
        y=expectancy,
        marker_color=["#3fb950" if value >= 0 else "#f85149" for value in expectancy],
        text=[f"n={count}" for count in counts],
        textposition="outside",
        name="Expectancy",
        hovertemplate="Qty %{x}<br>Expectancy: %{y:$,.2f}<br>%{text}<extra></extra>",
    ), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels,
        y=win_rates,
        mode="lines+markers",
        name="Win Rate",
        line=dict(color="#58a6ff", width=2),
        hovertemplate="Qty %{x}<br>Win rate: %{y:.1f}%<extra></extra>",
    ), row=1, col=1, secondary_y=True)
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.update_layout(**dict(_CHART_LAYOUT, height=320), xaxis_title="Contracts / Position Size", legend=dict(x=0.02, y=1.02, orientation="h"))
    fig.update_yaxes(title_text="Expectancy ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1, secondary_y=True)
    return _div(fig)


def _monthly_return_heatmap(equity_curve: list[dict]) -> str:
    series = _daily_equity_series(equity_curve)
    if len(series) < 2:
        return _empty_chart("Not enough equity history for monthly heatmap.")

    month_records: list[dict[str, Any]] = []
    current_month = None
    month_start_balance = None
    month_end_balance = None

    for point in series:
        month_key = point["date"].strftime("%Y-%m")
        if current_month != month_key:
            if current_month is not None and month_start_balance not in (None, 0):
                month_records.append({
                    "month_key": current_month,
                    "pnl": month_end_balance - month_start_balance,
                    "return_pct": (month_end_balance - month_start_balance) / month_start_balance * 100.0,
                })
            current_month = month_key
            month_start_balance = month_end_balance if month_end_balance is not None else point["balance"]
        month_end_balance = point["balance"]

    if current_month is not None and month_start_balance not in (None, 0) and month_end_balance is not None:
        month_records.append({
            "month_key": current_month,
            "pnl": month_end_balance - month_start_balance,
            "return_pct": (month_end_balance - month_start_balance) / month_start_balance * 100.0,
        })

    if not month_records:
        return _empty_chart("Not enough monthly return data.")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = sorted({record["month_key"][:4] for record in month_records})
    z = [[None for _ in range(12)] for _ in years]
    text = [["" for _ in range(12)] for _ in years]
    year_index = {year: idx for idx, year in enumerate(years)}

    for record in month_records:
        year, month = record["month_key"].split("-")
        row = year_index[year]
        col = int(month) - 1
        z[row][col] = round(record["return_pct"], 2)
        text[row][col] = f"{record['return_pct']:.2f}%<br>${record['pnl']:,.0f}"

    fig = go.Figure(go.Heatmap(
        x=month_names,
        y=years,
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0.0, "#f85149"], [0.5, "#30363d"], [1.0, "#3fb950"]],
        zmid=0,
        hovertemplate="%{y} %{x}<br>%{text}<extra></extra>",
        colorbar=dict(title="Return %"),
    ))
    fig.update_layout(**dict(_CHART_LAYOUT, height=360), xaxis_title="Month", yaxis_title="Year")
    return _div(fig)


# ---------------------------------------------------------------------------
# New enhanced charts
# ---------------------------------------------------------------------------

def _pnl_waterfall_chart(trades: list[dict]) -> str:
    """Bar chart of per-trade P&L with a cumulative P&L line overlay.

    Each bar is green (winner) or red (loser); the running total is plotted as a
    line on the secondary y-axis so traders can instantly spot whether profits are
    evenly distributed or concentrated in a handful of outlier trades.
    """
    if not trades:
        return _empty_chart("No trade data.")

    pnls = [t.get("gross_pnl", 0.0) for t in trades]
    labels = [f"T{t.get('trade_id', i + 1)}" for i, t in enumerate(trades)]
    cumulative = []
    running = 0.0
    for p in pnls:
        running += p
        cumulative.append(round(running, 2))

    bar_colors = ["#3fb950" if p >= 0 else "#f85149" for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=pnls,
        marker_color=bar_colors,
        name="Trade P&L",
        hovertemplate="Trade %{x}: %{y:$,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=cumulative,
        name="Cumulative P&L",
        mode="lines",
        line=dict(color="#d29922", width=2),
        yaxis="y2",
        hovertemplate="After %{x}: %{y:$,.2f}<extra></extra>",
    ))
    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["yaxis"] = {"title": "Trade P&L ($)", "color": "#c9d1d9"}
    layout["yaxis2"] = {"title": "Cumulative P&L ($)", "overlaying": "y", "side": "right",
                        "color": "#d29922"}
    layout["legend"] = {"x": 0.02, "y": 0.98}
    layout["bargap"] = 0.15
    fig.update_layout(**layout)
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _fee_drag_chart(trades: list[dict]) -> str:
    """Show cumulative gross P&L vs cumulative fees vs cumulative net P&L over trades.

    The shaded gap between gross and net lines visualises fee drag — useful for
    evaluating whether commission costs are materially eroding edge.
    """
    if not trades:
        return _empty_chart("No trade data.")

    labels = [f"T{t.get('trade_id', i + 1)}" for i, t in enumerate(trades)]
    cum_gross, cum_fees, cum_net = [], [], []
    g = f = n = 0.0
    for t in trades:
        g += t.get("gross_pnl", 0.0)
        f += t.get("fees", 0.0)
        n += t.get("net_pnl", 0.0)
        cum_gross.append(round(g, 2))
        cum_fees.append(round(f, 2))
        cum_net.append(round(n, 2))

    fig = go.Figure()
    # Gross P&L fill area
    fig.add_trace(go.Scatter(
        x=labels, y=cum_gross, name="Cumulative Gross P&L",
        line=dict(color="#58a6ff", width=2),
        fill=None,
        hovertemplate="%{x} gross: %{y:$,.2f}<extra></extra>",
    ))
    # Net P&L fill to gross — shaded region = fees
    fig.add_trace(go.Scatter(
        x=labels, y=cum_net, name="Cumulative Net P&L",
        line=dict(color="#3fb950", width=2),
        fill="tonexty",
        fillcolor="rgba(248,81,73,0.15)",
        hovertemplate="%{x} net: %{y:$,.2f}<extra></extra>",
    ))
    # Cumulative fees as a separate line for reference
    fig.add_trace(go.Scatter(
        x=labels, y=cum_fees, name="Cumulative Fees",
        line=dict(color="#f85149", width=1.5, dash="dot"),
        hovertemplate="%{x} fees: %{y:$,.2f}<extra></extra>",
    ))
    layout = dict(_CHART_LAYOUT)
    layout["height"] = 300
    layout["yaxis"] = {"title": "Cumulative P&L ($)"}
    layout["legend"] = {"x": 0.02, "y": 0.98}
    fig.update_layout(**layout)
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _win_loss_histogram_chart(trades: list[dict]) -> str:
    """Overlaid histograms of winner and loser P&L amounts.

    Overlaying the two distributions on the same axis shows the shape of the
    edge: whether wins/losses are fat-tailed, whether the distributions overlap,
    and how cleanly winners are separated from losers.
    """
    if not trades:
        return _empty_chart("No trade data.")

    winners = [t["gross_pnl"] for t in trades if (t.get("gross_pnl") or 0) > 0]
    losers = [t["gross_pnl"] for t in trades if (t.get("gross_pnl") or 0) < 0]

    if not winners and not losers:
        return _empty_chart("No P&L data.")

    nbins = min(30, max(8, int(math.sqrt(len(trades)) * 1.5)))
    fig = go.Figure()
    if winners:
        fig.add_trace(go.Histogram(
            x=winners, name="Winners",
            nbinsx=nbins,
            marker_color="#3fb950",
            opacity=0.65,
            hovertemplate="P&L: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
    if losers:
        fig.add_trace(go.Histogram(
            x=losers, name="Losers",
            nbinsx=nbins,
            marker_color="#f85149",
            opacity=0.65,
            hovertemplate="P&L: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
    layout = dict(_CHART_LAYOUT)
    layout["barmode"] = "overlay"
    layout["bargap"] = 0.05
    layout["legend"] = {"x": 0.75, "y": 0.98}
    fig.update_layout(**layout, xaxis_title="Gross P&L ($)", yaxis_title="Trades")
    fig.add_vline(x=0, line_color="#30363d", line_width=1)
    return _div(fig)


def _exit_type_chart(segmentation: dict | None) -> str:
    """Horizontal bar chart comparing win rate and avg P&L across exit types.

    Uses the precomputed segment_by_exit_type data to avoid re-scanning trades.
    Traders can immediately see which exit type (target / stop / manual) is most
    and least profitable.
    """
    if not segmentation:
        return _empty_chart("No segmentation data.")

    by_exit = segmentation.get("by_exit_type") or {}
    if not by_exit:
        return _empty_chart("No exit-type breakdown available.")

    labels = list(by_exit.keys())
    avg_pnls = [by_exit[k].get("avg_pnl", 0.0) for k in labels]
    win_rates = [by_exit[k].get("win_rate", 0.0) * 100 for k in labels]
    n_trades = [by_exit[k].get("n_trades", 0) for k in labels]

    pnl_colors = ["#3fb950" if v >= 0 else "#f85149" for v in avg_pnls]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg_pnls, y=labels,
        orientation="h",
        name="Avg P&L ($)",
        marker_color=pnl_colors,
        text=[f"${v:,.2f}  n={n}" for v, n in zip(avg_pnls, n_trades)],
        textposition="outside",
        hovertemplate="%{y}: %{x:$,.2f} avg P&L<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=win_rates, y=labels,
        mode="markers+text",
        name="Win Rate (%)",
        marker=dict(color="#58a6ff", size=10, symbol="diamond"),
        text=[f"{v:.0f}%" for v in win_rates],
        textposition="middle right",
        xaxis="x2",
        hovertemplate="%{y}: %{x:.1f}% win rate<extra></extra>",
    ))
    layout = dict(_CHART_LAYOUT)
    layout["height"] = max(200, 60 + 55 * len(labels))
    layout["xaxis"] = {"title": "Avg P&L ($)", "zeroline": True, "zerolinecolor": "#30363d"}
    layout["xaxis2"] = {"title": "Win Rate (%)", "overlaying": "x", "side": "top",
                        "range": [0, 110], "color": "#58a6ff"}
    layout["legend"] = {"x": 0.02, "y": 0.02}
    layout["margin"] = dict(l=80, r=24, t=32, b=48)
    fig.update_layout(**layout)
    return _div(fig)


def _daily_distribution_chart(trades: list[dict]) -> str:
    """Histogram of daily P&L (net, aggregated by exit date).

    Per-trade distributions can hide regime effects; this view shows how
    profitable days compare to losing days and whether daily P&L is normally
    distributed or skewed.
    """
    if not trades:
        return _empty_chart("No trade data.")

    import pandas as pd

    daily: dict[str, float] = {}
    for t in trades:
        et = t.get("exit_time")
        if et is None:
            continue
        d = str(pd.Timestamp(et).date())
        daily[d] = daily.get(d, 0.0) + (t.get("net_pnl") or 0.0)

    if not daily:
        return _empty_chart("No daily P&L data.")

    values = list(daily.values())
    mean = sum(values) / len(values)
    nbins = min(30, max(8, len(values)))

    colors_bar = ["#3fb950" if v >= 0 else "#f85149" for v in values]

    fig = go.Figure(go.Histogram(
        x=values,
        nbinsx=nbins,
        marker=dict(
            color=colors_bar[:1],   # single colour fallback; Plotly ignores per-bin colours in histogram
            colorscale=[[0, "#f85149"], [0.5, "#30363d"], [1, "#3fb950"]],
        ),
        marker_color="#58a6ff",
        opacity=0.85,
        hovertemplate="Daily P&L: %{x:$.2f}<br>Days: %{y}<extra></extra>",
    ))
    layout = dict(_CHART_LAYOUT)
    layout["bargap"] = 0.05
    fig.update_layout(**layout, xaxis_title="Daily Net P&L ($)", yaxis_title="Days")
    fig.add_vline(x=0, line_color="#30363d", line_width=1)
    fig.add_vline(
        x=mean,
        line_color="#3fb950" if mean >= 0 else "#f85149",
        line_width=2,
        annotation_text=f"mean ${mean:,.0f}",
        annotation_position="top right",
    )
    return _div(fig)


# ---------------------------------------------------------------------------
# Win Rate Over Time
# ---------------------------------------------------------------------------

def _win_rate_over_time_chart(segmentation: dict | None) -> str:
    """Monthly win-rate bar chart, reusing the precomputed by_month segmentation."""
    by_month: dict = (segmentation or {}).get("by_month") or {}
    if not by_month:
        return _empty_chart("No monthly trade data available.")

    months = sorted(by_month.keys())   # "YYYY-MM", lexicographic = chronological
    win_rates = [round((by_month[mo].get("win_rate") or 0) * 100, 1) for mo in months]
    n_trades_list = [by_month[mo].get("n_trades", 0) for mo in months]
    colors = ["#3fb950" if wr >= 50 else "#f85149" for wr in win_rates]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=months,
        y=win_rates,
        marker_color=colors,
        name="Win Rate",
        text=[f"{wr:.0f}%" for wr in win_rates],
        textposition="outside",
        textfont=dict(size=10),
        customdata=n_trades_list,
        hovertemplate="Month: %{x}<br>Win Rate: %{y:.1f}%<br>Trades: %{customdata}<extra></extra>",
    ))

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 280
    layout["yaxis"] = {"title": "Win Rate (%)", "range": [0, 118], "ticksuffix": "%"}
    layout["xaxis"] = {"title": "", "tickangle": -45}
    layout["bargap"] = 0.3
    fig.update_layout(**layout)
    fig.add_hline(
        y=50,
        line_color="#d29922",
        line_width=1.5,
        line_dash="dash",
        annotation_text="50 %",
        annotation_position="top right",
        annotation_font_size=10,
    )
    return _div(fig)


def _monte_carlo_chart(mc_results: dict) -> str:
    """Fan chart of bootstrap equity-curve distribution."""
    if not mc_results or not mc_results.get("percentile_curves"):
        return _empty_chart("Insufficient data for Monte Carlo simulation (< 5 trades).")

    curves = mc_results["percentile_curves"]
    actual = mc_results.get("actual_curve", [])
    n = len(curves.get("p50", [])) - 1
    if n < 1:
        return _empty_chart("Insufficient trade data for Monte Carlo simulation.")

    x = list(range(n + 1))
    fig = go.Figure()

    # Outer band P5–P95
    fig.add_trace(go.Scatter(
        x=x, y=curves["p95"], name="P95",
        mode="lines", line=dict(color="rgba(56,139,253,0.25)", width=1),
        hovertemplate="Trade %{x}: $%{y:,.0f}<extra>P95</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curves["p5"], name="P5",
        mode="lines", line=dict(color="rgba(248,81,73,0.25)", width=1),
        fill="tonexty", fillcolor="rgba(99,110,123,0.15)",
        hovertemplate="Trade %{x}: $%{y:,.0f}<extra>P5</extra>",
    ))

    # Inner band P25–P75
    fig.add_trace(go.Scatter(
        x=x, y=curves["p75"], name="P75",
        mode="lines", line=dict(color="rgba(63,185,80,0.5)", width=1.5),
        hovertemplate="Trade %{x}: $%{y:,.0f}<extra>P75</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=curves["p25"], name="P25",
        mode="lines", line=dict(color="rgba(63,185,80,0.5)", width=1.5),
        fill="tonexty", fillcolor="rgba(63,185,80,0.18)",
        hovertemplate="Trade %{x}: $%{y:,.0f}<extra>P25</extra>",
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=x, y=curves["p50"], name="Median (P50)",
        mode="lines", line=dict(color="#d29922", width=2),
        hovertemplate="Trade %{x}: $%{y:,.0f}<extra>Median</extra>",
    ))

    # Actual historical overlay
    if actual:
        actual_x = list(range(len(actual)))
        fig.add_trace(go.Scatter(
            x=actual_x, y=actual, name="Actual",
            mode="lines", line=dict(color="#58a6ff", width=2.5),
            hovertemplate="Trade %{x}: $%{y:,.0f}<extra>Actual</extra>",
        ))

    layout = dict(_CHART_LAYOUT)
    layout["height"] = 340
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    fig.update_layout(**layout, xaxis_title="Trade Number", yaxis_title="Balance ($)")
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

    pnls = [t.get("gross_pnl", 0.0) for t in trades]
    starting_bal = equity_curve[0]["balance"] if equity_curve else 0.0
    monte_carlo = run_monte_carlo(pnls, starting_bal) if len(pnls) >= 5 else {}
    mc_chart = _monte_carlo_chart(monte_carlo)

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
        mae_winners_scatter_chart=_mae_winners_scatter_chart(trades),
        drawdown_recovery_chart=_drawdown_recovery_chart(equity_curve),
        time_bucket_expectancy_chart=_time_bucket_expectancy_chart(segmentation_data),
        excursion_percentile_chart=_excursion_percentile_chart(trades),
        holding_time_efficiency_chart=_holding_time_efficiency_chart(trades),
        streak_state_chart=_streak_state_chart(trades),
        exit_efficiency_chart=_exit_efficiency_chart(trades),
        concentration_lorenz_chart=_lorenz_curve_chart(trades),
        position_size_chart=_position_size_sensitivity_chart(trades),
        monthly_return_heatmap_chart=_monthly_return_heatmap(equity_curve),
        timing_heatmap_chart=_timing_heatmap(trades),
        direction_mix_chart=_direction_mix_chart(trades),
        session_mix_chart=_session_mix_chart(trades),
        outcome_mix_chart=_outcome_mix_chart(trades),
        pnl_waterfall_chart=_pnl_waterfall_chart(trades),
        fee_drag_chart=_fee_drag_chart(trades),
        win_loss_histogram_chart=_win_loss_histogram_chart(trades),
        exit_type_chart=_exit_type_chart(segmentation_data),
        daily_distribution_chart=_daily_distribution_chart(trades),
        win_rate_over_time_chart=_win_rate_over_time_chart(segmentation_data),
        rolling_volatility_chart=_rolling_volatility_chart(equity_curve, benchmark_data),
        rolling_sharpe_chart=_rolling_sharpe_chart(equity_curve),
        rolling_sortino_chart=_rolling_sortino_chart(equity_curve),
        worst_drawdown_periods_chart=_worst_drawdown_periods_chart(equity_curve),
        eoy_returns_chart=_eoy_returns_chart(equity_curve, benchmark_data),
        monthly_returns_dist_chart=_monthly_returns_dist_chart(equity_curve, benchmark_data),
        daily_active_returns_chart=_daily_active_returns_chart(equity_curve, benchmark_data),
        rolling_window=20,
        rolling_metrics=rolling_metrics,
        has_r_multiples=has_r_multiples,
        benchmark_metrics=benchmark_metrics,
        calendar_data=calendar_data,
        plotly_js=plotly_js,
        monthly_summary=monthly_summary,
        sc_statistics=sc_statistics,
        monte_carlo=monte_carlo,
        monte_carlo_chart=mc_chart,
    )

    Path(output_path).write_text(html, encoding="utf-8")
