"""Orchestration pipeline — ties all modules together."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tearsheet.dataio.loader import load_file
from tearsheet.normalize.events import split_events
from tearsheet.normalize.orders import normalize_orders
from tearsheet.recon.trades import reconstruct_trades, enrich_trades
from tearsheet.recon.equity import build_equity_curve, detect_cash_flows, adjust_equity_curve
from tearsheet.metrics.performance import compute_metrics
from tearsheet.metrics.monthly_summary import compute_monthly_summary
from tearsheet.metrics.sc_statistics import compute_sc_statistics
from tearsheet.metrics.execution import compute_execution_metrics
from tearsheet.metrics.segmentation import (
    segment_by_direction, segment_by_instrument, segment_by_note, segment_by_session,
    segment_by_date, segment_by_day_of_week, segment_by_week, segment_by_month,
    segment_by_hour, segment_by_exit_type,
    pct_profitable_periods,
)
from tearsheet.metrics.rolling import compute_rolling_metrics
from tearsheet.dataio.benchmark import fetch_benchmark, compute_benchmark_metrics
from tearsheet.report.render import render_report


_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _build_calendar_data(trades: list[dict]) -> dict:
    """Build a JSON-serializable calendar data structure for the interactive calendar.

    Returns ``{"years": {str_year: {..., "months": {str_month: {...}}}},
               "days":  {date_str: {..., "trades": [trade_record, ...]}}}``.

    All stats (win rate, profit factor, max drawdown, is_profitable) are based
    on **net** P&L so every metric is consistent with the calendar tile colour.
    """
    import pandas as pd

    days: dict = {}

    # --- Pass 1: accumulate per-day raw stats from individual trades ---
    for t in trades:
        et = t.get("exit_time")
        if et is None:
            continue
        try:
            ts = pd.Timestamp(et)
        except Exception:
            continue

        y, m, d = ts.year, ts.month, ts.day
        date_str = ts.strftime("%Y-%m-%d")

        if date_str not in days:
            days[date_str] = {
                "year": y, "month": m, "day": d,
                "total_pnl": 0.0, "net_pnl": 0.0,
                "n_trades": 0, "n_wins": 0,      # n_wins = gross-based (legacy)
                "_net_wins": 0,                    # net-based, for calendar display
                "_net_gross_wins": 0.0,            # sum of net_pnl where net_pnl > 0
                "_net_gross_losses": 0.0,          # sum of net_pnl where net_pnl < 0
                "trades": [],
            }

        gp = float(t.get("gross_pnl", 0.0) or 0.0)
        np_ = float(t.get("net_pnl", 0.0) or 0.0)

        days[date_str]["total_pnl"] += gp
        days[date_str]["net_pnl"] += np_
        days[date_str]["n_trades"] += 1
        if gp > 0:
            days[date_str]["n_wins"] += 1
        if np_ > 0:
            days[date_str]["_net_wins"] += 1
            days[date_str]["_net_gross_wins"] += np_
        elif np_ < 0:
            days[date_str]["_net_gross_losses"] += np_   # stays negative

        # Full trade record — same schema as the main trade log
        rec: dict = {
            "trade_id": t.get("trade_id"),
            "symbol": t.get("symbol", ""),
            "direction": t.get("direction", ""),
            "entry_time": str(t.get("entry_time", "")),
            "exit_time": str(t.get("exit_time", "")),
            "duration_s": t.get("duration_s"),
            "total_qty": t.get("total_qty", 0),
            "avg_entry": round(float(t.get("avg_entry", 0.0)), 2),
            "avg_exit": round(float(t.get("avg_exit", 0.0)), 2),
            "gross_pnl": round(float(t.get("gross_pnl", 0.0)), 2),
            "fees": round(float(t.get("fees", 0.0)), 2),
            "net_pnl": round(float(t.get("net_pnl", 0.0)), 2),
            "mfe": round(float(t.get("mfe", 0.0)), 2),
            "mae": round(float(t.get("mae", 0.0)), 2),
            "exit_type": t.get("exit_type") or "",
            "note": t.get("note") or "",
            "entry_chase_pts": None,
            "exit_chase_pts": None,
            "stop_price": None,
            "initial_risk": None,
            "r_multiple": None,
        }
        for fld in ("entry_chase_pts", "exit_chase_pts", "stop_price", "initial_risk"):
            v = t.get(fld)
            rec[fld] = round(float(v), 4) if v is not None else None
        v = t.get("r_multiple")
        rec["r_multiple"] = round(float(v), 2) if v is not None else None
        days[date_str]["trades"].append(rec)

    # --- Pass 2: compute derived day-level stats ---
    for dd in days.values():
        n = dd["n_trades"]
        dd["total_pnl"] = round(dd["total_pnl"], 2)
        dd["net_pnl"] = round(dd["net_pnl"], 2)
        dd["is_profitable"] = dd["net_pnl"] > 0
        # Net-based win rate for calendar display
        dd["win_rate"] = round(dd["_net_wins"] / n, 4) if n else 0.0
        # Net-based profit factor — None when no winners or no losers
        gw = dd["_net_gross_wins"]
        gl = abs(dd["_net_gross_losses"])
        dd["profit_factor"] = round(gw / gl, 2) if gl > 0 and gw > 0 else None
        # Max intra-day drawdown on cumulative net_pnl, anchored at 0
        # Sort trades by exit_time then trade_id for a stable chronological order.
        recs = sorted(
            dd["trades"],
            key=lambda r: (r.get("exit_time", ""), r.get("trade_id") or 0),
        )
        peak = 0.0
        running = 0.0
        max_dd = 0.0
        for r in recs:
            running += r.get("net_pnl", 0.0)
            if running > peak:
                peak = running
            val = running - peak
            if val < max_dd:
                max_dd = val
        dd["max_drawdown"] = round(max_dd, 2)

    # --- Pass 3: build year/month aggregates from chronologically sorted days ---
    # Separate accumulator for the ordered daily net_pnl sequence per month
    # (kept outside the public dict to avoid JSON leakage).
    month_day_pnls: dict[str, list[float]] = {}

    str_years: dict = {}
    for date_str in sorted(days.keys()):   # ISO sort = chronological order
        dd = days[date_str]
        ys = str(dd["year"])
        m = dd["month"]
        ms = str(m)
        mo_key = f"{ys}-{ms}"

        if ys not in str_years:
            str_years[ys] = {
                "total_pnl": 0.0, "net_pnl": 0.0,
                "n_trades": 0, "n_days": 0, "months": {},
            }
        yr = str_years[ys]
        yr["total_pnl"] += dd["total_pnl"]
        yr["net_pnl"] += dd["net_pnl"]
        yr["n_trades"] += dd["n_trades"]
        yr["n_days"] += 1

        if ms not in yr["months"]:
            yr["months"][ms] = {
                "total_pnl": 0.0, "net_pnl": 0.0,
                "n_trades": 0, "n_days": 0,
                "name": _MONTH_ABBR[m - 1], "month_num": m,
                "_net_wins": 0,
                "_net_gross_wins": 0.0,
                "_net_gross_losses": 0.0,
            }
            month_day_pnls[mo_key] = []

        mo = yr["months"][ms]
        mo["total_pnl"] += dd["total_pnl"]
        mo["net_pnl"] += dd["net_pnl"]
        mo["n_trades"] += dd["n_trades"]
        mo["n_days"] += 1
        mo["_net_wins"] += dd["_net_wins"]
        mo["_net_gross_wins"] += dd["_net_gross_wins"]
        mo["_net_gross_losses"] += dd["_net_gross_losses"]
        month_day_pnls[mo_key].append(dd["net_pnl"])   # already in order (sorted loop)

    # --- Pass 4: finalise year and month derived stats ---
    for ys, yr in str_years.items():
        yr["total_pnl"] = round(yr["total_pnl"], 2)
        yr["net_pnl"] = round(yr["net_pnl"], 2)
        yr["is_profitable"] = yr["net_pnl"] > 0

        for ms, mo in yr["months"].items():
            mo_key = f"{ys}-{ms}"
            mo["total_pnl"] = round(mo["total_pnl"], 2)
            mo["net_pnl"] = round(mo["net_pnl"], 2)
            mo["is_profitable"] = mo["net_pnl"] > 0

            n = mo["n_trades"]
            mo["win_rate"] = round(mo["_net_wins"] / n, 4) if n else 0.0
            gw = mo["_net_gross_wins"]
            gl = abs(mo["_net_gross_losses"])
            mo["profit_factor"] = round(gw / gl, 2) if gl > 0 and gw > 0 else None

            # Max drawdown from sequential daily net_pnl (already chronologically ordered).
            peak = 0.0
            running = 0.0
            max_dd = 0.0
            for dp in month_day_pnls.get(mo_key, []):
                running += dp
                if running > peak:
                    peak = running
                val = running - peak
                if val < max_dd:
                    max_dd = val
            mo["max_drawdown"] = round(max_dd, 2)

            del mo["_net_wins"], mo["_net_gross_wins"], mo["_net_gross_losses"]

    # --- Final cleanup: remove internal accumulators from day records ---
    for dd in days.values():
        del dd["_net_wins"], dd["_net_gross_wins"], dd["_net_gross_losses"]

    return {"years": str_years, "days": days}


def run(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Execute the full pipeline and write *output_path*.

    Returns a summary dict with ``trades``, ``metrics`` keys for testing.
    """
    import pandas as pd

    input_path = Path(input_path)
    output_path = Path(output_path)

    df = load_file(input_path)
    fills, cash_events = split_events(df)
    trades = reconstruct_trades(fills, cash_events)

    # Phase 2: order-level enrichment; Phase 3: adds r_multiple
    orders = normalize_orders(df)
    enriched_trades = enrich_trades(trades, orders)

    equity_curve = build_equity_curve(df)
    cash_flows = detect_cash_flows(df)
    adjust_equity_curve(equity_curve, cash_flows)
    metrics = compute_metrics(enriched_trades, equity_curve)

    monthly_summary = compute_monthly_summary(enriched_trades)

    sc_statistics = compute_sc_statistics(enriched_trades)

    exec_metrics = compute_execution_metrics(enriched_trades, orders)

    rolling_metrics = compute_rolling_metrics(enriched_trades, window=20)

    # Segmentation
    by_date = segment_by_date(enriched_trades)
    by_dow = segment_by_day_of_week(enriched_trades)
    by_week = segment_by_week(enriched_trades)
    by_month = segment_by_month(enriched_trades)

    segmentation = {
        "by_direction": segment_by_direction(enriched_trades),
        "by_instrument": segment_by_instrument(enriched_trades),
        "by_note": segment_by_note(enriched_trades),
        "by_session": segment_by_session(enriched_trades),
        "by_date": by_date,
        "by_day_of_week": by_dow,
        "by_week": by_week,
        "by_month": by_month,
        "by_hour": segment_by_hour(enriched_trades),
        "by_exit_type": segment_by_exit_type(enriched_trades),
        "pct_profitable_days": pct_profitable_periods(by_date),
        "pct_profitable_weeks": pct_profitable_periods(by_week),
        "pct_profitable_months": pct_profitable_periods(by_month),
    }

    calendar_data = _build_calendar_data(enriched_trades)

    # Benchmark (only for multi-day data)
    benchmark_data = None
    benchmark_metrics = None
    exit_dates = sorted(set(
        pd.Timestamp(t["exit_time"]).date()
        for t in enriched_trades if t.get("exit_time")
    ))
    if exit_dates:
        benchmark_data = fetch_benchmark(exit_dates[0], exit_dates[-1])
        if benchmark_data:
            start_balance = equity_curve[0]["balance"] if equity_curve else 18000.0
            daily_pnl: dict = {}
            for t in enriched_trades:
                if t.get("exit_time"):
                    d = pd.Timestamp(t["exit_time"]).date()
                    daily_pnl[d] = daily_pnl.get(d, 0.0) + t["gross_pnl"]
            benchmark_metrics = compute_benchmark_metrics(daily_pnl, start_balance, benchmark_data)

    render_report(
        enriched_trades, equity_curve, metrics, output_path,
        source_file=input_path.name,
        exec_metrics=exec_metrics,
        segmentation=segmentation,
        rolling_metrics=rolling_metrics,
        benchmark_data=benchmark_data,
        benchmark_metrics=benchmark_metrics,
        calendar_data=calendar_data,
        cash_flows=cash_flows,
        monthly_summary=monthly_summary,
        sc_statistics=sc_statistics,
    )

    print(f"[tearsheet] {len(enriched_trades)} trades processed → {output_path}")
    return {
        "trades": enriched_trades,
        "metrics": metrics,
        "equity_curve": equity_curve,
        "exec_metrics": exec_metrics,
        "segmentation": segmentation,
        "rolling_metrics": rolling_metrics,
        "benchmark_data": benchmark_data,
        "benchmark_metrics": benchmark_metrics,
        "calendar_data": calendar_data,
        "monthly_summary": monthly_summary,
        "sc_statistics": sc_statistics,
    }
