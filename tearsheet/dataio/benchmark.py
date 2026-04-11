"""Optional benchmark (SPY) data fetcher. Degrades gracefully if yfinance unavailable."""

from __future__ import annotations

import datetime
from typing import Optional


def fetch_benchmark(
    start_date: datetime.date,
    end_date: datetime.date,
    ticker: str = "SPY",
) -> Optional[dict]:
    """
    Fetch benchmark daily data.

    Returns None if:
    - yfinance is not installed
    - Network error
    - No overlapping dates

    Returns:
    {
        "ticker": str,
        "dates": list[str],          # ISO date strings
        "closes": list[float],
        "normalized": list[float],   # 100 * close / close[0]
        "total_return_pct": float,   # (last/first - 1) * 100
    }
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        # Fetch with 1 extra day buffer for alignment
        fetch_start = start_date - datetime.timedelta(days=5)
        fetch_end = end_date + datetime.timedelta(days=1)
        hist = yf.download(ticker, start=str(fetch_start), end=str(fetch_end),
                           progress=False, auto_adjust=True)
        if hist is None or len(hist) < 2:
            return None

        # Filter to date range
        hist.index = [d.date() if hasattr(d, 'date') else d for d in hist.index]
        matching_dates = [d for d in hist.index if start_date <= d <= end_date]
        if not matching_dates:
            return None

        if len(matching_dates) == 1:
            match_pos = hist.index.get_loc(matching_dates[0])
            if isinstance(match_pos, slice):
                match_pos = match_pos.start
            try:
                match_pos = int(match_pos)
            except (TypeError, ValueError):
                return None
            if match_pos <= 0:
                return None
            hist = hist.iloc[match_pos - 1:match_pos + 1]
        else:
            mask = [(d >= start_date and d <= end_date) for d in hist.index]
            hist = hist[mask]
            if len(hist) < 2:
                return None

        close_values = hist["Close"]
        # yfinance may return a DataFrame with MultiIndex columns; flatten if needed
        if hasattr(close_values, "columns"):
            close_values = close_values.iloc[:, 0]
        closes = [float(c) for c in close_values.tolist()]

        dates = [str(d) for d in hist.index]
        base = closes[0]
        normalized = [100.0 * c / base for c in closes]

        return {
            "ticker": ticker,
            "dates": dates,
            "closes": [round(c, 4) for c in closes],
            "normalized": [round(n, 4) for n in normalized],
            "total_return_pct": round((closes[-1] / closes[0] - 1) * 100, 4),
        }
        return None

    except Exception:
        return None


def compute_benchmark_metrics(
    daily_pnl: dict,
    start_balance: float,
    benchmark_data: dict,
) -> dict:
    """Compute comparison metrics between strategy and benchmark."""
    total_strategy_pnl = sum(daily_pnl.values()) if daily_pnl else 0.0
    strategy_total_return_pct = (total_strategy_pnl / start_balance) if start_balance else 0.0
    benchmark_total_return_pct = (benchmark_data.get("total_return_pct", 0.0) or 0.0) / 100.0
    ticker = benchmark_data.get("ticker", "SPY")

    beta = None
    alpha_annualized = None
    treynor_ratio = None
    m2_ratio = None
    correlation = None

    try:
        bench_dates = benchmark_data.get("dates", [])
        bench_normalized = benchmark_data.get("normalized", [])
        if len(bench_dates) >= 2 and len(bench_normalized) == len(bench_dates):
            import datetime as _dt
            import math
            bench_rets = []
            bench_date_keys = []
            for i in range(1, len(bench_normalized)):
                br = (bench_normalized[i] - bench_normalized[i - 1]) / bench_normalized[i - 1]
                d = bench_dates[i]
                if isinstance(d, str):
                    d = _dt.date.fromisoformat(d)
                bench_rets.append(br)
                bench_date_keys.append(d)

            if bench_date_keys and start_balance > 0:
                strat_rets = [daily_pnl.get(d, 0.0) / start_balance for d in bench_date_keys]
                n = len(strat_rets)
                if n >= 3:
                    sm = sum(strat_rets) / n
                    bm = sum(bench_rets) / n
                    cov = sum((s - sm) * (b - bm) for s, b in zip(strat_rets, bench_rets)) / n
                    sv = sum((s - sm) ** 2 for s in strat_rets) / n
                    bv = sum((b - bm) ** 2 for b in bench_rets) / n
                    if sv > 0 and bv > 0:
                        correlation = round(cov / math.sqrt(sv * bv), 4)
                        beta = round(cov / bv, 4)
                        alpha_annualized = round(sm * 252 - beta * bm * 252, 4)
                        strat_ann = sm * 252
                        treynor_ratio = round(strat_ann / beta, 4) if beta != 0 else None
                        # M² = Sharpe * σm_ann  (with Rf = 0), expressed in percentage points
                        m2_ratio = round(strat_ann * math.sqrt(bv) / math.sqrt(sv) * 100, 4)
    except Exception:
        pass

    return {
        "strategy_total_return_pct": round(strategy_total_return_pct * 100, 4),
        "benchmark_total_return_pct": round(benchmark_total_return_pct * 100, 4),
        "alpha": round((strategy_total_return_pct - benchmark_total_return_pct) * 100, 4),
        "beta": beta,
        "alpha_annualized": alpha_annualized,
        "treynor_ratio": treynor_ratio,
        "m2_ratio": m2_ratio,
        "correlation": correlation,
        "ticker": ticker,
    }
