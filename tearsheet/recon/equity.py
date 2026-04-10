"""Build an equity curve from AccountBalance rows."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Sierra Chart re-posts historical balance events for "Teton CME Routing".
# These rows duplicate already-recorded P&L/fee events at slightly different
# timestamps and create artificial zigzags in the raw equity curve.
_TETON_PREFIX = "Teton CME Routing historical balance"

_FEE_RE = re.compile(r"Trade Fee:\s*([\d.]+)", re.IGNORECASE)
_PNL_RE = re.compile(r"Closed Trade Profit/Loss:\s*(-?[\d]+(?:\.\d+)?)", re.IGNORECASE)


def _parse_expected_delta(source: str) -> float | None:
    """Return the expected account-balance change implied by *source* text.

    Returns ``None`` when the source is unrecognised (caller should treat the
    full observed delta as a potential cash flow).
    """
    if not source:
        return None
    m = _FEE_RE.search(source)
    if m:
        return -float(m.group(1))
    m = _PNL_RE.search(source)
    if m:
        return float(m.group(1))
    if "current account balance data request" in source.lower():
        return 0.0
    return None


def detect_cash_flows(df: pd.DataFrame, threshold: float = 50.0) -> list[dict[str, Any]]:
    """Detect deposits and withdrawals as unexplained account-balance changes.

    Teton CME Routing historical re-post rows are excluded before analysis
    because they duplicate already-recorded events and would produce false
    positives.

    Returns a list of ``{DateTime, amount}`` dicts where:

    * ``amount > 0``  — deposit (balance increase not due to trading)
    * ``amount < 0``  — withdrawal (balance decrease not due to trading)
    """
    ab = df[df["ActivityType"] == "Account Balance"].copy()
    ab = ab[~ab["OrderActionSource"].str.startswith(_TETON_PREFIX, na=False)]
    ab = ab.dropna(subset=["AccountBalance", "DateTime"])
    ab = ab.sort_values("DateTime").reset_index(drop=True)

    cash_flows: list[dict[str, Any]] = []
    prev_balance: float | None = None

    for _, row in ab.iterrows():
        curr = float(row["AccountBalance"])
        src = str(row.get("OrderActionSource", ""))
        expected = _parse_expected_delta(src)

        if prev_balance is not None:
            actual_delta = curr - prev_balance
            residual = actual_delta - expected if expected is not None else actual_delta
            if abs(residual) >= threshold:
                cash_flows.append({"DateTime": row["DateTime"], "amount": round(residual, 2)})

        prev_balance = curr

    return cash_flows


def adjust_equity_curve(
    curve: list[dict[str, Any]],
    cash_flows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add an ``adjusted_balance`` field to every point in *curve* (in-place).

    The adjusted balance strips out cumulative cash flows so that the curve
    reflects only trading P&L starting from the initial account balance.

    Example: a $7 500 deposit on day 30 lifts the raw curve by $7 500 but the
    adjusted curve continues from where it left off — the deposit appears only
    as a clearly-labelled annotation on the chart.
    """
    if not cash_flows:
        for point in curve:
            point["adjusted_balance"] = point["balance"]
        return curve

    sorted_flows = sorted(cash_flows, key=lambda x: x["DateTime"])
    cumulative = 0.0
    flow_idx = 0
    n_flows = len(sorted_flows)

    # curve is already sorted chronologically by build_equity_curve()
    for point in curve:
        while flow_idx < n_flows and sorted_flows[flow_idx]["DateTime"] <= point["DateTime"]:
            cumulative += sorted_flows[flow_idx]["amount"]
            flow_idx += 1
        point["adjusted_balance"] = round(point["balance"] - cumulative, 2)

    return curve


def build_equity_curve(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a chronological list of ``{DateTime, balance}`` dicts.

    Uses only *Account Balance* rows that carry a numeric ``AccountBalance``.
    Teton CME Routing historical re-post rows are filtered out to prevent
    artificial spikes and duplicates in the curve.
    """
    ab = df[df["ActivityType"] == "Account Balance"].copy()
    ab = ab[~ab["OrderActionSource"].str.startswith(_TETON_PREFIX, na=False)]
    ab = ab.dropna(subset=["AccountBalance", "DateTime"])
    ab = ab.sort_values("DateTime")

    curve = [
        {"DateTime": row["DateTime"], "balance": float(row["AccountBalance"])}
        for _, row in ab.iterrows()
    ]
    return curve


def daily_returns(equity_curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return per-day P&L list ``{date, pnl}`` from the equity curve.

    Prefers ``adjusted_balance`` (trading P&L only) when available.
    """
    if not equity_curve:
        return []

    balance_key = "adjusted_balance" if "adjusted_balance" in equity_curve[0] else "balance"

    by_date: dict[Any, list[float]] = {}
    for point in equity_curve:
        date = pd.Timestamp(point["DateTime"]).date()
        by_date.setdefault(date, []).append(point[balance_key])

    dates = sorted(by_date)
    result = []
    for i, date in enumerate(dates):
        balances = by_date[date]
        if i == 0:
            prev = balances[0]
        else:
            prev = by_date[dates[i - 1]][-1]
        result.append({"date": date, "pnl": balances[-1] - prev})
    return result
