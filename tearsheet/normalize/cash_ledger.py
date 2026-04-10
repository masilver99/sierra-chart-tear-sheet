"""Parse fee and P&L events from AccountBalance rows, and compute fees from fills."""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

_FEE_RE = re.compile(r"Trade Fee:\s*([\d.]+)\s*USD", re.IGNORECASE)
_PNL_RE = re.compile(r"Closed Trade Profit/Loss:\s*(-?[\d]+(?:\.\d+)?)", re.IGNORECASE)

# Broker-confirmed commission rates per side per contract.
# Each fill (entry or exit) is charged this amount per lot.
COMMISSION_PER_SIDE: dict[str, float] = {
    "MES": 0.50,
    "ES":  1.90,
    "1OZ": 0.42,
    "1oz": 0.42,
}


def _symbol_rate(symbol: str) -> float | None:
    """Return the per-side commission rate for *symbol*, or None if unknown."""
    for prefix, rate in COMMISSION_PER_SIDE.items():
        if symbol.startswith(prefix):
            return rate
    return None


def compute_fee_events_from_fills(fills_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute fee events from fill data using broker-confirmed per-side rates.

    This approach is more accurate than parsing Account Balance fee rows because:
    - Account Balance records slightly inflated rates (e.g. MES $0.52 vs actual $0.50)
    - Teton CME Routing phantom fills have real commissions that AB never records

    Each fill (entry and exit) incurs ``Quantity × rate_per_side`` in fees.

    Symbols not found in :data:`COMMISSION_PER_SIDE` are skipped with a warning.

    Returns
    -------
    list[dict]
        One entry per fill with keys ``DateTime``, ``kind`` (``'fee'``), ``amount``.
    """
    events: list[dict[str, Any]] = []
    unknown: set[str] = set()

    for _, row in fills_df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        rate = _symbol_rate(symbol)
        if rate is None:
            unknown.add(symbol)
            continue
        qty = int(row["Quantity"])
        dt = row["DateTime"]
        events.append({"DateTime": dt, "kind": "fee", "amount": round(qty * rate, 4)})

    if unknown:
        warnings.warn(
            f"Commission rate unknown for symbol(s): {', '.join(sorted(unknown))}. "
            "These fills contributed no commission. Add rates to COMMISSION_PER_SIDE.",
            stacklevel=2,
        )

    return events


def extract_cash_events(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a list of dicts with keys ``DateTime``, ``kind``, ``amount``.

    *kind* is ``'fee'`` or ``'pnl'``.
    Rows that match neither pattern are silently skipped.
    """
    ab = df[df["ActivityType"] == "Account Balance"].copy()
    events: list[dict[str, Any]] = []

    for _, row in ab.iterrows():
        # Fee/P&L data lives in OrderActionSource, not Note
        note = str(row.get("OrderActionSource", ""))
        dt = row["DateTime"]

        m_fee = _FEE_RE.search(note)
        if m_fee:
            events.append({"DateTime": dt, "kind": "fee", "amount": float(m_fee.group(1))})
            continue

        m_pnl = _PNL_RE.search(note)
        if m_pnl:
            events.append({"DateTime": dt, "kind": "pnl", "amount": float(m_pnl.group(1))})

    return events
