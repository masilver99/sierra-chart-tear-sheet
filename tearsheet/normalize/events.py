"""General event helpers — currently a thin convenience wrapper."""

from __future__ import annotations

import pandas as pd

from tearsheet.normalize.fills import extract_fills
from tearsheet.normalize.cash_ledger import compute_fee_events_from_fills


def split_events(df: pd.DataFrame):
    """Return ``(fills_df, cash_events)`` from a raw loaded DataFrame.

    Fee events are computed from fills at broker-confirmed per-side rates
    rather than from Account Balance rows, which use slightly inflated rates
    and miss commissions for Teton CME Routing phantom fills.
    """
    fills = extract_fills(df)
    cash = compute_fee_events_from_fills(fills)
    return fills, cash
