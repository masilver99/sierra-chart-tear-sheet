"""Pytest fixtures shared across the test suite.

The *three_trades* fixture provides exactly the 3 trades described in the
spec so that exact-value assertions work independently of the real sample
file (which has 8+ trades).
"""

from __future__ import annotations

import datetime
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal fills DataFrame reproducing the spec's 3 trades exactly
# ---------------------------------------------------------------------------

def _row(dt_str, bs, qty, fill_price, pos_qty, symbol="MESM26_FUT_CME",
         oc="Open", order_type="Limit", high=None, low=None, exec_id=""):
    return {
        "DateTime": pd.to_datetime(dt_str),
        "BuySell": bs,
        "Quantity": qty,           # incremental per-fill quantity (spec: use this for P&L)
        "FilledQuantity": qty,     # same in fixture (cumulative = incremental for single fills)
        "FillPrice": fill_price,
        "PositionQuantity": pos_qty,
        "Symbol": symbol,
        "OpenClose": oc,
        "OrderType": order_type,
        "HighDuringPosition": high,
        "LowDuringPosition": low,
        "FillExecutionServiceID": exec_id,
    }


@pytest.fixture
def spec_fills_df():
    """Fills DataFrame for exactly 3 spec trades (T1 has 2-lot entry splits)."""
    rows = [
        # T1: Long 2 lots @ 6812.75 → exit @ 6818.00
        _row("2026-04-09 10:10:07", "Buy", 1, 6812.75, 1.0, exec_id="e1"),
        _row("2026-04-09 10:10:08", "Buy", 1, 6812.75, 2.0, exec_id="e2"),
        _row("2026-04-09 10:12:51", "Sell", 2, 6818.00, None, oc="Close",
             order_type="Limit", high=6818.00, low=6811.00, exec_id="e3"),
        # T2: Short 2 lots @ 6806.25 → exit @ 6814.25
        _row("2026-04-09 10:41:51", "Sell", 2, 6806.25, -2.0, oc="Open",
             order_type="Stop Limit", exec_id="e4"),
        _row("2026-04-09 10:50:38", "Buy", 2, 6814.25, None, oc="Close",
             order_type="Stop Limit", high=6814.25, low=6805.50, exec_id="e5"),
        # T3: Long 2 lots @ 6815.00 → exit @ 6811.50
        _row("2026-04-09 10:54:16", "Buy", 2, 6815.00, 2.0, exec_id="e6"),
        _row("2026-04-09 11:04:42", "Sell", 2, 6811.50, None, oc="Close",
             order_type="Stop Limit", high=6819.00, low=6809.50, exec_id="e7"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def three_trades(spec_fills_df):
    """Exactly 3 reconstructed trade dicts matching the spec."""
    from tearsheet.recon.trades import reconstruct_trades
    return reconstruct_trades(spec_fills_df)
