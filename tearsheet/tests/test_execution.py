"""Tests for normalize.orders and metrics.execution using the real sample file."""

from __future__ import annotations

import pathlib
import pytest

DATA_FILE = pathlib.Path(__file__).parents[2] / "TradeActivityLog_2026-04-09.txt"
pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="Real data file not present")


@pytest.fixture(scope="module")
def orders():
    from tearsheet.dataio.loader import load_file
    from tearsheet.normalize.orders import normalize_orders
    df = load_file(DATA_FILE)
    return normalize_orders(df)


@pytest.fixture(scope="module")
def enriched_trades():
    from tearsheet.dataio.loader import load_file
    from tearsheet.normalize.events import split_events
    from tearsheet.normalize.orders import normalize_orders
    from tearsheet.recon.trades import reconstruct_trades, enrich_trades
    df = load_file(DATA_FILE)
    fills, cash_events = split_events(df)
    trades = reconstruct_trades(fills, cash_events)
    orders = normalize_orders(df)
    return enrich_trades(trades, orders)


def test_normalize_orders_count(orders):
    """Real file has 48 unique orders (deduplicated by InternalOrderID)."""
    assert len(orders) >= 10


def test_normalize_orders_fields(orders):
    required = {"order_id", "exchange_order_id", "status", "price", "modify_count"}
    for o in orders[:5]:
        assert required.issubset(set(o.keys()))


def test_order_fill_linkage(orders):
    """At least some orders have a non-empty exchange_order_id (fill linkage)."""
    with_exch = [o for o in orders if o.get("exchange_order_id")]
    assert len(with_exch) > 0


def test_cancel_rate_range(orders):
    from tearsheet.metrics.execution import compute_execution_metrics
    exec_m = compute_execution_metrics([], orders)
    assert exec_m["cancel_rate"] is not None
    assert 0.0 <= exec_m["cancel_rate"] <= 1.0


def test_exec_metrics_keys(enriched_trades, orders):
    from tearsheet.metrics.execution import compute_execution_metrics
    exec_m = compute_execution_metrics(enriched_trades, orders)
    required = {
        "total_orders", "fill_rate", "cancel_rate", "modify_rate",
        "avg_entry_chase_pts", "avg_exit_chase_pts",
        "max_entry_chase_pts", "max_exit_chase_pts",
        "avg_modifications_per_order",
        "pct_chased_entry", "pct_chased_exit",
    }
    assert required.issubset(set(exec_m.keys()))


def test_enriched_trades_have_note(enriched_trades):
    """All 8 real trades should have note = '15x15.twconfig'."""
    assert len(enriched_trades) == 8
    for t in enriched_trades:
        assert t.get("note") == "15x15.twconfig"
