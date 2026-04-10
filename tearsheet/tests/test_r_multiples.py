"""Tests for R-multiple enrichment."""

from __future__ import annotations

import pytest


def test_r_multiple_field_present(three_trades):
    # three_trades uses reconstruct_trades without enrich, so r_multiple won't be present.
    # But enrich_trades adds it. Use enrich_trades with empty orders.
    from tearsheet.recon.trades import enrich_trades
    enriched = enrich_trades(three_trades, [])
    for t in enriched:
        assert "r_multiple" in t


def test_r_multiple_sign_from_real_file():
    from pathlib import Path
    p = Path("TradeActivityLog_2026-04-09.txt")
    if not p.exists():
        pytest.skip("sample file not available")
    from tearsheet.dataio.loader import load_file
    from tearsheet.normalize.events import split_events
    from tearsheet.normalize.orders import normalize_orders
    from tearsheet.recon.trades import reconstruct_trades, enrich_trades
    df = load_file(p)
    fills, cash = split_events(df)
    orders = normalize_orders(df)
    trades = reconstruct_trades(fills, cash)
    trades = enrich_trades(trades, orders)
    for t in trades:
        if t.get("r_multiple") is not None:
            if t["gross_pnl"] > 0:
                assert t["r_multiple"] > 0, f"Trade {t['trade_id']} winner has negative R"
            elif t["gross_pnl"] < 0:
                assert t["r_multiple"] < 0, f"Trade {t['trade_id']} loser has positive R"


def test_initial_stop_price_reasonable():
    from pathlib import Path
    p = Path("TradeActivityLog_2026-04-09.txt")
    if not p.exists():
        pytest.skip("sample file not available")
    from tearsheet.dataio.loader import load_file
    from tearsheet.normalize.events import split_events
    from tearsheet.normalize.orders import normalize_orders
    from tearsheet.recon.trades import reconstruct_trades, enrich_trades
    df = load_file(p)
    fills, cash = split_events(df)
    orders = normalize_orders(df)
    trades = reconstruct_trades(fills, cash)
    trades = enrich_trades(trades, orders)
    for t in trades:
        if t.get("initial_stop_price") is not None:
            if t["direction"] == "long":
                assert t["initial_stop_price"] < t["avg_entry"], \
                    f"T{t['trade_id']} stop above entry"
            else:
                assert t["initial_stop_price"] > t["avg_entry"], \
                    f"T{t['trade_id']} stop below entry"
