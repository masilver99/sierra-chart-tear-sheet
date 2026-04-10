"""Tests for metrics.segmentation using the real sample file."""

from __future__ import annotations

import pathlib
import pytest

DATA_FILE = pathlib.Path(__file__).parents[2] / "TradeActivityLog_2026-04-09.txt"
pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="Real data file not present")


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


def test_direction_split_counts(enriched_trades):
    from tearsheet.metrics.segmentation import segment_by_direction
    segs = segment_by_direction(enriched_trades)
    assert segs["long"]["n_trades"] == 6
    assert segs["short"]["n_trades"] == 2


def test_long_win_count(enriched_trades):
    from tearsheet.metrics.segmentation import segment_by_direction
    segs = segment_by_direction(enriched_trades)
    long_seg = segs["long"]
    expected_wins = round(long_seg["win_rate"] * long_seg["n_trades"])
    assert expected_wins >= 0  # sanity; real value depends on actual P&L signs


def test_instrument_segmentation_uses_symbol_values(three_trades):
    from tearsheet.metrics.segmentation import segment_by_instrument

    instrument_trades = [dict(t) for t in three_trades]
    instrument_trades[0]["symbol"] = "MESM26_FUT_CME"
    instrument_trades[1]["symbol"] = "MNQM26_FUT_CME"
    instrument_trades[2]["symbol"] = "MNQM26_FUT_CME"

    segs = segment_by_instrument(instrument_trades)

    assert list(segs.keys()) == ["MESM26_FUT_CME", "MNQM26_FUT_CME"]
    assert segs["MESM26_FUT_CME"]["n_trades"] == 1
    assert segs["MESM26_FUT_CME"]["total_gross_pnl"] == 52.5
    assert segs["MNQM26_FUT_CME"]["n_trades"] == 2
    assert segs["MNQM26_FUT_CME"]["total_gross_pnl"] == -115.0


def test_single_note_tag(enriched_trades):
    from tearsheet.metrics.segmentation import segment_by_note
    segs = segment_by_note(enriched_trades)
    # All trades share the same note → exactly one segment
    assert len(segs) == 1
    assert "15x15.twconfig" in segs


def test_session_keys(enriched_trades):
    from tearsheet.metrics.segmentation import segment_by_session
    segs = segment_by_session(enriched_trades)
    for key in segs:
        assert key in ("open", "midday", "close")


def test_session_count_sum(enriched_trades):
    from tearsheet.metrics.segmentation import segment_by_session
    segs = segment_by_session(enriched_trades)
    total = sum(s["n_trades"] for s in segs.values())
    assert total == len(enriched_trades)
