"""Tests for trade reconstruction — fixture-based (exact spec values) + real-file (8 trades)."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_FILE = Path(__file__).parents[2] / "TradeActivityLog_2026-04-09.txt"
REAL_FILE_PRESENT = SAMPLE_FILE.exists()

# ---------------------------------------------------------------------------
# Ground-truth for the 8 flat-to-flat trades from TradesList.txt
# Source: Sierra Chart TradesList.txt (authoritative validation)
#
# Format: (direction, avg_entry, avg_exit, total_qty, gross_pnl)
# Gross = FlatToFlat net + commission.  Commission = $1/contract/roundtrip
# so gross = net + (total_qty * 1.00) for each trade.
#   Trade 1: Long  2@6812.75 -> 6818.00          gross=+52.50  net=+50.50
#   Trade 2: Short 2@6806.25 -> 6814.25          gross= -80.00 net= -82.00
#   Trade 3: Long  2@6815.00 -> 6811.50          gross= -35.00 net= -37.00
#   Trade 4: Long  2@6829.50 -> 6839.25          gross= +97.50 net= +95.50
#   Trade 5: Long  2@6853.50+2@6860.50 -> 6864   gross=+140.00 net=+136.00
#   Trade 6: Short 2@6855.50 -> 6860.75          gross= -52.50 net= -54.50
#   Trade 7: Long  2@6862.00 -> 6862.50          gross=  +5.00 net=  +3.00
#   Trade 8: Long  2@6866.50+2@6872.00 -> 6875.25 gross=+120.00 net=+116.00
#
# Total gross = 247.50, total net (TradesList) = 227.50
# ---------------------------------------------------------------------------

EXPECTED_GROSS = [52.50, -80.00, -35.00, 97.50, 140.00, -52.50, 5.00, 120.00]
TOTAL_GROSS_EXPECTED = 247.50
TOTAL_NET_TRADELIST = 227.50   # TradesList uses $1/contract roundtrip commission


def _load_real_trades():
    from tearsheet.dataio.loader import load_file
    from tearsheet.normalize.events import split_events
    from tearsheet.recon.trades import reconstruct_trades

    df = load_file(SAMPLE_FILE)
    fills, cash = split_events(df)
    return reconstruct_trades(fills, cash)


# ===========================================================================
# Fixture-based tests (conftest spec_fills_df / three_trades)
# These run without the real sample file.
# ===========================================================================

def test_three_trades_count(three_trades):
    assert len(three_trades) == 3


def test_t1_long(three_trades):
    t = three_trades[0]
    assert t["direction"] == "long"
    assert abs(t["avg_entry"] - 6812.75) < 0.01
    assert abs(t["avg_exit"] - 6818.00) < 0.01
    assert abs(t["gross_pnl"] - 52.50) < 0.01
    assert t["exit_type"] == "target"


def test_t2_short(three_trades):
    t = three_trades[1]
    assert t["direction"] == "short"
    assert abs(t["avg_entry"] - 6806.25) < 0.01
    assert abs(t["avg_exit"] - 6814.25) < 0.01
    assert abs(t["gross_pnl"] - (-80.00)) < 0.01
    assert t["exit_type"] == "stop"


def test_t3_long(three_trades):
    t = three_trades[2]
    assert t["direction"] == "long"
    assert abs(t["avg_entry"] - 6815.00) < 0.01
    assert abs(t["avg_exit"] - 6811.50) < 0.01
    assert abs(t["gross_pnl"] - (-35.00)) < 0.01
    assert t["exit_type"] == "stop"


def test_spec_total_gross(three_trades):
    total = sum(t["gross_pnl"] for t in three_trades)
    assert abs(total - (-62.50)) < 0.01


def test_t1_mfe_mae(three_trades):
    """MFE/MAE per unit: (High-AvgEntry)*pv and (Low-AvgEntry)*pv."""
    t = three_trades[0]
    # High=6818, Low=6811, avg_entry=6812.75, pv=5
    # mfe = (6818.00 - 6812.75) * 5 = 26.25
    # mae = (6811.00 - 6812.75) * 5 = -8.75
    assert abs(t["mfe"] - 26.25) < 0.01
    assert t["mfe"] > 0
    assert t["mae"] < 0


def test_t1_total_qty(three_trades):
    """Trade 1 has two 1-lot partial fills -> total_qty should be 2."""
    assert three_trades[0]["total_qty"] == 2


def test_trade_dict_keys(three_trades):
    required = {
        "trade_id", "symbol", "direction", "entry_time", "exit_time",
        "total_qty", "avg_entry", "avg_exit", "gross_pnl", "fees",
        "net_pnl", "mfe", "mae", "exit_type", "point_value",
    }
    assert required.issubset(set(three_trades[0].keys()))


# ===========================================================================
# Real-file integration tests -- 8 flat-to-flat trades
# ===========================================================================

@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_exactly_8_trades():
    """The sample file contains exactly 8 flat-to-flat trade groups."""
    trades = _load_real_trades()
    assert len(trades) == 8, (
        f"Expected 8 trades, got {len(trades)}. "
        f"Gross PnLs: {[round(t['gross_pnl'], 2) for t in trades]}"
    )


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_trade_directions():
    """Verify the long/short pattern for all 8 trades."""
    trades = _load_real_trades()
    expected_dir = ["long", "short", "long", "long", "long", "short", "long", "long"]
    actual_dir = [t["direction"] for t in trades]
    assert actual_dir == expected_dir, f"Direction mismatch: {actual_dir}"


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_gross_pnl_per_trade():
    """Verify gross P&L for every trade (+-$0.50 tolerance)."""
    trades = _load_real_trades()
    for i, (t, expected) in enumerate(zip(trades, EXPECTED_GROSS), start=1):
        assert abs(t["gross_pnl"] - expected) < 0.50, (
            f"Trade {i}: expected gross={expected}, got {round(t['gross_pnl'], 2)}"
        )


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_total_gross_pnl():
    trades = _load_real_trades()
    total = sum(t["gross_pnl"] for t in trades)
    assert abs(total - TOTAL_GROSS_EXPECTED) < 0.50, (
        f"Expected total gross={TOTAL_GROSS_EXPECTED}, got {round(total, 2)}"
    )


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_total_net_pnl():
    """Net P&L from AccountBalance fees should be within $5 of TradesList total."""
    trades = _load_real_trades()
    total_net = sum(t["net_pnl"] for t in trades)
    assert abs(total_net - TOTAL_NET_TRADELIST) < 5.0, (
        f"Expected total net~={TOTAL_NET_TRADELIST}, got {round(total_net, 2)}"
    )


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_scale_in_trade5():
    """Trade 5: scaled-in long (2@6853.50 + 2@6860.50 -> all exit @6864.00).
    Expected gross = (6864-6853.50)*2*5 + (6864-6860.50)*2*5 = 105 + 35 = $140.
    """
    trades = _load_real_trades()
    t5 = trades[4]
    assert t5["direction"] == "long"
    assert t5["total_qty"] == 4, f"Expected 4 contracts, got {t5['total_qty']}"
    assert abs(t5["avg_entry"] - 6857.00) < 0.05, f"avg_entry={t5['avg_entry']}"
    assert abs(t5["avg_exit"] - 6864.00) < 0.05,  f"avg_exit={t5['avg_exit']}"
    assert abs(t5["gross_pnl"] - 140.00) < 0.50,  f"gross_pnl={t5['gross_pnl']}"
    # Net within $1.50 of TradesList $136.00 (fee rate difference: $0.52 vs $0.50/contract)
    assert abs(t5["net_pnl"] - 136.00) < 1.50,    f"net_pnl={t5['net_pnl']}"


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_scale_in_trade8():
    """Trade 8: scaled-in long (2@6866.50 + 2@6872.00 -> all exit @6875.25).
    Expected gross = (6875.25-6866.50)*2*5 + (6875.25-6872.00)*2*5 = 87.50 + 32.50 = $120.
    """
    trades = _load_real_trades()
    t8 = trades[7]
    assert t8["direction"] == "long"
    assert t8["total_qty"] == 4, f"Expected 4 contracts, got {t8['total_qty']}"
    assert abs(t8["avg_entry"] - 6869.25) < 0.05, f"avg_entry={t8['avg_entry']}"
    assert abs(t8["avg_exit"] - 6875.25) < 0.05,  f"avg_exit={t8['avg_exit']}"
    assert abs(t8["gross_pnl"] - 120.00) < 0.50,  f"gross_pnl={t8['gross_pnl']}"
    assert abs(t8["net_pnl"] - 116.00) < 1.50,    f"net_pnl={t8['net_pnl']}"


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_trade4_gross():
    """Trade 4: Long 2@6829.50 -> 6839.25, gross = $97.50."""
    trades = _load_real_trades()
    t4 = trades[3]
    assert t4["direction"] == "long"
    assert abs(t4["gross_pnl"] - 97.50) < 0.50


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_trade6_short():
    """Trade 6: Short 2@6855.50 -> 6860.75, gross = -$52.50."""
    trades = _load_real_trades()
    t6 = trades[5]
    assert t6["direction"] == "short"
    assert abs(t6["gross_pnl"] - (-52.50)) < 0.50


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_all_trades_have_exit_time():
    trades = _load_real_trades()
    for t in trades:
        assert t["exit_time"] is not None, f"Trade {t['trade_id']} has no exit_time"


@pytest.mark.skipif(not REAL_FILE_PRESENT, reason="Real sample file not present")
def test_real_file_first_three_trades():
    """Backward-compat guard: first 3 trades still match original spec values."""
    trades = _load_real_trades()
    t1, t2, t3 = trades[0], trades[1], trades[2]

    assert t1["direction"] == "long"
    assert abs(t1["gross_pnl"] - 52.50) < 0.50

    assert t2["direction"] == "short"
    assert abs(t2["gross_pnl"] - (-80.00)) < 0.50

    assert t3["direction"] == "long"
    assert abs(t3["gross_pnl"] - (-35.00)) < 0.50
