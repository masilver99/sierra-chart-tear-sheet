"""Tests for tearsheet.dataio.loader against the real sample file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

SAMPLE_FILE = Path(__file__).parents[2] / "TradeActivityLog_2026-04-09.txt"

pytestmark = pytest.mark.skipif(
    not SAMPLE_FILE.exists(),
    reason="Real sample file not present",
)


def test_load_returns_dataframe():
    from tearsheet.dataio.loader import load_file
    df = load_file(SAMPLE_FILE)
    assert isinstance(df, pd.DataFrame)


def test_column_count():
    from tearsheet.dataio.loader import load_file
    df = load_file(SAMPLE_FILE)
    assert len(df.columns) == 29


def test_expected_activity_types():
    from tearsheet.dataio.loader import load_file
    df = load_file(SAMPLE_FILE)
    types = set(df["ActivityType"].dropna().unique())
    assert {"Account Balance", "Fills", "Orders", "Positions"}.issubset(types)


def test_datetime_no_nat():
    from tearsheet.dataio.loader import load_file
    df = load_file(SAMPLE_FILE)
    # Rows that have a DateTime value should parse cleanly
    dt_col = df["DateTime"]
    non_empty = dt_col[df["DateTime"].notna()]
    assert non_empty.dtype == "datetime64[ns]"
    assert non_empty.isna().sum() == 0


def test_fills_have_numeric_fill_price():
    from tearsheet.dataio.loader import load_file
    df = load_file(SAMPLE_FILE)
    fills = df[df["ActivityType"] == "Fills"]
    assert fills["FillPrice"].notna().all()


def test_position_quantity_numeric():
    from tearsheet.dataio.loader import load_file
    df = load_file(SAMPLE_FILE)
    fills = df[df["ActivityType"] == "Fills"]
    # PositionQuantity is NaN for flat-closing fills or numeric otherwise
    non_null = fills["PositionQuantity"].dropna()
    assert pd.to_numeric(non_null, errors="coerce").notna().all()
