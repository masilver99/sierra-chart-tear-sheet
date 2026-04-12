"""Tests for tearsheet.dataio.loader.validate_file."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from tearsheet.dataio.loader import REQUIRED_COLUMNS, validate_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_valid_df(**overrides) -> pd.DataFrame:
    """Return a minimal DataFrame that passes validate_file."""
    data = {col: [""] for col in REQUIRED_COLUMNS}
    data["ActivityType"] = ["Fills"]
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# validate_file: happy path
# ---------------------------------------------------------------------------

def test_valid_df_passes():
    df = _minimal_valid_df()
    validate_file(df)  # must not raise


def test_multiple_activity_types_pass():
    """A realistic file has several ActivityType values; only 'Fills' is required."""
    df = _minimal_valid_df()
    extra_rows = pd.DataFrame(
        [{col: "" for col in df.columns} for _ in range(3)]
    )
    extra_rows["ActivityType"] = ["Orders", "Positions", "Account Balance"]
    combined = pd.concat([df, extra_rows], ignore_index=True)
    validate_file(combined)  # must not raise


# ---------------------------------------------------------------------------
# validate_file: missing required columns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_col", sorted(REQUIRED_COLUMNS))
def test_missing_required_column_raises(missing_col):
    df = _minimal_valid_df()
    df = df.drop(columns=[missing_col])
    with pytest.raises(ValueError, match="Missing required column"):
        validate_file(df)


def test_error_message_names_missing_columns():
    df = _minimal_valid_df()
    df = df.drop(columns=["BuySell", "FillPrice"])
    with pytest.raises(ValueError) as exc_info:
        validate_file(df)
    msg = str(exc_info.value)
    assert "BuySell" in msg
    assert "FillPrice" in msg


def test_empty_dataframe_raises():
    """An empty DataFrame has no columns and should fail."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Missing required column"):
        validate_file(df)


def test_single_column_df_raises():
    """A non-TSV file loaded as TSV will produce a single-column DataFrame."""
    df = pd.DataFrame({"not_a_valid_column": ["some data", "more data"]})
    with pytest.raises(ValueError, match="Missing required column"):
        validate_file(df)


# ---------------------------------------------------------------------------
# validate_file: no Fills rows
# ---------------------------------------------------------------------------

def test_no_fills_rows_raises():
    df = _minimal_valid_df()
    df["ActivityType"] = ["Orders"]  # no Fills
    with pytest.raises(ValueError, match="no 'Fills' rows"):
        validate_file(df)


def test_empty_activity_type_raises():
    df = _minimal_valid_df()
    df["ActivityType"] = [""]
    with pytest.raises(ValueError, match="no 'Fills' rows"):
        validate_file(df)


# ---------------------------------------------------------------------------
# load_file: integration — bad file raises ValueError
# ---------------------------------------------------------------------------

def test_load_file_wrong_format_raises(tmp_path):
    """A comma-separated file loaded by load_file should raise ValueError."""
    from tearsheet.dataio.loader import load_file

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
    with pytest.raises(ValueError):
        load_file(bad_file)


def test_load_file_tsv_no_fills_raises(tmp_path):
    """A valid-looking TSV with no Fills rows should raise ValueError."""
    from tearsheet.dataio.loader import load_file

    cols = sorted(REQUIRED_COLUMNS)
    header = "\t".join(cols) + "\n"
    row_values = {col: "" for col in cols}
    row_values["ActivityType"] = "Orders"
    row = "\t".join(row_values[c] for c in cols) + "\n"

    bad_file = tmp_path / "no_fills.txt"
    bad_file.write_text(header + row)
    with pytest.raises(ValueError, match="no 'Fills' rows"):
        load_file(bad_file)
