"""Load a Sierra Chart TradeActivityLog_*.txt file into a raw DataFrame."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_NUMERIC_COLS = [
    "Quantity",
    "Price",
    "Price2",
    "FillPrice",
    "FilledQuantity",
    "PositionQuantity",
    "AccountBalance",
    "HighDuringPosition",
    "LowDuringPosition",
]

_DATETIME_COLS = ["DateTime", "TransDateTime"]

# Columns that must be present in a valid Sierra Chart TradeActivityLog file.
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "ActivityType",
    "DateTime",
    "Symbol",
    "BuySell",
    "Quantity",
    "FillPrice",
    "FilledQuantity",
})


def validate_file(df: pd.DataFrame) -> None:
    """Validate that *df* looks like a Sierra Chart TradeActivityLog export.

    Raises:
        ValueError: with a human-readable message describing the problem.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "The file does not appear to be a Sierra Chart TradeActivityLog export. "
            f"Missing required column(s): {', '.join(sorted(missing))}.\n"
            "Expected a tab-separated TradeActivityLog_*.txt file exported from "
            "Sierra Chart (Trade → Activity Log)."
        )

    if "Fills" not in df["ActivityType"].values:
        raise ValueError(
            "The file contains no 'Fills' rows. "
            "Please export the Trade Activity Log from Sierra Chart with at least "
            "one completed trade (Trade → Activity Log)."
        )


def load_file(path: str | Path) -> pd.DataFrame:
    """Parse *path* and return a tidy DataFrame.

    All columns arrive as ``str`` first; then numeric / datetime coercion is
    applied with ``errors='coerce'`` so bad/empty cells become NaN / NaT.

    Raises:
        ValueError: if the file is not a valid Sierra Chart TradeActivityLog.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        header=0,
        keep_default_na=False,  # keep empty strings as "" not NaN
    )

    # Normalise column names (strip surrounding whitespace)
    df.columns = [c.strip() for c in df.columns]

    validate_file(df)

    # DateTime: Sierra Chart writes two spaces between date and time
    for col in _DATETIME_COLS:
        if col in df.columns:
            raw = df[col].str.strip()
            # Normalise the two-space separator to a single space
            raw = raw.str.replace(r"\s{2,}", " ", regex=True)
            df[col] = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
