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


def load_file(path: str | Path) -> pd.DataFrame:
    """Parse *path* and return a tidy DataFrame.

    All columns arrive as ``str`` first; then numeric / datetime coercion is
    applied with ``errors='coerce'`` so bad/empty cells become NaN / NaT.
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
