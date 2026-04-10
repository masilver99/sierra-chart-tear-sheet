"""Normalise raw rows into Fills events."""

from __future__ import annotations

import pandas as pd


def extract_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Return only *Fills* rows, deduplicated by FillExecutionServiceID.

    Rows without a FillExecutionServiceID (empty string) are kept as-is
    because some brokers omit that field.
    """
    fills = df[df["ActivityType"] == "Fills"].copy()

    if fills.empty:
        return fills

    has_id = fills["FillExecutionServiceID"].str.strip().ne("")
    with_id = fills[has_id].drop_duplicates(subset=["FillExecutionServiceID"])
    without_id = fills[~has_id]

    result = pd.concat([with_id, without_id]).sort_values("DateTime").reset_index(drop=True)
    return result
