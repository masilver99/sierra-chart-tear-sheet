"""Normalize Orders rows into order lifecycle dicts."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def normalize_orders(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Filter Orders rows and build one dict per unique InternalOrderID.

    Parameters
    ----------
    df:
        Full raw DataFrame from :func:`tearsheet.dataio.loader.load_file`.

    Returns
    -------
    list[dict]
        One dict per unique InternalOrderID (see field descriptions below).
    """
    orders_df = df[df["ActivityType"] == "Orders"].copy()
    if orders_df.empty:
        return []

    # Ensure DateTime is sorted correctly
    if "DateTime" in orders_df.columns:
        orders_df = orders_df.sort_values("DateTime").reset_index(drop=True)

    result: list[dict[str, Any]] = []

    for order_id, group in orders_df.groupby("InternalOrderID", sort=False):
        group = group.sort_values("DateTime").reset_index(drop=True)
        first = group.iloc[0]
        last = group.iloc[-1]

        def _float(val) -> Optional[float]:
            v = pd.to_numeric(str(val).strip(), errors="coerce")
            return float(v) if pd.notna(v) and str(val).strip() != "" else None

        def _int(val) -> Optional[int]:
            v = pd.to_numeric(str(val).strip(), errors="coerce")
            return int(v) if pd.notna(v) and str(val).strip() != "" else None

        def _str(val) -> str:
            return str(val).strip() if pd.notna(val) else ""

        price = _float(first.get("Price", ""))
        price2 = _float(first.get("Price2", ""))
        quantity = _int(first.get("Quantity", ""))
        parent_id = _str(first.get("ParentInternalOrderID", "")) or None

        # Modifications: count "Pending Modify" status rows
        modify_count = int((group["OrderStatus"] == "Pending Modify").sum())

        # Fill information from the final "Filled" status row
        filled_rows = group[group["OrderStatus"] == "Filled"]
        is_filled = len(filled_rows) > 0

        fill_price: Optional[float] = None
        fill_qty: Optional[int] = None
        fill_exec_id: Optional[str] = None
        exchange_order_id: Optional[str] = None
        fill_time = None

        if is_filled:
            fr = filled_rows.iloc[-1]
            fill_price = _float(fr.get("FillPrice", ""))
            fill_qty = _int(fr.get("FilledQuantity", ""))
            fill_exec_id = _str(fr.get("FillExecutionServiceID", "")) or None
            exchange_order_id = _str(fr.get("ExchangeOrderID", "")) or None
            if not exchange_order_id and fill_exec_id:
                exchange_order_id = fill_exec_id.split("_")[0]
            fill_time = fr.get("DateTime")
        else:
            # For non-filled orders that have an ExchangeOrderID set (e.g., Working)
            has_exch = group[group.get("ExchangeOrderID", pd.Series(dtype=str)).ne("")]
            if "ExchangeOrderID" in group.columns:
                non_empty = group[group["ExchangeOrderID"].str.strip().ne("")]
                if len(non_empty) > 0:
                    exchange_order_id = _str(non_empty.iloc[-1]["ExchangeOrderID"]) or None

        # Partial fill: partially filled and not fully filled
        is_partial = (
            len(group[group["OrderStatus"].str.contains("Partial", case=False, na=False)]) > 0
            and not is_filled
        )

        final_status = _str(last.get("OrderStatus", ""))
        is_canceled = final_status == "Canceled"

        submit_time = group["DateTime"].iloc[0]
        last_update_time = group["DateTime"].iloc[-1]

        time_to_fill: Optional[float] = None
        if (
            is_filled
            and fill_time is not None
            and pd.notna(fill_time)
            and pd.notna(submit_time)
        ):
            time_to_fill = (fill_time - submit_time).total_seconds()

        result.append({
            "order_id": str(order_id),
            "parent_id": parent_id,
            "symbol": _str(first.get("Symbol", "")),
            "order_type": _str(first.get("OrderType", "")),
            "buy_sell": _str(first.get("BuySell", "")),
            "price": price,
            "price2": price2,
            "quantity": quantity,
            "open_close": _str(first.get("OpenClose", "")),
            "note": _str(first.get("Note", "")),
            "submit_time": submit_time,
            "last_update_time": last_update_time,
            "status": final_status,
            "fill_price": fill_price,
            "fill_qty": fill_qty,
            "fill_exec_id": fill_exec_id,
            "exchange_order_id": exchange_order_id,
            "modify_count": modify_count,
            "is_filled": is_filled,
            "is_canceled": is_canceled,
            "is_partial": is_partial,
            "time_to_fill": time_to_fill,
        })

    return result


def build_order_index(orders: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a dict mapping both ``fill_exec_id`` and ``exchange_order_id`` to
    order dicts for fast lookup.

    The ``exchange_order_id`` key (first ``_``-delimited segment of
    ``fill_exec_id``) is indexed so that multi-lot fills from the same order
    all resolve to the same order dict even when their full exec IDs differ.
    """
    index: dict[str, Any] = {}
    for o in orders:
        if o.get("fill_exec_id"):
            index[o["fill_exec_id"]] = o
        if o.get("exchange_order_id"):
            # Prefer first-set mapping (exchange_order_id → order is 1-to-1)
            index.setdefault(o["exchange_order_id"], o)
    return index
