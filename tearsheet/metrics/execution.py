"""Execution quality metrics — slippage, order behaviour, fill statistics."""

from __future__ import annotations

from typing import Any, Optional


def compute_execution_metrics(
    trades: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return execution-quality metrics derived from enriched trades and orders.

    Parameters
    ----------
    trades:
        Enriched trade dicts (output of
        :func:`tearsheet.recon.trades.enrich_trades`).
    orders:
        Normalised order dicts from
        :func:`tearsheet.normalize.orders.normalize_orders`.
    """
    # ------------------------------------------------------------------
    # Chase / slippage from enriched trades
    # ------------------------------------------------------------------
    entry_chases = [
        t["entry_chase_pts"]
        for t in trades
        if t.get("entry_chase_pts") is not None
    ]
    exit_chases = [
        t["exit_chase_pts"]
        for t in trades
        if t.get("exit_chase_pts") is not None
    ]

    def _avg(lst: list) -> Optional[float]:
        return sum(lst) / len(lst) if lst else None

    def _max(lst: list) -> Optional[float]:
        return max(lst) if lst else None

    # ------------------------------------------------------------------
    # Order behaviour from raw orders
    # ------------------------------------------------------------------
    total_orders = len(orders)
    filled_orders = sum(1 for o in orders if o["is_filled"])
    canceled_orders = sum(1 for o in orders if o["is_canceled"])
    cancel_rate = canceled_orders / total_orders if total_orders else 0.0

    modify_counts = [o["modify_count"] for o in orders]
    avg_modify = sum(modify_counts) / len(modify_counts) if modify_counts else 0.0
    orders_with_mods = sum(1 for c in modify_counts if c > 0)

    partial_fill_count = sum(1 for o in orders if o["is_partial"])
    partial_fill_rate = partial_fill_count / total_orders if total_orders else 0.0

    # ------------------------------------------------------------------
    # Time to fill — split by entry/exit (Open vs Close)
    # ------------------------------------------------------------------
    entry_ttf = [
        o["time_to_fill"]
        for o in orders
        if o["is_filled"]
        and o.get("open_close", "").lower() == "open"
        and o["time_to_fill"] is not None
    ]
    exit_ttf = [
        o["time_to_fill"]
        for o in orders
        if o["is_filled"]
        and o.get("open_close", "").lower() == "close"
        and o["time_to_fill"] is not None
    ]

    # ------------------------------------------------------------------
    # Exit type breakdown
    # ------------------------------------------------------------------
    n_total = len(trades)
    n_target = sum(1 for t in trades if t.get("exit_type") == "target")
    n_stop = sum(1 for t in trades if t.get("exit_type") == "stop")
    n_manual = sum(1 for t in trades if t.get("exit_type") == "manual")

    def _pct(n: int) -> float:
        return (n / n_total) if n_total else 0.0

    return {
        # Chase
        "avg_entry_chase_pts": round(_avg(entry_chases), 4) if _avg(entry_chases) is not None else None,
        "avg_exit_chase_pts": round(_avg(exit_chases), 4) if _avg(exit_chases) is not None else None,
        "max_entry_chase_pts": round(_max(entry_chases), 4) if _max(entry_chases) is not None else None,
        "max_exit_chase_pts": round(_max(exit_chases), 4) if _max(exit_chases) is not None else None,
        "pct_chased_entry": round(sum(1 for c in entry_chases if c > 0) / len(entry_chases), 4) if entry_chases else None,
        "pct_chased_exit": round(sum(1 for c in exit_chases if c > 0) / len(exit_chases), 4) if exit_chases else None,
        # Order behaviour
        "total_orders": total_orders,
        "filled_orders": filled_orders,
        "canceled_orders": canceled_orders,
        "fill_rate": round(filled_orders / total_orders, 4) if total_orders else None,
        "cancel_rate": round(cancel_rate, 4),
        "modify_rate": round(orders_with_mods / total_orders, 4) if total_orders else None,
        "avg_modifications_per_order": round(avg_modify, 4),
        "avg_modify_count": round(avg_modify, 4),  # backward-compat alias
        "orders_with_modifications": orders_with_mods,
        "partial_fill_count": partial_fill_count,
        "partial_fill_rate": round(partial_fill_rate, 4),
        # Time to fill
        "avg_entry_time_to_fill_s": round(_avg(entry_ttf), 2) if _avg(entry_ttf) is not None else None,
        "avg_exit_time_to_fill_s": round(_avg(exit_ttf), 2) if _avg(exit_ttf) is not None else None,
        # Exit type breakdown
        "pct_target_exits": round(_pct(n_target), 4),
        "pct_stop_exits": round(_pct(n_stop), 4),
        "pct_manual_exits": round(_pct(n_manual), 4),
        "n_target_exits": n_target,
        "n_stop_exits": n_stop,
        "n_manual_exits": n_manual,
    }
