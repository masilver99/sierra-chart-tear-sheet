"""Flat-to-flat trade reconstruction using a FIFO state machine."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Point-value lookup
# ---------------------------------------------------------------------------

POINT_VALUES: dict[str, float] = {
    "MES": 5.0,
    "ES": 50.0,
    "MNQ": 2.0,
    "NQ": 20.0,
    "MYM": 0.5,
    "YM": 5.0,
    "M2K": 5.0,
    "RTY": 50.0,
    "MCL": 100.0,
    "CL": 1000.0,
    "MGC": 10.0,
    "GC": 100.0,
    "ZB": 1000.0,
    "ZN": 1000.0,
    "ZF": 1000.0,
    "ZT": 2000.0,
}


def get_point_value(symbol: str) -> float:
    """Return the dollar-per-point value for *symbol* (e.g. ``MESM26_FUT_CME``)."""
    root = symbol.split("_")[0]  # e.g. MESM26
    m = re.match(r"^([A-Za-z]+)", root)
    if not m:
        return 1.0
    alpha = m.group(1)  # e.g. MES
    for length in range(len(alpha), 0, -1):
        candidate = alpha[:length]
        if candidate in POINT_VALUES:
            return POINT_VALUES[candidate]
    return 1.0


# ---------------------------------------------------------------------------
# Trade dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    trade_id: int
    symbol: str
    direction: str          # 'long' | 'short'
    entry_time: Any         # pandas Timestamp
    exit_time: Any = None
    point_value: float = 1.0

    # FIFO lots: (price, qty)
    entry_lots: deque = field(default_factory=deque)
    # exit fills: (price, qty, high_during, low_during, order_type)
    exit_fills: list = field(default_factory=list)

    gross_pnl: float = 0.0
    fees: float = 0.0
    total_qty: int = 0      # total contracts traded (entry side)
    avg_entry: float = 0.0
    avg_exit: float = 0.0
    mfe: float = 0.0        # max favourable excursion in $
    mae: float = 0.0        # max adverse excursion in $ (negative)
    exit_type: str = "manual"  # 'target' | 'stop' | 'manual'

    # Phase 2: execution tracking
    entry_exec_ids: list = field(default_factory=list)   # one per entry fill event
    entry_fill_qtys: list = field(default_factory=list)  # parallel qty per entry fill
    exit_exec_ids: list = field(default_factory=list)    # one per exit fill event
    note: str = ""           # config tag from first non-empty entry fill

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "point_value": self.point_value,
            "total_qty": self.total_qty,
            "avg_entry": round(self.avg_entry, 4),
            "avg_exit": round(self.avg_exit, 4),
            "gross_pnl": round(self.gross_pnl, 4),
            "fees": round(self.fees, 4),
            "net_pnl": round(self.net_pnl, 4),
            "mfe": round(self.mfe, 4),
            "mae": round(self.mae, 4),
            "exit_type": self.exit_type,
            # Phase 2 additions
            "entry_exec_ids": list(self.entry_exec_ids),
            "entry_fill_qtys": list(self.entry_fill_qtys),
            "exit_exec_ids": list(self.exit_exec_ids),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exit_type(order_type: str) -> str:
    ot = str(order_type).lower()
    if "stop" in ot:
        return "stop"
    if "limit" in ot:
        return "target"
    return "manual"


def _compute_pnl(entry_lots: deque, exit_fills, direction: str, pv: float) -> tuple[float, float, float]:
    """Return ``(gross_pnl, avg_entry, avg_exit)`` via FIFO matching."""
    lots = deque((p, q) for p, q in entry_lots)
    gross = 0.0
    total_exit_qty = 0
    total_exit_value = 0.0
    total_entry_qty = 0
    total_entry_value = 0.0

    for ex_price, ex_qty, *_ in exit_fills:
        remaining = ex_qty
        total_exit_qty += ex_qty
        total_exit_value += ex_price * ex_qty
        while remaining > 0 and lots:
            en_price, en_qty = lots[0]
            matched = min(en_qty, remaining)
            if direction == "long":
                gross += (ex_price - en_price) * matched * pv
            else:
                gross += (en_price - ex_price) * matched * pv
            remaining -= matched
            if en_qty - matched == 0:
                total_entry_qty += en_qty
                total_entry_value += en_price * en_qty
                lots.popleft()
            else:
                lots[0] = (en_price, en_qty - matched)
                total_entry_qty += matched
                total_entry_value += en_price * matched

    avg_entry = total_entry_value / total_entry_qty if total_entry_qty else 0.0
    avg_exit = total_exit_value / total_exit_qty if total_exit_qty else 0.0
    return gross, avg_entry, avg_exit


def _compute_mfe_mae(exit_fills, direction: str, avg_entry: float, pv: float) -> tuple[float, float]:
    """Return ``(mfe, mae)`` in dollars."""
    if not exit_fills:
        return 0.0, 0.0

    highs = [f[2] for f in exit_fills if pd.notna(f[2])]
    lows = [f[3] for f in exit_fills if pd.notna(f[3])]

    if not highs or not lows:
        return 0.0, 0.0

    high = max(highs)
    low = min(lows)

    if direction == "long":
        mfe = (high - avg_entry) * pv
        mae = (low - avg_entry) * pv
    else:
        mfe = (avg_entry - low) * pv
        mae = (avg_entry - high) * pv

    return mfe, mae


# ---------------------------------------------------------------------------
# Reconstructor
# ---------------------------------------------------------------------------

class FlatToFlatReconstructor:
    """Reconstruct flat-to-flat trades from a sorted fills DataFrame."""

    def __init__(self) -> None:
        self._trades: list[Trade] = []
        self._current: dict[str, Optional[Trade]] = {}
        self._prev_position: dict[str, int] = {}
        self._trade_counter: int = 0

    # ------------------------------------------------------------------
    def reconstruct(self, fills_df: pd.DataFrame) -> list[dict[str, Any]]:
        """Process all fill rows and return a list of completed trade dicts."""
        fills_df = fills_df.sort_values("DateTime").reset_index(drop=True)

        for _, row in fills_df.iterrows():
            self._process_fill(row)

        # Force-close any lingering open trades (e.g. file truncated)
        for sym, trade in self._current.items():
            if trade is not None:
                self._close_trade(sym, trade)

        return [t.to_dict() for t in self._trades]

    # ------------------------------------------------------------------
    def _process_fill(self, row: pd.Series) -> None:
        symbol = str(row.get("Symbol", "")).strip()
        prev = self._prev_position.get(symbol, 0)
        bs = str(row.get("BuySell", "")).strip().lower()
        fill_price = float(row["FillPrice"])
        # Use Quantity (incremental per-fill), NOT FilledQuantity (cumulative for the order)
        fill_qty = int(row["Quantity"])

        pos_qty_raw = row["PositionQuantity"]
        if pd.notna(pos_qty_raw):
            new_position = int(pos_qty_raw)
        else:
            # PositionQuantity absent (e.g. Teton CME Routing phantom fills): infer from delta
            delta = fill_qty if bs == "buy" else -fill_qty
            new_position = prev + delta

        dt = row["DateTime"]
        order_type = str(row.get("OrderType", "")).strip()
        high = row.get("HighDuringPosition", float("nan"))
        low = row.get("LowDuringPosition", float("nan"))

        # Phase 2: capture exec ID and note from the fill row
        exec_id = str(row.get("FillExecutionServiceID", "") or "").strip()
        note = str(row.get("Note", "") or "").strip()

        current = self._current.get(symbol)

        # Determine transition type
        if prev == 0 and new_position != 0:
            # Opening a new position
            direction = "long" if new_position > 0 else "short"
            self._open_trade(symbol, direction, dt, fill_price, fill_qty, exec_id, note)

        elif prev != 0 and new_position == 0:
            # Closing to flat
            if current is None:
                self._open_trade(symbol, "long" if bs == "buy" else "short", dt, fill_price, fill_qty, exec_id, note)
                current = self._current.get(symbol)
            current.exit_fills.append((fill_price, fill_qty, high, low, order_type))
            current.exit_exec_ids.append(exec_id)
            self._close_trade(symbol, current, exit_time=dt)

        elif prev != 0 and new_position != 0:
            if abs(new_position) > abs(prev) and (new_position > 0) == (prev > 0):
                # Scale-in — same direction, position growing
                if current is not None:
                    current.entry_lots.append((fill_price, fill_qty))
                    current.total_qty += fill_qty
                    current.entry_exec_ids.append(exec_id)
                    current.entry_fill_qtys.append(fill_qty)
                    if note and not current.note:
                        current.note = note

            elif abs(new_position) < abs(prev) and (new_position > 0) == (prev > 0):
                # Partial close — same direction, position shrinking
                if current is not None:
                    current.exit_fills.append((fill_price, fill_qty, high, low, order_type))
                    current.exit_exec_ids.append(exec_id)

            else:
                # Reversal — close existing then open opposite
                if current is not None:
                    close_qty = abs(prev)
                    current.exit_fills.append((fill_price, close_qty, high, low, order_type))
                    current.exit_exec_ids.append(exec_id)
                    self._close_trade(symbol, current, exit_time=dt)

                new_open_qty = abs(new_position)
                direction = "long" if new_position > 0 else "short"
                self._open_trade(symbol, direction, dt, fill_price, new_open_qty, exec_id, note)

        self._prev_position[symbol] = new_position

    # ------------------------------------------------------------------
    def _open_trade(self, symbol: str, direction: str, dt, price: float, qty: int,
                    exec_id: str = "", note: str = "") -> None:
        self._trade_counter += 1
        pv = get_point_value(symbol)
        t = Trade(
            trade_id=self._trade_counter,
            symbol=symbol,
            direction=direction,
            entry_time=dt,
            point_value=pv,
            note=note,
        )
        t.entry_lots.append((price, qty))
        t.total_qty = qty
        t.entry_exec_ids.append(exec_id)
        t.entry_fill_qtys.append(qty)
        self._current[symbol] = t

    # ------------------------------------------------------------------
    def _close_trade(self, symbol: str, trade: Trade, exit_time=None) -> None:
        if exit_time is not None:
            trade.exit_time = exit_time

        gross, avg_entry, avg_exit = _compute_pnl(
            trade.entry_lots, trade.exit_fills, trade.direction, trade.point_value
        )
        trade.gross_pnl = gross
        trade.avg_entry = avg_entry
        trade.avg_exit = avg_exit

        mfe, mae = _compute_mfe_mae(trade.exit_fills, trade.direction, avg_entry, trade.point_value)
        trade.mfe = mfe
        trade.mae = mae

        # Exit type from last exit fill's OrderType
        if trade.exit_fills:
            trade.exit_type = _exit_type(trade.exit_fills[-1][4])

        self._trades.append(trade)
        self._current[symbol] = None

    # ------------------------------------------------------------------
    def assign_fees(self, cash_events: list[dict]) -> None:
        """Assign fee events to trades by timestamp overlap.

        A small epsilon (500 ms) is added to the exit_time upper bound to
        capture AccountBalance rows that are stamped a few microseconds after
        the closing fill.
        """
        import pandas as pd
        _eps = pd.Timedelta(milliseconds=500)
        for ev in cash_events:
            if ev["kind"] != "fee":
                continue
            fee_time = ev["DateTime"]
            fee_amt = ev["amount"]
            for t in self._trades:
                if t.entry_time <= fee_time <= t.exit_time + _eps:
                    t.fees += fee_amt
                    break


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------

def reconstruct_trades(fills_df: pd.DataFrame, cash_events: list[dict] | None = None) -> list[dict[str, Any]]:
    """Reconstruct flat-to-flat trades and optionally attach fees.

    Parameters
    ----------
    fills_df:
        DataFrame of fill rows (from :func:`tearsheet.normalize.fills.extract_fills`).
    cash_events:
        Optional list from :func:`tearsheet.normalize.cash_ledger.extract_cash_events`.

    Returns
    -------
    list[dict]
        One dict per completed trade (see :meth:`Trade.to_dict`).
    """
    rec = FlatToFlatReconstructor()
    trades = rec.reconstruct(fills_df)

    if cash_events:
        rec.assign_fees(cash_events)
        # Refresh dicts after fee assignment
        trades = [t.to_dict() for t in rec._trades]

    return trades


def compute_r_multiples(trades: list[dict], orders: list[dict]) -> list[dict]:
    """Attach stop-price, initial risk, and R-multiple to each trade dict.

    For each trade, finds the stop-loss child order linked to the entry order
    via the parent→child order hierarchy, then computes:

    * ``stop_price``    — price of the stop-loss child order
    * ``initial_risk``  — ``abs(avg_entry - stop_price) * total_qty * point_value``
    * ``r_multiple``    — ``gross_pnl / initial_risk`` (None when risk == 0 or no stop found)

    Parameters
    ----------
    trades:
        List of trade dicts (already enriched with chase distances).
    orders:
        Raw order list from :func:`tearsheet.normalize.orders.normalize_orders`.

    Returns
    -------
    list[dict]
        Copies of trade dicts with the three R-multiple fields added.
    """
    from tearsheet.normalize.orders import build_order_index

    order_index = build_order_index(orders)

    # parent_id → list of child orders
    parent_children: dict[str, list[dict]] = {}
    for o in orders:
        pid = o.get("parent_id")
        if pid:
            parent_children.setdefault(pid, []).append(o)

    result = []
    for t in trades:
        t = dict(t)  # shallow copy

        stop_price = None
        initial_risk = None
        r_multiple = None

        entry_exec_ids = t.get("entry_exec_ids", [])
        if entry_exec_ids and entry_exec_ids[0]:
            exch_id = entry_exec_ids[0].split("_")[0]
            entry_order = order_index.get(exch_id) or order_index.get(entry_exec_ids[0])
            if entry_order is not None:
                entry_order_id = entry_order["order_id"]
                for child in parent_children.get(entry_order_id, []):
                    if ("stop" in child["order_type"].lower()
                            and child["open_close"] == "Close"):
                        stop_price = child["price"]
                        break

        if stop_price is not None:
            initial_risk = abs(t["avg_entry"] - stop_price) * t["total_qty"] * t["point_value"]
            if initial_risk > 0:
                r_multiple = t["gross_pnl"] / initial_risk

        t["stop_price"] = stop_price
        t["initial_risk"] = round(initial_risk, 4) if initial_risk is not None else None
        t["r_multiple"] = round(r_multiple, 4) if r_multiple is not None else None
        result.append(t)

    return result


def enrich_trades(trades: list[dict], orders: list[dict]) -> list[dict]:
    """Attach execution metadata (chase distances) to each trade dict.

    Uses the order index from :func:`tearsheet.normalize.orders.build_order_index`
    to resolve entry/exit orders via ExchangeOrderID (first ``_`` segment of each
    exec ID).  Chase distances use the convention *positive = adverse slippage*.

    Parameters
    ----------
    trades:
        Output from :func:`reconstruct_trades`.
    orders:
        Output from :func:`tearsheet.normalize.orders.normalize_orders` — either
        a raw list of order dicts or a pre-built index dict.

    Returns
    -------
    list[dict]
        Augmented trade dicts with ``entry_order_prices``, ``exit_order_price``,
        ``entry_chase_pts``, ``exit_chase_pts``, ``stop_price``, ``initial_risk``,
        and ``r_multiple`` added.
    """
    from tearsheet.normalize.orders import build_order_index  # local import to avoid circular

    orders_list: list[dict] = orders if isinstance(orders, list) else []
    order_index = build_order_index(orders_list) if orders_list else (orders if not isinstance(orders, list) else {})

    enriched = []
    for t in trades:
        t = dict(t)  # shallow copy — don't mutate caller's list

        direction = t.get("direction", "long")
        avg_entry = t.get("avg_entry", 0.0)
        avg_exit = t.get("avg_exit", 0.0)

        # --- entry order lookup (one or more exec IDs for scale-ins) ---
        entry_exec_ids = t.get("entry_exec_ids", [])
        entry_fill_qtys = t.get("entry_fill_qtys", [])
        entry_order_prices: list[tuple[float, int]] = []  # (order_price, qty)

        for exec_id, qty in zip(entry_exec_ids, entry_fill_qtys):
            order = None
            if exec_id:
                exch_id = exec_id.split("_")[0]
                order = order_index.get(exch_id) or order_index.get(exec_id)
            if order is not None:
                entry_order_prices.append((float(order.get("price", 0.0) or 0.0), qty))

        # Weighted average order price across all entry lots
        if entry_order_prices:
            total_qty = sum(q for _, q in entry_order_prices)
            if total_qty > 0:
                wavg_entry_order = sum(p * q for p, q in entry_order_prices) / total_qty
                if direction == "long":
                    entry_chase_pts = round(avg_entry - wavg_entry_order, 4)
                else:
                    entry_chase_pts = round(wavg_entry_order - avg_entry, 4)
                t["entry_order_prices"] = [p for p, _ in entry_order_prices]
                t["entry_chase_pts"] = entry_chase_pts
            else:
                t["entry_order_prices"] = []
                t["entry_chase_pts"] = None
        else:
            t["entry_order_prices"] = []
            t["entry_chase_pts"] = None

        # --- exit order lookup (use last exit exec ID) ---
        exit_exec_ids = t.get("exit_exec_ids", [])
        exit_order = None
        if exit_exec_ids:
            last_exec = exit_exec_ids[-1]
            if last_exec:
                exch_id = last_exec.split("_")[0]
                exit_order = order_index.get(exch_id) or order_index.get(last_exec)

        if exit_order is not None:
            exit_order_price = float(exit_order.get("price", 0.0) or 0.0)
            if direction == "long":
                exit_chase_pts = round(exit_order_price - avg_exit, 4)
            else:
                exit_chase_pts = round(avg_exit - exit_order_price, 4)
            t["exit_order_price"] = exit_order_price
            t["exit_chase_pts"] = exit_chase_pts
        else:
            t["exit_order_price"] = None
            t["exit_chase_pts"] = None

        # Duration in seconds for template convenience
        import pandas as pd
        try:
            if t.get("entry_time") is not None and t.get("exit_time") is not None:
                t["duration_s"] = (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds()
            else:
                t["duration_s"] = None
        except Exception:
            t["duration_s"] = None

        enriched.append(t)

    # Phase 3: attach stop_price, initial_risk, r_multiple via compute_r_multiples
    if orders_list:
        enriched = compute_r_multiples(enriched, orders_list)
    else:
        for t in enriched:
            t.setdefault("stop_price", None)
            t.setdefault("initial_risk", None)
            t.setdefault("r_multiple", None)

    return enriched
