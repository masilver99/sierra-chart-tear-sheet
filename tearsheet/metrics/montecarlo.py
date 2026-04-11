"""Monte Carlo simulation via bootstrap resampling of trade P&Ls."""

from __future__ import annotations

from typing import Any


def run_monte_carlo(
    pnls: list[float],
    starting_balance: float,
    n_sims: int = 1000,
    ruin_threshold: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap-resample trade P&Ls to generate an equity-curve distribution.

    Parameters
    ----------
    pnls:
        Gross P&L per trade, in chronological order.
    starting_balance:
        Equity at the start of the period.
    n_sims:
        Number of bootstrap simulations.
    ruin_threshold:
        Max drawdown fraction that defines "ruin" (default 0.25 = 25 %).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ``percentile_curves`` – {"p5","p25","p50","p75","p95"} → list of
        balance values at each trade step (length n + 1).
        ``actual_curve`` – actual historical equity indexed by trade number.
        ``stats`` – summary statistics dict.
    """
    if len(pnls) < 5 or starting_balance <= 0:
        return {"percentile_curves": {}, "actual_curve": [], "stats": {}}

    n = len(pnls)

    # Actual historical equity curve (trade-indexed)
    actual_curve: list[float] = [starting_balance]
    bal = starting_balance
    for p in pnls:
        bal += p
        actual_curve.append(bal)

    try:
        import numpy as np

        rng = np.random.default_rng(seed)
        pnls_arr = np.array(pnls, dtype=float)
        idx = rng.integers(0, n, size=(n_sims, n))
        sampled = pnls_arr[idx]

        # Equity matrix: (n_sims, n+1)
        curves = np.empty((n_sims, n + 1), dtype=float)
        curves[:, 0] = starting_balance
        curves[:, 1:] = starting_balance + np.cumsum(sampled, axis=1)

        # Drawdown matrix
        running_peak = np.maximum.accumulate(curves, axis=1)
        safe_peak = np.where(running_peak > 0, running_peak, 1.0)
        max_dds = ((running_peak - curves) / safe_peak).max(axis=1)
        final_bals = curves[:, -1]

        pct_matrix = np.percentile(curves, [5, 25, 50, 75, 95], axis=0)
        percentile_curves = {
            "p5":  pct_matrix[0].tolist(),
            "p25": pct_matrix[1].tolist(),
            "p50": pct_matrix[2].tolist(),
            "p75": pct_matrix[3].tolist(),
            "p95": pct_matrix[4].tolist(),
        }

        fb_sorted = float_sorted = sorted(final_bals.tolist())
        dd_sorted = sorted(max_dds.tolist())
        ruin_count = int((max_dds >= ruin_threshold).sum())
        profit_count = int((final_bals > starting_balance).sum())

    except ImportError:
        import random

        rng_py = random.Random(seed)
        all_curves: list[list[float]] = []
        fb_sorted: list[float] = []
        dd_sorted_raw: list[float] = []
        ruin_count = 0
        profit_count = 0

        for _ in range(n_sims):
            sampled_py = rng_py.choices(pnls, k=n)
            balance = starting_balance
            peak = balance
            max_dd = 0.0
            curve = [balance]
            for p in sampled_py:
                balance += p
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                curve.append(balance)
            all_curves.append(curve)
            fb_sorted.append(balance)
            dd_sorted_raw.append(max_dd)
            if max_dd >= ruin_threshold:
                ruin_count += 1
            if balance > starting_balance:
                profit_count += 1

        fb_sorted = sorted(fb_sorted)
        dd_sorted = sorted(dd_sorted_raw)

        percentile_curves = {k: [] for k in ("p5", "p25", "p50", "p75", "p95")}
        for i in range(n + 1):
            col = sorted(c[i] for c in all_curves)
            m = len(col) - 1
            for name, q in (("p5", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95)):
                percentile_curves[name].append(col[int(q * m)])

    def _q(lst: list, q: float) -> float:
        return lst[int(q * (len(lst) - 1))]

    stats: dict[str, Any] = {
        "n_sims": n_sims,
        "n_trades": n,
        "starting_balance": round(starting_balance, 2),
        "median_final": round(_q(fb_sorted, 0.50), 2),
        "p5_final": round(_q(fb_sorted, 0.05), 2),
        "p95_final": round(_q(fb_sorted, 0.95), 2),
        "median_max_dd_pct": round(_q(dd_sorted, 0.50) * 100, 2),
        "p95_max_dd_pct": round(_q(dd_sorted, 0.95) * 100, 2),
        "ruin_probability": round(ruin_count / n_sims * 100, 2),
        "ruin_threshold_pct": round(ruin_threshold * 100, 1),
        "prob_profit": round(profit_count / n_sims * 100, 2),
    }

    return {
        "percentile_curves": percentile_curves,
        "actual_curve": actual_curve,
        "stats": stats,
    }
