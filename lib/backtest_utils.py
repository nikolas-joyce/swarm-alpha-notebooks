"""
Common backtest utilities shared across strategy notebooks.
"""

import pandas as pd
import numpy as np
from typing import Optional


def split_is_oos(
    df: pd.DataFrame, split_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into in-sample and out-of-sample periods."""
    split_idx = int(len(df) * split_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def compute_metrics(pf, label: str = "") -> dict:
    """
    Extract standard metrics from a vectorbt Portfolio object.

    Returns dict compatible with qc_gate.apply_qc_gate().
    """
    stats = pf.stats()

    # Handle both Series and scalar returns from vectorbt
    def _get(key, default=0.0):
        try:
            val = stats[key]
            return float(val) if not pd.isna(val) else default
        except (KeyError, TypeError):
            return default

    total_return = _get("Total Return [%]", 0.0) / 100
    max_dd = abs(_get("Max Drawdown [%]", 100.0) / 100)
    sharpe = _get("Sharpe Ratio", 0.0)
    total_trades = int(_get("Total Trades", 0))

    # Calmar = annualized return / max drawdown
    ann_return = _get("Annualized Return [%]", 0.0) / 100
    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    return {
        "label": label,
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "calmar_ratio": round(calmar, 4),
        "total_trades": total_trades,
        "win_rate": round(_get("Win Rate [%]", 0.0) / 100, 4),
        "profit_factor": round(_get("Profit Factor", 0.0), 4),
    }


def merge_is_oos_metrics(is_metrics: dict, oos_metrics: dict) -> dict:
    """
    Combine IS and OOS metrics into the format expected by qc_gate.

    Returns dict with sharpe_is, sharpe_oos, max_drawdown (worst of both),
    calmar_ratio (OOS), total_trades (sum), correlation_to_book (default 0).
    """
    return {
        "sharpe_is": is_metrics["sharpe_ratio"],
        "sharpe_oos": oos_metrics["sharpe_ratio"],
        "max_drawdown": max(is_metrics["max_drawdown"], oos_metrics["max_drawdown"]),
        "calmar_ratio": oos_metrics["calmar_ratio"],
        "total_trades": is_metrics["total_trades"] + oos_metrics["total_trades"],
        "correlation_to_book": 0.0,  # No book to compare against yet

        # Extra context (not used by QC gate but useful in vault notes)
        "is_total_return": is_metrics["total_return"],
        "oos_total_return": oos_metrics["total_return"],
        "is_win_rate": is_metrics["win_rate"],
        "oos_win_rate": oos_metrics["win_rate"],
        "is_profit_factor": is_metrics["profit_factor"],
        "oos_profit_factor": oos_metrics["profit_factor"],
    }


def print_comparison(is_metrics: dict, oos_metrics: dict):
    """Pretty-print IS vs OOS comparison table."""
    header = f"{'Metric':<25} {'In-Sample':>12} {'Out-of-Sample':>14} {'Delta':>10}"
    print("=" * 65)
    print(header)
    print("-" * 65)

    pairs = [
        ("Sharpe Ratio", "sharpe_ratio", ".3f"),
        ("Total Return", "total_return", ".2%"),
        ("Annualized Return", "annualized_return", ".2%"),
        ("Max Drawdown", "max_drawdown", ".2%"),
        ("Calmar Ratio", "calmar_ratio", ".3f"),
        ("Win Rate", "win_rate", ".2%"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Total Trades", "total_trades", "d"),
    ]

    for label, key, fmt in pairs:
        is_val = is_metrics.get(key, 0)
        oos_val = oos_metrics.get(key, 0)

        if fmt == "d":
            delta_str = f"{oos_val - is_val:+d}"
            print(f"{label:<25} {is_val:>12d} {oos_val:>14d} {delta_str:>10}")
        elif "%" in fmt:
            delta = oos_val - is_val
            print(
                f"{label:<25} {is_val:>12{fmt}} {oos_val:>14{fmt}} {delta:>+10{fmt}}"
            )
        else:
            delta = oos_val - is_val
            print(
                f"{label:<25} {is_val:>12{fmt}} {oos_val:>14{fmt}} {delta:>+10{fmt}}"
            )

    print("=" * 65)
