"""
QC Gate — mirrors Alpha Factory Station 4 thresholds.
Keep in sync with 12-Alpha-Factory/config/settings.py QC_THRESHOLDS.

Usage:
    from lib.qc_gate import apply_qc_gate, QC_THRESHOLDS
    passed, failures = apply_qc_gate(backtest_results)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Thresholds (mirror of settings.py) ───────────────────────────────
QC_THRESHOLDS = {
    "sharpe_is": 0.8,
    "sharpe_oos": 0.6,
    "oos_degradation": 0.35,
    "max_drawdown": 0.25,
    "calmar_ratio": 0.5,
    "min_trades": 100,
    "correlation_to_book": 0.4,
}

BACKTEST_OOS_SPLIT = 0.7
BACKTEST_INITIAL_CASH = 100_000


def apply_qc_gate(results: dict) -> tuple[bool, list[str]]:
    """
    Apply QC thresholds to backtest results.

    Args:
        results: dict with keys matching QC_THRESHOLDS
            Required: sharpe_is, sharpe_oos, max_drawdown, calmar_ratio,
                      total_trades, correlation_to_book (default 0.0)

    Returns:
        (passed: bool, failures: list[str])
    """
    failures = []

    sharpe_is = results.get("sharpe_is", 0)
    sharpe_oos = results.get("sharpe_oos", 0)

    if sharpe_is < QC_THRESHOLDS["sharpe_is"]:
        failures.append(f"sharpe_is={sharpe_is:.3f} < {QC_THRESHOLDS['sharpe_is']}")

    if sharpe_oos < QC_THRESHOLDS["sharpe_oos"]:
        failures.append(f"sharpe_oos={sharpe_oos:.3f} < {QC_THRESHOLDS['sharpe_oos']}")

    # OOS degradation
    if sharpe_is > 0:
        degradation = (sharpe_is - sharpe_oos) / sharpe_is
        if degradation > QC_THRESHOLDS["oos_degradation"]:
            failures.append(
                f"oos_degradation={degradation:.1%} > {QC_THRESHOLDS['oos_degradation']:.0%}"
            )

    max_dd = abs(results.get("max_drawdown", 1.0))
    if max_dd > QC_THRESHOLDS["max_drawdown"]:
        failures.append(f"max_drawdown={max_dd:.1%} > {QC_THRESHOLDS['max_drawdown']:.0%}")

    calmar = results.get("calmar_ratio", 0)
    if calmar < QC_THRESHOLDS["calmar_ratio"]:
        failures.append(f"calmar_ratio={calmar:.3f} < {QC_THRESHOLDS['calmar_ratio']}")

    trades = results.get("total_trades", 0)
    if trades < QC_THRESHOLDS["min_trades"]:
        failures.append(f"total_trades={trades} < {QC_THRESHOLDS['min_trades']}")

    corr = results.get("correlation_to_book", 0.0)
    if corr > QC_THRESHOLDS["correlation_to_book"]:
        failures.append(
            f"correlation_to_book={corr:.3f} > {QC_THRESHOLDS['correlation_to_book']}"
        )

    return (len(failures) == 0, failures)


def export_qc_result(
    strategy_name: str,
    variant_id: str,
    results: dict,
    passed: bool,
    failures: list[str],
    output_dir: str = "results",
) -> Path:
    """
    Export QC result as JSON compatible with Alpha Factory Station 5.

    The JSON matches the schema expected by obsidian_writer.py:
    {
        "variant_id", "strategy_name", "status", "qc_passed",
        "qc_failures", "backtest_results", "timestamp", "source"
    }
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    suffix = "pass" if passed else "fail"
    filename = f"{variant_id}_{suffix}.json"

    payload = {
        "variant_id": variant_id,
        "strategy_name": strategy_name,
        "status": "qc_passed" if passed else "qc_failed",
        "qc_passed": passed,
        "qc_failures": failures,
        "backtest_results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "colab_notebook",
    }

    filepath = output_path / filename
    filepath.write_text(json.dumps(payload, indent=2, default=str))
    print(f"{'✓ PASS' if passed else '✗ FAIL'} → {filepath}")
    return filepath
