"""Parameter robustness checker — overfitting / curve-fitting detection tool.

Usage (research time, NOT for CI — requires data files)::

    # Basic: check if strategy holds up under ±15% param perturbation
    python scripts/check_param_robustness.py

    # Custom perturbation range
    python scripts/check_param_robustness.py --pct 20

    # Custom baseline threshold
    python scripts/check_param_robustness.py --min-pf-ratio 0.80

Purpose
-------
A strategy that only works at the exact calibrated parameter values is
curve-fitted.  Any real edge should survive modest perturbation of its
inputs.

The standard quantitative rule used here:
  **All perturbed variants must have PF ≥ MIN_PF_RATIO × baseline PF**

If a ±15% change in a single parameter collapses PF by > 15%, the
strategy is almost certainly over-fit to historical data and will not
transfer to live trading.

Parameters tested
-----------------
``d_sl``         SL distance (directional)
``s_sl``         SL distance (straddle)
``tp``           TP distance
``s_vr``         VR squeeze threshold (entry sensitivity)
``s_dur``        Squeeze duration minimum (entry sensitivity)
``lp``           Long probability threshold
``sp``           Short probability threshold

Only parameters present in the cfg dict are tested — missing ones are
skipped with a warning.

Output
------
Prints a table of PF values for each (param, perturbation) combination,
with PASS / FAIL markers and a final verdict.

Exit codes
----------
0 — all variants pass
1 — one or more variants fail (strategy is fragile)
2 — data not found or configuration error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow running as script from project root
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from quantbox.backtest.trade_sim import collect_trades, portfolio_sim, compute_metrics
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig
except ImportError as exc:
    print(f"ERROR: Cannot import quantbox: {exc}", file=sys.stderr)
    print("Run from project root: python scripts/check_param_robustness.py", file=sys.stderr)
    sys.exit(2)

CACHE_DIR = ROOT / "data" / "cache_extended"
DATA_YEARS = 5.67  # approximate — based on 2190-day cache files

# Default robustness thresholds
DEFAULT_PERTURBATION_PCT = 15.0  # ±15%
DEFAULT_MIN_PF_RATIO = 0.85      # each variant must have PF ≥ 85% of baseline


def _load_data(cache_dir: Path, whitelist: set[str]) -> tuple[dict, dict, dict]:
    """Load 15m/1h/4h data from cache_extended for whitelisted symbols."""
    data_15m: dict[str, dict] = {}
    data_1h: dict[str, dict] = {}
    data_4h: dict[str, dict] = {}

    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        try:
            npz = np.load(f, allow_pickle=False)
            data_15m[sym] = {k: npz[k] for k in npz.files}
        except Exception as exc:
            print(f"  WARNING: skip {f.name}: {exc}", file=sys.stderr)

    for suffix, store in [("1h", data_1h), ("4h", data_4h)]:
        for f in sorted(cache_dir.glob(f"*_{suffix}_1095d_*.npz")):
            sym = f.name.split("_")[0]
            if sym not in whitelist:
                continue
            try:
                npz = np.load(f, allow_pickle=False)
                store[sym] = {k: npz[k] for k in npz.files}
            except Exception:
                pass

    return data_15m, data_1h, data_4h


def _run_cfg(
    cfg: dict,
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str,
    max_concurrent: int,
) -> float:
    """Run a single config and return PF. Returns 0.0 on error."""
    try:
        trades = collect_trades(cfg, data_15m, data_1h, data_4h, strat)
        if not trades:
            return 0.0
        pnl = portfolio_sim(trades, max_concurrent=max_concurrent)
        m = compute_metrics(pnl, DATA_YEARS)
        return float(m["PF"])
    except Exception as exc:
        print(f"    ERROR in _run_cfg: {exc}", file=sys.stderr)
        return 0.0


def check_robustness(
    baseline_cfg: dict[str, Any],
    data_15m: dict,
    data_1h: dict,
    data_4h: dict,
    strat: str = "D_mom8",
    max_concurrent: int = 10,
    perturbation_pct: float = DEFAULT_PERTURBATION_PCT,
    min_pf_ratio: float = DEFAULT_MIN_PF_RATIO,
    params_to_test: list[str] | None = None,
) -> dict[str, Any]:
    """Run parameter robustness sweep.

    Args:
        baseline_cfg: Base configuration dict (read from VolStraddleConfig).
        data_15m / data_1h / data_4h: Data dicts (symbol → arrays).
        strat: Strategy name ("D_mom8" or "S_d8").
        max_concurrent: Slot count for portfolio_sim.
        perturbation_pct: Perturbation percentage (e.g. 15 → ±15%).
        min_pf_ratio: Minimum PF / baseline_PF ratio to pass.
        params_to_test: List of param names to test (None → all available).

    Returns:
        Dict with keys: baseline_pf, results (list), passed (bool), fragile_params (list).
    """
    _ALL_PARAMS = ["d_sl", "s_sl", "tp", "s_vr", "s_dur", "lp", "sp"]
    if params_to_test is None:
        params_to_test = [p for p in _ALL_PARAMS if p in baseline_cfg]

    print(f"\nRunning baseline ({strat})...")
    baseline_pf = _run_cfg(baseline_cfg, data_15m, data_1h, data_4h, strat, max_concurrent)
    print(f"  Baseline PF = {baseline_pf:.3f}")

    if baseline_pf <= 0:
        print("ERROR: Baseline returned PF=0 — cannot run robustness check.", file=sys.stderr)
        return {"baseline_pf": 0.0, "results": [], "passed": False, "fragile_params": []}

    threshold = baseline_pf * min_pf_ratio
    results = []

    mult = perturbation_pct / 100.0
    for param in params_to_test:
        base_val = baseline_cfg[param]
        if not isinstance(base_val, (int, float)):
            continue

        for direction, factor in [("+", 1 + mult), ("-", 1 - mult)]:
            perturbed_val = base_val * factor
            # For integer params (s_dur), round and avoid zero
            if isinstance(base_val, int):
                perturbed_val = max(1, round(perturbed_val))
            else:
                perturbed_val = max(1e-6, perturbed_val)

            cfg_copy = {**baseline_cfg, param: perturbed_val}
            pf = _run_cfg(cfg_copy, data_15m, data_1h, data_4h, strat, max_concurrent)
            passed = pf >= threshold

            results.append({
                "param": param,
                "direction": direction,
                "base_val": base_val,
                "perturbed_val": perturbed_val,
                "pf": pf,
                "ratio": pf / baseline_pf if baseline_pf > 0 else 0,
                "passed": passed,
            })
            status = "✅" if passed else "❌ FRAGILE"
            print(
                f"  {param} {direction}{perturbation_pct:.0f}%: "
                f"{base_val} → {perturbed_val:.4g}  PF={pf:.3f} "
                f"({pf/baseline_pf*100:.0f}% of baseline) {status}"
            )

    fragile = sorted({r["param"] for r in results if not r["passed"]})
    all_passed = len(fragile) == 0

    return {
        "baseline_pf": baseline_pf,
        "results": results,
        "passed": all_passed,
        "fragile_params": fragile,
        "threshold_pf": threshold,
        "perturbation_pct": perturbation_pct,
        "min_pf_ratio": min_pf_ratio,
    }


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact summary table."""
    print("\n" + "═" * 60)
    print(f"ROBUSTNESS CHECK SUMMARY")
    print(f"  Baseline PF:     {report['baseline_pf']:.3f}")
    print(f"  Perturbation:    ±{report['perturbation_pct']:.0f}%")
    print(f"  Min PF ratio:    {report['min_pf_ratio']:.0%}")
    print(f"  PF threshold:    {report['threshold_pf']:.3f}")

    if report["fragile_params"]:
        print(f"\n  ❌ FRAGILE parameters: {', '.join(report['fragile_params'])}")
        print("     Strategy may be curve-fitted. Review before deploying.")
    else:
        print(f"\n  ✅ ALL VARIANTS PASS — strategy is robust to ±{report['perturbation_pct']:.0f}% perturbation")
    print("═" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check parameter robustness: ±N% perturbation of key strategy parameters."
    )
    parser.add_argument(
        "--pct", type=float, default=DEFAULT_PERTURBATION_PCT,
        help=f"Perturbation percentage (default: {DEFAULT_PERTURBATION_PCT})"
    )
    parser.add_argument(
        "--min-pf-ratio", type=float, default=DEFAULT_MIN_PF_RATIO,
        help=f"Min PF ratio vs baseline (default: {DEFAULT_MIN_PF_RATIO})"
    )
    parser.add_argument(
        "--strat", default="D_mom8", choices=["D_mom8", "S_d8"],
        help="Strategy to test (default: D_mom8)"
    )
    parser.add_argument(
        "--params", nargs="+",
        help="Specific params to test (default: all available)"
    )
    args = parser.parse_args()

    if not CACHE_DIR.exists():
        print(f"ERROR: {CACHE_DIR} not found.", file=sys.stderr)
        print("Run on a machine with data/cache_extended/ populated.", file=sys.stderr)
        return 2

    cfg_obj = VolStraddleConfig()
    whitelist = set(cfg_obj.symbol_whitelist)

    print(f"Loading data from {CACHE_DIR} ({len(whitelist)} symbols)...")
    data_15m, data_1h, data_4h = _load_data(CACHE_DIR, whitelist)
    print(f"  Loaded: {len(data_15m)} 15m, {len(data_1h)} 1h, {len(data_4h)} 4h symbols")

    if not data_15m:
        print("ERROR: No data loaded.", file=sys.stderr)
        return 2

    # Build baseline cfg from config — never hardcode
    baseline_cfg: dict[str, Any] = {
        "d_sl": cfg_obj.d_sl,
        "s_sl": cfg_obj.s_sl,
        "tp": cfg_obj.d_tp,
        "s_vr": cfg_obj.s_vr_threshold,
        "s_dur": cfg_obj.s_squeeze_dur,
        "lp": cfg_obj.d_long_prob,
        "sp": cfg_obj.d_short_prob,
        "d_vr": cfg_obj.d_vr_threshold,
        "lp": cfg_obj.d_long_prob,
        "sp": cfg_obj.d_short_prob,
        "strat": args.strat,
    }
    # Remove strat key — it's passed separately to collect_trades
    baseline_cfg.pop("strat", None)

    report = check_robustness(
        baseline_cfg=baseline_cfg,
        data_15m=data_15m,
        data_1h=data_1h,
        data_4h=data_4h,
        strat=args.strat,
        max_concurrent=cfg_obj.max_concurrent,
        perturbation_pct=args.pct,
        min_pf_ratio=args.min_pf_ratio,
        params_to_test=args.params,
    )

    print_summary(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
