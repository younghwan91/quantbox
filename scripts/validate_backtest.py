"""Backtest result plausibility validator.

Usage in research scripts::

    from scripts.validate_backtest import validate_backtest, PlausibilityError

    metrics = compute_metrics(pnl, data_years)
    validate_backtest(metrics, context="r58 baseline, 60 syms, 6yr")

    # Or as a soft check (returns errors instead of raising):
    errors = validate_backtest(metrics, context="...", raise_on_error=False)
    if errors:
        print("WARNING:", errors)

Thresholds encode empirical expectations for VR Squeeze-class strategies.
Values outside these ranges don't mean the result is wrong — they mean
the researcher should manually verify before documenting.

Statistical logic:
    - PF > 5.0:   almost always lookahead bias or a code bug (r53 incident)
    - WR > 88%:   physically implausible for a trend-following strategy with
                  hard SL/TP — indicates signal computed on exit bar's data
    - t-stat < 1.5: result is not statistically significant (N too small or
                   EV too noisy) — do not document or act on this
    - N < 20/yr:  fewer than 20 trades/year → insufficient for any inference
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow running as a script (python scripts/validate_backtest.py)
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))


# ── threshold definitions ──────────────────────────────────────────────────────

#: Minimum acceptable PF (below 1.0 = losing strategy)
PF_MIN: float = 1.00
#: Maximum plausible PF (above = suspect lookahead / code bug)
PF_MAX: float = 5.00
#: Minimum win rate (%)
WR_MIN: float = 30.0
#: Maximum win rate (%) — >88% = physically implausible for SL/TP strategies
WR_MAX: float = 88.0
#: Minimum t-statistic for statistical significance
T_STAT_MIN: float = 1.5
#: Minimum trades per year (fewer = unreliable stats)
MIN_TRADES_PER_YEAR: float = 20.0
#: Maximum drawdown — can't lose more than 100%
MAX_DD_CAP: float = 1.00


class PlausibilityError(ValueError):
    """Raised when backtest results fail plausibility checks."""

    def __init__(self, errors: list[str], context: str = ""):
        self.errors = errors
        self.context = context
        lines = [f"Backtest plausibility check FAILED" + (f" [{context}]" if context else "")]
        lines += [f"  • {e}" for e in errors]
        super().__init__("\n".join(lines))


def validate_backtest(
    metrics: dict[str, Any],
    data_years: float | None = None,
    context: str = "",
    raise_on_error: bool = True,
) -> list[str]:
    """Validate compute_metrics() output against plausibility thresholds.

    Args:
        metrics: Output dict from ``compute_metrics()``.  Must contain at
            minimum: N, PF, WR, t, max_dd_pct, ann_ret_pct.
        data_years: Optional explicit data duration in years.  If not
            provided, falls back to ``N / tpd / 365`` using the tpd field.
        context: Free-form description for error messages (e.g. "r58 baseline").
        raise_on_error: If True (default), raise PlausibilityError on failure.
            If False, return the list of error strings silently.

    Returns:
        List of error strings (empty = passed).

    Raises:
        PlausibilityError: When ``raise_on_error=True`` and checks fail.
        KeyError: If required metrics keys are missing.
    """
    errors: list[str] = []

    n = int(metrics["N"])
    pf = float(metrics["PF"])
    wr = float(metrics.get("WR", 0.0))
    t_stat = float(metrics.get("t", 0.0))
    max_dd = float(metrics.get("max_dd_pct", 0.0))

    # Estimate data_years if not provided
    if data_years is None:
        tpd = float(metrics.get("tpd", 0.0))
        data_years = n / (tpd * 365) if tpd > 0 and n > 0 else None

    # ── PF checks ──
    if pf < PF_MIN:
        errors.append(f"PF={pf:.3f} < {PF_MIN} — losing strategy (not worth documenting)")
    if pf > PF_MAX:
        errors.append(
            f"PF={pf:.3f} > {PF_MAX} — suspiciously high; verify for lookahead bias "
            f"or code bugs before documenting"
        )

    # ── WR checks ──
    if wr < WR_MIN:
        errors.append(f"WR={wr:.1f}% < {WR_MIN}% — unusually low (check SL/TP logic)")
    if wr > WR_MAX:
        errors.append(
            f"WR={wr:.1f}% > {WR_MAX}% — physically implausible for SL/TP strategy; "
            f"check for lookahead bias"
        )

    # ── Statistical significance ──
    if n > 0 and t_stat < T_STAT_MIN:
        errors.append(
            f"t-stat={t_stat:.2f} < {T_STAT_MIN} — not statistically significant; "
            f"need more trades or stronger edge"
        )

    # ── Sample size ──
    if data_years is not None and data_years > 0:
        trades_per_year = n / data_years
        if trades_per_year < MIN_TRADES_PER_YEAR:
            errors.append(
                f"N={n} over {data_years:.1f}yr = {trades_per_year:.0f} trades/yr "
                f"< {MIN_TRADES_PER_YEAR:.0f} minimum — unreliable statistics"
            )

    # ── Max drawdown cap ──
    if max_dd > MAX_DD_CAP * 100:
        errors.append(
            f"max_dd_pct={max_dd:.1f}% > {MAX_DD_CAP*100:.0f}% — strategy wipes account"
        )

    if errors and raise_on_error:
        raise PlausibilityError(errors, context=context)

    return errors


def print_validation_report(
    metrics: dict[str, Any],
    data_years: float | None = None,
    context: str = "",
) -> bool:
    """Print a compact validation report. Returns True if passed."""
    errors = validate_backtest(metrics, data_years=data_years, context=context, raise_on_error=False)
    n = int(metrics.get("N", 0))
    pf = float(metrics.get("PF", 0))
    wr = float(metrics.get("WR", 0))
    t = float(metrics.get("t", 0))

    label = f"[validate{' ' + context if context else ''}]"
    if not errors:
        print(f"{label} ✅ PASSED — N={n}, PF={pf:.3f}, WR={wr:.1f}%, t={t:.2f}")
    else:
        print(f"{label} ❌ FAILED — N={n}, PF={pf:.3f}, WR={wr:.1f}%, t={t:.2f}")
        for e in errors:
            print(f"  ⚠️  {e}")
    return not errors


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Validate backtest metrics JSON against plausibility thresholds."
    )
    parser.add_argument("metrics_json", nargs="?", help="Path to metrics JSON file (or stdin)")
    parser.add_argument("--years", type=float, default=None, help="Data duration in years")
    parser.add_argument("--context", default="", help="Description for error messages")
    args = parser.parse_args()

    if args.metrics_json:
        with open(args.metrics_json) as f:
            metrics = json.load(f)
    else:
        metrics = json.load(sys.stdin)

    passed = print_validation_report(metrics, data_years=args.years, context=args.context)
    sys.exit(0 if passed else 1)
