"""Tests for scripts/validate_backtest.py — backtest plausibility gate.

Verifies that the validator correctly flags:
  - PF too high (lookahead / code bug symptom)
  - PF below 1.0 (losing strategy)
  - WR above physical limit (lookahead symptom)
  - WR below minimum
  - Insufficient t-statistic (not significant)
  - Too few trades per year
  - Max drawdown > 100%

And confirms that legitimate strategy outputs pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing from scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_backtest import (  # noqa: E402
    PlausibilityError,
    validate_backtest,
    print_validation_report,
    PF_MAX,
    PF_MIN,
    WR_MAX,
    WR_MIN,
    T_STAT_MIN,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _metrics(
    N: int = 5000,
    PF: float = 1.55,
    WR: float = 57.0,
    t: float = 8.5,
    max_dd_pct: float = 60.0,
    tpd: float = 2.5,
) -> dict:
    """Build a realistic-looking metrics dict (baseline: r73-like)."""
    return {
        "N": N,
        "PF": PF,
        "WR": WR,
        "t": t,
        "max_dd_pct": max_dd_pct,
        "tpd": tpd,
        "EV_bp": 38.0,
        "ann_ret_pct": 120.0,
        "calmar": 2.0,
    }


# ── passing cases ─────────────────────────────────────────────────────────────


class TestValidBacktest:
    def test_realistic_baseline_passes(self):
        """r73-like metrics should pass all checks."""
        m = _metrics()
        errors = validate_backtest(m, data_years=5.5, raise_on_error=False)
        assert errors == [], errors

    def test_pf_at_lower_bound_passes(self):
        m = _metrics(PF=PF_MIN)
        errors = validate_backtest(m, raise_on_error=False)
        assert not any("PF" in e for e in errors)

    def test_pf_just_below_max_passes(self):
        m = _metrics(PF=PF_MAX - 0.01)
        errors = validate_backtest(m, raise_on_error=False)
        assert not any("PF" in e for e in errors)

    def test_wr_at_bounds_passes(self):
        for wr in (WR_MIN, WR_MAX - 1):
            m = _metrics(WR=wr)
            errors = validate_backtest(m, raise_on_error=False)
            assert not any("WR" in e for e in errors), f"WR={wr} should pass"

    def test_explicit_data_years(self):
        m = _metrics(N=300, tpd=0)  # tpd=0 → no auto-infer
        errors = validate_backtest(m, data_years=5.0, raise_on_error=False)
        assert not any("trades/yr" in e for e in errors)


# ── PF failures ───────────────────────────────────────────────────────────────


class TestPFFailures:
    def test_pf_below_1_flagged(self):
        m = _metrics(PF=0.98)
        errors = validate_backtest(m, raise_on_error=False)
        assert any("PF=" in e and "losing" in e for e in errors)

    def test_pf_too_high_flagged(self):
        m = _metrics(PF=PF_MAX + 0.01)
        errors = validate_backtest(m, raise_on_error=False)
        assert any("PF=" in e and "lookahead" in e.lower() for e in errors)

    def test_extreme_pf_suggests_lookahead(self):
        """PF=8 is the classic lookahead symptom (r53 incident)."""
        m = _metrics(PF=8.0)
        errors = validate_backtest(m, raise_on_error=False)
        assert errors  # must flag

    def test_raise_on_error_default(self):
        m = _metrics(PF=0.5)
        with pytest.raises(PlausibilityError):
            validate_backtest(m)

    def test_plausibility_error_contains_errors(self):
        m = _metrics(PF=0.5)
        with pytest.raises(PlausibilityError) as exc_info:
            validate_backtest(m, context="test ctx")
        err = exc_info.value
        assert err.errors
        assert "test ctx" in str(err)


# ── WR failures ───────────────────────────────────────────────────────────────


class TestWRFailures:
    def test_wr_above_max_flagged(self):
        m = _metrics(WR=WR_MAX + 0.1)
        errors = validate_backtest(m, raise_on_error=False)
        assert any("WR=" in e and "implausible" in e for e in errors)

    def test_wr_92pct_lookahead_symptom(self):
        """WR=92% is the classic MTF lookahead symptom (2026-02-26 incident)."""
        m = _metrics(WR=92.0)
        errors = validate_backtest(m, raise_on_error=False)
        assert errors  # must flag

    def test_wr_below_min_flagged(self):
        m = _metrics(WR=WR_MIN - 1)
        errors = validate_backtest(m, raise_on_error=False)
        assert any("WR=" in e for e in errors)


# ── t-statistic / significance ────────────────────────────────────────────────


class TestStatisticalSignificance:
    def test_low_t_stat_flagged(self):
        m = _metrics(t=T_STAT_MIN - 0.1)
        errors = validate_backtest(m, raise_on_error=False)
        assert any("t-stat" in e for e in errors)

    def test_t_stat_exactly_at_threshold_ok(self):
        m = _metrics(t=T_STAT_MIN)
        errors = validate_backtest(m, raise_on_error=False)
        assert not any("t-stat" in e for e in errors)

    def test_zero_n_skips_tstat(self):
        """N=0 → no t-stat check (no data)."""
        m = _metrics(N=0, t=0.0)
        errors = validate_backtest(m, raise_on_error=False)
        assert not any("t-stat" in e for e in errors)


# ── sample size ───────────────────────────────────────────────────────────────


class TestSampleSize:
    def test_too_few_trades_per_year_flagged(self):
        m = _metrics(N=50, tpd=0.05)  # → ~18 trades/yr over ~3yr
        errors = validate_backtest(m, data_years=3.0, raise_on_error=False)
        assert any("trades/yr" in e for e in errors)

    def test_sufficient_trades_passes(self):
        m = _metrics(N=1000, tpd=0.5)  # → ~200 trades/yr over ~5yr
        errors = validate_backtest(m, data_years=5.0, raise_on_error=False)
        assert not any("trades/yr" in e for e in errors)

    def test_no_data_years_skips_check(self):
        """Without data_years or tpd, skip the per-year check."""
        m = _metrics(N=10, tpd=0.0)
        errors = validate_backtest(m, data_years=None, raise_on_error=False)
        assert not any("trades/yr" in e for e in errors)


# ── max drawdown ──────────────────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_drawdown_over_100_flagged(self):
        m = _metrics(max_dd_pct=101.0)
        errors = validate_backtest(m, raise_on_error=False)
        assert any("max_dd" in e for e in errors)

    def test_drawdown_at_100_ok(self):
        m = _metrics(max_dd_pct=100.0)
        errors = validate_backtest(m, raise_on_error=False)
        assert not any("max_dd" in e for e in errors)


# ── multiple failures ──────────────────────────────────────────────────────────


class TestMultipleFailures:
    def test_multiple_failures_all_reported(self):
        """All failing checks should be reported, not just the first."""
        m = _metrics(PF=0.5, WR=95.0, t=0.5, N=5, tpd=0.01)
        errors = validate_backtest(m, data_years=1.0, raise_on_error=False)
        assert len(errors) >= 3  # PF, WR, t-stat at minimum


# ── print report ──────────────────────────────────────────────────────────────


class TestPrintValidationReport:
    def test_passes_returns_true(self, capsys):
        m = _metrics()
        result = print_validation_report(m, data_years=5.5, context="test")
        assert result is True
        out = capsys.readouterr().out
        assert "PASSED" in out

    def test_fails_returns_false(self, capsys):
        m = _metrics(PF=0.5)
        result = print_validation_report(m, context="bad test")
        assert result is False
        out = capsys.readouterr().out
        assert "FAILED" in out
