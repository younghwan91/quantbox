"""Data integrity tests for cached OHLCV .npz files.

Catches: NaN/Inf prices, zero/negative prices, invalid bar geometry
(high < low / close outside [low, high]), and extreme single-bar moves
that indicate corrupted data rather than real market events.

Tests run in two modes:
  1. Unit tests (always run) — validate the sanity-check logic itself
     using synthetic arrays.
  2. Integration tests (skip if cache_extended absent) — spot-check
     real cached files so CI on development machines stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


# ── constants ──────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache_extended"
# Flash-crash threshold: single-bar |return| ≥ this is treated as corrupt data.
MAX_SINGLE_BAR_MOVE = 0.70  # 70%
SAMPLE_SIZE = 10  # number of real files to spot-check in integration tests


# ── validation helpers (reusable in research scripts) ─────────────────────────


def check_price_array(arr: np.ndarray, name: str) -> list[str]:
    """Return list of error strings for a single price array."""
    errors: list[str] = []
    if not np.isfinite(arr).all():
        n = int(np.sum(~np.isfinite(arr)))
        errors.append(f"{name}: {n} NaN/Inf values")
    if np.any(arr <= 0):
        n = int(np.sum(arr <= 0))
        errors.append(f"{name}: {n} values ≤ 0")
    return errors


def check_bar_geometry(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> list[str]:
    """Validate OHLCV bar geometry: high≥low, low≤close≤high."""
    errors: list[str] = []
    bad_hl = np.sum(high < low)
    if bad_hl:
        errors.append(f"high < low in {bad_hl} bars")
    bad_ch = np.sum(close > high)
    if bad_ch:
        errors.append(f"close > high in {bad_ch} bars")
    bad_cl = np.sum(close < low)
    if bad_cl:
        errors.append(f"close < low in {bad_cl} bars")
    return errors


def check_extreme_moves(close: np.ndarray, threshold: float = MAX_SINGLE_BAR_MOVE) -> list[str]:
    """Detect single-bar returns that exceed threshold (flash-crash / bad data)."""
    if len(close) < 2:
        return []
    returns = np.abs(np.diff(close) / close[:-1])
    n_extreme = int(np.sum(returns >= threshold))
    if n_extreme:
        return [f"{n_extreme} bars with |return| ≥ {threshold * 100:.0f}%"]
    return []


def validate_npz_file(path: Path) -> list[str]:
    """Full validation of a single backtest .npz file. Returns error list."""
    try:
        npz = np.load(path, allow_pickle=False)
    except Exception as exc:
        return [f"Cannot load file: {exc}"]

    errors: list[str] = []
    required = {"close", "high", "low"}
    missing = required - set(npz.files)
    if missing:
        errors.append(f"Missing keys: {missing}")
        return errors

    close = npz["close"]
    high = npz["high"]
    low = npz["low"]

    for arr, name in ((close, "close"), (high, "high"), (low, "low")):
        errors.extend(check_price_array(arr, name))

    errors.extend(check_bar_geometry(high, low, close))
    errors.extend(check_extreme_moves(close))
    return errors


# ── unit tests (always run, synthetic data) ───────────────────────────────────


class TestCheckPriceArray:
    def test_clean_array_no_errors(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert check_price_array(arr, "x") == []

    def test_nan_detected(self):
        arr = np.array([1.0, np.nan, 3.0])
        errs = check_price_array(arr, "x")
        assert any("NaN" in e for e in errs)

    def test_inf_detected(self):
        arr = np.array([1.0, np.inf, 3.0])
        errs = check_price_array(arr, "x")
        assert any("NaN/Inf" in e for e in errs)

    def test_zero_price_detected(self):
        arr = np.array([1.0, 0.0, 3.0])
        errs = check_price_array(arr, "x")
        assert any("≤ 0" in e for e in errs)

    def test_negative_price_detected(self):
        arr = np.array([1.0, -0.5, 3.0])
        errs = check_price_array(arr, "x")
        assert any("≤ 0" in e for e in errs)


class TestCheckBarGeometry:
    def _make(self, high, low, close):
        return (
            np.array(high, dtype=float),
            np.array(low, dtype=float),
            np.array(close, dtype=float),
        )

    def test_valid_bars_no_errors(self):
        h, l, c = self._make([2, 3], [1, 2], [1.5, 2.5])
        assert check_bar_geometry(h, l, c) == []

    def test_high_below_low(self):
        h, l, c = self._make([1], [2], [1.5])  # high < low
        errs = check_bar_geometry(h, l, c)
        assert any("high < low" in e for e in errs)

    def test_close_above_high(self):
        h, l, c = self._make([2], [1], [3])  # close > high
        errs = check_bar_geometry(h, l, c)
        assert any("close > high" in e for e in errs)

    def test_close_below_low(self):
        h, l, c = self._make([2], [1], [0.5])  # close < low
        errs = check_bar_geometry(h, l, c)
        assert any("close < low" in e for e in errs)


class TestCheckExtremeMoves:
    def test_normal_moves_no_errors(self):
        close = np.array([100.0, 101.0, 99.0, 100.5])
        assert check_extreme_moves(close) == []

    def test_70pct_move_detected(self):
        close = np.array([100.0, 170.1])  # +70.1%
        errs = check_extreme_moves(close)
        assert errs  # must flag

    def test_exactly_threshold_is_flagged(self):
        close = np.array([100.0, 100.0 * (1 + MAX_SINGLE_BAR_MOVE)])
        errs = check_extreme_moves(close)
        assert errs

    def test_just_below_threshold_ok(self):
        close = np.array([100.0, 100.0 * (1 + MAX_SINGLE_BAR_MOVE - 0.001)])
        assert check_extreme_moves(close) == []

    def test_single_element_no_error(self):
        assert check_extreme_moves(np.array([100.0])) == []


# ── integration tests (skip if data not available) ────────────────────────────


def _get_sample_files() -> list[Path]:
    """Return up to SAMPLE_SIZE real 15m npz files for spot-checking."""
    if not CACHE_DIR.exists():
        return []
    files = sorted(CACHE_DIR.glob("*_15m_2190d_*.npz"))
    return files[:SAMPLE_SIZE]


@pytest.mark.skipif(
    not CACHE_DIR.exists() or not any(CACHE_DIR.glob("*_15m_2190d_*.npz")),
    reason="data/cache_extended not available (CI environment)",
)
class TestRealDataSanity:
    """Spot-check real cached .npz files for data quality issues."""

    def test_sample_files_exist(self):
        files = _get_sample_files()
        assert len(files) > 0, "No *_15m_2190d_*.npz files found in cache_extended"

    @pytest.mark.parametrize("path", _get_sample_files())
    def test_file_passes_sanity_check(self, path: Path):
        errors = validate_npz_file(path)
        assert errors == [], f"Data quality issues in {path.name}:\n" + "\n".join(f"  • {e}" for e in errors)

    def test_all_sample_files_clean(self):
        """Aggregate test: no errors in any sampled file."""
        all_errors: dict[str, list[str]] = {}
        for path in _get_sample_files():
            errs = validate_npz_file(path)
            if errs:
                all_errors[path.name] = errs

        if all_errors:
            msg = "\n".join(f"{name}: {errs}" for name, errs in all_errors.items())
            pytest.fail(f"Data quality issues found:\n{msg}")
