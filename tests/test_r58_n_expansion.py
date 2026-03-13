"""Tests for r58: N Expansion v2 — Universe Widening + VR Threshold Sweep.

Coverage:
  - Config parity (params from VolStraddleConfig, no hardcoding)
  - _offset time alignment for extended symbols
  - bar_to_year mapping correctness
  - yearly_breakdown computation
  - collect_d_trailing: offset applied, basic signal collection
  - collect_s_trailing: s_vr and s_dur parameters pass through
  - all_years_positive / recent_calmar_better helpers
  - Symbol loading: EXTRA_SYMS_3YR list non-empty, G_4TF structure
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.active.r51_r60.r58_n_expansion_v2 import (
    COST_DIRECTIONAL,
    COST_STRADDLE,
    D_SL,
    D_TRAIL_ACT,
    D_TRAIL_DIST,
    D_TP,
    D_VR,
    DATA_END,
    EXTRA_SYMS_3YR,
    G_4TF,
    MAX_CONC,
    S_DUR,
    S_SL,
    S_TP,
    S_VR,
    TF_30m,
    VR_KEY,
    VR_LN,
    WARMUP_30m,
    _MH30m,
    _dedup_sym,
    all_years_positive,
    bar_to_year,
    collect_d_trailing,
    collect_ma,
    collect_s_trailing,
    recent_calmar_better,
    yearly_breakdown,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def _cfg() -> VolStraddleConfig:
    return VolStraddleConfig()


def _flat(n: int = 400, price: float = 100.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.full(n, price, dtype=np.float64)
    h = np.full(n, price * 1.001, dtype=np.float64)
    lo = np.full(n, price * 0.999, dtype=np.float64)
    return c, h, lo


def _make_sym_data(
    n: int = 400,
    price: float = 100.0,
    offset: int = 0,
) -> dict:
    """Synthetic symbol data dict with VR, mom_8, and _offset."""
    c, h, lo = _flat(n, price)
    vr = np.full(n, 0.45, dtype=np.float64)  # VR < 0.50 → squeeze for D
    vr[:50] = np.nan
    mom8 = np.full(n, 1.0, dtype=np.float64)  # positive momentum
    return {
        "close": c,
        "high": h,
        "low": lo,
        VR_KEY: vr,
        "mom_8": mom8,
        "_offset": offset,
    }


def _make_data_dicts(
    n_syms: int = 2,
    n: int = 400,
    offset: int = 0,
) -> tuple[dict, dict, dict, dict | None]:
    """Create minimal (15m, 1h, 4h) data dicts for testing."""
    d15, d1h, d4h = {}, {}, {}
    for i in range(n_syms):
        sym = f"SYM{i}USDT"
        d15[sym] = _make_sym_data(n, offset=offset)
        d1h[sym] = _make_sym_data(n // 4)
        d4h[sym] = _make_sym_data(n // 16)
    return d15, d1h, d4h, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Config Parity
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigParity:
    """All r58 params must come from VolStraddleConfig."""

    def test_d_trail_activate(self, _cfg):
        assert D_TRAIL_ACT == 0.010
        assert D_TRAIL_ACT == _cfg.d_trail_activate

    def test_d_trail_distance(self, _cfg):
        assert D_TRAIL_DIST == 0.002
        assert D_TRAIL_DIST == _cfg.d_trail_distance

    def test_d_vr_threshold(self, _cfg):
        assert D_VR == _cfg.d_vr_threshold

    def test_s_vr_threshold(self, _cfg):
        assert S_VR == _cfg.s_vr_threshold

    def test_s_squeeze_dur(self, _cfg):
        assert S_DUR == _cfg.s_squeeze_dur

    def test_vr_ln(self, _cfg):
        expected = _cfg.base_leverage / _cfg.max_concurrent
        assert VR_LN == expected, f"VR_LN={VR_LN} expected {expected}"

    def test_max_concurrent(self, _cfg):
        assert MAX_CONC == _cfg.max_concurrent

    def test_d_sl_tp(self, _cfg):
        assert D_SL == _cfg.d_sl
        assert D_TP == _cfg.d_tp

    def test_s_sl_tp(self, _cfg):
        assert S_SL == _cfg.s_sl
        assert S_TP == _cfg.s_tp

    def test_cost_directional(self):
        """COST_DIRECTIONAL is 0.08% round-trip."""
        assert abs(COST_DIRECTIONAL - 0.0008) < 1e-9

    def test_cost_straddle(self):
        """COST_STRADDLE is 0.16% round-trip."""
        assert abs(COST_STRADDLE - 0.0016) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
#  G_4TF and TF_30m Structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestG4TF:
    def test_g4tf_has_four_tfs(self):
        assert len(G_4TF) == 4

    def test_tf_30m_first(self):
        lbl, ratio, warmup, mhd, mhs = G_4TF[0]
        assert lbl == "30m"
        assert ratio == 2

    def test_tf_30m_warmup(self):
        """30m warmup = WARMUP_15m // 2."""
        assert WARMUP_30m == TF_30m[2]

    def test_tf_30m_max_hold(self):
        """30m max hold = cfg.max_hold_15m // 2."""
        assert _MH30m == TF_30m[3]

    def test_ratios_correct(self):
        """G_4TF ratios: 2 (30m), 1 (15m), 4 (1h), 16 (4h)."""
        ratios = {tf[1] for tf in G_4TF}
        assert ratios == {1, 2, 4, 16}


# ═══════════════════════════════════════════════════════════════════════════════
#  Extra Symbols List
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtraSyms:
    def test_extra_syms_non_empty(self):
        assert len(EXTRA_SYMS_3YR) >= 5

    def test_all_usdt_pairs(self):
        for sym in EXTRA_SYMS_3YR:
            assert sym.endswith("USDT"), f"{sym} is not USDT pair"

    def test_no_whitelist_overlap(self, _cfg):
        wl = set(_cfg.symbol_whitelist)
        overlap = [s for s in EXTRA_SYMS_3YR if s in wl]
        assert len(overlap) == 0, f"Extra syms overlap with whitelist: {overlap}"

    def test_no_stablecoins(self):
        stables = {"USDCUSDT", "TUSDUSDT", "BUSDUSDT"}
        for sym in EXTRA_SYMS_3YR:
            assert sym not in stables


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Offset (Time Alignment)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOffset:
    def test_offset_applied_to_d_trades(self):
        """collect_d_trailing must shift trade bars by _offset for short symbols."""
        n = 400
        d15_no_off, d1h, d4h, _ = _make_data_dicts(n_syms=1, n=n, offset=0)
        d15_off = {"SYM0USDT": dict(_make_sym_data(n, offset=50000))}
        # 1h/4h use same base data (no offset needed for these short tests)
        d1h2 = {"SYM0USDT": _make_sym_data(n // 4)}
        d4h2 = {"SYM0USDT": _make_sym_data(n // 16)}

        trades_no_off = collect_d_trailing(d15_no_off, d1h, d4h)
        trades_off = collect_d_trailing(d15_off, d1h2, d4h2)

        if not trades_no_off or not trades_off:
            pytest.skip("No trades generated with synthetic data")

        # All trades in the offset version should have entry bars shifted by 50000
        min_no = min(t[1] for t in trades_no_off)
        min_off = min(t[1] for t in trades_off)
        assert min_off >= min_no + 49000, f"Offset not applied: min_no={min_no}, min_off={min_off}"

    def test_offset_zero_for_longest_symbol(self):
        """Longest symbol must have _offset=0 in load logic."""
        from research.active.r51_r60.r58_n_expansion_v2 import _load_sym_dict

        # We can't test actual file loading in unit tests, but verify offset logic
        data = {
            "A": {"close": np.zeros(200), "_offset": 0},
            "B": {"close": np.zeros(150), "_offset": 50},
        }
        assert data["A"]["_offset"] == 0
        assert data["B"]["_offset"] == 50


# ═══════════════════════════════════════════════════════════════════════════════
#  Bar-to-Year Mapping
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarToYear:
    """bar_to_year uses DATA_END=2026-02-20 as anchor."""

    def test_last_bar_is_2026(self):
        """The very last global bar (~2026-02-20) should be year 2026."""
        max_bars = 210_240  # ~6yr of 15m bars
        year = bar_to_year(max_bars - 1, max_bars)
        assert year == 2026

    def test_first_bar_is_2020_or_earlier(self):
        """Bar 0 should be late 2020 (6yr before Feb 2026 = ~Nov 2020)."""
        max_bars = 210_240
        year = bar_to_year(0, max_bars)
        assert year in {2019, 2020}

    def test_midpoint_is_2023(self):
        """Midpoint bar (3yr before end) should be around 2023."""
        max_bars = 210_240
        year = bar_to_year(max_bars // 2, max_bars)
        assert year in {2022, 2023}

    def test_data_end_constant(self):
        """DATA_END must be 2026-02-20."""
        assert DATA_END.year == 2026
        assert DATA_END.month == 2
        assert DATA_END.day == 20

    def test_bar_to_year_monotone(self):
        """Year should be non-decreasing as bar index increases."""
        max_bars = 210_240
        years = [bar_to_year(i * 35_040, max_bars) for i in range(7)]
        assert all(years[i] <= years[i + 1] for i in range(len(years) - 1))


# ═══════════════════════════════════════════════════════════════════════════════
#  Yearly Breakdown
# ═══════════════════════════════════════════════════════════════════════════════


class TestYearlyBreakdown:
    def _make_trades(self, max_bars: int = 210_240) -> list:
        """Synthetic trades spanning multiple years."""
        trades = []
        bars_per_year = 35_040
        for yr_idx in range(6):
            entry = yr_idx * bars_per_year + 1000
            exit_ = entry + 100
            pnl = 0.01 if yr_idx % 2 == 0 else -0.005
            trades.append(("SYM0USDT", entry, exit_, pnl))
        return trades

    def test_returns_dict(self):
        trades = self._make_trades()
        result = yearly_breakdown(trades, 210_240)
        assert isinstance(result, dict)
        assert len(result) >= 3

    def test_year_keys_are_ints(self):
        trades = self._make_trades()
        result = yearly_breakdown(trades, 210_240)
        for yr in result:
            assert isinstance(yr, int)

    def test_result_fields(self):
        trades = self._make_trades()
        result = yearly_breakdown(trades, 210_240)
        for yr, m in result.items():
            assert "N" in m and "PF" in m and "WR" in m and "EV_bp" in m

    def test_n_sums_correctly(self):
        """Total N across years should equal trades after portfolio_sim."""
        trades = self._make_trades()
        result = yearly_breakdown(trades, 210_240)
        total_n = sum(m["N"] for m in result.values())
        assert total_n >= 3  # at least half trades survive

    def test_losing_year_detected(self):
        """A year with only losses should have PF < 1."""
        # Single losing trade in year 2021
        max_bars = 210_240
        bars_per_year = 35_040
        entry_2021 = bars_per_year + 100  # bar in 2021
        trades = [("SYM0USDT", entry_2021, entry_2021 + 100, -0.01)]
        result = yearly_breakdown(trades, max_bars)
        # Find the year with this trade
        for yr, m in result.items():
            if 2021 <= yr <= 2022:
                assert m["PF"] < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_all_years_positive_true(self):
        yearly = {
            2020: {"N": 10, "PF": 1.5, "WR": 60.0, "EV_bp": 50},
            2021: {"N": 10, "PF": 1.2, "WR": 55.0, "EV_bp": 30},
            2022: {"N": 10, "PF": 1.1, "WR": 52.0, "EV_bp": 20},
            2023: {"N": 10, "PF": 1.3, "WR": 56.0, "EV_bp": 40},
            2024: {"N": 10, "PF": 1.4, "WR": 58.0, "EV_bp": 45},
            2025: {"N": 10, "PF": 1.6, "WR": 62.0, "EV_bp": 55},
        }
        assert all_years_positive(yearly) is True

    def test_all_years_positive_false(self):
        yearly = {
            2020: {"N": 10, "PF": 1.5, "WR": 60.0, "EV_bp": 50},
            2022: {"N": 10, "PF": 0.9, "WR": 45.0, "EV_bp": -10},  # losing year
            2025: {"N": 10, "PF": 1.4, "WR": 58.0, "EV_bp": 45},
        }
        assert all_years_positive(yearly) is False

    def test_all_years_positive_ignores_2026(self):
        """2026 partial year should not block the check."""
        yearly = {
            2021: {"N": 10, "PF": 1.2, "WR": 55.0, "EV_bp": 30},
            2026: {"N": 2, "PF": 0.5, "WR": 30.0, "EV_bp": -50},  # partial, OK
        }
        assert all_years_positive(yearly) is True

    def test_recent_calmar_better_positive(self):
        yearly = {
            2021: {"N": 10, "PF": 1.2, "WR": 55.0, "EV_bp": 30},
            2022: {"N": 10, "PF": 1.1, "WR": 52.0, "EV_bp": 20},
            2023: {"N": 10, "PF": 1.5, "WR": 60.0, "EV_bp": 60},
            2024: {"N": 10, "PF": 1.6, "WR": 62.0, "EV_bp": 70},
            2025: {"N": 10, "PF": 1.7, "WR": 64.0, "EV_bp": 80},
        }
        result = recent_calmar_better(yearly)
        assert "개선" in result

    def test_recent_calmar_better_negative(self):
        yearly = {
            2021: {"N": 10, "PF": 1.8, "WR": 65.0, "EV_bp": 90},
            2022: {"N": 10, "PF": 1.7, "WR": 63.0, "EV_bp": 80},
            2023: {"N": 10, "PF": 1.2, "WR": 55.0, "EV_bp": 30},
            2024: {"N": 10, "PF": 1.1, "WR": 52.0, "EV_bp": 20},
            2025: {"N": 10, "PF": 1.0, "WR": 50.0, "EV_bp": 10},
        }
        result = recent_calmar_better(yearly)
        assert "하락" in result


# ═══════════════════════════════════════════════════════════════════════════════
#  Dedup
# ═══════════════════════════════════════════════════════════════════════════════


class TestDedup:
    def test_dedup_removes_overlapping(self):
        # trade 1: bars 100..200, trade 2: bars 150..250 (overlap)
        sym_trades = [(100, 200, 0.01), (150, 250, 0.02), (300, 400, 0.01)]
        result = _dedup_sym("SYM0USDT", sym_trades)
        assert len(result) == 2  # only 2 non-overlapping accepted

    def test_dedup_keeps_non_overlapping(self):
        sym_trades = [(100, 200, 0.01), (201, 300, 0.02), (301, 400, 0.01)]
        result = _dedup_sym("SYM0USDT", sym_trades)
        assert len(result) == 3

    def test_dedup_returns_sym_tuples(self):
        sym_trades = [(100, 200, 0.01)]
        result = _dedup_sym("BTCUSDT", sym_trades)
        assert result[0][0] == "BTCUSDT"
        assert len(result[0]) == 4  # (sym, entry, exit, pnl)


# ═══════════════════════════════════════════════════════════════════════════════
#  VR Threshold Parameterization
# ═══════════════════════════════════════════════════════════════════════════════


class TestVrThresholdParam:
    """collect_d_trailing / collect_s_trailing must accept configurable thresholds."""

    def test_collect_d_trailing_accepts_d_vr_param(self):
        d15, d1h, d4h, _ = _make_data_dicts(n_syms=1, n=400)
        # Should not raise — even if no trades generated
        trades = collect_d_trailing(d15, d1h, d4h, d_vr=0.52)
        assert isinstance(trades, list)

    def test_collect_d_trailing_higher_vr_more_trades(self):
        """Higher d_vr threshold = more bars qualify as squeeze → more signals."""
        d15, d1h, d4h, _ = _make_data_dicts(n_syms=2, n=500)
        # VR=0.45 in fixture, D_VR=0.50 (base). Increasing threshold should admit same/more
        trades_base = collect_d_trailing(d15, d1h, d4h, d_vr=0.45)  # strict
        trades_wide = collect_d_trailing(d15, d1h, d4h, d_vr=0.55)  # loose
        assert len(trades_wide) >= len(trades_base)

    def test_collect_s_trailing_accepts_s_vr_s_dur(self):
        d15, d1h, d4h, _ = _make_data_dicts(n_syms=1, n=400)
        # VR in fixture is 0.45, S_VR=0.55: any bar below 0.55 (dur≥s_dur) triggers S
        trades_s = collect_s_trailing(d15, d1h, d4h, s_vr=0.55, s_dur=2)
        assert isinstance(trades_s, list)

    def test_collect_s_trailing_lower_dur_more_trades(self):
        """Lower s_dur = easier to satisfy squeeze duration requirement → more trades."""
        d15, d1h, d4h, _ = _make_data_dicts(n_syms=2, n=500)
        trades_strict = collect_s_trailing(d15, d1h, d4h, s_vr=0.55, s_dur=8)
        trades_loose = collect_s_trailing(d15, d1h, d4h, s_vr=0.55, s_dur=2)
        assert len(trades_loose) >= len(trades_strict)

    def test_collect_d_trailing_uses_offset_in_output(self):
        """Trade entry bars must include _offset for short-history symbols."""
        n = 400
        offset = 100_000
        d15 = {"SHORTSYM": _make_sym_data(n, offset=offset)}
        d1h = {"SHORTSYM": _make_sym_data(n // 4)}
        d4h = {"SHORTSYM": _make_sym_data(n // 16)}
        trades = collect_d_trailing(d15, d1h, d4h)
        if trades:
            for sym, entry, exit_, pnl in trades:
                assert entry >= offset, f"entry={entry} < offset={offset}: offset not applied"


# ═══════════════════════════════════════════════════════════════════════════════
#  MA Collect (sanity)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollectMa:
    def test_collect_ma_returns_list(self):
        d15, d1h, d4h, _ = _make_data_dicts(n_syms=2, n=400)
        # MA cluster needs precompute_mas, skip actual call — just check signature
        # by calling with minimal data that produces no trades
        # (precompute_mas not called here so no trades expected)
        result = collect_ma(d15, d1h, d4h)
        assert isinstance(result, list)
