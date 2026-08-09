"""Bit-parity: new generic engine (sim_core) vs existing monolithic sims.

North star: 틀리지 않은 백테스팅.  The generic ``sim_core`` engine must
reproduce ``trade_sim``'s existing sims to the trade and to the PnL — otherwise
the refactor would silently change historical backtest numbers.

Three levels:
  1. Kernel parity — ``ma_cluster_conditions`` + ``sim_drive`` vs
     ``sim_ma_cluster_trailing`` on identical arrays (deterministic, fast).
  2. State-coupling regression — a crafted case where a trade spans in-cluster
     bars, proving the freeze / ``count=0`` reset / ``i=eb+1`` jump is
     reproduced (a naive whole-array precompute would diverge here).
  3. End-to-end parity on REAL cached data — ``run_ma_cluster`` vs
     ``collect_ma_cluster_trades`` through the full resample+offset pipeline.

Levels 1–2 feed the SAME synthetic array to two implementations; that is a
parity (plumbing) check, not a performance backtest, so deterministic input is
appropriate and standard here (cf. tests/test_trade_sim_parity.py).  Level 3
uses only real Binance data.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pytest

from quantbox.backtest.sim_core import ma_cluster_conditions, run_ma_cluster, sim_drive
from quantbox.backtest.trade_sim import (
    WARMUP_4h,
    collect_ma_cluster_trades,
    sim_ma_cluster_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import compute_momentum

CACHE_DIR = "data/cache_extended"


# ── helpers ──────────────────────────────────────────────────────────────────


def _sma_rolling(c: np.ndarray, period: int) -> np.ndarray:
    """Same inline rolling-sum SMA as collect_ma_cluster_trades."""
    n = len(c)
    out = np.full(n, np.nan)
    if n >= period:
        s = float(np.sum(c[:period]))
        out[period - 1] = s / period
        for idx in range(period, n):
            s += c[idx] - c[idx - period]
            out[idx] = s / period
    return out


def _make_4h_series(n: int = 4000, seed: int = 7) -> tuple[np.ndarray, ...]:
    """Deterministic 4h OHLC that actually forms MA clusters + breakouts.

    Parity fixture only (same array fed to both engines) — not a perf backtest.
    """
    rng = np.random.RandomState(seed)
    # Low-vol random walk → SMAs converge → clusters form; occasional jumps
    # produce breakouts and trailing activations.
    steps = rng.normal(0, 0.004, n)
    jump_mask = rng.rand(n) < 0.03
    steps[jump_mask] += rng.normal(0, 0.03, int(jump_mask.sum()))
    close = 100.0 * np.cumprod(1.0 + steps)
    high = close * (1.0 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.006, n)))
    return close, high, low


def _cfg_params() -> dict:
    c = VolStraddleConfig()
    return {
        "cluster_pct": c.ma_cluster_pct,
        "min_duration": c.ma_cluster_min_duration,
        "max_duration": c.ma_cluster_max_duration,
        "min_momentum": c.ma_cluster_min_momentum,
        "sl": c.ma_cluster_sl,
        "tp": c.ma_cluster_tp,
        "trail_activate": c.ma_cluster_trail_activate,
        "trail_distance": c.ma_cluster_trail_distance,
        "max_hold": c.ma_cluster_max_hold_4h,
    }


def _run_split(close, high, low, p, cost) -> tuple[np.ndarray, ...]:
    """Run the sim_core split path directly on 4h arrays."""
    sma20 = _sma_rolling(close, 20)
    sma50 = _sma_rolling(close, 50)
    sma100 = _sma_rolling(close, 100)
    sma200 = _sma_rolling(close, 200)
    mom = compute_momentum(close, 8)
    bar_kind, entry_ok, direction = ma_cluster_conditions(
        close, sma20, sma50, sma100, sma200, mom, p["cluster_pct"], p["min_momentum"]
    )
    return sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        p["min_duration"],
        p["max_duration"],
        p["sl"],
        p["tp"],
        p["trail_activate"],
        p["trail_distance"],
        p["max_hold"],
        cost,
        WARMUP_4h,
    )


def _run_monolith(close, high, low, p, cost) -> tuple[np.ndarray, ...]:
    sma20 = _sma_rolling(close, 20)
    sma50 = _sma_rolling(close, 50)
    sma100 = _sma_rolling(close, 100)
    sma200 = _sma_rolling(close, 200)
    mom = compute_momentum(close, 8)
    return sim_ma_cluster_trailing(
        close,
        high,
        low,
        sma20,
        sma50,
        sma100,
        sma200,
        mom,
        p["cluster_pct"],
        p["min_duration"],
        p["max_duration"],
        p["min_momentum"],
        p["sl"],
        p["tp"],
        p["trail_activate"],
        p["trail_distance"],
        p["max_hold"],
        cost,
        WARMUP_4h,
    )


# ── Level 1: kernel bit-parity ───────────────────────────────────────────────


def test_kernel_bit_parity_default_params():
    """sim_core split == sim_ma_cluster_trailing on identical arrays."""
    close, high, low = _make_4h_series()
    p = _cfg_params()
    cost = 0.0008

    mp, me, mx = _run_monolith(close, high, low, p, cost)
    sp, se, sx, _sd = _run_split(close, high, low, p, cost)

    assert len(sp) == len(mp), f"trade count differs: split={len(sp)} monolith={len(mp)}"
    assert len(mp) > 0, "fixture produced no trades — parity check would be vacuous"
    np.testing.assert_array_equal(se, me)
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_allclose(sp, mp, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 3, 11, 42])
def test_kernel_bit_parity_multi_seed(seed):
    """Parity holds across varied price paths."""
    close, high, low = _make_4h_series(seed=seed)
    p = _cfg_params()
    cost = 0.0008
    mp, me, mx = _run_monolith(close, high, low, p, cost)
    sp, se, sx, _sd = _run_split(close, high, low, p, cost)
    assert len(sp) == len(mp)
    np.testing.assert_array_equal(se, me)
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_allclose(sp, mp, rtol=0, atol=1e-12)


@pytest.mark.parametrize("min_dur,max_dur", [(1, 0), (3, 20), (6, 24), (10, 0)])
def test_kernel_bit_parity_duration_gates(min_dur, max_dur):
    """Counter gate (min/max_duration) reproduced across settings."""
    close, high, low = _make_4h_series(seed=5)
    p = _cfg_params()
    p["min_duration"] = min_dur
    p["max_duration"] = max_dur
    cost = 0.0008
    mp, me, mx = _run_monolith(close, high, low, p, cost)
    sp, se, sx, _sd = _run_split(close, high, low, p, cost)
    assert len(sp) == len(mp)
    np.testing.assert_array_equal(se, me)
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_allclose(sp, mp, rtol=0, atol=1e-12)


# ── Level 2: state-coupling regression ───────────────────────────────────────


def test_state_coupling_freeze_reset_jump():
    """A trade spans in-cluster bars: freeze + count=0 reset + i=eb+1 jump.

    This is the exact scenario the design flagged (parity trap): if entry state
    were precomputed independently of exits, post-exit signals would diverge.
    Here sim_core (which routes entry state through sim_drive) must still match
    the monolith trade-for-trade.
    """
    # A long, tight cluster (SMAs converged) with a mid-series breakout that
    # opens a trade; the cluster continues *through* the trade window.
    n = 1200
    close = np.full(n, 100.0)
    # tiny drift keeps SMAs clustered
    close = close + np.linspace(0, 1.0, n)
    # inject a breakout jump at bar 900 to trigger an entry, then keep clustering
    close[900:] += 3.0
    high = close * 1.001
    low = close * 0.999
    p = _cfg_params()
    p["min_duration"] = 3
    p["max_duration"] = 0
    cost = 0.0008

    mp, me, mx = _run_monolith(close, high, low, p, cost)
    sp, se, sx, _sd = _run_split(close, high, low, p, cost)

    np.testing.assert_array_equal(se, me)
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_allclose(sp, mp, rtol=0, atol=1e-12)


def test_state_coupling_naive_precompute_would_diverge():
    """Guard has teeth: a naive whole-array entry precompute DIFFERS from monolith.

    Confirms the parity test is not vacuously passing — the naive approach the
    design warned against genuinely produces a different result on some input,
    so matching the monolith is a real constraint.
    """
    close, high, low = _make_4h_series(seed=13)
    p = _cfg_params()
    cost = 0.0008
    sma20 = _sma_rolling(close, 20)
    sma50 = _sma_rolling(close, 50)
    sma100 = _sma_rolling(close, 100)
    sma200 = _sma_rolling(close, 200)
    mom = compute_momentum(close, 8)

    # Naive: count clusters over the WHOLE array ignoring trade windows, mark an
    # entry wherever count>=min_duration & breakout & momentum, then take trades
    # greedily. This is what the design says NOT to do.
    n = len(close)
    naive_entries = []
    count = 0
    for i in range(n):
        if np.isnan(sma200[i]) or close[i] <= 0:
            count = 0
            continue
        ma_max = max(sma20[i], sma50[i], sma100[i], sma200[i])
        ma_min = min(sma20[i], sma50[i], sma100[i], sma200[i])
        if (ma_max - ma_min) / close[i] > p["cluster_pct"]:
            count = 0
            continue
        count += 1  # NEVER frozen/reset by trades — the bug
        if count >= p["min_duration"] and not (ma_min <= close[i] <= ma_max) and abs(mom[i]) >= p["min_momentum"]:
            naive_entries.append(i)

    _mp, me, _mx = _run_monolith(close, high, low, p, cost)
    # The naive entry-bar set should NOT equal the monolith's entry bars.
    assert set(naive_entries) != set(int(x) for x in me), (
        "naive precompute matched monolith — fixture doesn't exercise the coupling"
    )


# ── Directional (VR) kernel bit-parity vs sim_directional_trailing ────────────


def _make_directional_signals(n: int = 4000, seed: int = 7) -> tuple[np.ndarray, ...]:
    """Deterministic close/high/low + per-bar VR sig/dire arrays (parity input)."""
    rng = np.random.RandomState(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.006, n))
    high = close * (1.0 + np.abs(rng.normal(0, 0.007, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.007, n)))
    # ~8% of bars carry a directional signal, direction ±1.
    sig = (rng.rand(n) < 0.08).astype(np.int64)
    dire = np.where(rng.rand(n) < 0.5, 1, -1).astype(np.int64) * sig
    return close, high, low, sig, dire


@pytest.mark.parametrize("seed", [7, 1, 2, 42])
def test_directional_kernel_bit_parity(seed):
    """directional_conditions + sim_drive == sim_directional_trailing."""
    from quantbox.backtest.sim_core import directional_conditions
    from quantbox.backtest.trade_sim import sim_directional_trailing

    close, high, low, sig, dire = _make_directional_signals(seed=seed)
    sl, tp, max_hold, cost, tr_act, tr_dist = 0.02, 0.20, 8, 0.0008, 0.02, 0.01

    mp, me, mx = sim_directional_trailing(close, high, low, sig, dire, sl, tp, max_hold, cost, tr_act, tr_dist)

    bar_kind, entry_ok, direction = directional_conditions(close, sig, dire)
    sp, se, sx, _sd = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,
        sl,
        tp,
        tr_act,
        tr_dist,
        max_hold,
        cost,
        0,
        2,  # min_dur=1, max_dur=0, warmup=0, cap_divisor=2
    )

    assert len(sp) == len(mp), f"count differs: split={len(sp)} monolith={len(mp)}"
    assert len(mp) > 0, "no directional trades — parity check would be vacuous"
    np.testing.assert_array_equal(se, me)
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_allclose(sp, mp, rtol=0, atol=1e-12)


# ── Donchian kernel bit-parity vs sim_donchian_trailing ──────────────────────


def _make_donchian_series(n: int = 4000, seed: int = 7) -> tuple[np.ndarray, ...]:
    """Deterministic OHLCV with breakout jumps + volume spikes (parity input)."""
    rng = np.random.RandomState(seed)
    steps = rng.normal(0, 0.004, n)
    jump_idx = np.where(rng.rand(n) < 0.04)[0]
    steps[jump_idx] += rng.choice([-1.0, 1.0], size=len(jump_idx)) * 0.05
    close = 100.0 * np.cumprod(1.0 + steps)
    high = close * (1.0 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.uniform(1e6, 3e6, n)
    volume[jump_idx] *= 4.0  # spikes coincide with breakouts
    return close, high, low, volume


@pytest.mark.parametrize("seed", [7, 1, 2, 42])
def test_donchian_kernel_bit_parity(seed):
    """donchian_conditions + sim_drive == sim_donchian_trailing (incl cooldown)."""
    from quantbox.backtest.sim_core import donchian_conditions
    from quantbox.backtest.trade_sim import sim_donchian_trailing

    close, high, low, volume = _make_donchian_series(seed=seed)
    lookback, vol_sma_period, vol_mult, breakout_pct, cooldown_bars = 50, 20, 2.5, 0.02, 3
    sl, tp, tr_act, tr_dist, max_hold, cost, warmup = 0.015, 0.20, 0.02, 0.008, 6, 0.0008, 50

    mp, me, mx = sim_donchian_trailing(
        close,
        high,
        low,
        volume,
        lookback,
        vol_sma_period,
        vol_mult,
        breakout_pct,
        cooldown_bars,
        sl,
        tp,
        tr_act,
        tr_dist,
        max_hold,
        cost,
        warmup,
    )

    bar_kind, entry_ok, direction = donchian_conditions(
        close, high, low, volume, lookback, vol_sma_period, vol_mult, breakout_pct
    )
    sp, se, sx, _sd = sim_drive(
        close,
        high,
        low,
        bar_kind,
        entry_ok,
        direction,
        1,
        0,
        sl,
        tp,
        tr_act,
        tr_dist,
        max_hold,
        cost,
        warmup,
        2,
        cooldown_bars,
    )

    assert len(sp) == len(mp), f"count differs: split={len(sp)} monolith={len(mp)}"
    assert len(mp) > 0, "no donchian trades — parity check would be vacuous"
    np.testing.assert_array_equal(se, me)
    np.testing.assert_array_equal(sx, mx)
    np.testing.assert_allclose(sp, mp, rtol=0, atol=1e-12)


# ── Level 3: end-to-end parity on real data ──────────────────────────────────


def _load_real_15m(n_symbols: int = 6) -> dict:
    """Load a few real symbols from cache into the data_15m dict format."""
    files = sorted(f for f in glob.glob(os.path.join(CACHE_DIR, "*_15m_2190d_*.npz")) if "numtrades" not in f)
    data: dict = {}
    for f in files:
        sym = os.path.basename(f).split("_")[0]
        if sym in data:
            continue
        z = np.load(f, allow_pickle=False)
        if not {"close", "high", "low"}.issubset(set(z.files)):
            continue
        c = z["close"].astype(np.float64)
        if len(c) < 5000:
            continue
        data[sym] = {
            "close": c,
            "high": z["high"].astype(np.float64),
            "low": z["low"].astype(np.float64),
            "volume": z["volume"].astype(np.float64) if "volume" in z.files else np.ones_like(c),
        }
        if len(data) >= n_symbols:
            break
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


@pytest.mark.slow
def test_real_data_end_to_end_parity():
    """run_ma_cluster == collect_ma_cluster_trades on real cached symbols."""
    data_15m = _load_real_15m(n_symbols=6)
    if len(data_15m) < 3:
        pytest.skip("real cache data not available (need data/cache_extended npz)")

    golden = collect_ma_cluster_trades(dict(data_15m))
    candidate = run_ma_cluster(dict(data_15m))

    assert len(candidate) == len(golden), f"count differs: cand={len(candidate)} gold={len(golden)}"
    assert len(golden) > 0, "no trades on real data — parity check would be vacuous"
    for (gs, ge, gx, gp), (cs, ce, cx, cp) in zip(golden, candidate):
        assert gs == cs and ge == ce and gx == cx, f"trade index mismatch: {(gs, ge, gx)} vs {(cs, ce, cx)}"
        assert abs(gp - cp) <= 1e-12, f"pnl mismatch {gp} vs {cp}"


@pytest.mark.slow
def test_real_data_deterministic_golden():
    """Golden reference is deterministic across two runs (US-001)."""
    data_15m = _load_real_15m(n_symbols=6)
    if len(data_15m) < 3:
        pytest.skip("real cache data not available")
    a = collect_ma_cluster_trades(dict(data_15m))
    b = collect_ma_cluster_trades(dict(data_15m))
    assert a == b


# ── US-005: TF parameterization (per-TF without forking a sim) ────────────────


@pytest.mark.slow
def test_tf_param_4h_matches_default():
    """run_ma_cluster(ratio=16) == default path == collect_ma_cluster_trades."""
    data_15m = _load_real_15m(n_symbols=6)
    if len(data_15m) < 3:
        pytest.skip("real cache data not available")
    golden = collect_ma_cluster_trades(dict(data_15m))
    explicit_4h = run_ma_cluster(dict(data_15m), ratio=16)
    assert explicit_4h == golden


@pytest.mark.slow
@pytest.mark.parametrize("ratio,label", [(4, "1h"), (2, "30m"), (1, "15m")])
def test_tf_param_per_tf_no_fork(ratio, label):
    """r120's per-TF question is answered through sim_core with NO forked sim.

    Just routing a different ``ratio`` through the same run_ma_cluster /
    sim_drive path must yield a sane (non-negative count) trade set — the whole
    point of TF parameterization replacing forked per-TF sims.
    """
    data_15m = _load_real_15m(n_symbols=6)
    if len(data_15m) < 3:
        pytest.skip("real cache data not available")
    trades = run_ma_cluster(dict(data_15m), ratio=ratio)
    # Deterministic + well-formed (entry < exit, finite pnl).
    assert isinstance(trades, list)
    for sym, e, x, pnl in trades:
        assert x >= e
        assert np.isfinite(pnl)
    # Determinism across two runs at this TF.
    assert trades == run_ma_cluster(dict(data_15m), ratio=ratio)
