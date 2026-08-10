"""Generic backtest engine — bit-parity core (Phase 1a).

This module is the *shared* backtest core the refactor is built around
(design: ``docs/refactor/shared_core_harness_design.md``).  Its job is to
replace the three copy-pasted monolithic sims in ``trade_sim.py``
(``sim_ma_cluster_trailing`` / ``sim_directional_trailing`` /
``sim_donchian_trailing``) with ONE generic driver, without changing any
existing backtest number.

Design (verified against trade_sim.py):
  * The **exit** logic (peak_gain → trail_activate → SL-first → TP → max-hold)
    is byte-identical across all three sims.  It lives here exactly once
    inside :func:`sim_drive`.
  * The **entry** logic differs per strategy, but its *cross-bar state*
    (consecutive-count → min/max_duration gate → post-trade reset → ``i=eb+1``
    jump) is the same pattern everywhere.  That state is owned by
    :func:`sim_drive`.  Each strategy contributes only **stateless per-bar
    arrays** via a condition kernel.

Contract of a condition kernel — three per-bar arrays:
  * ``bar_kind`` (int8): ``0`` = RESET_SKIP (counter → 0, skip bar),
    ``1`` = HOLD_SKIP (skip bar, counter untouched),
    ``2`` = ACTIVE (counter += 1, bar is an entry candidate).
  * ``entry_ok`` (bool): extra per-bar gate checked only on ACTIVE candidate
    bars; failing it skips the bar **without** resetting the counter.
  * ``direction`` (int8): +1 long / -1 short at the entry bar.

**ADDITIVE ONLY** — this module never modifies ``trade_sim.py``.  Bit-parity
against the existing collectors is enforced by ``tests/test_sim_core_parity.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np

from quantbox.strategy.vol_straddle.signals import check_ma_cluster

if TYPE_CHECKING:
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# Bar-kind enum (kept as plain ints for numba nopython).
KIND_RESET_SKIP = 0
KIND_HOLD_SKIP = 1
KIND_ACTIVE = 2


# ── Generic driver ───────────────────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def sim_drive(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    bar_kind: np.ndarray,  # int8: 0=RESET_SKIP, 1=HOLD_SKIP, 2=ACTIVE
    entry_ok: np.ndarray,  # bool: extra entry gate on ACTIVE candidate bars
    direction: np.ndarray,  # int8: +1/-1 at entry bar
    min_duration: int,
    max_duration: int,
    sl: float,
    tp: float,
    trail_activate: float,
    trail_distance: float,
    max_hold: int,
    cost: float,
    warmup: int,
    cap_divisor: int = 4,
    cooldown_bars: int = 0,
    trail_peak_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generic entry-episode + trailing-exit driver (strategy/TF agnostic).

    Owns ALL cross-bar state so per-strategy condition kernels stay stateless.
    Reproduces the monolithic sims' scan semantics exactly:
      - consecutive ACTIVE bars increment a counter; RESET_SKIP zeroes it,
      - entry fires when counter in [min_duration, max_duration] and entry_ok,
      - after a trade the counter resets to 0 and the scan jumps to ``eb+1``
        (in-trade bars are never re-scanned — entry state is frozen).

    The exit block is the shared trailing-stop logic (G5 parity: once trailing
    is active the trail level is tighter than SL, so it is checked first).

    ``cap_divisor`` sets the output preallocation ``n // cap_divisor + 1`` to
    match each monolith's cap exactly (ma_cluster=4, directional/donchian=2) so
    truncation behaviour is bit-identical.

    ``cooldown_bars`` skips that many bars after each trade (Donchian's
    post-signal cooldown).  ``0`` disables it (ma_cluster / directional).

    ``trail_peak_lag`` (G6) controls the intra-bar ordering assumption.  Live
    trails tick-by-tick; bars only give high/low with unknown order.  ``0``
    (default, bit-identical to the monolith) raises the peak with this bar's
    high before testing the trail — i.e. assumes high came first, the optimistic
    bound.  ``1`` uses only the peak as of the previous bar, the pessimistic
    bound.  The spread between the two is the uncertainty this assumption
    creates; it cannot be removed with bar data.

    Returns:
        (pnl, entry_bars, exit_bars, dirs, exit_reasons) — 1-D arrays of length
        N_trades, bar indices in the input array's own bar units.  ``exit_reasons``
        holds ``EXIT_TRAIL`` / ``EXIT_SL`` / ``EXIT_TP`` / ``EXIT_MAXHOLD`` and is
        purely diagnostic: it records which branch ended each trade so the
        resolution-dependence of the exit model can be attributed per path.
    """
    n = len(close)
    cap = n // cap_divisor + 1
    pnl: np.ndarray = np.empty(cap, dtype=np.float64)
    ent: np.ndarray = np.empty(cap, dtype=np.int64)
    ext: np.ndarray = np.empty(cap, dtype=np.int64)
    drs: np.ndarray = np.empty(cap, dtype=np.int64)
    rsn: np.ndarray = np.empty(cap, dtype=np.int8)
    cnt = 0
    i = warmup
    count = 0
    cooldown = 0

    while i < n - 1:
        if cooldown > 0:
            cooldown -= 1
            i += 1
            continue

        k = bar_kind[i]
        if k == KIND_RESET_SKIP:
            count = 0
            i += 1
            continue
        if k == KIND_HOLD_SKIP:
            i += 1
            continue

        # k == KIND_ACTIVE
        count += 1
        if count < min_duration:
            i += 1
            continue
        if max_duration > 0 and count > max_duration:
            i += 1
            continue
        if not entry_ok[i]:
            i += 1
            continue

        d = direction[i]
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        trailing_active = False
        peak_gain = 0.0
        reason = EXIT_MAXHOLD

        for j in range(i + 1, end):
            if d == 1:
                prev_peak = peak_gain
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                # G6: 트레일 판정에 쓸 고점 — lag면 이 봉의 고점은 아직 못 봤다고 본다.
                eff_peak = prev_peak if trail_peak_lag > 0 else peak_gain

                if not trailing_active and eff_peak >= trail_activate:
                    trailing_active = True

                if trailing_active:
                    # G5 parity: trail first when active (tighter stop).
                    trail_level = entry * (1.0 + eff_peak - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(eff_peak - trail_distance, -sl)
                        eb = j
                        reason = EXIT_TRAIL
                        break
                else:
                    if low[j] <= entry * (1.0 - sl):
                        trade_pnl = -sl
                        eb = j
                        reason = EXIT_SL
                        break

                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    reason = EXIT_TP
                    break
            else:
                prev_peak = peak_gain
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain

                eff_peak = prev_peak if trail_peak_lag > 0 else peak_gain

                if not trailing_active and eff_peak >= trail_activate:
                    trailing_active = True

                if trailing_active:
                    trail_level = entry * (1.0 - eff_peak + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(eff_peak - trail_distance, -sl)
                        eb = j
                        reason = EXIT_TRAIL
                        break
                else:
                    if high[j] >= entry * (1.0 + sl):
                        trade_pnl = -sl
                        eb = j
                        reason = EXIT_SL
                        break

                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    reason = EXIT_TP
                    break
        else:
            # Max-hold exit at close.
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl[cnt] = trade_pnl - cost
            ent[cnt] = i
            ext[cnt] = eb
            drs[cnt] = d
            rsn[cnt] = reason
            cnt += 1

        # One signal per episode — reset after trade, resume past the exit.
        count = 0
        cooldown = cooldown_bars
        i = eb + 1

    return pnl[:cnt], ent[:cnt], ext[:cnt], drs[:cnt], rsn[:cnt]


#: 청산 사유 태그 — 어느 경로로 트레이드가 끝났는지 (해상도 아티팩트 분해용).
EXIT_TRAIL = 0
EXIT_SL = 1
EXIT_TP = 2
EXIT_MAXHOLD = 3


# ── MA cluster condition kernel (stateless) ──────────────────────────────────


@numba.jit(nopython=True, cache=True)
def ma_cluster_conditions(
    close: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_momentum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar stateless conditions for the MA-cluster squeeze strategy.

    Mirrors the inlined signal detection of ``sim_ma_cluster_trailing`` but with
    NO cross-bar state (that belongs to :func:`sim_drive`).  Order of the skip
    tests matches the monolith exactly so bit-parity holds:
      isnan(sma200) → RESET_SKIP; close<=0 → HOLD_SKIP (no reset);
      spread>cluster_pct → RESET_SKIP; else ACTIVE (in-cluster).
    ``entry_ok`` = breakout (price outside cluster zone) AND |momentum| gate.
    """
    n = len(close)
    bar_kind: np.ndarray = np.empty(n, dtype=np.int8)
    entry_ok: np.ndarray = np.zeros(n, dtype=np.bool_)
    direction: np.ndarray = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if np.isnan(sma200[i]):
            bar_kind[i] = KIND_RESET_SKIP
            continue
        if close[i] <= 0.0:
            bar_kind[i] = KIND_HOLD_SKIP
            continue

        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        spread = (ma_max - ma_min) / close[i]

        if spread > cluster_pct:
            bar_kind[i] = KIND_RESET_SKIP
            continue

        bar_kind[i] = KIND_ACTIVE

        # Entry gate (does not affect the counter): breakout + momentum.
        breakout = not (ma_min <= close[i] <= ma_max)
        mom_ok = abs(momentum[i]) >= min_momentum
        if breakout and mom_ok:
            entry_ok[i] = True
            direction[i] = 1 if momentum[i] > 0 else -1

    return bar_kind, entry_ok, direction


# ── Directional (VR) condition kernel (stateless) ────────────────────────────


@numba.jit(nopython=True, cache=True)
def directional_conditions(
    close: np.ndarray,
    sig: np.ndarray,
    dire: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar stateless conditions for VR directional entries.

    Mirrors ``sim_directional_trailing``: entry fires when ``sig != 0`` and
    ``dire != 0`` and ``close > 0`` — there is no consecutive-count, so every
    qualifying bar is an ACTIVE candidate and the driver runs with
    ``min_duration=1, max_duration=0`` (gate always open).  Non-qualifying bars
    are HOLD_SKIP (directional has no counter to reset).
    """
    n = len(close)
    bar_kind: np.ndarray = np.full(n, KIND_HOLD_SKIP, dtype=np.int8)
    entry_ok: np.ndarray = np.zeros(n, dtype=np.bool_)
    direction: np.ndarray = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if close[i] > 0.0 and sig[i] != 0 and dire[i] != 0:
            bar_kind[i] = KIND_ACTIVE
            entry_ok[i] = True
            direction[i] = 1 if dire[i] > 0 else -1

    return bar_kind, entry_ok, direction


# ── Donchian condition kernel (stateless) ────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def donchian_conditions(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    lookback: int,
    vol_sma_period: int,
    vol_mult: float,
    breakout_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar stateless conditions for Donchian breakout + volume spike.

    Mirrors the inlined detection of ``sim_donchian_trailing`` (channel from
    completed bars, volume-spike, breakout direction) with NO cross-bar state.
    The post-signal cooldown is owned by :func:`sim_drive` (``cooldown_bars``),
    so a qualifying breakout bar is simply ACTIVE with ``min_duration=1``.
    """
    n = len(close)
    bar_kind: np.ndarray = np.full(n, KIND_HOLD_SKIP, dtype=np.int8)
    entry_ok: np.ndarray = np.zeros(n, dtype=np.bool_)
    direction: np.ndarray = np.zeros(n, dtype=np.int8)

    for i in range(n):
        if i < lookback or i < vol_sma_period:
            continue

        # Donchian channel from completed bars (no lookahead).
        chan_high = high[i - lookback]
        chan_low = low[i - lookback]
        for k in range(i - lookback + 1, i):
            if high[k] > chan_high:
                chan_high = high[k]
            if low[k] < chan_low:
                chan_low = low[k]

        # Volume spike vs SMA of completed bars.
        vol_sum = 0.0
        for k in range(i - vol_sma_period, i):
            vol_sum += volume[k]
        vol_sma = vol_sum / vol_sma_period
        if not (volume[i] > vol_mult * vol_sma):
            continue

        d = 0
        if close[i] > chan_high and (close[i] - chan_high) / chan_high > breakout_pct:
            d = 1
        elif close[i] < chan_low and chan_low > 0.0 and (chan_low - close[i]) / chan_low > breakout_pct:
            d = -1
        if d == 0:
            continue

        bar_kind[i] = KIND_ACTIVE
        entry_ok[i] = True
        direction[i] = d

    return bar_kind, entry_ok, direction


# ── MA cluster adapter (parity with collect_ma_cluster_trades) ────────────────


def _rolling_sma(c: np.ndarray, period: int) -> np.ndarray:
    """SMA identical to the inline rolling-sum in ``collect_ma_cluster_trades``.

    Replicated (not imported) to keep this module additive; the arithmetic and
    accumulation order match the monolith so results are bit-identical.
    """
    n = len(c)
    out = np.full(n, np.nan)
    if n >= period:
        s = float(np.sum(c[:period]))
        out[period - 1] = s / period
        for idx in range(period, n):
            s += c[idx] - c[idx - period]
            out[idx] = s / period
    return out


def run_ma_cluster(
    data_15m: dict,
    data_4h: dict | None = None,
    *,
    cluster_pct: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_momentum: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    trail_activate: float | None = None,
    trail_distance: float | None = None,
    max_hold: int | None = None,
    cost: float | None = None,
    ratio: int = 16,
    cw_sizing: bool = False,
    return_directions: bool = False,
    trail_peak_lag: int = 0,
) -> list[tuple[str, int, int, float]] | tuple[list[tuple[str, int, int, float]], list[int]]:
    """MA-cluster trades via the generic engine — bit-parity with the monolith.

    Drop-in equivalent of ``trade_sim.collect_ma_cluster_trades``: same config
    defaults, same SMA/momentum precompute, same 4h→15m index conversion, but
    routed through :func:`ma_cluster_conditions` + :func:`sim_drive`.

    ``ratio`` selects the timeframe (16 = 4h, 4 = 1h, 2 = 30m, 1 = 15m).  At
    ``ratio=16`` the output matches ``collect_ma_cluster_trades`` exactly.

    ``cw_sizing`` (G8) scales each trade's PnL by live's conviction multiplier.
    It defaults to **off** here because this function's contract is bit-parity
    with the monolith; the live-faithful path (:func:`run_ma_cluster_live`)
    applies it by default instead.
    """
    from quantbox.backtest.trade_sim import build_resampled, get_trade_sim_config
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig
    from quantbox.strategy.vol_straddle.signals import compute_momentum

    cfg = VolStraddleConfig()
    _cluster_pct = cluster_pct if cluster_pct is not None else cfg.ma_cluster_pct
    _min_duration = min_duration if min_duration is not None else cfg.ma_cluster_min_duration
    _max_duration = max_duration if max_duration is not None else cfg.ma_cluster_max_duration
    _min_momentum = min_momentum if min_momentum is not None else cfg.ma_cluster_min_momentum
    _sl = sl if sl is not None else cfg.ma_cluster_sl
    _tp = tp if tp is not None else cfg.ma_cluster_tp
    _trail_activate = trail_activate if trail_activate is not None else cfg.ma_cluster_trail_activate
    _trail_distance = trail_distance if trail_distance is not None else cfg.ma_cluster_trail_distance
    _max_hold = max_hold if max_hold is not None else cfg.ma_cluster_max_hold_4h
    # G4: 호출 시점에 읽는다 — 모듈 전역은 import 시점에 굳어 설정 변경을 무시한다.
    _cost = cost if cost is not None else cfg.round_trip_cost
    WARMUP_4h = get_trade_sim_config()["tf_4h"][2]

    if data_4h is None:
        data_4h = build_resampled(data_15m, ratio)

    # Precompute SMAs / momentum on TF bars (idempotent; matches monolith).
    for _sym, d in data_4h.items():
        c = d["close"]
        for period, key in ((20, "sma_20"), (50, "sma_50"), (100, "sma_100"), (200, "sma_200")):
            if key not in d:
                d[key] = _rolling_sma(c, period)
        if "mom_8" not in d:
            d["mom_8"] = compute_momentum(c, 8)

    all_trades: list[tuple[str, int, int, float]] = []
    all_dirs: list[int] = []
    for sym in sorted(data_15m.keys()):
        d = data_4h.get(sym)
        if d is None or "sma_200" not in d or "mom_8" not in d:
            continue

        offset = int(data_15m[sym].get("_offset", 0))

        bar_kind, entry_ok, direction = ma_cluster_conditions(
            d["close"],
            d["sma_20"],
            d["sma_50"],
            d["sma_100"],
            d["sma_200"],
            d["mom_8"],
            _cluster_pct,
            _min_momentum,
        )
        p, e, x, dr, _rsn = sim_drive(
            d["close"],
            d["high"],
            d["low"],
            bar_kind,
            entry_ok,
            direction,
            _min_duration,
            _max_duration,
            _sl,
            _tp,
            _trail_activate,
            _trail_distance,
            _max_hold,
            _cost,
            WARMUP_4h,
            trail_peak_lag=trail_peak_lag,
        )

        for kk in range(len(p)):
            entry_bar = int(e[kk]) * ratio + offset
            exit_bar = int(x[kk]) * ratio + ratio - 1 + offset
            trade_pnl = float(p[kk])
            if cw_sizing:
                trade_pnl *= _cw_mult_at(
                    d, d["sma_20"], d["sma_50"], d["sma_100"], d["sma_200"], d["mom_8"], cfg, int(e[kk])
                )
            all_trades.append((sym, entry_bar, exit_bar, trade_pnl))
            all_dirs.append(int(dr[kk]))

    return (all_trades, all_dirs) if return_directions else all_trades


def _cw_mult_at(
    d: dict,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    cfg: "VolStraddleConfig",
    idx: int,
) -> float:
    """Conviction multiplier at one entry bar (G8) — CALLS live's own judgement.

    This does not re-derive live's CW conditions; it feeds the entry bar's state to
    ``strategy.vol_straddle.conviction``, the same functions ``strategy.py`` calls.
    Evaluating per entry bar (not per bar) keeps that call affordable — live only
    scores bars where a signal actually fired.

    Args:
        d: TF bar dict; needs ``close`` and ``volume``.
        sma20: SMA(20) on TF bars.
        sma50: SMA(50).
        sma100: SMA(100).
        sma200: SMA(200).
        momentum: Momentum array on TF bars.
        cfg: Config supplying the CW thresholds and multiplier table.
        idx: Entry bar index into the TF arrays.

    Returns:
        Nominal-size multiplier for that entry.
    """
    from quantbox.strategy.vol_straddle.conviction import (
        ma_cluster_cw_multiplier,
        ma_cluster_cw_score,
        ma_cluster_spread_pct,
    )

    spread = ma_cluster_spread_pct(
        float(d["close"][idx]), float(sma20[idx]), float(sma50[idx]), float(sma100[idx]), float(sma200[idx])
    )
    m = float(momentum[idx])
    # 라이브도 NaN 모멘텀을 0.0으로 정규화한다 (`strategy.py`).
    mom_val = 0.0 if np.isnan(m) else m
    score = ma_cluster_cw_score(spread, mom_val, d["volume"][: idx + 1], cfg)
    return ma_cluster_cw_multiplier(score, cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  Live-faithful path — CALLS the live signal function instead of re-deriving it
# ══════════════════════════════════════════════════════════════════════════════
#
#  Why this exists (measured 2026-07-17): the monolith's MA-cluster entry scan
#  fires **2.32x** the trades that live ``signals.check_ma_cluster`` fires — 2,184
#  of 3,836 backtest trades (56.9%) would never be taken live.
#
#  Live fires once per cluster episode: on signal it sets ``in_cooldown_arr[0]=1``
#  and clears it only when ``duration == 0`` (the cluster dissolves).  The monolith
#  only does ``cluster_count = 0`` after a trade, which does NOT require dissolution
#  — the count rebuilds inside the same episode and re-enters ``min_duration`` bars
#  later (worst observed: 17 entries in one episode).  Its own comment claims
#  "One signal per cluster episode"; the code does not implement it.  ``max_duration``
#  (zombie filter) is bypassed the same way.
#
#  The fix is not to re-derive live's rule a third time — that is the disease.
#  ``check_ma_cluster`` is ``@numba.jit(nopython=True)``, so a jitted driver can
#  call it directly.  Then backtest == live by construction.
#
#  NOTE: live's entry rule is **trade-independent** (cooldown reacts to cluster
#  dissolution, not to trade boundaries) — that is exactly why live is safe to call
#  at any cadence.  So precomputing signals over the whole array is correct HERE,
#  even though it breaks parity against the monolith (whose entry state is coupled
#  to trade boundaries — see the parity trap note above).  That coupling is itself
#  a symptom of the bug.


@numba.jit(nopython=True, cache=True)
def live_ma_cluster_signals(
    close: np.ndarray,
    sma20: np.ndarray,
    sma50: np.ndarray,
    sma100: np.ndarray,
    sma200: np.ndarray,
    momentum: np.ndarray,
    cluster_pct: float,
    min_duration: int,
    max_duration: int,
    min_momentum: float,
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay live ``check_ma_cluster`` bar by bar; return per-bar (entry_ok, direction).

    This CALLS the live function — it does not reimplement it.  The cooldown array
    persists across bars exactly like ``strategy.py``'s per-symbol
    ``self._ma_cluster_cooldown``.  Live evaluates on the last completed 4h bar every
    15m cycle; within one 4h bar the inputs don't change, so one call per 4h bar is
    equivalent.
    """
    n = len(close)
    entry_ok: np.ndarray = np.zeros(n, dtype=np.bool_)
    direction: np.ndarray = np.zeros(n, dtype=np.int8)
    cooldown: np.ndarray = np.zeros(1, dtype=np.int64)
    dur_out: np.ndarray = np.zeros(1, dtype=np.int64)

    for t in range(warmup, n):
        sig = check_ma_cluster(
            close[: t + 1],
            sma20[: t + 1],
            sma50[: t + 1],
            sma100[: t + 1],
            sma200[: t + 1],
            momentum[: t + 1],
            cluster_pct,
            min_duration,
            cooldown,
            dur_out,
            max_duration,
            min_momentum,
        )
        if sig != 0:
            entry_ok[t] = True
            direction[t] = sig
    return entry_ok, direction


def run_ma_cluster_live(
    data_15m: dict,
    data_4h: dict | None = None,
    *,
    cluster_pct: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_momentum: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    trail_activate: float | None = None,
    trail_distance: float | None = None,
    max_hold: int | None = None,
    cost: float | None = None,
    ratio: int = 16,
    cw_sizing: bool = True,
    return_directions: bool = False,
    trail_peak_lag: int = 0,
) -> list[tuple[str, int, int, float]] | tuple[list[tuple[str, int, int, float]], list[int]]:
    """MA-cluster trades using the **live** entry rule (``check_ma_cluster``).

    Same signature/output as :func:`run_ma_cluster`, but entries come from replaying
    the production signal function rather than the monolith's re-derived scan.  SMAs
    use production ``compute_sma`` (the monolith seeds its inline rolling sum with
    ``np.sum``, which diverges from live in float64 — real but ~1e-15).

    This will NOT match ``collect_ma_cluster_trades`` — that is the point: measured
    2026-07-17, the monolith takes 2.32x live's trades.  Use this to see what the
    strategy's edge actually is under live semantics.

    Exit handling reuses :func:`sim_drive` unchanged (no second copy of exit logic):
    ``bar_kind`` is ACTIVE everywhere with ``min_duration=1``, so the driver's
    counter never gates — the gate is ``entry_ok``, which already encodes live's
    duration/cooldown/breakout/momentum decision.  The driver's post-trade
    ``i = eb + 1`` jump reproduces live's inability to re-enter a symbol while a
    position is open.

    ``cw_sizing`` (G8) applies live's conviction-weighted sizing — each trade's
    PnL is scaled by ``ma_cluster_cw_multipliers[score]``.  It defaults to **on**
    because live runs with ``ma_cluster_cw_enabled=True``; without it the backtest
    sizes every trade at 1.0x while live sizes 0.5x-1.5x.  Pass ``False`` to see
    the unweighted edge.
    """
    from quantbox.backtest.trade_sim import build_resampled, get_trade_sim_config
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig
    from quantbox.strategy.vol_straddle.signals import compute_momentum, compute_sma

    cfg = VolStraddleConfig()
    _cluster_pct = cluster_pct if cluster_pct is not None else cfg.ma_cluster_pct
    _min_duration = min_duration if min_duration is not None else cfg.ma_cluster_min_duration
    _max_duration = max_duration if max_duration is not None else cfg.ma_cluster_max_duration
    _min_momentum = min_momentum if min_momentum is not None else cfg.ma_cluster_min_momentum
    _sl = sl if sl is not None else cfg.ma_cluster_sl
    _tp = tp if tp is not None else cfg.ma_cluster_tp
    _trail_activate = trail_activate if trail_activate is not None else cfg.ma_cluster_trail_activate
    _trail_distance = trail_distance if trail_distance is not None else cfg.ma_cluster_trail_distance
    _max_hold = max_hold if max_hold is not None else cfg.ma_cluster_max_hold_4h
    # G4: 호출 시점에 읽는다 — 모듈 전역은 import 시점에 굳어 설정 변경을 무시한다.
    _cost = cost if cost is not None else cfg.round_trip_cost
    WARMUP_4h = get_trade_sim_config()["tf_4h"][2]

    if data_4h is None:
        data_4h = build_resampled(data_15m, ratio)

    all_trades: list[tuple[str, int, int, float]] = []
    all_dirs: list[int] = []
    for sym in sorted(data_15m.keys()):
        d = data_4h.get(sym)
        if d is None:
            continue
        c = d["close"]
        if len(c) < WARMUP_4h + 2:
            continue

        # Live's own SMA/momentum implementations — not the monolith's inline copies.
        sma20 = compute_sma(c, 20)
        sma50 = compute_sma(c, 50)
        sma100 = compute_sma(c, 100)
        sma200 = compute_sma(c, 200)
        mom = compute_momentum(c, 8)

        entry_ok, direction = live_ma_cluster_signals(
            c,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            _cluster_pct,
            _min_duration,
            _max_duration,
            _min_momentum,
            WARMUP_4h,
        )

        bar_kind: np.ndarray = np.full(len(c), KIND_ACTIVE, dtype=np.int8)
        p, e, x, dr, _rsn = sim_drive(
            c,
            d["high"],
            d["low"],
            bar_kind,
            entry_ok,
            direction,
            1,
            0,  # min_duration=1 / max_duration=0 — gating already done by entry_ok
            _sl,
            _tp,
            _trail_activate,
            _trail_distance,
            _max_hold,
            _cost,
            WARMUP_4h,
            trail_peak_lag=trail_peak_lag,
        )

        offset = int(data_15m[sym].get("_offset", 0))
        for kk in range(len(p)):
            entry_bar = int(e[kk]) * ratio + offset
            exit_bar = int(x[kk]) * ratio + ratio - 1 + offset
            trade_pnl = float(p[kk])
            if cw_sizing:
                trade_pnl *= _cw_mult_at(d, sma20, sma50, sma100, sma200, mom, cfg, int(e[kk]))
            all_trades.append((sym, entry_bar, exit_bar, trade_pnl))
            all_dirs.append(int(dr[kk]))

    return (all_trades, all_dirs) if return_directions else all_trades


# ══════════════════════════════════════════════════════════════════════════════
#  G6 narrowing — resolve 4h exits on the 15m grid
# ══════════════════════════════════════════════════════════════════════════════
#
#  ``trail_peak_lag`` bounds the intra-bar ordering assumption but cannot remove
#  it: within one 4h bar we never learn whether the high or the low came first.
#  Measured on the full symbol set that assumption alone spans PF 1.618-2.078.
#
#  The band shrinks with bar resolution.  Entries stay on 4h (live evaluates the
#  MA cluster on completed 4h bars), but exits — SL, TP, trailing — are checked
#  against 15m bars, which is 16x finer and closer to live's tick-level brackets.
#  The residual uncertainty is then one 15m bar wide instead of one 4h bar.
#
#  This does NOT re-implement exits: the same ``sim_drive`` runs on the 15m grid.


def run_ma_cluster_live_fine_exits(
    data_15m: dict,
    data_4h: dict | None = None,
    *,
    cluster_pct: float | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
    min_momentum: float | None = None,
    sl: float | None = None,
    tp: float | None = None,
    trail_activate: float | None = None,
    trail_distance: float | None = None,
    max_hold: int | None = None,
    cost: float | None = None,
    ratio: int = 16,
    cw_sizing: bool = True,
    return_directions: bool = False,
    trail_peak_lag: int = 0,
    exit_ratio: int = 1,
    return_reasons: bool = False,
) -> (
    list[tuple[str, int, int, float]]
    | tuple[list[tuple[str, int, int, float]], list[int]]
    | tuple[list[tuple[str, int, int, float]], list[int], list[int]]
):
    """MA-cluster with live entries on 4h bars but exits resolved on finer bars.

    Entry selection is identical to :func:`run_ma_cluster_live` — the same
    ``check_ma_cluster`` replay on completed 4h bars.  Only the exit grid changes:
    a trade entered at the close of 4h bar ``e`` starts on the exit-grid bar whose
    close IS that 4h close, and its SL/TP/trail are then evaluated bar by bar at
    the finer resolution.

    ``exit_ratio`` is 15m bars per exit bar: 1 = 15m, 2 = 30m, 4 = 1h, 8 = 2h,
    16 = 4h (which reproduces :func:`run_ma_cluster_live`\'s exit behaviour).
    Sweeping it shows how much of the measured edge depends on bar aggregation.

    Exit logic is not duplicated — ``sim_drive`` is driven on the 15m arrays.

    Args:
        data_15m: 15m OHLCV dict keyed by symbol.
        data_4h: Pre-resampled 4h dict; built from ``data_15m`` when omitted.
        cluster_pct: MA cluster width. Defaults to config.
        min_duration: Min cluster duration in 4h bars. Defaults to config.
        max_duration: Zombie-cluster cutoff. Defaults to config.
        min_momentum: Momentum floor for entry. Defaults to config.
        sl: Stop-loss fraction. Defaults to config.
        tp: Take-profit fraction. Defaults to config.
        trail_activate: Gain at which trailing arms. Defaults to config.
        trail_distance: Trail give-back. Defaults to config.
        max_hold: Max hold in **4h** bars; converted to 15m internally.
        cost: Round-trip cost. Defaults to config.
        ratio: 15m bars per entry-TF bar (16 = 4h).
        cw_sizing: Apply live conviction-weighted sizing.
        return_directions: Also return per-trade ±1 directions.
        trail_peak_lag: Intra-bar ordering assumption, now on exit-grid bars.
        exit_ratio: 15m bars per exit bar. Must divide ``ratio``.
        return_reasons: Also return per-trade exit reason tags (``EXIT_*``).

    Returns:
        ``[(symbol, entry_bar_15m, exit_bar_15m, pnl)]`` — bar indices are in 15m
        units regardless of ``exit_ratio`` so results stay comparable and feed
        ``portfolio_sim`` unchanged. Plus a directions list when requested.

    Raises:
        ValueError: ``exit_ratio`` does not divide ``ratio``.
    """
    if exit_ratio < 1 or ratio % exit_ratio != 0:
        raise ValueError(f"exit_ratio must divide ratio ({ratio}), got {exit_ratio}")
    from quantbox.backtest.trade_sim import build_resampled, get_trade_sim_config
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig
    from quantbox.strategy.vol_straddle.signals import compute_momentum, compute_sma

    cfg = VolStraddleConfig()
    _cluster_pct = cluster_pct if cluster_pct is not None else cfg.ma_cluster_pct
    _min_duration = min_duration if min_duration is not None else cfg.ma_cluster_min_duration
    _max_duration = max_duration if max_duration is not None else cfg.ma_cluster_max_duration
    _min_momentum = min_momentum if min_momentum is not None else cfg.ma_cluster_min_momentum
    _sl = sl if sl is not None else cfg.ma_cluster_sl
    _tp = tp if tp is not None else cfg.ma_cluster_tp
    _trail_activate = trail_activate if trail_activate is not None else cfg.ma_cluster_trail_activate
    _trail_distance = trail_distance if trail_distance is not None else cfg.ma_cluster_trail_distance
    _max_hold_4h = max_hold if max_hold is not None else cfg.ma_cluster_max_hold_4h
    _cost = cost if cost is not None else cfg.round_trip_cost
    WARMUP_4h = get_trade_sim_config()["tf_4h"][2]

    if data_4h is None:
        data_4h = build_resampled(data_15m, ratio)
    #: 진입 4h 봉 하나가 청산 격자에서 차지하는 봉 수.
    per_entry = ratio // exit_ratio
    data_exit = data_15m if exit_ratio == 1 else build_resampled(data_15m, exit_ratio)

    all_trades: list[tuple[str, int, int, float]] = []
    all_dirs: list[int] = []
    all_reasons: list[int] = []
    for sym in sorted(data_15m.keys()):
        d4 = data_4h.get(sym)
        if d4 is None:
            continue
        c4 = d4["close"]
        if len(c4) < WARMUP_4h + 2:
            continue

        sma20 = compute_sma(c4, 20)
        sma50 = compute_sma(c4, 50)
        sma100 = compute_sma(c4, 100)
        sma200 = compute_sma(c4, 200)
        mom = compute_momentum(c4, 8)

        entry_ok_4h, direction_4h = live_ma_cluster_signals(
            c4,
            sma20,
            sma50,
            sma100,
            sma200,
            mom,
            _cluster_pct,
            _min_duration,
            _max_duration,
            _min_momentum,
            WARMUP_4h,
        )

        dx = data_exit.get(sym)
        if dx is None:
            continue
        cx = dx["close"]
        nx = len(cx)

        # 4h 봉 e의 종가 = 청산 격자 봉 (e+1)*per_entry - 1 의 종가. 진입을 그리로 옮긴다.
        entry_ok: np.ndarray = np.zeros(nx, dtype=np.bool_)
        direction: np.ndarray = np.zeros(nx, dtype=np.int8)
        idx4 = np.nonzero(entry_ok_4h)[0]
        idxx = (idx4 + 1) * per_entry - 1
        keep = idxx < nx
        entry_ok[idxx[keep]] = True
        direction[idxx[keep]] = direction_4h[idx4[keep]]

        bar_kind: np.ndarray = np.full(nx, KIND_ACTIVE, dtype=np.int8)
        p, e, x, dr, _rsn = sim_drive(
            cx,
            dx["high"],
            dx["low"],
            bar_kind,
            entry_ok,
            direction,
            1,
            0,  # gating already encoded in entry_ok
            _sl,
            _tp,
            _trail_activate,
            _trail_distance,
            _max_hold_4h * per_entry,
            _cost,
            WARMUP_4h * per_entry,
            trail_peak_lag=trail_peak_lag,
        )

        offset = int(data_15m[sym].get("_offset", 0))
        for kk in range(len(p)):
            ex = int(e[kk])
            trade_pnl = float(p[kk])
            if cw_sizing:
                # CW는 진입한 4h 봉의 상태로 판정한다 — 라이브와 같은 시점.
                trade_pnl *= _cw_mult_at(d4, sma20, sma50, sma100, sma200, mom, cfg, (ex + 1) // per_entry - 1)
            # 봉 인덱스는 항상 15m 단위로 환산해 내보낸다 (해상도 간 비교 가능하게).
            entry_bar = ex * exit_ratio + exit_ratio - 1 + offset
            exit_bar = int(x[kk]) * exit_ratio + exit_ratio - 1 + offset
            all_trades.append((sym, entry_bar, exit_bar, trade_pnl))
            all_dirs.append(int(dr[kk]))
            all_reasons.append(int(_rsn[kk]))

    if return_reasons:
        return all_trades, all_dirs, all_reasons
    return (all_trades, all_dirs) if return_directions else all_trades


def run_ma_cluster_live_15m_exits(data_15m: dict, data_4h: dict | None = None, **kwargs):
    """Backward-compatible alias: :func:`run_ma_cluster_live_fine_exits` at 15m.

    Args:
        data_15m: 15m OHLCV dict keyed by symbol.
        data_4h: Pre-resampled 4h dict; built when omitted.
        **kwargs: Forwarded unchanged; ``exit_ratio`` is pinned to 1.

    Returns:
        Same as :func:`run_ma_cluster_live_fine_exits`.
    """
    kwargs.pop("exit_ratio", None)
    return run_ma_cluster_live_fine_exits(data_15m, data_4h, exit_ratio=1, **kwargs)
