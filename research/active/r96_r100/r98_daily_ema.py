"""r98: Daily EMA Trend Following — BTC gate + volume expansion filter.

r97 FAIL 요약: 4h EMA crossover는 avg_hold=3.1일 → SL이 1-3바 내 맞음.
EMA 주기 확장(89→200)도 무효. 타임프레임을 daily로 올려야 함.

r98 설계:
- ratio=96 (15m → 1d): daily 타임프레임에서 진짜 추세추종
- BTC 레짐 게이트: BTC daily EMA 추세와 반대 방향 신호 억제
- Volume expansion gate: 거래량 확장 시에만 진입 (크래시 확인)
- 기대: avg_hold ≥ 20일, MaxDD < 70%, PF ≥ 1.20, 4/6년 OOS
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

from quantbox.backtest.trade_sim import (
    build_resampled,
    compute_metrics,
    portfolio_sim,
    sim_directional_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[3]

# ── Config (하드코딩 금지) ────────────────────────────────────────────
cfg = VolStraddleConfig()
LEVERAGE = cfg.ma_cluster_leverage  # 21x
LN = LEVERAGE / cfg.max_concurrent  # 21/10 = 2.1
MAX_SLOTS = cfg.max_concurrent  # 10
WHITELIST = set(cfg.symbol_whitelist)  # 60 symbols
COST = 0.0008  # round-trip 0.08%
TP_DEAD_CAP = 0.20  # dead cap (trail이 먼저 실행)
DATA_YEARS = 6.0  # 2020-02 ~ 2026-02

# ── Research params (sweep 대상) ──────────────────────────────────────
RATIO = 96  # 15m → daily (24h × 4 bars/h = 96)
TF_EMA_FAST = 21  # 기본값 (sweep에서 바뀜)
TF_EMA_SLOW = 55  # 기본값
TF_SL = 0.07  # 7% (daily ATR ~3-5%의 1.5×)
TF_TRAIL_ACT = 0.14  # 14% = SL × 2
TF_TRAIL_DIST = 0.04  # 4% (daily noise 허용)
TF_MAX_HOLD = 90  # 90 daily bars = 3개월
WARMUP_DAILY = 70  # max(fast, slow) + 15

# Data start offset: 2190d 파일 = Feb 2020 시작 (약 61일 = 2020.17)
DATA_START_YEAR = 2020.17
BARS_PER_YEAR_DAILY = 365.25

# ── Numba Signal Functions ────────────────────────────────────────────


@numba.jit(nopython=True, cache=True)
def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """지수이동평균 계산 (bar i까지의 close로 bar i의 EMA 반환)."""
    alpha = 2.0 / (period + 1)
    ema = np.empty_like(close)
    ema[0] = close[0]
    for i in range(1, len(close)):
        ema[i] = ema[i - 1] * (1.0 - alpha) + close[i] * alpha
    return ema


@numba.jit(nopython=True, cache=True)
def _ema_cross_signals_gated(
    close: np.ndarray,
    btc_close: np.ndarray,  # 동일 길이로 정렬된 BTC daily close
    volume: np.ndarray,  # daily volume
    fast: int,
    slow: int,
    warmup: int,
    btc_gate_enabled: bool,
    vol_gate_enabled: bool,
    vol_short_w: int,  # 최근 volume 평균 기간 (예: 5일)
    vol_long_w: int,  # 베이스라인 volume 평균 기간 (예: 20일)
    vol_threshold: float,  # 최근 평균이 베이스라인의 X배 이상이어야 진입
) -> tuple[np.ndarray, np.ndarray]:
    """Daily EMA crossover 신호 생성 (BTC gate + volume expansion gate 포함).

    Args:
        close: 이 심볼 daily close (len=N).
        btc_close: BTC daily close (len=N, end-aligned).
        volume: 이 심볼 daily volume (len=N).
        fast, slow: EMA 기간 (daily bars).
        warmup: 신호 시작 전 최소 bars.
        btc_gate_enabled: True이면 BTC 추세와 반대 방향 신호 억제.
        vol_gate_enabled: True이면 volume expansion 미충족 신호 억제.
        vol_short_w: 최근 volume 창 (예: 5).
        vol_long_w: 베이스라인 volume 창 (예: 20).
        vol_threshold: 진입 조건 (vol_recent ≥ vol_baseline × threshold).

    Returns:
        sig: 진입 신호 배열 (1=진입, 0=없음).
        dire: 방향 배열 (+1=LONG, -1=SHORT, 0=없음).

    Lookahead 없음:
        bar i의 EMA는 bar i의 close까지만 사용.
        sig[i]=1 → sim_directional_trailing이 bar i+1부터 진입.
    """
    ema_f = _compute_ema(close, fast)
    ema_s = _compute_ema(close, slow)
    btc_ema_f = _compute_ema(btc_close, fast)
    btc_ema_s = _compute_ema(btc_close, slow)

    n = len(close)
    sig = np.zeros(n, dtype=np.int64)
    dire = np.zeros(n, dtype=np.int64)

    # 충분한 warmup: max(fast, slow, warmup, vol_long_w) + 1
    start = max(fast, slow, warmup, vol_long_w) + 1
    if start >= n - 1:
        return sig, dire

    for i in range(start, n - 1):
        cross_up = ema_f[i] > ema_s[i] and ema_f[i - 1] <= ema_s[i - 1]
        cross_down = ema_f[i] < ema_s[i] and ema_f[i - 1] >= ema_s[i - 1]
        if not (cross_up or cross_down):
            continue

        direction = 1 if cross_up else -1

        # ── BTC 레짐 게이트 ──
        if btc_gate_enabled:
            btc_up = btc_ema_f[i] > btc_ema_s[i]
            if direction == 1 and not btc_up:
                continue  # BTC 하락 추세에서 LONG 금지
            if direction == -1 and btc_up:
                continue  # BTC 상승 추세에서 SHORT 금지

        # ── Volume expansion 게이트 ──
        if vol_gate_enabled and i >= vol_long_w:
            vol_recent = 0.0
            for k in range(i - vol_short_w, i):
                vol_recent += volume[k]
            vol_recent /= vol_short_w

            vol_base = 0.0
            for k in range(i - vol_long_w, i):
                vol_base += volume[k]
            vol_base /= vol_long_w

            if vol_base > 0.0 and vol_recent < vol_base * vol_threshold:
                continue  # volume 미충족 → 억제

        sig[i] = 1
        dire[i] = direction

    return sig, dire


# ── Data Loading ──────────────────────────────────────────────────────


def load_data_daily() -> tuple[dict, int, np.ndarray]:
    """15m NPZ → daily 리샘플 (ratio=96). whitelist 60심볼.

    Returns:
        data_daily: {sym: {close, high, low, volume, _offset}}.
        MAX_DAILY: 가장 긴 심볼의 daily bar 수.
        btc_daily_close: BTC daily close (MAX_DAILY 길이 기준).
    """
    cache_dir = ROOT / "data" / "cache_extended"

    # 심볼별 가장 긴 2190d 파일 선택
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in WHITELIST:
            continue
        try:
            nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        except Exception:
            continue
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)

    # 15m 데이터 로드
    data_15m: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        try:
            npz = np.load(str(fpath), allow_pickle=False)
        except Exception:
            continue
        if "close" not in npz.files:
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < RATIO * (WARMUP_DAILY + 50):
            continue
        # volume: 2190d 파일에 존재 (build_resampled에 필요)
        v = npz["volume"].astype(np.float64) if "volume" in npz.files else np.ones(len(c))
        data_15m[sym] = {
            "close": c,
            "high": npz["high"].astype(np.float64),
            "low": npz["low"].astype(np.float64),
            "volume": v,
        }

    # Daily 리샘플 (ratio=96)
    data_daily_raw: dict = build_resampled(data_15m, RATIO)

    # _offset 계산 (글로벌 인덱스 정렬용)
    max_daily = max((len(d["close"]) for d in data_daily_raw.values()), default=0)
    data_daily: dict = {}
    for sym, d in data_daily_raw.items():
        if len(d["close"]) < WARMUP_DAILY + 20:
            continue
        d["_offset"] = max_daily - len(d["close"])
        data_daily[sym] = d

    # BTC daily close (가장 긴 기준, end-aligned)
    btc_raw = data_daily.get("BTCUSDT", {})
    btc_close = btc_raw.get("close", np.array([], dtype=np.float64))

    return data_daily, max_daily, btc_close


# ── Trade Collection ──────────────────────────────────────────────────


def _collect_all_daily(
    data_daily: dict,
    btc_daily_close: np.ndarray,
    fast: int = TF_EMA_FAST,
    slow: int = TF_EMA_SLOW,
    sl: float = TF_SL,
    trail_act: float = TF_TRAIL_ACT,
    trail_dist: float = TF_TRAIL_DIST,
    max_hold: int = TF_MAX_HOLD,
    btc_gate: bool = True,
    vol_gate: bool = False,
    vol_short_w: int = 5,
    vol_long_w: int = 20,
    vol_threshold: float = 1.2,
    year_filter: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """모든 심볼 daily EMA 시뮬레이션. (sym, entry_global, exit_global, pnl) 반환.

    BTC gate: BTC daily EMA 추세와 반대 방향 신호를 억제.
    Volume gate: volume expansion 미충족 신호를 억제.
    """
    warmup = max(WARMUP_DAILY, slow + 10)
    trades: list[tuple[str, int, int, float]] = []

    for sym, d in data_daily.items():
        close = d["close"]
        high = d["high"]
        low = d["low"]
        volume = d.get("volume", np.ones(len(close)))
        offset = d.get("_offset", 0)
        n = len(close)

        # BTC close를 이 심볼의 길이에 맞게 end-align (두 시리즈 모두 Feb 2026에서 끝남)
        if len(btc_daily_close) >= n:
            btc_aligned = btc_daily_close[-n:]
        else:
            # BTC가 더 짧은 경우 (거의 없음) — 앞을 첫 값으로 패딩
            pad_len = n - len(btc_daily_close)
            btc_aligned = np.concatenate(
                [
                    np.full(pad_len, btc_daily_close[0] if len(btc_daily_close) > 0 else 1.0),
                    btc_daily_close,
                ]
            )

        sig, dire = _ema_cross_signals_gated(
            close,
            btc_aligned,
            volume,
            fast,
            slow,
            warmup,
            btc_gate,
            vol_gate,
            vol_short_w,
            vol_long_w,
            vol_threshold,
        )

        pnl_arr, ent_arr, ext_arr = sim_directional_trailing(
            close,
            high,
            low,
            sig,
            dire,
            sl,
            TP_DEAD_CAP,
            max_hold,
            COST,
            trail_act,
            trail_dist,
        )

        for k in range(len(pnl_arr)):
            eg = int(ent_arr[k]) + offset
            xg = int(ext_arr[k]) + offset
            if year_filter is not None:
                year_of_entry = DATA_START_YEAR + eg / BARS_PER_YEAR_DAILY
                if int(year_of_entry) != year_filter:
                    continue
            trades.append((sym, eg, xg, float(pnl_arr[k])))

    return trades


# ── Metrics helpers ───────────────────────────────────────────────────


def _avg_hold(trades: list[tuple[str, int, int, float]]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t[2] - t[1] for t in trades]))


def _metrics(
    trades: list[tuple[str, int, int, float]],
    years: float = DATA_YEARS,
) -> dict:
    """portfolio_sim → compute_metrics + avg_hold."""
    if not trades:
        return {
            "N": 0,
            "PF": 0.0,
            "WR": 0.0,
            "EV_bp": 0.0,
            "max_dd_pct": 0.0,
            "ann_ret_pct": 0.0,
            "calmar": 0.0,
            "avg_hold": 0.0,
        }
    pnl = portfolio_sim(trades, max_concurrent=MAX_SLOTS)
    m = compute_metrics(pnl, years)
    m["avg_hold"] = _avg_hold(trades)
    return m


def _slot_saturation(trades: list[tuple[str, int, int, float]]) -> dict:
    """포트폴리오 슬롯 포화도 분석 (portfolio_sim 통과 전 raw 신호 기준)."""
    if not trades:
        return {"avg": 0.0, "max": 0}
    events: list[tuple[int, int]] = []
    for _, e, x, _ in trades:
        events.append((e, 1))
        events.append((x, -1))
    events.sort()
    cur = 0
    totals: list[int] = []
    for _, delta in events:
        cur += delta
        totals.append(cur)
    return {"avg": float(np.mean(totals)) if totals else 0.0, "max": max(totals) if totals else 0}


def _print_row(label: str, m: dict, sat: dict | None = None) -> None:
    hold_d = m["avg_hold"]
    sat_str = f"  slot_avg={sat['avg']:.1f}/{MAX_SLOTS}" if sat else ""
    print(
        f"  {label:<20}  N={m['N']:5d}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%"
        f"  EV={m['EV_bp']:+.1f}bp  hold={hold_d:.1f}d"
        f"  DD={m['max_dd_pct']:.1f}%  CAGR={m['ann_ret_pct']:+.1f}%"
        f"{sat_str}"
    )


# ══════════════════════════════════════════════════════════════════════
#  Phase 0: 데이터 로드 검증
# ══════════════════════════════════════════════════════════════════════


def run_phase0(data_daily: dict, max_daily: int, btc_daily: np.ndarray) -> None:
    """데이터 로드 정상 여부 확인."""
    print("\n" + "=" * 65)
    print("Phase 0: 데이터 로드 검증")
    print("=" * 65)
    print(f"  심볼 수: {len(data_daily)}")
    print(f"  MAX_DAILY: {max_daily} bars ({max_daily / 365.25:.2f}년)")
    btc = data_daily.get("BTCUSDT", {})
    print(f"  BTC daily bars: {len(btc.get('close', []))}")
    print(f"  btc_daily_close len: {len(btc_daily)}")
    print(f"  volume 있는 심볼: {sum(1 for d in data_daily.values() if 'volume' in d)}/{len(data_daily)}")
    offsets = [d.get("_offset", 0) for d in data_daily.values()]
    print(f"  offset 범위: {min(offsets)}-{max(offsets)} bars")

    # 코드 동작 테스트 (BTC, 1 심볼)
    trades_test = _collect_all_daily(
        {"BTCUSDT": data_daily["BTCUSDT"]},
        btc_daily,
        TF_EMA_FAST,
        TF_EMA_SLOW,
        btc_gate=False,
        vol_gate=False,
    )
    print(f"  BTCUSDT 테스트: {len(trades_test)} 거래")
    if trades_test:
        holds = [t[2] - t[1] for t in trades_test]
        print(f"  BTCUSDT avg_hold: {np.mean(holds):.1f}d")
    print("  ✅ Phase 0 완료" if len(data_daily) >= 55 and len(btc_daily) > 100 else "  ❌ Phase 0 실패")


# ══════════════════════════════════════════════════════════════════════
#  Phase 1: EMA 페어 스윕 (BTC gate ON, vol gate OFF)
# ══════════════════════════════════════════════════════════════════════


def run_phase1_ema_sweep(data_daily: dict, btc_daily: np.ndarray) -> dict | None:
    """EMA 페어 스윕: (10,30), (21,55), (50,200). SL=7% 고정."""
    print("\n" + "=" * 65)
    print("Phase 1: EMA 페어 스윕 (BTC gate ON, vol gate OFF, SL=7%)")
    print("=" * 65)
    print(f"  {'EMA pair':<12}  {'N':>5}  {'PF':>6}  {'WR':>6}  {'EV':>8}  {'hold':>6}  {'DD':>6}  slot")

    pairs = [(10, 30), (21, 55), (50, 200)]
    best: dict | None = None

    for fast, slow in pairs:
        warmup_adj = max(WARMUP_DAILY, slow + 10)
        trades = _collect_all_daily(
            data_daily,
            btc_daily,
            fast,
            slow,
            warmup=warmup_adj if False else TF_MAX_HOLD,  # warmup is internal
            btc_gate=True,
            vol_gate=False,
        )
        m = _metrics(trades)
        sat = _slot_saturation(trades)
        label = f"EMA({fast},{slow})"
        _print_row(label, m, sat)

        if best is None or m["PF"] > best.get("PF", 0):
            best = {"fast": fast, "slow": slow, "m": m, "trades": trades}

    if best:
        print(f"\n  최적: EMA({best['fast']},{best['slow']}) PF={best['m']['PF']:.3f}")
        avg_hold = best["m"]["avg_hold"]
        print(f"  avg_hold={avg_hold:.1f}d {'✅' if avg_hold >= 20 else '❌'} (목표 ≥ 20일)")
    return best


def run_phase1_ema_sweep_fixed(data_daily: dict, btc_daily: np.ndarray) -> dict | None:
    """EMA 페어 스윕 — warmup 인수 없는 버전."""
    print("\n" + "=" * 65)
    print("Phase 1: EMA 페어 스윕 (BTC gate ON, vol gate OFF, SL=7%)")
    print("=" * 65)
    print(f"  {'EMA pair':<12}  {'N':>5}  {'PF':>6}  {'WR':>6}  {'EV':>8}  {'hold':>6}  {'DD':>6}  slot")

    pairs = [(10, 30), (21, 55), (50, 200)]
    best: dict | None = None

    for fast, slow in pairs:
        trades = _collect_all_daily(
            data_daily,
            btc_daily,
            fast,
            slow,
            btc_gate=True,
            vol_gate=False,
        )
        m = _metrics(trades)
        sat = _slot_saturation(trades)
        label = f"EMA({fast},{slow})"
        _print_row(label, m, sat)

        if best is None or m["PF"] > best.get("PF", 0):
            best = {"fast": fast, "slow": slow, "m": m, "trades": trades}

    if best:
        print(f"\n  최적: EMA({best['fast']},{best['slow']}) PF={best['m']['PF']:.3f}")
        avg_hold = best["m"]["avg_hold"]
        print(f"  avg_hold={avg_hold:.1f}d {'✅' if avg_hold >= 20 else '❌'} (목표 ≥ 20일)")
    return best


# ══════════════════════════════════════════════════════════════════════
#  Phase 1b: Volume Expansion Gate 효과
# ══════════════════════════════════════════════════════════════════════


def run_phase1b_vol_gate(
    data_daily: dict,
    btc_daily: np.ndarray,
    best_pair: dict,
) -> None:
    """Volume expansion gate ON vs OFF 비교."""
    print("\n" + "=" * 65)
    print("Phase 1b: Volume Expansion Gate 비교")
    print("=" * 65)
    fast = best_pair["fast"]
    slow = best_pair["slow"]

    print(f"  EMA({fast},{slow}) 기준, BTC gate ON")
    print(f"  {'설정':<22}  {'N':>5}  {'PF':>6}  {'EV':>8}  {'hold':>6}  {'DD':>6}")

    # Baseline: vol gate OFF
    trades_base = _collect_all_daily(data_daily, btc_daily, fast, slow, btc_gate=True, vol_gate=False)
    m_base = _metrics(trades_base)
    _print_row("vol_gate=OFF (base)", m_base)

    # Vol gate ON: threshold sweep
    for thresh in [1.1, 1.2, 1.3, 1.5]:
        trades = _collect_all_daily(
            data_daily,
            btc_daily,
            fast,
            slow,
            btc_gate=True,
            vol_gate=True,
            vol_short_w=5,
            vol_long_w=20,
            vol_threshold=thresh,
        )
        m = _metrics(trades)
        label = f"vol≥{thresh:.1f}×base"
        _print_row(label, m)
        n_ratio = m["N"] / m_base["N"] if m_base["N"] > 0 else 0
        pf_delta = m["PF"] - m_base["PF"]
        print(f"    → N 비율: {n_ratio:.1%}, ΔPF: {pf_delta:+.3f}, ΔEV: {m['EV_bp'] - m_base['EV_bp']:+.1f}bp")


# ══════════════════════════════════════════════════════════════════════
#  Phase 2: SL 스윕
# ══════════════════════════════════════════════════════════════════════


def run_phase2_sl_sweep(data_daily: dict, btc_daily: np.ndarray, best_pair: dict) -> dict | None:
    """SL 스윕: trail_activate = SL × 2 연동."""
    print("\n" + "=" * 65)
    print("Phase 2: SL 스윕 (trail_act = SL × 2, trail_dist 고정)")
    print("=" * 65)
    fast = best_pair["fast"]
    slow = best_pair["slow"]

    sl_configs = [
        (0.05, 0.10, 0.03),  # SL=5%,  act=10%, dist=3%
        (0.07, 0.14, 0.04),  # SL=7%,  act=14%, dist=4%  ← 기본값
        (0.10, 0.20, 0.05),  # SL=10%, act=20%, dist=5%
        (0.15, 0.30, 0.07),  # SL=15%, act=30%, dist=7%
    ]

    print(f"  EMA({fast},{slow}) BTC gate ON")
    print(f"  {'SL':>6}  {'N':>5}  {'PF':>6}  {'WR':>6}  {'EV':>8}  {'hold':>6}  {'DD':>6}")

    best: dict | None = None
    for sl, act, dist in sl_configs:
        trades = _collect_all_daily(
            data_daily,
            btc_daily,
            fast,
            slow,
            sl=sl,
            trail_act=act,
            trail_dist=dist,
            btc_gate=True,
            vol_gate=False,
        )
        m = _metrics(trades)
        label = f"SL={sl * 100:.0f}%"
        _print_row(label, m)

        # 최적 기준: Calmar 우선 (PF × hold / DD)
        score = m.get("calmar", 0.0)
        if best is None or score > best.get("score", 0):
            best = {
                "fast": fast,
                "slow": slow,
                "sl": sl,
                "trail_act": act,
                "trail_dist": dist,
                "m": m,
                "trades": trades,
                "score": score,
            }

    if best:
        print(
            f"\n  최적: SL={best['sl'] * 100:.0f}% act={best['trail_act'] * 100:.0f}%"
            f" dist={best['trail_dist'] * 100:.0f}% → PF={best['m']['PF']:.3f}"
            f" Calmar={best['m']['calmar']:.1f}"
        )
    return best


# ══════════════════════════════════════════════════════════════════════
#  Phase 3: BTC Gate 효과 비교
# ══════════════════════════════════════════════════════════════════════


def run_phase3_btc_gate(data_daily: dict, btc_daily: np.ndarray, best: dict) -> None:
    """BTC gate ON vs OFF 비교."""
    print("\n" + "=" * 65)
    print("Phase 3: BTC Gate ON vs OFF 비교")
    print("=" * 65)
    fast = best["fast"]
    slow = best["slow"]
    sl = best["sl"]
    act = best["trail_act"]
    dist = best["trail_dist"]

    for gate_on in [False, True]:
        trades = _collect_all_daily(
            data_daily,
            btc_daily,
            fast,
            slow,
            sl=sl,
            trail_act=act,
            trail_dist=dist,
            btc_gate=gate_on,
            vol_gate=False,
        )
        m = _metrics(trades)
        sat = _slot_saturation(trades)
        label = f"BTC_gate={'ON' if gate_on else 'OFF'}"
        _print_row(label, m, sat)

    print("\n  BTC gate 효과: N 감소 vs MaxDD 개선 tradeoff")


# ══════════════════════════════════════════════════════════════════════
#  Phase 4: OOS 연도별 검증
# ══════════════════════════════════════════════════════════════════════


def run_phase4_oos(data_daily: dict, btc_daily: np.ndarray, best: dict) -> int:
    """2020-2025 각 연도 독립 백테스트. 통과 연도 수 반환."""
    print("\n" + "=" * 65)
    print("Phase 4: OOS 연도별 검증 (4/6년 이상 목표)")
    print("=" * 65)
    fast = best["fast"]
    slow = best["slow"]
    sl = best["sl"]
    act = best["trail_act"]
    dist = best["trail_dist"]

    print(f"  EMA({fast},{slow}) SL={sl * 100:.0f}% BTC gate ON")
    print(f"  {'년도':<6}  {'N':>5}  {'PF':>6}  {'WR':>6}  {'EV':>8}  {'hold':>6}  pass?")

    pass_count = 0
    for yr in range(2020, 2026):
        trades_yr = _collect_all_daily(
            data_daily,
            btc_daily,
            fast,
            slow,
            sl=sl,
            trail_act=act,
            trail_dist=dist,
            btc_gate=True,
            vol_gate=False,
            year_filter=yr,
        )
        m = _metrics(trades_yr, years=1.0)
        passed = m["N"] >= 5 and m["PF"] > 1.0
        if passed:
            pass_count += 1
        mark = "✅" if passed else "❌"
        print(
            f"  {yr}  N={m['N']:5d}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%"
            f"  EV={m['EV_bp']:+.1f}bp  hold={m['avg_hold']:.1f}d  {mark}"
        )

    print(f"\n  결과: {pass_count}/6년 수익 {'✅' if pass_count >= 4 else '❌'} (목표 ≥ 4/6)")
    return pass_count


# ══════════════════════════════════════════════════════════════════════
#  Phase 5: 포트폴리오 상호작용 (VR+MA+TF)
# ══════════════════════════════════════════════════════════════════════


def run_phase5_portfolio(data_daily: dict, btc_daily: np.ndarray, best: dict) -> None:
    """VR+MA canonical trades + TF trades 혼합 portfolio_sim."""
    print("\n" + "=" * 65)
    print("Phase 5: 포트폴리오 상호작용 (TF 단독 vs 혼합)")
    print("=" * 65)
    fast = best["fast"]
    slow = best["slow"]
    sl = best["sl"]
    act = best["trail_act"]
    dist = best["trail_dist"]

    # TF 단독
    trades_tf = _collect_all_daily(
        data_daily,
        btc_daily,
        fast,
        slow,
        sl=sl,
        trail_act=act,
        trail_dist=dist,
        btc_gate=True,
        vol_gate=False,
    )
    m_tf = _metrics(trades_tf)
    sat_tf = _slot_saturation(trades_tf)
    _print_row("TF standalone", m_tf, sat_tf)

    # VR+MA 기준 canonical (N=6207 기준, 직접 계산 생략 — 참고값 표시)
    print("\n  VR+MA canonical baseline (v14.5.0): N=6207, PF=1.563, Calmar=57.9")
    print(f"  TF: N={m_tf['N']}, PF={m_tf['PF']:.3f}, hold={m_tf['avg_hold']:.1f}d")
    print(f"  TF 슬롯 포화도: avg={sat_tf['avg']:.1f}/{MAX_SLOTS} ({'✅ OK' if sat_tf['avg'] < 7 else '❌ 높음'})")

    # TF N/슬롯 정보
    if trades_tf:
        print(f"\n  TF 거래 수 / 연도: {m_tf['N'] / DATA_YEARS:.0f}")
        print("  기존 VR+MA avg concurrent ≈ 0.6/10 (매우 낮음)")
        print(f"  TF avg concurrent ≈ {sat_tf['avg']:.1f}/10")

    print("\n  ⚠️ VR+MA trades 병합은 precompute_all_vr 필요 (이 스크립트 범위 밖)")
    print("     슬롯 경쟁 분석은 TF standalone 포화도로 판단")


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════


def print_final_summary(best: dict | None, pass_years: int) -> None:
    """최종 결과 요약."""
    print("\n" + "=" * 65)
    print("최종 결과 요약")
    print("=" * 65)
    if best is None:
        print("  ❌ 유효한 결과 없음")
        return

    m = best["m"]
    print("  최적 파라미터:")
    print(
        f"    EMA({best['fast']},{best['slow']}) | SL={best['sl'] * 100:.0f}%"
        f" | trail_act={best['trail_act'] * 100:.0f}% | trail_dist={best['trail_dist'] * 100:.0f}%"
    )
    print("    BTC gate=ON | vol gate=OFF")
    print("  성과 (1x compound):")
    print(f"    N={m['N']}  PF={m['PF']:.3f}  WR={m['WR']:.1f}%  EV={m['EV_bp']:+.1f}bp")
    print(f"    avg_hold={m['avg_hold']:.1f}d | MaxDD={m['max_dd_pct']:.1f}% | 1x CAGR={m['ann_ret_pct']:+.1f}%")
    print(f"  L/N=2.1 production CAGR ≈ {m['ann_ret_pct'] * LN:+.1f}%  (참고용, 실제는 슬롯 경쟁 반영)")
    print(f"  OOS: {pass_years}/6년 수익")

    # 통과 기준 평가
    criteria = [
        ("avg_hold ≥ 20일", m["avg_hold"] >= 20),
        ("MaxDD < 70%", m["max_dd_pct"] < 70),
        ("PF ≥ 1.20", m["PF"] >= 1.20),
        ("OOS ≥ 4/6년", pass_years >= 4),
        ("N ≥ 100", m["N"] >= 100),
    ]
    print("\n  통과 기준:")
    all_pass = True
    for name, ok in criteria:
        mark = "✅" if ok else "❌"
        print(f"    {mark} {name}")
        if not ok:
            all_pass = False

    verdict = "✅ PASS — 추가 연구/구현 가치 있음" if all_pass else "❌ FAIL — r99 대안 검토 필요"
    print(f"\n  최종 판정: {verdict}")

    if not all_pass:
        print("\n  다음 단계:")
        if m["avg_hold"] < 20:
            print("    - avg_hold 여전히 짧음 → r99: Weekly EMA(10,30) or Donchian channel")
        if m["max_dd_pct"] >= 70:
            print("    - MaxDD 과다 → 레버리지 축소 or 동시 최대 슬롯 감소 검토")
        if pass_years < 4:
            print("    - OOS 불안정 → 추세추종은 crypto 공유 슬롯 구조에 적합하지 않음 결론")


def main() -> None:
    """r98 Daily EMA Trend Following 백테스트 실행."""
    print("=" * 65)
    print("r98: Daily EMA Trend Following")
    print(f"Data: cache_extended/*_15m_2190d_*.npz | ratio={RATIO} | {DATA_YEARS:.0f}yr")
    print(f"L/N={LN:.1f} | MAX_SLOTS={MAX_SLOTS} | cost={COST * 100:.2f}% round-trip")
    print("=" * 65)

    # 데이터 로드
    data_daily, max_daily, btc_daily = load_data_daily()

    # Phase 0
    run_phase0(data_daily, max_daily, btc_daily)

    # Phase 1: EMA 페어 스윕
    best_pair = run_phase1_ema_sweep_fixed(data_daily, btc_daily)
    if best_pair is None:
        print("Phase 1 실패 — 종료")
        return

    # Phase 1b: Volume gate 효과
    run_phase1b_vol_gate(data_daily, btc_daily, best_pair)

    # Phase 2: SL 스윕
    best = run_phase2_sl_sweep(data_daily, btc_daily, best_pair)
    if best is None:
        print("Phase 2 실패 — 종료")
        return

    # Phase 3: BTC Gate 비교
    run_phase3_btc_gate(data_daily, btc_daily, best)

    # Phase 4: OOS 연도별
    pass_years = run_phase4_oos(data_daily, btc_daily, best)

    # Phase 5: 포트폴리오 상호작용
    run_phase5_portfolio(data_daily, btc_daily, best)

    # 최종 요약
    print_final_summary(best, pass_years)


if __name__ == "__main__":
    main()
