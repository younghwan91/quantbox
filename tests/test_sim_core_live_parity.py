"""라이브 충실 경로 검증 — 백테스트가 라이브 규칙을 그대로 재현하는가.

`run_ma_cluster` (모놀리식 재현) 는 기존 백테스트 숫자와의 bit-parity를 지킨다.
`run_ma_cluster_live` 는 다른 목적이다: **라이브가 실제로 낼 시그널**을 재현한다.
둘은 일치하지 않으며, 그 불일치가 바로 이 모듈이 존재하는 이유다 (2026-07-17 실측:
모놀리식이 라이브의 2.32배 거래).

여기서 지켜야 할 계약:
  1. live 경로의 시그널 = 라이브 check_ma_cluster를 직접 리플레이한 것과 동일.
  2. 클러스터 에피소드당 최대 1 시그널 (라이브의 in_cooldown 의미).
  3. exit 로직은 sim_drive 하나만 쓴다 (사본 없음).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quantbox.backtest.sim_core import KIND_ACTIVE, live_ma_cluster_signals, run_ma_cluster_live
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import check_ma_cluster, compute_momentum, compute_sma

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"
WARMUP = 200


def _synthetic_4h(n: int = 1200, seed: int = 7) -> np.ndarray:
    """가격 시계열. 값이 아니라 상태기계 동작만 본다 — 실데이터 테스트는 아래 @slow."""
    rng = np.random.RandomState(seed)
    steps = rng.normal(0, 0.004, n)
    # 구간마다 변동성을 눌러 클러스터 에피소드가 생기도록 한다
    for start in range(100, n, 300):
        steps[start : start + 150] *= 0.15
    return 100.0 * np.exp(np.cumsum(steps))


def _live_replay(c: np.ndarray, cfg: VolStraddleConfig) -> list[int]:
    """라이브가 하는 그대로: 봉마다 check_ma_cluster 호출, 쿨다운은 영속."""
    sma = [compute_sma(c, p) for p in (20, 50, 100, 200)]
    mom = compute_momentum(c, 8)
    cooldown = np.zeros(1, dtype=np.int64)
    dur_out = np.zeros(1, dtype=np.int64)
    out = []
    for t in range(WARMUP, len(c)):
        sig = check_ma_cluster(
            c[: t + 1],
            sma[0][: t + 1],
            sma[1][: t + 1],
            sma[2][: t + 1],
            sma[3][: t + 1],
            mom[: t + 1],
            cfg.ma_cluster_pct,
            cfg.ma_cluster_min_duration,
            cooldown,
            dur_out,
            max_duration=cfg.ma_cluster_max_duration,
            min_momentum=cfg.ma_cluster_min_momentum,
        )
        if sig != 0:
            out.append((t, sig))
    return out


# ──────────────────────────────────────────────────────────────────────────
#  계약 1 — live 경로의 시그널은 라이브 함수 리플레이와 동일
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [7, 1, 2, 42])
def test_live_signals_match_direct_replay(seed):
    cfg = VolStraddleConfig()
    c = _synthetic_4h(seed=seed)
    sma = [compute_sma(c, p) for p in (20, 50, 100, 200)]
    mom = compute_momentum(c, 8)

    entry_ok, direction = live_ma_cluster_signals(
        c,
        sma[0],
        sma[1],
        sma[2],
        sma[3],
        mom,
        cfg.ma_cluster_pct,
        cfg.ma_cluster_min_duration,
        cfg.ma_cluster_max_duration,
        cfg.ma_cluster_min_momentum,
        WARMUP,
    )

    expected = _live_replay(c, cfg)
    got = [(int(t), int(direction[t])) for t in np.flatnonzero(entry_ok)]
    assert got == expected


def test_live_signals_fire_at_most_once_per_episode():
    """라이브의 핵심 의미: 에피소드당 1회. 모놀리식은 이걸 어긴다(최대 17회 관측)."""
    cfg = VolStraddleConfig()
    c = _synthetic_4h(seed=7)
    sma = [compute_sma(c, p) for p in (20, 50, 100, 200)]
    mom = compute_momentum(c, 8)

    entry_ok, _ = live_ma_cluster_signals(
        c,
        sma[0],
        sma[1],
        sma[2],
        sma[3],
        mom,
        cfg.ma_cluster_pct,
        cfg.ma_cluster_min_duration,
        cfg.ma_cluster_max_duration,
        cfg.ma_cluster_min_momentum,
        WARMUP,
    )

    ma_max = np.max(np.vstack(sma), axis=0)
    ma_min = np.min(np.vstack(sma), axis=0)
    with np.errstate(invalid="ignore"):
        spread = (ma_max - ma_min) / c
    in_cluster = (spread <= cfg.ma_cluster_pct) & ~np.isnan(sma[3]) & (c > 0)

    # 에피소드 = 연속 in_cluster 구간
    ep = np.full(len(c), -1, dtype=np.int64)
    cur, prev = -1, False
    for i in range(len(c)):
        if in_cluster[i]:
            if not prev:
                cur += 1
            ep[i] = cur
        prev = in_cluster[i]

    fired = np.flatnonzero(entry_ok)
    assert len(fired) > 0, "시그널이 하나도 없다 — 테스트가 공허하다"
    eps = [int(ep[t]) for t in fired]
    assert all(e >= 0 for e in eps), "클러스터 밖에서 발화했다"
    assert len(eps) == len(set(eps)), "한 에피소드에서 2회 이상 발화했다 — 라이브 의미 위반"


def test_signals_empty_before_warmup():
    cfg = VolStraddleConfig()
    c = _synthetic_4h()
    sma = [compute_sma(c, p) for p in (20, 50, 100, 200)]
    mom = compute_momentum(c, 8)
    entry_ok, _ = live_ma_cluster_signals(
        c,
        sma[0],
        sma[1],
        sma[2],
        sma[3],
        mom,
        cfg.ma_cluster_pct,
        cfg.ma_cluster_min_duration,
        cfg.ma_cluster_max_duration,
        cfg.ma_cluster_min_momentum,
        WARMUP,
    )
    assert not entry_ok[:WARMUP].any()


# ──────────────────────────────────────────────────────────────────────────
#  계약 3 — exit 로직 사본이 늘지 않았다
# ──────────────────────────────────────────────────────────────────────────


def test_no_second_exit_implementation():
    """sim_drive 하나만 존재해야 한다. live 경로도 그걸 재사용한다."""
    src = (ROOT / "quantbox" / "backtest" / "sim_core.py").read_text()
    assert src.count("def sim_drive(") == 1
    # trail_level은 sim_drive의 long/short 두 갈래에만 있어야 한다
    assert src.count("trail_level = entry") == 2


def test_live_path_uses_active_bar_kind():
    """게이팅은 entry_ok가 전담한다 — 드라이버 카운터는 관여하지 않는다."""
    assert KIND_ACTIVE == 2


# ──────────────────────────────────────────────────────────────────────────
#  실데이터 — 확정된 발산 규모를 회귀로 고정
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_real_data_live_path_diverges_from_monolith():
    """모놀리식이 라이브보다 유의하게 많이 거래한다 — 이 사실 자체를 고정한다.

    이 테스트가 깨지면 (a) 모놀리식이 고쳐졌거나 (b) 라이브 규칙이 바뀐 것이다.
    둘 다 사람이 판단할 일이다.
    """
    from quantbox.backtest.trade_sim import collect_ma_cluster_trades
    from quantbox.research.harness import MarketData

    cfg = VolStraddleConfig()
    md = MarketData.load(cfg.symbol_whitelist)

    mono = collect_ma_cluster_trades(md.data)
    live = run_ma_cluster_live(md.data)

    assert len(live) > 0 and len(mono) > 0
    ratio = len(mono) / len(live)
    assert ratio > 1.5, f"모놀리식/라이브 = {ratio:.2f}x — 발산이 사라졌다면 이 테스트를 재검토하라"

    # 라이브 경로는 심볼별로 에피소드당 1회이므로 트레이드가 훨씬 적다
    assert len(live) < len(mono)
