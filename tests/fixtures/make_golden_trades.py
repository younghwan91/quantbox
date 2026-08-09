"""골든 트레이드 스냅샷 생성기 — 현행 엔진의 출력을 박제한다.

왜 필요한가: `tests/test_sim_core_parity.py`는 모놀리식 sim과 sim_core를 맞대어
bit-parity를 본다. 그런데 모놀리식이 sim_core에 위임하게 되면 그 비교는 자기 자신과의
비교가 되어 의미를 잃는다. 그래서 **위임 전에** 현행 출력을 파일로 박제해 두고,
위임 후에는 그 박제본과 대조한다. 이게 "기존 백테스트 숫자가 안 바뀌었다"의 유일한 근거다.

재생성 (엔진 동작을 의도적으로 바꿀 때만):
    uv run python tests/fixtures/make_golden_trades.py --regen

의도적 변경이 아니라면 이 파일을 재생성하지 마라 — 테스트가 빨간 게 정답이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "golden_trades.npz"


def build() -> dict[str, np.ndarray]:
    from quantbox.backtest.trade_sim import (
        build_resampled,
        collect_donchian_trades,
        collect_ma_cluster_trades,
        collect_trades,
        precompute,
        precompute_all_vr,
    )
    from quantbox.research.harness import MarketData
    from quantbox.strategy.vol_straddle.config import VolStraddleConfig

    cfg = VolStraddleConfig()
    md = MarketData.load(cfg.symbol_whitelist)
    data_15m = md.data

    # collect_trades 는 vr_X_Y / mom_8 피처가 미리 계산돼 있어야 한다
    # (준비 순서는 research/active/r116_willy_nfi/r116_prep_baseline.py 와 동일).
    sp, lp = cfg.vr_short, cfg.vr_long
    precompute_all_vr([data_15m], [(sp, lp)])
    precompute(data_15m, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_1h, data_4h], [(sp, lp)])
    for ds in (data_1h, data_4h):
        precompute(ds, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    # 파라미터는 config가 유일 출처.
    sim_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "d_sl": cfg.d_sl,
        "tp": cfg.d_tp,
        "d_trail_activate": cfg.d_trail_activate,
        "d_trail_distance": cfg.d_trail_distance,
    }
    sets = {
        "ma_cluster": collect_ma_cluster_trades(data_15m),
        "donchian": collect_donchian_trades(data_15m),
        "directional": collect_trades(sim_cfg, data_15m, data_1h, data_4h, "D_mom8"),
    }

    out: dict[str, np.ndarray] = {}
    for name, trades in sets.items():
        trades = sorted(trades)
        out[f"{name}_sym"] = np.array([t[0] for t in trades], dtype="<U16")
        out[f"{name}_entry"] = np.array([t[1] for t in trades], dtype=np.int64)
        out[f"{name}_exit"] = np.array([t[2] for t in trades], dtype=np.int64)
        out[f"{name}_pnl"] = np.array([t[3] for t in trades], dtype=np.float64)
        print(f"  {name:12} {len(trades):6} trades")
    return out


def main() -> int:
    if "--regen" not in sys.argv:
        print("의도적 재생성만 허용한다. --regen 을 붙여라.")
        return 1
    data = build()
    np.savez_compressed(FIXTURE, **data)
    size_kb = FIXTURE.stat().st_size / 1024
    print(f"\n박제 완료: {FIXTURE.name}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
