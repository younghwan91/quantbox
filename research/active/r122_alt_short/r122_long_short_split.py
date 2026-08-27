"""r122 — "알트코인 숏 절대 금지"를 우리 데이터로 검증한다.

실전 트레이더 룰북(2026-08-27 사용자 제공)이 절대 금기로 못박은 항목:
    "알트코인은 세력의 의도적 숏스퀴즈와 비이성적 급등이 빈번하므로,
     선물 숏 포지션은 원금 청산의 주원인이다."

프로덕션 MA Cluster는 지금 양방향으로 들어간다(BTC 게이트는 **롱만** 막는다).
그래서 이 주장은 config 한 줄로 반증 가능한 형태다: 숏 트레이드의 EV가 음수이거나
롱보다 유의하게 나쁘면 `ma_cluster` 숏을 끄면 되고, 아니면 룰북 주장이 우리 계좌에서는
성립하지 않는 것이다.

방법:
  - 엔트리는 `run_ma_cluster_live` — 라이브 `check_ma_cluster`를 그대로 리플레이한다
    (모놀리스 재구현 경로는 라이브의 2.32배를 발화한다, project_ma_cluster_parity_break).
  - BTC 게이트는 걸지 않는다. 게이트는 롱만 차단하므로 방향 비교에 교락(confound)을
    만든다 — 소지(素地) 상태의 방향별 EV가 이 질문의 답이다.
  - 두 유니버스에서 각각 측정한다: 프로덕션 화이트리스트(60) / 전 종목(760, 폐지 포함).
    후자는 생존편향이 없어 "숏이 스퀴즈로 죽는다"는 주장에 더 가혹한 시험대다.

방향별 지표는 두 가지로 본다:
  raw   — portfolio_sim 없이 트레이드 그 자체의 EV. 신호의 순수 엣지.
  slot  — 해당 방향만 남겼을 때의 10슬롯 포트폴리오. 실제로 숏을 끄면 얻는 것.
"""

from __future__ import annotations

import numpy as np

from quantbox.research.harness import MarketData, experiment, report
from quantbox.research.harness.data import UNIVERSE_CACHE_DIR, UNIVERSE_PATTERN
from quantbox.backtest.sim_core import run_ma_cluster_live

#: (라벨, 캐시 디렉토리, 글로브 패턴, 심볼 제한)
UNIVERSES: list[tuple[str, str | None, str | None, bool]] = [
    ("production-60", None, None, True),
    ("universe-all", UNIVERSE_CACHE_DIR, UNIVERSE_PATTERN, False),
]


def _split(trades: list, dirs: list[int]) -> dict[str, list]:
    """방향별로 트레이드를 가른다. direction: +1 롱 / -1 숏."""
    out: dict[str, list] = {"both": list(trades), "long": [], "short": []}
    for t, d in zip(trades, dirs, strict=True):
        out["long" if d > 0 else "short"].append(t)
    return out


def _raw(trades: list) -> dict[str, float]:
    """슬롯 경합 없이 트레이드 자체의 EV/PF/t — 신호의 순수 엣지."""
    if not trades:
        return {"N": 0, "EV_bp": 0.0, "PF": 0.0, "t": 0.0, "WR": 0.0}
    p = np.array([t[3] for t in trades], dtype=np.float64)
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    return {
        "N": len(p),
        "EV_bp": round(float(p.mean()) * 1e4, 2),
        "PF": round(float(wins / losses) if losses > 0 else float("inf"), 3),
        "t": round(float(p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))) if len(p) > 1 else 0.0, 2),
        "WR": round(float((p > 0).mean()) * 100, 1),
    }


@experiment(name="r122_long_short_split")
def run(cfg):
    results: dict[str, dict] = {}
    for label, cache_dir, pattern, restrict in UNIVERSES:
        kw = {}
        if cache_dir:
            kw = {"cache_dir": cache_dir, "pattern": pattern}
        data = MarketData.load(cfg.symbol_whitelist if restrict else None, **kw)
        trades, dirs = run_ma_cluster_live(data.data, return_directions=True)
        by_dir = _split(trades, dirs)

        results[label] = {
            "n_symbols": len(data.data),
            "years": round(report.data_years(data), 2),
            "raw": {k: _raw(v) for k, v in by_dir.items()},
            "slot": {
                k: report.standard(v, data, cfg, n_boot=1000) if v else None for k, v in by_dir.items()
            },
            "gate": {k: report.robust_gate(v, data, cfg) if v else None for k, v in by_dir.items()},
        }
    return results


if __name__ == "__main__":
    res = run()
    for label, r in res.value.items():
        print(f"\n{'=' * 72}\n{label}  —  {r['n_symbols']} symbols, {r['years']}y\n{'=' * 72}")
        print("\n[raw — 슬롯 경합 없는 신호 자체의 엣지]")
        for k in ("both", "long", "short"):
            m = r["raw"][k]
            print(f"  {k:6s} N={m['N']:6d}  EV={m['EV_bp']:8.2f}bp  PF={m['PF']:6.3f}  t={m['t']:6.2f}  WR={m['WR']:5.1f}%")
        print("\n[slot — 해당 방향만 남긴 10슬롯 포트폴리오]")
        for k in ("both", "long", "short"):
            m = r["slot"][k]
            if m is None:
                print(f"  {k}: 트레이드 없음")
                continue
            print("\n" + report.render(m, title=f"  {label} / {k}"))
            g = r["gate"][k]
            print(f"  robust_gate: passed={g['passed']}  walkforward PF={g['walkforward_pf']}")
