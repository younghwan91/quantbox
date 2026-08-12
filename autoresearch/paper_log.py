"""횡단면 리버설 페이퍼 기록기 — 매일 실행.

전방 검증의 실체. 매일 목표 포지션을 기록하고, HOLD_D일 전에 연 포지션의 **실현 손익**을
같은 파일에 남긴다. 이 기록이 쌓여야 최근 구간 효과크기(3일당 +52.2bp, t=1.51)가
진짜인지 판정할 수 있다. 지평 격자에서 3일이 최적이었고, 이 설정이면
**약 1.1년**에 t=2 판정이 난다(7일 설정은 6.7년이었다).

## 오늘 배운 것을 반영한다

`core/types.py`가 **가정한 비용 상수**를 저널에 적어서 6년간 비용을 검증할 수 없었다.
여기서는 그 실수를 반복하지 않는다 — 진입가·청산가·수수료 가정을 **분리해서** 적고,
나중에 실체결과 대조할 수 있게 한다.

## 출력

`data/paper_reversal.jsonl` 에 하루 한 줄:
  {date, equity, targets:[{symbol, side, notional, entry_px, rank_return}], realized:[...]}
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from reversal_live import HOLD_D, K_PER_LEG, LOOKBACK_D, _get, build_targets, lookback_returns, universe

LOG = pathlib.Path(__file__).resolve().parents[1] / "data" / "paper_reversal.jsonl"
MAKER_FEE = 2e-4  # 지정가 진입/청산 각 1회


def marks(symbols: list[str]) -> dict[str, float]:
    """현재 마크 가격 (전 종목 1회 호출)."""
    try:
        rows = _get("https://fapi.binance.com/fapi/v1/premiumIndex")
    except Exception:
        return {}
    want = set(symbols)
    return {r["symbol"]: float(r["markPrice"]) for r in rows if r["symbol"] in want}  # type: ignore[union-attr,index]


def load_history() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()]


def main() -> None:
    today = dt.date.today().isoformat()
    hist = load_history()
    if any(h["date"] == today for h in hist):
        print(f"{today}: 이미 기록됨 — 중복 실행 방지")
        return

    syms = universe()
    rets = lookback_returns(syms)
    px = marks(syms)
    equity = 1000.00

    targets = build_targets(rets, equity, K_PER_LEG)
    tgt_out = [
        {
            "symbol": t.symbol,
            "side": t.side,
            "notional": round(t.notional, 4),
            "entry_px": px.get(t.symbol),
            "rank_return": round(t.rank_return, 6),
        }
        for t in targets
        if px.get(t.symbol)
    ]

    # HOLD_D일 전 포지션의 실현 손익
    realized, summary = [], None
    cutoff = (dt.date.today() - dt.timedelta(days=HOLD_D)).isoformat()
    old = next((h for h in hist if h["date"] == cutoff), None)
    if old:
        rs = []
        for p in old["targets"]:
            now = px.get(p["symbol"])
            if not now or not p.get("entry_px"):
                continue
            raw = now / p["entry_px"] - 1.0
            pnl = (raw if p["side"] == "LONG" else -raw) - 2 * MAKER_FEE
            realized.append({"symbol": p["symbol"], "side": p["side"], "pnl": round(pnl, 6)})
            rs.append(pnl)
        if rs:
            a = np.array(rs)
            summary = {
                "opened": cutoff,
                "n": len(a),
                "mean_bp": round(float(a.mean()) * 1e4, 2),
                "median_bp": round(float(np.median(a)) * 1e4, 2),
                "win_rate": round(float((a > 0).mean()), 4),
                "portfolio_pnl_usd": round(float(a.mean()) * equity, 4),
            }

    rec = {
        "date": today,
        "equity": equity,
        "params": {"k": K_PER_LEG, "lookback_d": LOOKBACK_D, "hold_d": HOLD_D, "maker_fee": MAKER_FEE},
        "n_universe": len(rets),
        "targets": tgt_out,
        "realized": realized,
        "summary": summary,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"{today}: 목표 {len(tgt_out)}개 기록")
    if summary:
        print(
            f"  {cutoff} 개시분 실현: N={summary['n']} 평균 {summary['mean_bp']}bp "
            f"승률 {summary['win_rate'] * 100:.1f}% 포트 손익 ${summary['portfolio_pnl_usd']}"
        )
    # 누적
    done = [h["summary"] for h in hist if h.get("summary")]
    if summary:
        done = [*done, summary]
    if len(done) >= 2:
        m = np.array([d["mean_bp"] for d in done])
        print(
            f"  누적 {len(m)}회: 평균 {m.mean():.1f}bp/{HOLD_D}일  양수 {(m > 0).mean() * 100:.0f}%  "
            f"연환산 {m.mean() / 1e4 * (365 / HOLD_D) * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
