"""매크로 타이밍 알트 숏 — 바이낸스 실행기.

오늘 12개 계열을 검정해 **독립 표본에서 재현된 유일한 신호**.

## 왜 이것만 살아남았나

죽은 11개는 전부 **크립토 가격**에서 신호를 뽑았다. 그건 이미 가격에 반영돼 있어
거래소를 바꿔도 같은 결론이 났다(리버설: 원표본 t=3.33 → Bybit 0.75 → 현물 -0.33).

이 신호는 **미국 주식시장 구조**에서 온다. 크립토 참가자가 보지 않는 정보이므로
세 표본 모두에서 같은 크기로 살아남는다.

## 신호

  EQ_VOL = 미국 보통주 동일가중 지수의 20일 실현변동성.
  그날까지의 이력으로 만든 **확장 창 20% 분위** 아래면 진입 (룩어헤드 없음).

  해석: 저변동은 랠리가 끝난 국면이고 그 뒤 크립토가 크게 빠진다.

## 전략

  진입  EQ_VOL이 임계 아래인 날, 유동성 상위 알트를 동일가중 **숏**
  보유  20일
  비용  왕복 12.6bp (20일 보유라 회전이 낮아 민감하지 않다)

## 검증 상태 (확장창 임계, Newey-West lag=20)

  표본             진입   신호숏    t      무조건   초과     t
  바이낸스 무기한(원)  303   +6.62%  3.32   +1.67%  +4.94%  2.48
  바이낸스 현물       250   +8.64%  3.93   +1.69%  +6.95%  3.16
  Bybit 무기한      174  +10.67%  3.66   +4.09%  +6.57%  2.25

세 표본 초과가 +4.9~7.0%로 크기까지 일관된다. 단 이 계열에서 27셀을 검정했으므로
Bonferroni 임계 t≈3.2이고, 현물만 그 선에 닿는다. **확정은 아니다.**
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.request

import numpy as np

DB = "/home/user/data/us_micro.duckdb"
MIN_HIST = 250
QUANTILE = 0.20
HOLD_D = 20
TOP_N = 40  # 숏할 종목 수
MIN_DOLLAR_VOL = 1e7
FAPI = "https://fapi.binance.com/fapi/v1"
LOG = pathlib.Path(__file__).resolve().parents[1] / "data" / "paper_macro.jsonl"


def signal_today() -> tuple[bool, float, float]:
    """(진입여부, 오늘 EQ_VOL, 임계). macro_timing 의 정의를 그대로 쓴다."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from macro_timing import equity_macro

    macro = equity_macro(DB)
    items = sorted((d, m["EQ_VOL"]) for d, m in macro.items() if m.get("EQ_VOL") and np.isfinite(m["EQ_VOL"]))
    if len(items) < MIN_HIST + 1:
        return False, float("nan"), float("nan")
    hist = [v for _d, v in items[:-1]]
    today_v = items[-1][1]
    thr = float(np.quantile(hist, QUANTILE))
    return today_v <= thr, today_v, thr


def liquid_universe() -> list[str]:
    """유동성 상위 무기한. 24h 거래대금 기준."""
    with urllib.request.urlopen(f"{FAPI}/ticker/24hr", timeout=60) as r:
        rows = json.load(r)
    out = [
        (x["symbol"], float(x["quoteVolume"]))
        for x in rows
        if x["symbol"].endswith("USDT") and float(x["quoteVolume"]) >= MIN_DOLLAR_VOL
    ]
    out.sort(key=lambda kv: -kv[1])
    return [s for s, _v in out[:TOP_N]]


def main() -> None:
    today = dt.date.today().isoformat()
    hist = [json.loads(x) for x in LOG.read_text().splitlines()] if LOG.exists() else []
    if any(h["date"] == today for h in hist):
        print(f"{today}: 이미 기록됨")
        return

    on, v, thr = signal_today()
    print(f"{today}  EQ_VOL {v:.5f}  임계 {thr:.5f}  → {'진입(알트 숏)' if on else '대기'}")

    targets = []
    if on:
        syms = liquid_universe()
        equity = 1000.00
        per = equity / max(len(syms), 1)
        targets = [{"symbol": s, "side": "SHORT", "notional": round(per, 4)} for s in syms]
        print(f"  숏 {len(targets)}종목 × ${per:.2f}  보유 {HOLD_D}일")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(
            json.dumps(
                {"date": today, "eq_vol": v, "threshold": thr, "entry": on, "targets": targets, "hold_d": HOLD_D}
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
