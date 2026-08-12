"""횡단면 리버설 — 페이퍼 실행기.

오늘 여덟 계열을 검정해 유일하게 살아남은 전략. 백테스트는 여기서 끝이고,
필요한 것은 **새 데이터**다. 소진된 홀드아웃을 더 파면 노이즈를 맞추게 된다.

## 전략

  매일  전 USDT 무기한을 직전 **3일** 수익률로 순위
        하위 K개 롱 / 상위 K개 숏 (동일 명목, 순노출 0)
  보유  3일. 자본을 3등분해 하루 1/3씩 굴린다(중첩).
  체결  전량 지정가. 3일 창이라 여유가 있다 (실측 체결률 5분 98.5%).

## 파라미터 근거

  K=40  다리당 종목 수. 통계 최적(t_NW 4.64)이자 실행 경계다.
        5~10종목으로 집중하면 최근 구간이 -130%, -40%로 죽는다 — 분산이 본체.
        145종목이면 포지션이 3.25달러로 최소 명목가(5달러) 미달.
  LB=HOLD=3일. 지평 격자를 4h~168h로 훑은 결과:
        12h 이하는 **유의하게 음수**(12h t_NW -3.81) — 초단기는 모멘텀이다.
        3일이 리버설의 최적점이고, 7일보다 최근 t가 높고(1.51 vs 0.65)
        전방 검증 기간이 6.7년 → **1.1년**으로 짧다. 관측이 연 299 → 704회.

## 성과 (백테스트, 2,208 관측)

  전체    3일당 +55.3bp  t(NW) 3.33  연 67.3%
  최근1/4 3일당 +52.2bp  t(NW) 1.51  (미확정 — 전방 검증 1.1년이면 판정)

전방 검증 전에는 실계좌에 올리지 않는다.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import numpy as np

K_PER_LEG = 40
LOOKBACK_D = 3
HOLD_D = 3
MIN_NOTIONAL = 5.0
FAPI = "https://fapi.binance.com/fapi/v1"


@dataclass(frozen=True)
class Target:
    """목표 포지션 하나."""

    symbol: str
    side: str  # LONG / SHORT
    notional: float
    rank_return: float  # 순위 근거가 된 7일 수익률


def _get(url: str) -> object:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def universe() -> list[str]:
    """거래 중인 USDT 무기한 전체."""
    info = _get(f"{FAPI}/exchangeInfo")
    return sorted(
        s["symbol"]
        for s in info["symbols"]  # type: ignore[index]
        if s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
    )


def lookback_returns(symbols: list[str], days: int = LOOKBACK_D) -> dict[str, float]:
    """심볼별 직전 `days` 수익률. 일봉 종가 기준, 미래 미사용."""
    out: dict[str, float] = {}
    need = days + 1
    for s in symbols:
        try:
            kl = _get(f"{FAPI}/klines?symbol={s}&interval=1d&limit={need}")
        except Exception:
            continue
        if not isinstance(kl, list) or len(kl) < need:
            continue
        first, last = float(kl[0][4]), float(kl[-1][4])
        if first > 0:
            out[s] = last / first - 1.0
    return out


def build_targets(rets: dict[str, float], equity: float, k: int = K_PER_LEG) -> list[Target]:
    """순위 → 목표 포지션. 롱숏 명목 균형, 순노출 0.

    Args:
        rets: 심볼별 순위 수익률.
        equity: 배분할 자기자본 (USD).
        k: 다리당 종목 수.

    Returns:
        목표 포지션 목록. 최소 명목가 미달이면 빈 목록.
    """
    if len(rets) < 2 * k + 10:
        return []
    ranked = sorted(rets.items(), key=lambda kv: kv[1])
    longs, shorts = ranked[:k], ranked[-k:]

    per = equity / (2 * k)
    if per < MIN_NOTIONAL:
        return []

    return [Target(s, "LONG", per, r) for s, r in longs] + [Target(s, "SHORT", per, r) for s, r in shorts]


def main() -> None:
    syms = universe()
    print(f"# USDT 무기한 (거래중) {len(syms)}종목")
    rets = lookback_returns(syms)
    print(f"# {LOOKBACK_D}일 수익률 확보 {len(rets)}종목")

    equity = 1000.00  # 현재 계좌 (페이퍼 기준선)
    targets = build_targets(rets, equity)
    if not targets:
        print("!! 목표 생성 실패 — 종목 부족 또는 최소 명목가 미달")
        return

    per = targets[0].notional
    longs = [t for t in targets if t.side == "LONG"]
    shorts = [t for t in targets if t.side == "SHORT"]
    print(f"# 포지션 {len(targets)}개 × ${per:.2f}  총명목 ${per * len(targets):.2f}  순노출 $0.00")
    print(f"# 보유 {HOLD_D}일, 자본 1/{HOLD_D} 씩 회전 → 일일 신규 ${per * len(targets) / HOLD_D:.2f}")
    print()

    print(f"롱  {len(longs)}종목  7일수익 중앙 {np.median([t.rank_return for t in longs]) * 100:>7.2f}%")
    print(f"숏  {len(shorts)}종목  7일수익 중앙 {np.median([t.rank_return for t in shorts]) * 100:>7.2f}%")
    print(
        f"스프레드(숏중앙 − 롱중앙) {(np.median([t.rank_return for t in shorts]) - np.median([t.rank_return for t in longs])) * 100:.2f}%p"
    )
    print()
    print("롱 상위 10 (가장 많이 떨어진 순):")
    for t in longs[:10]:
        print(f"  {t.symbol:<14}{t.rank_return * 100:>8.2f}%  ${t.notional:.2f}")
    print("숏 상위 10 (가장 많이 오른 순):")
    for t in sorted(shorts, key=lambda x: -x.rank_return)[:10]:
        print(f"  {t.symbol:<14}{t.rank_return * 100:>8.2f}%  ${t.notional:.2f}")
    print()
    print("# 전량 지정가로 체결. 순노출 0이므로 시장 방향과 무관.")
    print("# 백테스트 최근 1/4 성과는 t=0.65로 미확정 — 페이퍼 전방 검증 단계다.")


if __name__ == "__main__":
    main()
