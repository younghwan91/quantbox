#!/usr/bin/env python3
"""기록된 상장 이벤트의 비용 프로파일과 판정을 출력한다.

수동 실행이다. 3건이 모이면 돌려 판정한다.

사용:
    uv run python scripts/listing_report.py
    uv run python scripts/listing_report.py --symbol DOSUSDT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quantbox.listing.analyze import ENTRY_WINDOW_H, hourly_profile, read_depth_samples, verdict
from quantbox.listing.events import EventLog

ROOT = Path(__file__).resolve().parent.parent
SOCKET = ROOT / "data" / "socket"
STATE_DIR = ROOT / "data" / "listing"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", help="특정 심볼만")
    ap.add_argument("--socket", default=str(SOCKET))
    args = ap.parse_args()

    events = EventLog(STATE_DIR / "events.json").all()
    if args.symbol:
        events = {k: v for k, v in events.items() if k == args.symbol}
    if not events:
        print("기록된 이벤트가 없다. scripts/listing_probe.py 가 돌고 있는지 확인할 것.")
        return 1

    lo, hi = ENTRY_WINDOW_H
    all_samples = []
    for sym, ev in sorted(events.items()):
        onboard = ev.get("onboard_ms")
        if not onboard:
            print(f"{sym}: onboard_ms 없음 — 경과시간을 계산할 수 없다")
            continue

        samples = read_depth_samples(sym, Path(args.socket), onboard)
        all_samples.extend(samples)
        v = verdict(samples)

        print(f"\n{'=' * 72}\n{sym}  (상장 {ev['underlying_type']}, 표본 {len(samples):,})")
        print(f"{'시간':>5} {'n':>7} {'중앙 bp':>9} {'p25':>8} {'p75':>8} {'체결불가':>8}")
        for r in hourly_profile(samples):
            mark = " ←" if lo <= r["hour"] <= hi else ""
            print(
                f"{r['hour']:>5} {r['n']:>7,} {r['median_bp']:>9.2f} "
                f"{r['p25_bp']:>8.2f} {r['p75_bp']:>8.2f} {r['unfillable_frac']:>7.1%}{mark}"
            )
        print(f"\n  진입 창({lo:.0f}~{hi:.0f}h) 판정: {v['decision']}  "
              f"중앙 {v['median_bp']:.2f}bp  체결불가 {v['unfillable_frac']:.1%}  n={v['n']:,}")

    if len(events) >= 3:
        combined = verdict(all_samples)
        print(f"\n{'=' * 72}\n종합 판정 ({len(events)}건): {combined['decision']}")
        print(f"  중앙 {combined['median_bp']:.2f}bp / 기준 ≤25 진행, ≤50 재검토, >50 종료")
        print(f"  체결불가 {combined['unfillable_frac']:.1%} / 기준 >10% 종료")
    else:
        print(f"\n판정에는 3건이 필요하다 (현재 {len(events)}건). 실측 상장 빈도는 주당 1.09건.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
