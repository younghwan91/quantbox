"""실행 가능 353심볼의 최근 펀딩 이력 수집 (공개 API, 키 불필요).

`fapi/v1/fundingRate`는 호출당 최대 1000건. 심볼당 1회만 호출해 **가장 최근 1000건**을
받는다 — 4h 주기면 약 5.5개월, 8h면 약 11개월. 차입금리가 현재 스냅샷이므로
시대를 맞추려면 오히려 최근 구간이 맞다.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request

SP = pathlib.Path(__file__).parent
EXE = json.loads((SP / "exe.json").read_text())
OUT = SP / "funding_all.json"
URL = "https://fapi.binance.com/fapi/v1/fundingRate?symbol={}&limit=1000"


def main() -> None:
    syms = EXE["exe"]
    out: dict[str, list] = {}
    if OUT.exists():
        out = json.loads(OUT.read_text())
    todo = [s for s in syms if s not in out]
    print(f"대상 {len(syms)}  이미받음 {len(out)}  남음 {len(todo)}", flush=True)

    fail = 0
    for i, s in enumerate(todo, 1):
        try:
            with urllib.request.urlopen(URL.format(s), timeout=30) as r:
                d = json.load(r)
            out[s] = [(int(x["fundingTime"]), float(x["fundingRate"])) for x in d]
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            fail += 1
            print(f"  FAIL {s}: {type(e).__name__}", flush=True)
            time.sleep(1.0)
            continue
        if i % 50 == 0:
            OUT.write_text(json.dumps(out))
            print(f"  [{i}/{len(todo)}] 저장 (누적 {len(out)})", flush=True)
        time.sleep(0.12)  # 레이트리밋 여유

    OUT.write_text(json.dumps(out))
    print(f"완료: {len(out)}심볼 저장, 실패 {fail}", flush=True)


if __name__ == "__main__":
    main()
