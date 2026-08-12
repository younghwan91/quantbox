"""신규 크립토 무기한 감지 — exchangeInfo가 유일한 진실 원천.

공지 제목 파싱은 조용히 틀린다. 2026-08-13 실측에서 GIGADEVUSDT를 크립토로
분류했으나 거래소 기준은 HK_EQUITY였고, 제목에 TradFi 표시가 없는 주식·원자재
상장이 다수였으며(AMDUSDT·NATGASUSDT·XPTUSDT), 복수 심볼 공지 16건이 통째로
누락됐다. 거래소가 스스로 붙인 분류만 쓴다.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

logger = logging.getLogger("listing.universe")

EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
_UA = {"User-Agent": "Mozilla/5.0"}

TARGET_TYPE = "COIN"
TARGET_CONTRACT = "PERPETUAL"
TARGET_QUOTE = "USDT"

#: 2026-08-13 실측된 `underlyingType` 전체 (859심볼).
#: 여기 없는 값이 나오면 거래소가 분류 체계를 바꾼 것이므로 경보한다.
KNOWN_TYPES = frozenset({"COIN", "EQUITY", "HK_EQUITY", "KR_EQUITY", "COMMODITY", "INDEX", "PREMARKET"})


def is_target(sym: dict) -> bool:
    """크립토 USDT 무기한인가.

    세 조건을 모두 봐야 한다. `contractType`만 보면 TRADIFI_PERPETUAL이 걸러지지만
    정산 자산이 USDC인 크립토가 남고, `underlyingType`만 보면 COIN 중 분기물이 남는다.
    """
    return (
        sym.get("underlyingType") == TARGET_TYPE
        and sym.get("contractType") == TARGET_CONTRACT
        and sym.get("quoteAsset") == TARGET_QUOTE
    )


def target_symbols(exchange_info: dict) -> dict[str, dict]:
    """대상 심볼만 `{symbol: 원본}` 으로."""
    return {s["symbol"]: s for s in exchange_info.get("symbols", []) if is_target(s)}


def unknown_types(exchange_info: dict) -> set[str]:
    """처음 보는 `underlyingType` 집합.

    비어 있지 않으면 거래소가 분류를 추가한 것이다. **조용히 제외하면 새로운 크립토
    분류가 생겼을 때 이벤트를 통째로 놓친다.**
    """
    seen = {s.get("underlyingType") for s in exchange_info.get("symbols", [])}
    return {t for t in seen if t is not None and t not in KNOWN_TYPES}


def fetch_exchange_info(timeout: float = 30.0) -> dict:
    """공개 엔드포인트. API 키 불필요, 계정 레이트리밋 무관, 가중치 1."""
    req = urllib.request.Request(EXCHANGE_INFO_URL, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def detect_new(current: dict[str, dict], seen: set[str]) -> dict[str, dict]:
    """`seen`에 없는 대상 심볼."""
    return {k: v for k, v in current.items() if k not in seen}


class SeenStore:
    """본 심볼 집합의 영속 저장소.

    재시작해도 같은 이벤트를 다시 기록하지 않게 한다.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            return set(json.loads(self.path.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"본 심볼 파일 읽기 실패({e}) — 빈 집합으로 시작한다")
            return set()

    def save(self, symbols: set[str]) -> None:
        """원자적 쓰기 — 중간에 죽어도 파일이 깨지지 않는다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(symbols)))
        tmp.replace(self.path)

    def bootstrap(self, exchange_info: dict) -> set[str]:
        """첫 실행: **전** 심볼을 이미 본 것으로 등록한다.

        대상 심볼만 등록하면 안 된다 — 오늘 EQUITY인 심볼이 나중에 분류가 바뀌면
        신규로 오인된다. 그리고 이 가드가 없으면 첫 폴링에서 700개를 구독한다.

        저장 실패는 **삼키지 않는다.** 상태 디렉터리가 쓰기 불가면 매 재시작마다
        조용히 다시 부트스트랩하게 되고, 중복 방지 보장이 사라진 것을 아무도 모른다.
        기동 시점에 시끄럽게 죽는 편이 낫다. (이벤트 감지 중의 `save` 실패는
        다르다 — 그때는 기록을 잃지 않는 게 우선이므로 호출자가 감싼다. Task 5 참조.)
        """
        seen = self.load()
        if seen:
            return seen
        seen = {s["symbol"] for s in exchange_info.get("symbols", [])}
        self.save(seen)
        logger.info(f"첫 실행 — 기존 {len(seen)}심볼을 등록하고 시작한다")
        return seen
