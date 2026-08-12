# 신규 상장 비용 측정기 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 바이낸스 신규 크립토 무기한 상장을 자동 포착해 상장 직후 호가·체결을 소켓으로 기록하고, 진입 창의 **실효 스프레드**를 산출해 사전 등록된 기준으로 판정한다.

**Architecture:** `exchangeInfo`를 60초 폴링하는 감시 프로세스가 신규 크립토 무기한을 감지하면, 그 심볼만 구독하는 **자식 프로세스**를 띄워 7일간 `bookTicker`·`trade`·`depth20@500ms`를 기록한다. 기존 `scripts/collect_book_data.py` 수집기는 건드리지 않고 저장 레이아웃만 공유한다. 분석은 수동 실행이며, 판정값은 호가창에 24달러를 실제로 체결시켜 얻은 편도 실효 스프레드다.

**Tech Stack:** Python 3.13, `websockets`(기존 의존성), 표준 라이브러리 `urllib`·`gzip`·`json`, `numpy`, pytest.

**설계 문서:** `docs/superpowers/specs/2026-08-13-listing-cost-probe-design.md`

## Global Constraints

- **주문을 넣지 않는다.** 이 프로젝트에 거래 코드는 없다. `quantbox/execution/`·`quantbox/scalper.py`를 import하지 않는다.
- **기존 수집기 프로세스를 재시작하지 않는다.** tmux `collector` 세션은 2026-08-10부터 무중단 가동 중이며 복구 불가능한 데이터를 쌓고 있다(바이낸스는 2024-03-30자로 `bookTicker` 아카이브를 중단했다). 파일 편집은 안전하지만(실행 중 프로세스는 이미 코드를 로드했다) **`tmux kill-session`·재시작 금지**.
- **Mock 데이터 금지.** 테스트는 실제 API 응답을 고정 자산으로 저장해 쓰거나 실구독한다. `np.random` 합성 데이터 금지.
- **판정 기준을 사후에 조정하지 않는다.** 임계값은 스펙에 사전 등록돼 있다: 편도 실효 스프레드 중앙값 ≤25bp 진행 / 25~50bp 재검토 / >50bp 종료. 체결 불가 비율 >10%면 스프레드와 무관하게 종료.
- **줄 길이 120** (ruff 강제). 타입 힌트는 Python 3.13 제네릭(`list[T]`, `dict[K, V]`). 독스트링 Google 스타일. 주석·로그는 한국어, 식별자는 영어.
- **커밋 전 `uv run pytest tests/ -v` 통과.** pre-commit이 ruff format → ruff check → mypy → pytest-fast를 돌린다.
- **대상 필터는 세 조건 동시 충족**: `underlyingType == "COIN"`, `contractType == "PERPETUAL"`, `quoteAsset == "USDT"`.
- **소켓 콜백은 절대 블로킹하지 않는다.** 콜백은 큐에 append만 하고 압축·디스크 쓰기는 별도 스레드가 한다(기존 수집기가 이 함정으로 한 번 터졌다).

## File Structure

| 파일 | 책임 |
|---|---|
| `quantbox/collect/rotating_writer.py` | 시간 단위 gzip JSONL writer. `collect_book_data.py`에서 추출해 공유 |
| `quantbox/listing/universe.py` | exchangeInfo 조회 · 대상 필터 · 신규 심볼 감지 · 본 심볼 상태 파일 |
| `quantbox/listing/book.py` | 호가창 → 편도 실효 스프레드. **순수 함수, 판정의 핵심** |
| `quantbox/listing/recorder.py` | 소켓 구독·기록 (한 심볼, 정해진 기간) |
| `quantbox/listing/analyze.py` | 기록 파일 → 경과시간별 표 → 판정 |
| `scripts/listing_probe.py` | 감시기 진입점 (tmux `listing` 세션) |
| `tests/fixtures/exchange_info_20260813.json` | exchangeInfo 859심볼 고정 자산 |
| `tests/test_listing_universe.py` | 필터 · 신규 감지 · 첫 실행 폭주 방지 |
| `tests/test_listing_book.py` | 실효 스프레드 (합성 호가창 + 실구독) |
| `tests/test_listing_recorder.py` | 실구독 30초 경로·형식 |
| `tests/test_listing_analyze.py` | 기존 58종목 실데이터로 시간대별 표 |
| `tests/test_rotating_writer.py` | 추출 전후 동작 동일성 |

의존 방향은 `analyze → book`, `recorder → rotating_writer`, `probe → universe + recorder`. `book.py`는 아무것도 import하지 않는다(numpy만).

---

### Task 1: 실효 스프레드 계산 (`book.py`)

판정의 핵심 로직이고 순수 함수라 가장 먼저, 가장 단단하게 만든다. 다른 어떤 것도 이것에 의존하지 않고, 이것은 아무것에도 의존하지 않는다.

**Files:**
- Create: `quantbox/listing/__init__.py`
- Create: `quantbox/listing/book.py`
- Test: `tests/test_listing_book.py`

**Interfaces:**
- Consumes: 없음 (표준 라이브러리 + numpy만)
- Produces:
  - `PriceLevel = tuple[float, float]` — `(price, qty)`
  - `parse_levels(raw: list[list[str]]) -> list[PriceLevel]` — 바이낸스 문자열 호가를 float로
  - `fill_vwap(levels: list[PriceLevel], notional_usd: float) -> tuple[float, bool]` — `(vwap, filled)`
  - `effective_half_spread_bp(bids: list[PriceLevel], asks: list[PriceLevel], notional_usd: float) -> tuple[float, bool]` — `(bp, filled)`
  - `top_half_spread_bp(bids: list[PriceLevel], asks: list[PriceLevel]) -> float` — 최우선 호가 기준 (대조용)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_listing_book.py`:

```python
"""호가창을 실제로 걸어 얻는 편도 실효 스프레드.

최우선 호가 스프레드만 보면 "1bp인데 그 가격에 100달러어치뿐"인 상황을 못 거른다.
신규 상장 초기는 호가가 얇아 이 구분이 판정을 뒤집는다. 그래서 판정값은
목표 금액을 실제로 체결시킨 체결가중평균과 중간가의 괴리다.
"""

from __future__ import annotations

import math

import pytest

from quantbox.listing.book import (
    effective_half_spread_bp,
    fill_vwap,
    parse_levels,
    top_half_spread_bp,
)


def _book(spread_bp: float, qty_usd_per_level: float, n: int = 20, mid: float = 100.0):
    """중간가 ``mid``, 지정 스프레드, 각 호가에 균일 금액이 쌓인 호가창."""
    half = mid * spread_bp / 1e4
    tick = mid * 1e-4
    bids = [(mid - half - i * tick, qty_usd_per_level / (mid - half - i * tick)) for i in range(n)]
    asks = [(mid + half + i * tick, qty_usd_per_level / (mid + half + i * tick)) for i in range(n)]
    return bids, asks


def test_parse_levels_converts_binance_strings() -> None:
    """바이낸스는 호가를 문자열로 준다 — float 변환이 없으면 산술이 조용히 문자열 연결이 된다."""
    assert parse_levels([["0.070710", "1205399"], ["0.070700", "1231221"]]) == [
        (0.070710, 1205399.0),
        (0.070700, 1231221.0),
    ]


def test_thick_top_level_equals_top_of_book_spread() -> None:
    """최우선 호가에서 다 채워지면 실효 스프레드 == 최우선 호가 스프레드.

    이게 성립하지 않으면 계산이 틀린 것이다 — 깊이가 충분할 때는 두 값이 같아야 한다.
    """
    bids, asks = _book(spread_bp=10.0, qty_usd_per_level=10_000.0)
    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)

    assert filled
    assert eff == pytest.approx(top_half_spread_bp(bids, asks), rel=1e-9)
    assert eff == pytest.approx(10.0, rel=1e-6)  # 편도 = 스프레드의 절반 → 여기선 half=10bp


def test_thin_book_costs_more_than_top_of_book() -> None:
    """호가가 얇으면 실효 스프레드가 최우선 호가 스프레드보다 크다 — 깊이 부족이 자동 반영된다."""
    bids, asks = _book(spread_bp=10.0, qty_usd_per_level=2.0)  # 레벨당 2달러뿐
    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)

    assert filled, "20호가 × 2달러 = 40달러이므로 24달러는 채워져야 한다"
    assert eff > top_half_spread_bp(bids, asks), "얇은데 최우선 호가와 같은 값이 나왔다"


def test_unfillable_book_reports_not_filled() -> None:
    """20호가를 다 먹어도 목표 금액이 안 되면 체결 불가로 보고한다."""
    bids, asks = _book(spread_bp=10.0, qty_usd_per_level=0.5)  # 20호가 합쳐도 10달러
    _, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)

    assert not filled


def test_fill_vwap_stops_at_target() -> None:
    """목표 금액에 도달하면 남은 호가를 먹지 않는다 — 다 먹으면 비용이 과대평가된다."""
    levels = [(100.0, 1.0), (200.0, 1.0)]  # 100달러 + 200달러
    vwap, filled = fill_vwap(levels, notional_usd=100.0)

    assert filled
    assert vwap == pytest.approx(100.0), "첫 호가에서 끝나야 하는데 둘째까지 먹었다"


def test_empty_book_is_nan_not_crash() -> None:
    """빈 호가창은 예외가 아니라 nan이다 — 상장 직후 첫 스냅샷이 비어 있을 수 있다."""
    eff, filled = effective_half_spread_bp([], [], notional_usd=24.0)

    assert math.isnan(eff)
    assert not filled


def test_crossed_book_is_rejected() -> None:
    """교차 호가(bid >= ask)는 데이터 오류다 — 음수 스프레드를 조용히 내보내지 않는다."""
    eff, filled = effective_half_spread_bp([(101.0, 100.0)], [(100.0, 100.0)], notional_usd=24.0)

    assert math.isnan(eff)
    assert not filled
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

```bash
uv run pytest tests/test_listing_book.py -v
```

기대: `ModuleNotFoundError: No module named 'quantbox.listing'` 로 전부 실패.

- [ ] **Step 3: 최소 구현을 쓴다**

`quantbox/listing/__init__.py`:

```python
"""신규 상장 비용 측정기.

주문을 넣지 않는다 — 상장 직후 호가·체결을 기록하고 실효 스프레드를 재는 것이 전부다.
설계: docs/superpowers/specs/2026-08-13-listing-cost-probe-design.md
"""
```

`quantbox/listing/book.py`:

```python
"""호가창 → 편도 실효 스프레드.

최우선 호가 스프레드는 "그 가격에 얼마나 있는가"를 말해주지 않는다. 신규 상장
초기처럼 얇은 호가에서는 그 차이가 판정을 뒤집으므로, 목표 금액을 호가창에 실제로
걸어 체결시킨 값을 쓴다. 깊이가 충분하면 두 값은 같아지고, 얇으면 실효 쪽이 커진다.
따라서 **깊이 부족을 별도 조건으로 둘 필요가 없다.**

순수 함수만 둔다 — 네트워크도 파일도 만지지 않으므로 테스트가 빠르고 결정적이다.
"""

from __future__ import annotations

PriceLevel = tuple[float, float]
"""``(price, qty)``. qty는 코인 수량이지 금액이 아니다."""

_EPS = 1e-12


def parse_levels(raw: list[list[str]]) -> list[PriceLevel]:
    """바이낸스 호가(문자열 쌍)를 float 쌍으로.

    바이낸스는 ``[["0.070710", "1205399"], ...]`` 형태로 준다. 문자열 그대로 두면
    비교·산술이 사전순으로 동작해 조용히 틀린다.
    """
    return [(float(p), float(q)) for p, q in raw]


def fill_vwap(levels: list[PriceLevel], notional_usd: float) -> tuple[float, bool]:
    """``levels``를 순서대로 먹어 ``notional_usd``를 채운 체결가중평균가.

    Args:
        levels: 매수측은 가격 내림차순, 매도측은 오름차순.
        notional_usd: 채울 명목 금액(USD).

    Returns:
        ``(vwap, filled)``. 호가를 다 먹어도 목표에 못 미치면 ``filled=False``이고
        vwap은 먹은 만큼의 평균이다(호가창이 비었으면 nan).
    """
    if notional_usd <= 0:
        return float("nan"), False

    remaining = notional_usd
    spent = 0.0
    qty = 0.0
    for price, size in levels:
        if price <= 0 or size <= 0:
            continue
        take = min(remaining, price * size)
        spent += take
        qty += take / price
        remaining -= take
        if remaining <= _EPS:
            return spent / qty, True

    return (spent / qty if qty > 0 else float("nan")), False


def top_half_spread_bp(bids: list[PriceLevel], asks: list[PriceLevel]) -> float:
    """최우선 호가 기준 편도 스프레드(bp). 대조용."""
    if not bids or not asks:
        return float("nan")
    bid, ask = bids[0][0], asks[0][0]
    if bid <= 0 or ask <= bid:
        return float("nan")
    mid = (bid + ask) / 2.0
    return (ask - bid) / 2.0 / mid * 1e4


def effective_half_spread_bp(
    bids: list[PriceLevel], asks: list[PriceLevel], notional_usd: float
) -> tuple[float, bool]:
    """``notional_usd``를 양방향으로 체결시킨 **편도** 실효 스프레드(bp).

    매수는 매도호가를, 매도는 매수호가를 걸어 각각 중간가와의 괴리를 구하고 평균낸다.
    왕복 비용은 이 값의 2배에 수수료를 더한 것이다.

    Returns:
        ``(bp, filled)``. 어느 한쪽이라도 채워지지 않으면 ``filled=False``.
        호가창이 비었거나 교차했으면 ``(nan, False)``.
    """
    if not bids or not asks:
        return float("nan"), False

    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0 or best_ask <= best_bid:
        return float("nan"), False  # 교차·비정상 호가는 음수 스프레드를 만든다

    mid = (best_bid + best_ask) / 2.0
    buy_vwap, buy_ok = fill_vwap(asks, notional_usd)
    sell_vwap, sell_ok = fill_vwap(bids, notional_usd)
    if buy_vwap != buy_vwap or sell_vwap != sell_vwap:  # nan 검사
        return float("nan"), False

    half = ((buy_vwap - mid) + (mid - sell_vwap)) / 2.0
    return half / mid * 1e4, (buy_ok and sell_ok)
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

```bash
uv run pytest tests/test_listing_book.py -v
```

기대: 7개 전부 PASS.

- [ ] **Step 5: 실제 호가로 검증한다 (No Mock Data 규칙)**

`tests/test_listing_book.py` 끝에 추가:

```python
@pytest.mark.slow
def test_live_book_gives_sane_spread() -> None:
    """실제 depth20 스냅샷에서 값이 합리적인가.

    합성 호가창은 내가 만든 가정을 검증할 뿐이다. 실호가 구조(가격·수량 문자열,
    레벨 수, 틱 크기)에서도 동작하는지는 실구독으로만 확인된다.
    """
    import asyncio
    import json

    import websockets

    async def snap() -> dict:
        url = "wss://fstream.binance.com/stream?streams=btcusdt@depth20@500ms"
        async with websockets.connect(url, ping_interval=20) as ws:
            raw = await asyncio.wait_for(ws.recv(), timeout=20)
            return json.loads(raw)["data"]

    d = asyncio.run(snap())
    bids, asks = parse_levels(d["b"]), parse_levels(d["a"])
    assert len(bids) == 20 and len(asks) == 20, "depth20인데 20레벨이 아니다"

    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)
    assert filled, "BTCUSDT 최우선 호가로 24달러가 안 채워질 리 없다"
    assert 0.0 < eff < 20.0, f"BTCUSDT 편도 스프레드가 비현실적이다: {eff}bp"
    # 24달러는 BTC 최우선 호가에 묻히므로 최우선 호가 값과 일치해야 한다.
    assert eff == pytest.approx(top_half_spread_bp(bids, asks), rel=1e-6)
```

실행:

```bash
uv run pytest tests/test_listing_book.py -m slow -v
```

기대: PASS. (`pyproject.toml`이 기본 실행에서 `@slow`를 제외하므로 pre-commit은 느려지지 않는다.)

- [ ] **Step 6: 커밋**

```bash
uv run ruff format quantbox/listing tests/test_listing_book.py
uv run ruff check quantbox/listing --fix
git add quantbox/listing/__init__.py quantbox/listing/book.py tests/test_listing_book.py
git commit -m "feat(listing): 호가창 실효 스프레드 계산 — 깊이 부족이 자동 반영된다"
```

---

### Task 2: 대상 필터와 신규 감지 (`universe.py`)

**Files:**
- Create: `quantbox/listing/universe.py`
- Create: `tests/fixtures/exchange_info_20260813.json`
- Test: `tests/test_listing_universe.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `TARGET_TYPE = "COIN"`, `TARGET_CONTRACT = "PERPETUAL"`, `TARGET_QUOTE = "USDT"`
  - `KNOWN_TYPES: frozenset[str]` — 2026-08-13 실측된 `underlyingType` 전체
  - `is_target(sym: dict) -> bool`
  - `target_symbols(exchange_info: dict) -> dict[str, dict]` — `{symbol: 원본 dict}`
  - `unknown_types(exchange_info: dict) -> set[str]` — 처음 보는 분류(경보용)
  - `fetch_exchange_info(timeout: float = 30.0) -> dict`
  - `SeenStore` — `path`, `load() -> set[str]`, `save(symbols: set[str]) -> None`
  - `detect_new(current: dict[str, dict], seen: set[str]) -> dict[str, dict]`

- [ ] **Step 1: 고정 자산을 만든다**

실제 응답을 저장한다. 합성 데이터를 쓰지 않는다.

```bash
mkdir -p tests/fixtures
uv run python -c "
import json, urllib.request
r = urllib.request.Request('https://fapi.binance.com/fapi/v1/exchangeInfo',
                           headers={'User-Agent': 'Mozilla/5.0'})
d = json.load(urllib.request.urlopen(r, timeout=30))
json.dump(d, open('tests/fixtures/exchange_info_20260813.json', 'w'))
print('심볼', len(d['symbols']))
"
```

기대 출력: `심볼 859` 내외 (날짜에 따라 다를 수 있다 — 아래 테스트는 하드코딩된 총계 대신 분류별 관계를 검증한다).

- [ ] **Step 2: 실패하는 테스트를 쓴다**

`tests/test_listing_universe.py`:

```python
"""대상 필터와 신규 상장 감지.

제목 파싱으로 크립토/주식을 구분하려던 초안은 조용히 틀렸다 (GIGADEVUSDT를
크립토로 분류했으나 거래소 기준은 HK_EQUITY). 거래소 자신의 분류만 믿는다.

두 번째 함정: 빈 상태로 시작하면 첫 폴링에서 기존 700여 심볼이 전부 "신규"가 된다.
그대로 두면 700개를 구독한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quantbox.listing.universe import (
    KNOWN_TYPES,
    SeenStore,
    detect_new,
    is_target,
    target_symbols,
    unknown_types,
)

FIXTURE = Path(__file__).parent / "fixtures" / "exchange_info_20260813.json"


@pytest.fixture
def info() -> dict:
    return json.loads(FIXTURE.read_text())


def test_crypto_perpetual_is_target(info: dict) -> None:
    """알려진 크립토 무기한이 통과한다."""
    by_sym = {s["symbol"]: s for s in info["symbols"]}
    for sym in ("DOGEUSDT", "BTCUSDT", "DOSUSDT", "GRVTUSDT"):
        assert is_target(by_sym[sym]), f"{sym}이 대상에서 빠졌다"


def test_equity_and_commodity_are_excluded(info: dict) -> None:
    """주식·원자재 무기한은 제외된다 — 섞이면 −26.59% 측정이 오염된다.

    GIGADEVUSDT가 핵심 케이스다. 제목에는 TradFi 표시가 없어 정규식 필터를 통과했지만
    거래소 분류는 HK_EQUITY다.
    """
    by_sym = {s["symbol"]: s for s in info["symbols"]}
    for sym in ("AMDUSDT", "MSFTUSDT", "QQQUSDT", "NATGASUSDT", "XPTUSDT", "GIGADEVUSDT"):
        if sym in by_sym:
            assert not is_target(by_sym[sym]), f"{sym}이 대상에 들어왔다 — 크립토가 아니다"


def test_target_set_is_a_strict_subset(info: dict) -> None:
    """대상이 전체보다 작고 비어 있지 않다 — 필터가 통과만 시키거나 전부 막지 않는다."""
    targets = target_symbols(info)
    assert 0 < len(targets) < len(info["symbols"])
    assert all(s["underlyingType"] == "COIN" for s in targets.values())
    assert all(s["contractType"] == "PERPETUAL" for s in targets.values())
    assert all(s["quoteAsset"] == "USDT" for s in targets.values())


def test_non_usdt_quote_is_excluded(info: dict) -> None:
    """USDC·USD1 등 다른 정산 자산은 제외한다 — 측정 기준이 USDT다."""
    targets = target_symbols(info)
    assert not any(s.endswith("USDC") for s in targets)


def test_unknown_type_is_reported_not_swallowed(info: dict) -> None:
    """거래소가 새 분류를 추가하면 알아차려야 한다 — 조용히 제외하면 모르고 지나간다."""
    assert unknown_types(info) == set(), f"고정 자산에 미등록 분류가 있다: {unknown_types(info)}"

    mutated = {"symbols": [dict(info["symbols"][0], underlyingType="QUANTUM_FOAM")]}
    assert unknown_types(mutated) == {"QUANTUM_FOAM"}


def test_known_types_matches_fixture(info: dict) -> None:
    """상수가 실측과 일치한다 — 손으로 적은 목록이 낡으면 경보가 오작동한다."""
    assert {s["underlyingType"] for s in info["symbols"]} <= KNOWN_TYPES


def test_first_run_does_not_flood(info: dict, tmp_path: Path) -> None:
    """빈 상태 첫 실행에서 기존 심볼이 신규로 잡히면 안 된다.

    이 가드가 없으면 첫 폴링에서 700개 구독을 시도한다.

    쓰기 가능한 경로를 쓴다 — `bootstrap`은 등록 결과를 실제로 저장해야 하고,
    저장 실패를 삼키면 재시작마다 조용히 다시 부트스트랩하게 된다.
    """
    store = SeenStore(tmp_path / "seen.json")
    seen = store.bootstrap(info)

    assert len(seen) == len(info["symbols"]), "부트스트랩이 전 심볼을 등록하지 않았다"
    assert detect_new(target_symbols(info), seen) == {}


def test_detect_new_finds_only_the_addition(info: dict) -> None:
    """정말 새 심볼만 잡는다."""
    targets = target_symbols(info)
    victim = next(iter(targets))
    seen = {s["symbol"] for s in info["symbols"]} - {victim}

    assert set(detect_new(targets, seen)) == {victim}


def test_seen_store_roundtrip(tmp_path: Path) -> None:
    """재시작해도 본 심볼을 기억한다 — 잊으면 같은 이벤트를 다시 기록한다."""
    store = SeenStore(tmp_path / "seen.json")
    assert store.load() == set()

    store.save({"AAAUSDT", "BBBUSDT"})
    assert SeenStore(tmp_path / "seen.json").load() == {"AAAUSDT", "BBBUSDT"}
```

- [ ] **Step 3: 테스트가 실패하는지 확인한다**

```bash
uv run pytest tests/test_listing_universe.py -v
```

기대: `ModuleNotFoundError: quantbox.listing.universe` 로 전부 실패.

- [ ] **Step 4: 구현을 쓴다**

`quantbox/listing/universe.py`:

```python
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
KNOWN_TYPES = frozenset(
    {"COIN", "EQUITY", "HK_EQUITY", "KR_EQUITY", "COMMODITY", "INDEX", "PREMARKET"}
)


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
```

- [ ] **Step 5: 테스트가 통과하는지 확인한다**

```bash
uv run pytest tests/test_listing_universe.py -v
```

기대: 9개 전부 PASS. `test_first_run_does_not_flood`가 실패하면 `bootstrap`이 대상만 등록하고 있는 것이다.

- [ ] **Step 6: 커밋**

```bash
uv run ruff format quantbox/listing tests/test_listing_universe.py
uv run ruff check quantbox/listing --fix
git add quantbox/listing/universe.py tests/test_listing_universe.py tests/fixtures/exchange_info_20260813.json
git commit -m "feat(listing): exchangeInfo 기반 대상 필터 — 제목 파싱을 쓰지 않는다"
```

---

### Task 3: `RotatingWriter` 공유 모듈로 추출

기록기가 같은 writer를 써야 저장 레이아웃이 갈라지지 않는다. 복사하면 두 벌이 서서히 어긋난다.

**Files:**
- Create: `quantbox/collect/__init__.py`
- Create: `quantbox/collect/rotating_writer.py`
- Modify: `scripts/collect_book_data.py:111-199` (클래스·헬퍼 제거 후 import)
- Test: `tests/test_rotating_writer.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `RotatingWriter(out_root: Path)` — `submit(stream, symbol, recv_ns, payload)`, `flush()`, `close()`, `qsize`, `counts`, `dropped`, `bytes_written`
  - `disk_used_gb(path: Path) -> float`
  - `MAX_QUEUE: int`

> **경고**: `scripts/collect_book_data.py`는 지금 tmux `collector` 세션에서 돌고 있다. 파일 편집은 안전하다(실행 중 프로세스는 이미 코드를 로드했다). **세션을 죽이거나 재시작하지 말 것.**

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_rotating_writer.py`:

```python
"""시간 단위 gzip JSONL writer.

`scripts/collect_book_data.py`에서 추출했다. 그 수집기는 2026-08-10부터 무중단
가동 중이고 바이낸스가 2024-03-30자로 bookTicker 아카이브를 중단했으므로 여기
쌓이는 데이터는 복구 불가능하다. 추출이 동작을 바꾸면 안 된다.
"""

from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from quantbox.collect.rotating_writer import MAX_QUEUE, RotatingWriter, disk_used_gb


def _drain(w: RotatingWriter, timeout: float = 5.0) -> None:
    """쓰기 스레드가 큐를 비울 때까지 기다린다."""
    end = time.monotonic() + timeout
    while w.qsize and time.monotonic() < end:
        time.sleep(0.02)
    w.flush()


def test_writes_expected_path_and_format(tmp_path: Path) -> None:
    """`<root>/<stream>/<symbol>/<YYYY-MM-DD-HH>.jsonl.gz`, 각 줄 `{recv_ns, d}`.

    기존 분석 코드(`autoresearch/spread_est.py`)가 이 레이아웃을 전제한다.
    """
    w = RotatingWriter(tmp_path)
    w.submit("bookTicker", "TESTUSDT", 123456789, {"b": "1.0", "a": "1.1"})
    _drain(w)
    w.close()

    hour = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    path = tmp_path / "bookTicker" / "TESTUSDT" / f"{hour}.jsonl.gz"
    assert path.exists(), f"기대 경로에 파일이 없다: {path}"

    with gzip.open(path, "rt") as fh:
        rec = json.loads(fh.readline())
    assert rec == {"recv_ns": 123456789, "d": {"b": "1.0", "a": "1.1"}}


def test_streams_and_symbols_are_separated(tmp_path: Path) -> None:
    """스트림·심볼별로 파일이 갈린다 — 섞이면 분석에서 되돌릴 수 없다."""
    w = RotatingWriter(tmp_path)
    w.submit("bookTicker", "AAAUSDT", 1, {"x": 1})
    w.submit("depth20", "AAAUSDT", 2, {"x": 2})
    w.submit("bookTicker", "BBBUSDT", 3, {"x": 3})
    _drain(w)
    w.close()

    files = {p.relative_to(tmp_path).parts[:2] for p in tmp_path.rglob("*.jsonl.gz")}
    assert files == {("bookTicker", "AAAUSDT"), ("depth20", "AAAUSDT"), ("bookTicker", "BBBUSDT")}
    assert w.counts["bookTicker"] == 2
    assert w.counts["depth20"] == 1


def test_overflow_is_counted_not_silent(tmp_path: Path) -> None:
    """큐가 넘치면 드랍을 센다 — 조용한 유실은 데이터에 구멍을 남기고 알려주지 않는다."""
    w = RotatingWriter(tmp_path)
    w._stop.set()  # 배수 스레드를 멈춰 큐를 강제로 채운다
    w._thread.join(timeout=5)
    for i in range(MAX_QUEUE + 10):
        w.submit("bookTicker", "TESTUSDT", i, {"i": i})

    assert w.dropped > 0, "큐 상한을 넘겼는데 드랍이 0이다"


def test_disk_used_counts_only_gz(tmp_path: Path) -> None:
    """`.gz`만 센다 — 상한 판정이 다른 파일에 오염되면 안 된다."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.jsonl.gz").write_bytes(b"0" * 1000)
    (tmp_path / "a" / "notes.txt").write_bytes(b"0" * 10_000)

    assert disk_used_gb(tmp_path) == 1000 / 1e9
    assert disk_used_gb(tmp_path / "missing") == 0.0
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

```bash
uv run pytest tests/test_rotating_writer.py -v
```

기대: `ModuleNotFoundError: quantbox.collect` 로 전부 실패.

- [ ] **Step 3: 클래스를 그대로 옮긴다**

`quantbox/collect/__init__.py`:

```python
"""소켓 수집 공용 부품."""
```

`quantbox/collect/rotating_writer.py` — `scripts/collect_book_data.py`의 다음을 **로직 변경 없이** 옮긴다:
- 상수 `MAX_QUEUE`, `WARN_THROTTLE_S`
- `_last_warn`, `_warn_throttled()`
- `class RotatingWriter` 전체
- `disk_used_gb()`

파일 상단 독스트링:

```python
"""스트림×심볼별 시간 단위 gzip JSONL writer.

`scripts/collect_book_data.py`에서 추출했다. 상장 비용 측정기가 같은 저장
레이아웃을 써야 기존 분석 코드가 그대로 동작하고, 복사본 두 벌이 서서히
어긋나는 일이 없다.

소켓 콜백은 절대 블로킹하면 안 된다 — bookTicker는 심볼당 초당 수십 건이 오므로
콜백 안에서 gzip 압축을 하면 큐가 넘친다. 콜백은 deque에 append만 하고(O(1),
GIL 하에서 원자적) 압축·디스크 쓰기는 별도 스레드가 맡는다.
"""
```

로거 이름만 `logging.getLogger("collect.writer")`로 바꾼다.

- [ ] **Step 4: 수집기가 새 모듈을 쓰게 한다**

`scripts/collect_book_data.py`에서 옮긴 정의들을 지우고 import로 대체한다:

```python
from quantbox.collect.rotating_writer import (  # noqa: F401  (수집기 로그가 _warn_throttled를 쓴다)
    RotatingWriter,
    _warn_throttled,
    disk_used_gb,
)
```

`MAX_QUEUE`·`WARN_THROTTLE_S`·`_last_warn` 정의와 `RotatingWriter`·`disk_used_gb`·`_warn_throttled` 본문을 삭제한다. `gzip`·`defaultdict`·`deque`가 더 이상 쓰이지 않으면 import도 지운다(ruff가 잡아준다).

- [ ] **Step 5: 테스트와 수집기 무결성을 확인한다**

```bash
uv run pytest tests/test_rotating_writer.py -v
uv run python -c "import ast,sys; ast.parse(open('scripts/collect_book_data.py').read()); print('구문 OK')"
uv run python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('cbd', 'scripts/collect_book_data.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('import OK — RotatingWriter:', m.RotatingWriter.__module__)
"
uv run ruff check scripts/collect_book_data.py
```

기대: 테스트 4개 PASS, `구문 OK`, `import OK — RotatingWriter: quantbox.collect.rotating_writer`, ruff 통과.

수집기가 여전히 살아 있는지 확인한다 (**절대 재시작하지 않는다**):

```bash
tmux list-sessions | grep collector
ls -t data/socket/bookTicker/BTCUSDT/ | head -2
```

기대: `collector` 세션 존재, 최신 파일의 mtime이 1시간 이내.

- [ ] **Step 6: 커밋**

```bash
uv run ruff format quantbox/collect tests/test_rotating_writer.py scripts/collect_book_data.py
git add quantbox/collect/ tests/test_rotating_writer.py scripts/collect_book_data.py
git commit -m "refactor(collect): RotatingWriter 공유 모듈로 추출 — 저장 레이아웃 단일화"
```

---

### Task 4: 소켓 기록기 (`recorder.py`)

**Files:**
- Create: `quantbox/listing/recorder.py`
- Test: `tests/test_listing_recorder.py`

**Interfaces:**
- Consumes: `quantbox.collect.rotating_writer.RotatingWriter`, `disk_used_gb`
- Produces:
  - `STREAM_SUFFIXES: tuple[str, ...] = ("bookTicker", "trade", "depth20@500ms")`
  - `stream_kind(stream_name: str) -> str` — `"btcusdt@depth20@500ms"` → `"depth20"`
  - `record(symbol: str, out_root: Path, duration_s: float, max_gb: float = 5.0, stop: threading.Event | None = None) -> dict[str, int]`

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_listing_recorder.py`:

```python
"""한 심볼을 정해진 기간 소켓 기록한다.

이 환경은 aggTrade·kline_1m·forceOrder가 네트워크 레벨에서 차단돼 0건이 온다
(`scripts/collect_book_data.py` 독스트링). depth 계열도 막혔을 가능성이 있었으나
2026-08-13 실측에서 전부 정상임을 확인했다. 회귀하면 여기서 잡힌다.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from quantbox.listing.book import effective_half_spread_bp, parse_levels
from quantbox.listing.recorder import STREAM_SUFFIXES, record, stream_kind


def test_stream_kind_strips_symbol_and_speed() -> None:
    """저장 디렉터리 이름은 갱신 주기를 포함하지 않는다 — 주기를 바꿔도 경로가 갈리면 안 된다."""
    assert stream_kind("btcusdt@bookTicker") == "bookTicker"
    assert stream_kind("btcusdt@trade") == "trade"
    assert stream_kind("btcusdt@depth20@500ms") == "depth20"
    assert stream_kind("btcusdt@depth20") == "depth20"


def test_three_streams_are_configured() -> None:
    """세 스트림을 다 구독한다 — depth20이 빠지면 실효 스프레드를 못 잰다."""
    assert STREAM_SUFFIXES == ("bookTicker", "trade", "depth20@500ms")


@pytest.mark.slow
@pytest.mark.timeout(120)  # 전역 timeout=30 < 기록 30초 — 명시하지 않으면 반드시 타임아웃난다
def test_records_all_three_streams_live(tmp_path: Path) -> None:
    """실구독 30초. 세 스트림이 모두 파일로 떨어지는가.

    실데이터만 쓴다 (프로젝트 규칙: No Mock Data). 이 환경에서 여러 스트림이 조용히
    0건을 반환하므로, 목으로는 이 검증이 성립하지 않는다.
    """
    counts = record("BTCUSDT", tmp_path, duration_s=30.0)

    for kind in ("bookTicker", "trade", "depth20"):
        assert counts.get(kind, 0) > 0, f"{kind} 스트림이 0건이다 — 차단됐거나 배선이 틀렸다"
        files = list((tmp_path / kind / "BTCUSDT").glob("*.jsonl.gz"))
        assert files, f"{kind} 파일이 없다"


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_recorded_depth_feeds_the_spread_calculation(tmp_path: Path) -> None:
    """기록한 depth20이 실효 스프레드 계산에 그대로 들어가는가.

    기록 형식과 계산 입력이 어긋나면 3주를 모으고 나서야 알게 된다. 지금 이어본다.
    """
    record("BTCUSDT", tmp_path, duration_s=30.0)
    f = next(iter((tmp_path / "depth20" / "BTCUSDT").glob("*.jsonl.gz")))

    with gzip.open(f, "rt") as fh:
        d = json.loads(fh.readline())["d"]

    bids, asks = parse_levels(d["b"]), parse_levels(d["a"])
    assert len(bids) == 20 and len(asks) == 20

    eff, filled = effective_half_spread_bp(bids, asks, notional_usd=24.0)
    assert filled and 0.0 < eff < 20.0, f"BTCUSDT 편도 스프레드가 비현실적이다: {eff}bp"
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

```bash
uv run pytest tests/test_listing_recorder.py -v
```

기대: `ModuleNotFoundError: quantbox.listing.recorder`.

- [ ] **Step 3: 구현을 쓴다**

`quantbox/listing/recorder.py`:

```python
"""한 심볼의 호가·체결을 정해진 기간 기록한다.

기존 수집기(`scripts/collect_book_data.py`)를 재사용하지 않는다. 심볼을 추가하려면
재시작해야 하고, 그 프로세스는 2026-08-10부터 복구 불가능한 데이터를 쌓고 있다.
저장 레이아웃만 공유해 기존 분석 코드가 그대로 동작하게 한다.

재연결·백오프는 기존 수집기에서 검증된 패턴을 따른다 — python-binance 래퍼는
내부 큐가 100건이라 15스트림에서 17초 만에 넘쳤으므로 raw websockets를 쓴다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import websockets

from quantbox.collect.rotating_writer import RotatingWriter, disk_used_gb

logger = logging.getLogger("listing.recorder")

STREAM_BASE = "wss://fstream.binance.com"

#: 구독할 스트림. `depth20@500ms`는 실측으로 고른 값이다 (2026-08-13, DOGEUSDT):
#: 기본 250ms 3.73건/s·58MB/일, **500ms 1.97건/s·31MB/일**, 100ms 8.33건/s·130MB/일.
#: 시간대별 중앙값을 재는 데 초당 2표본이면 충분하고 100ms는 10배 용량으로 같은 답을 준다.
STREAM_SUFFIXES: tuple[str, ...] = ("bookTicker", "trade", "depth20@500ms")

RECONNECT_AFTER_S = 23 * 3600  # 바이낸스는 24h에 강제 종료한다 — 그 전에 먼저 재연결
FLUSH_INTERVAL_S = 5.0
DISK_CHECK_INTERVAL_S = 60.0


def stream_kind(stream_name: str) -> str:
    """스트림 이름 → 저장 디렉터리 이름.

    `"btcusdt@depth20@500ms"` → `"depth20"`. 갱신 주기를 경로에 넣지 않는다 —
    주기를 바꿨을 때 같은 심볼이 두 디렉터리로 갈리면 분석이 조용히 반쪽이 된다.
    """
    parts = stream_name.split("@")
    return parts[1] if len(parts) > 1 else stream_name


def record(
    symbol: str,
    out_root: Path,
    duration_s: float,
    max_gb: float = 5.0,
    stop: threading.Event | None = None,
) -> dict[str, int]:
    """`symbol`을 `duration_s`초 동안 기록하고 스트림별 건수를 반환한다.

    Args:
        symbol: 대문자 심볼 (예: `"DOSUSDT"`).
        out_root: `data/socket` 루트. 하위에 `<stream>/<symbol>/<hour>.jsonl.gz`.
        duration_s: 기록 시간. 스펙 기본은 7일(604800).
        max_gb: 이 이벤트가 쓸 수 있는 상한. 폭주 방어용.
        stop: 외부 중단 신호.
    """
    symbol = symbol.upper()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    writer = RotatingWriter(out_root)
    stop = stop or threading.Event()

    streams = [f"{symbol.lower()}@{s}" for s in STREAM_SUFFIXES]
    logger.info(f"{symbol} 기록 시작 — {len(streams)}스트림, {duration_s / 3600:.1f}시간, 상한 {max_gb}GB")

    def handle(msg: dict) -> None:
        recv_ns = time.time_ns()
        data = msg.get("data", msg)
        name = msg.get("stream", "")
        if data.get("e") == "error":
            logger.warning(f"스트림 에러: {str(data)[:200]}")
            return
        writer.submit(stream_kind(name), data.get("s", symbol), recv_ns, data)

    async def consume() -> None:
        backoff = 1.0
        url = STREAM_BASE + "/stream?streams=" + "/".join(streams)
        while not stop.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=None) as ws:
                    logger.info(f"{symbol} 연결됨")
                    backoff = 1.0
                    deadline = time.monotonic() + RECONNECT_AFTER_S
                    while not stop.is_set() and time.monotonic() < deadline:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            logger.warning(f"{symbol} 30초간 수신 없음")
                            continue
                        handle(json.loads(raw))
            except Exception as e:
                if stop.is_set():
                    return
                logger.warning(f"{symbol} 연결 끊김: {type(e).__name__} — {backoff:.0f}s 후 재시도")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    t = threading.Thread(target=lambda: asyncio.run(consume()), name=f"ws-{symbol}", daemon=True)
    t.start()

    started = time.monotonic()
    last_disk = started
    try:
        while not stop.is_set() and time.monotonic() - started < duration_s:
            time.sleep(FLUSH_INTERVAL_S)
            writer.flush()
            now = time.monotonic()
            if now - last_disk >= DISK_CHECK_INTERVAL_S:
                last_disk = now
                gb = disk_used_gb(out_root)
                if gb >= max_gb:
                    logger.error(f"{symbol} 디스크 상한 {max_gb}GB 도달 — 기록 중단")
                    break
    finally:
        stop.set()
        t.join(timeout=10)
        writer.flush()
        writer.close()
        logger.info(f"{symbol} 기록 종료 — {dict(writer.counts)}")

    return dict(writer.counts)
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

```bash
uv run pytest tests/test_listing_recorder.py -v
uv run pytest tests/test_listing_recorder.py -m slow -v
```

기대: 빠른 테스트 2개 PASS. slow 2개 PASS(각 30초). `depth20이 0건`이면 바이낸스가 이 환경에서 depth를 차단하기 시작한 것이므로 **진행하지 말고 보고한다** — 설계 전제가 무너진다.

- [ ] **Step 5: 커밋**

```bash
uv run ruff format quantbox/listing tests/test_listing_recorder.py
uv run ruff check quantbox/listing --fix
git add quantbox/listing/recorder.py tests/test_listing_recorder.py
git commit -m "feat(listing): 이벤트 심볼 소켓 기록기 — 기존 수집기와 분리된 프로세스"
```

---

### Task 5: 감시기 진입점 (`scripts/listing_probe.py`)

**Files:**
- Create: `scripts/listing_probe.py`
- Create: `quantbox/listing/events.py`
- Test: `tests/test_listing_events.py`

**Interfaces:**
- Consumes: `universe.{fetch_exchange_info, target_symbols, detect_new, unknown_types, SeenStore}`, `recorder.record`
- Produces:
  - `EventLog(path: Path)` — `add(symbol: str, info: dict, detected_ns: int) -> None`, `all() -> dict[str, dict]`
  - 이벤트 레코드: `{"symbol", "onboard_ms", "detected_ns", "underlying_type", "status_at_detect"}`

이벤트 메타데이터가 없으면 분석기가 "상장 후 몇 시간"을 계산할 수 없다. 기록과 같은 트랜잭션으로 남긴다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_listing_events.py`:

```python
"""기록한 상장 이벤트의 메타데이터.

`onboard_ms`가 없으면 분석기가 "상장 후 몇 시간"을 계산할 수 없고, 진입 창
(1~24시간) 자체가 정의되지 않는다. 소켓 파일만 남기면 3주 뒤에 이 사실을 알게 된다.
"""

from __future__ import annotations

from pathlib import Path

from quantbox.listing.events import EventLog


def test_records_onboard_time(tmp_path: Path) -> None:
    """진입 창 계산에 필요한 상장 시각이 남는다."""
    log = EventLog(tmp_path / "events.json")
    log.add("DOSUSDT", {"onboardDate": 1754924400000, "underlyingType": "COIN", "status": "TRADING"}, 999)

    rec = log.all()["DOSUSDT"]
    assert rec["onboard_ms"] == 1754924400000
    assert rec["underlying_type"] == "COIN"
    assert rec["detected_ns"] == 999


def test_survives_restart(tmp_path: Path) -> None:
    """재시작해도 이벤트가 남는다."""
    p = tmp_path / "events.json"
    EventLog(p).add("AAAUSDT", {"onboardDate": 1, "underlyingType": "COIN", "status": "TRADING"}, 1)
    EventLog(p).add("BBBUSDT", {"onboardDate": 2, "underlyingType": "COIN", "status": "TRADING"}, 2)

    assert set(EventLog(p).all()) == {"AAAUSDT", "BBBUSDT"}


def test_first_detection_wins(tmp_path: Path) -> None:
    """같은 심볼을 두 번 넣어도 첫 감지 시각을 유지한다 — 재시작이 시각을 덮으면 안 된다."""
    log = EventLog(tmp_path / "events.json")
    log.add("AAAUSDT", {"onboardDate": 1, "underlyingType": "COIN", "status": "TRADING"}, 100)
    log.add("AAAUSDT", {"onboardDate": 1, "underlyingType": "COIN", "status": "TRADING"}, 200)

    assert log.all()["AAAUSDT"]["detected_ns"] == 100
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

```bash
uv run pytest tests/test_listing_events.py -v
```

기대: `ModuleNotFoundError: quantbox.listing.events`.

- [ ] **Step 3: `events.py`를 쓴다**

```python
"""기록한 상장 이벤트의 메타데이터.

소켓 파일만으로는 "상장 후 몇 시간"을 알 수 없다. 진입 창(1~24시간)이 판정의
정의 자체이므로 상장 시각을 이벤트와 함께 남긴다.
"""

from __future__ import annotations

import json
from pathlib import Path


class EventLog:
    """`{symbol: 레코드}` JSON 저장소."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def all(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return dict(json.loads(self.path.read_text()))
        except (json.JSONDecodeError, OSError):
            return {}

    def add(self, symbol: str, info: dict, detected_ns: int) -> None:
        """이벤트를 등록한다. 이미 있으면 **덮지 않는다** — 첫 감지 시각이 진실이다."""
        events = self.all()
        if symbol in events:
            return
        events[symbol] = {
            "symbol": symbol,
            "onboard_ms": info.get("onboardDate"),
            "detected_ns": detected_ns,
            "underlying_type": info.get("underlyingType"),
            "status_at_detect": info.get("status"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(events, indent=2, sort_keys=True))
        tmp.replace(self.path)
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

```bash
uv run pytest tests/test_listing_events.py -v
```

기대: 3개 PASS.

- [ ] **Step 5: 감시기 진입점을 쓴다**

`scripts/listing_probe.py`:

```python
#!/usr/bin/env python3
"""신규 크립토 무기한 상장 감시기 (연구 전용, 주문 없음).

`exchangeInfo`를 60초마다 조회해 처음 보는 크립토 USDT 무기한을 감지하면
그 심볼만 구독하는 기록 스레드를 띄운다. 공개 엔드포인트라 API 키가 필요 없고
계정 레이트리밋과 무관하다(가중치 1).

기존 수집기(tmux `collector`)와 **완전히 분리된 프로세스**다. 저장 레이아웃만
공유하므로 기존 분석 코드가 그대로 동작한다.

사용:
    uv run python scripts/listing_probe.py              # 상시 감시 (tmux 'listing')
    uv run python scripts/listing_probe.py --once       # 한 번만 확인하고 종료
    uv run python scripts/listing_probe.py --days 7     # 이벤트당 기록 기간
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from quantbox.listing.events import EventLog
from quantbox.listing.recorder import record
from quantbox.listing.universe import (
    SeenStore,
    detect_new,
    fetch_exchange_info,
    target_symbols,
    unknown_types,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "socket"
STATE_DIR = ROOT / "data" / "listing"

POLL_INTERVAL_S = 60.0
PENDING_TIMEOUT_S = 48 * 3600

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("listing.probe")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="한 번만 확인하고 종료")
    ap.add_argument("--days", type=float, default=7.0, help="이벤트당 기록 기간(일)")
    ap.add_argument("--max-gb", type=float, default=5.0, help="이벤트당 디스크 상한(GB)")
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out)
    seen_store = SeenStore(STATE_DIR / "seen_symbols.json")
    events = EventLog(STATE_DIR / "events.json")

    stop = threading.Event()

    def shutdown(signum, frame) -> None:  # noqa: ANN001
        logger.info(f"시그널 {signum} 수신 — 종료 중")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("상장 감시 시작 — exchangeInfo 60초 폴링, 주문 없음")
    logger.info(f"기존 'collector' 세션은 건드리지 않는다. 기록 → {out_root}")

    # 처음 본 심볼이 아직 TRADING이 아니면 여기서 대기시킨다.
    pending: dict[str, float] = {}
    recorders: list[threading.Thread] = []

    def remember(symbols: set[str]) -> None:
        """본 심볼을 저장하되, 실패해도 감시를 멈추지 않는다.

        `bootstrap`의 저장 실패는 기동을 막아야 하지만(중복 방지가 아예 성립하지
        않으므로) 이벤트 감지 중의 실패는 다르다. 기록을 놓치는 것보다 중복 위험을
        감수하는 편이 낫다 — 상장은 주당 1건뿐이고 놓치면 몇 주를 다시 기다린다.
        """
        try:
            seen_store.save(symbols)
        except OSError as e:
            logger.error(f"본 심볼 저장 실패({e}) — 재시작 시 이 이벤트를 다시 기록할 수 있다")

    while not stop.is_set():
        try:
            info = fetch_exchange_info()
        except Exception as e:
            logger.warning(f"exchangeInfo 조회 실패: {type(e).__name__} {str(e)[:80]}")
            if args.once:
                return 1
            stop.wait(POLL_INTERVAL_S)
            continue

        if unknown := unknown_types(info):
            logger.error(f"⚠ 처음 보는 underlyingType: {unknown} — 분류 체계가 바뀌었다. KNOWN_TYPES 갱신 필요")

        seen = seen_store.bootstrap(info)
        targets = target_symbols(info)
        by_sym = {s["symbol"]: s for s in info["symbols"]}

        for sym, meta in detect_new(targets, seen).items():
            pending.setdefault(sym, time.monotonic())
            logger.info(f"신규 대상 감지: {sym} status={meta.get('status')}")

        for sym in list(pending):
            meta = by_sym.get(sym)
            if meta is None:
                continue
            if meta.get("status") == "TRADING":
                logger.info(f"▶ {sym} 거래 시작 — 기록 개시 ({args.days}일)")
                events.add(sym, meta, time.time_ns())
                remember(seen | {sym})
                t = threading.Thread(
                    target=record,
                    args=(sym, out_root, args.days * 86400, args.max_gb, stop),
                    name=f"rec-{sym}",
                    daemon=True,
                )
                t.start()
                recorders.append(t)
                del pending[sym]
            elif time.monotonic() - pending[sym] > PENDING_TIMEOUT_S:
                logger.warning(f"{sym} 48시간 동안 TRADING이 되지 않았다 — 포기")
                remember(seen | {sym})
                del pending[sym]

        if args.once:
            logger.info(f"1회 확인 완료 — 대상 {len(targets)}개, 대기 {len(pending)}개")
            return 0

        stop.wait(POLL_INTERVAL_S)

    logger.info("종료 대기 중 (기록 스레드)...")
    for t in recorders:
        t.join(timeout=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 실제로 한 번 돌려본다**

```bash
uv run python scripts/listing_probe.py --once
```

기대 출력:
```
... | INFO    | 상장 감시 시작 — exchangeInfo 60초 폴링, 주문 없음
... | INFO    | 첫 실행 — 기존 859심볼을 등록하고 시작한다
... | INFO    | 1회 확인 완료 — 대상 651개, 대기 0개
```

(`COIN`은 699개지만 `PERPETUAL`+`USDT` 조건이 48개를 더 걸러내 651개가 된다.
심볼 수는 날짜에 따라 달라지므로 정확히 일치할 필요는 없다 — 전체보다 작고
0이 아니면 된다.)

`⚠ 처음 보는 underlyingType`이 뜨면 거래소가 분류를 추가한 것이니 `KNOWN_TYPES`를 갱신한다.

두 번째 실행이 신규를 만들지 않는지 확인한다(멱등성):

```bash
uv run python scripts/listing_probe.py --once
```

기대: `첫 실행` 줄이 없고 `신규 대상 감지`도 없다.

- [ ] **Step 7: 커밋**

```bash
uv run ruff format quantbox/listing scripts/listing_probe.py tests/test_listing_events.py
uv run ruff check quantbox/listing scripts/listing_probe.py --fix
git add quantbox/listing/events.py scripts/listing_probe.py tests/test_listing_events.py
git commit -m "feat(listing): 상장 감시기 진입점 — 60초 폴링, 감지 시 기록 스레드 기동"
```

`data/listing/`은 커밋하지 않는다. `.gitignore`에 `data/`가 이미 있는지 확인하고 없으면 추가한다.

---

### Task 6: 비용 분석기와 판정 (`analyze.py`)

**Files:**
- Create: `quantbox/listing/analyze.py`
- Test: `tests/test_listing_analyze.py`

**Interfaces:**
- Consumes: `book.{effective_half_spread_bp, parse_levels, top_half_spread_bp}`, `events.EventLog`
- Produces:
  - `ENTRY_WINDOW_H = (1.0, 24.0)`, `TARGET_NOTIONAL_USD = 24.0`
  - `PASS_BP = 25.0`, `EXTEND_BP = 50.0`, `MAX_UNFILLABLE_FRAC = 0.10`
  - `read_depth_samples(symbol, socket_root, onboard_ms, max_files=0) -> list[tuple[float, float, bool]]` — `(hours_since_listing, half_spread_bp, filled)`
  - `read_bookticker_samples(symbol, socket_root, onboard_ms, max_files=0) -> list[tuple[float, float, bool]]` — 기존 58종목 대조용 (1레벨 호가창)
  - `hourly_profile(samples) -> list[dict]` — `{"hour", "n", "median_bp", "p25_bp", "p75_bp", "unfillable_frac"}`
  - `verdict(samples) -> dict` — `{"median_bp", "unfillable_frac", "n", "decision"}` (`decision` ∈ `"진행"`/`"재검토"`/`"종료"`/`"표본부족"`)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_listing_analyze.py`:

```python
"""기록 파일 → 시간대별 스프레드 표 → 사전 등록된 판정.

임계값을 결과를 보고 바꾸지 않는다. 2026-08-12에 과거 리포트에서 합격 기준이
사후 하향된 사례를 발견했고(OOS/IS 0.85→0.50, WR 42%→40%), 같은 날 나 자신도
두 번 그 함정에 빠졌다.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from quantbox.listing.analyze import (
    ENTRY_WINDOW_H,
    EXTEND_BP,
    MAX_UNFILLABLE_FRAC,
    PASS_BP,
    hourly_profile,
    read_depth_samples,
    verdict,
)

SOCKET = Path(__file__).resolve().parents[1] / "data" / "socket"


def test_thresholds_match_the_spec() -> None:
    """사전 등록된 값이다 — 바뀌면 스펙과 대조해야 한다."""
    assert (PASS_BP, EXTEND_BP, MAX_UNFILLABLE_FRAC) == (25.0, 50.0, 0.10)
    assert ENTRY_WINDOW_H == (1.0, 24.0)


def test_verdict_boundaries() -> None:
    """경계값이 어느 쪽에 속하는지 못 박는다 — 25.0bp가 진행인가 재검토인가."""
    assert verdict([(2.0, 10.0, True)] * 200)["decision"] == "진행"
    assert verdict([(2.0, 25.0, True)] * 200)["decision"] == "진행"  # 이하 = 진행
    assert verdict([(2.0, 25.1, True)] * 200)["decision"] == "재검토"
    assert verdict([(2.0, 50.0, True)] * 200)["decision"] == "재검토"  # 이하 = 재검토
    assert verdict([(2.0, 50.1, True)] * 200)["decision"] == "종료"


def test_unfillable_overrides_a_good_spread() -> None:
    """체결 불가가 10%를 넘으면 스프레드가 아무리 좋아도 종료다."""
    samples = [(2.0, 1.0, True)] * 170 + [(2.0, 1.0, False)] * 30  # 15% 체결 불가
    v = verdict(samples)

    assert v["unfillable_frac"] == pytest.approx(0.15)
    assert v["decision"] == "종료", "스프레드 1bp에 가려 체결 불가가 무시됐다"


def test_only_entry_window_counts() -> None:
    """진입 창(1~24h) 밖 표본은 판정에 안 들어간다."""
    samples = [(0.5, 500.0, True)] * 200 + [(5.0, 10.0, True)] * 200  # 상장 30분은 제외
    v = verdict(samples)

    assert v["n"] == 200
    assert v["median_bp"] == pytest.approx(10.0)


def test_too_few_samples_is_not_a_pass() -> None:
    """표본이 적으면 '진행'이 아니라 '표본부족'이다 — 우연히 좋은 값이 통과하면 안 된다."""
    assert verdict([(2.0, 1.0, True)] * 3)["decision"] == "표본부족"


def test_hourly_profile_buckets_by_hour() -> None:
    """시간대별로 나뉘고 각 구간의 분위수가 나온다."""
    samples = [(1.5, 10.0, True), (1.9, 20.0, True), (3.2, 40.0, True)]
    rows = hourly_profile(samples)

    assert [r["hour"] for r in rows] == [1, 3]
    assert rows[0]["n"] == 2
    assert rows[0]["median_bp"] == pytest.approx(15.0)


def test_reads_recorded_depth_files(tmp_path: Path) -> None:
    """기록 형식을 그대로 읽는다 — 기록기 출력과 분석기 입력이 어긋나면 안 된다."""
    d = tmp_path / "depth20" / "TESTUSDT"
    d.mkdir(parents=True)
    onboard_ms = 1_754_924_400_000
    rec = {
        "recv_ns": (onboard_ms + 2 * 3_600_000) * 1_000_000,  # 상장 2시간 후
        "d": {"b": [["100.0", "10.0"]], "a": [["100.1", "10.0"]]},
    }
    with gzip.open(d / "2026-08-11-17.jsonl.gz", "wt") as fh:
        fh.write(json.dumps(rec) + "\n")

    samples = read_depth_samples("TESTUSDT", tmp_path, onboard_ms)

    assert len(samples) == 1
    hours, bp, filled = samples[0]
    assert hours == pytest.approx(2.0)
    assert filled
    assert bp == pytest.approx(5.0, rel=1e-3)  # 스프레드 0.1/100.05 → 편도 ≈ 5bp


@pytest.mark.slow
@pytest.mark.skipif(not (SOCKET / "bookTicker").is_dir(), reason="소켓 데이터 없음")
def test_runs_on_real_collected_data() -> None:
    """기존 58종목 실데이터에서 파이프라인이 끝까지 돈다.

    depth20은 없으므로(수집기가 최우선 호가만 받는다) bookTicker를 1레벨 호가창으로
    읽어 경로만 검증한다. 실효 스프레드 로직 자체는 test_listing_book.py의 실구독이 맡는다.
    """
    from quantbox.listing.analyze import read_bookticker_samples

    sym = next(p.name for p in (SOCKET / "bookTicker").iterdir() if p.is_dir())
    samples = read_bookticker_samples(sym, SOCKET, onboard_ms=0, max_files=1)

    assert samples, f"{sym}에서 표본이 하나도 안 나왔다"
    rows = hourly_profile(samples)
    assert rows and all(r["n"] > 0 for r in rows)
```

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

```bash
uv run pytest tests/test_listing_analyze.py -v
```

기대: `ModuleNotFoundError: quantbox.listing.analyze`.

- [ ] **Step 3: 구현을 쓴다**

`quantbox/listing/analyze.py`:

```python
"""기록 파일 → 경과시간별 스프레드 표 → 사전 등록된 판정.

임계값은 데이터가 오기 전에 정해졌고 결과를 보고 조정하지 않는다. 근거는
어제 실측한 손익분기 왕복 100bp이며, 편도 46bp + 수수료 8.6bp가 그 구성이다.
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
import os
from pathlib import Path

import numpy as np

from quantbox.listing.book import effective_half_spread_bp, parse_levels

logger = logging.getLogger("listing.analyze")

#: 진입 창 — 상장 후 이 구간의 스프레드로 판정한다.
ENTRY_WINDOW_H: tuple[float, float] = (1.0, 24.0)

#: 판정 기준 (사전 등록, 스펙과 일치해야 한다)
TARGET_NOTIONAL_USD = 24.0
PASS_BP = 25.0
EXTEND_BP = 50.0
MAX_UNFILLABLE_FRAC = 0.10
MIN_SAMPLES = 100

Sample = tuple[float, float, bool]
"""``(상장 후 경과 시간(h), 편도 실효 스프레드(bp), 체결 여부)``."""


def _samples_from_file(path: str, onboard_ms: int, extract) -> list[Sample]:
    out: list[Sample] = []
    try:
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bids, asks = extract(rec["d"])
                bp, filled = effective_half_spread_bp(bids, asks, TARGET_NOTIONAL_USD)
                if bp != bp:  # nan — 교차·빈 호가창
                    continue
                hours = (rec["recv_ns"] / 1e6 - onboard_ms) / 3_600_000.0
                out.append((hours, bp, filled))
    except (OSError, EOFError, gzip.BadGzipFile) as e:
        # 수집기가 지금 쓰고 있는 파일은 gzip 스트림이 미완결이다 — 흔한 일이므로 건너뛴다.
        logger.debug(f"{path} 읽기 건너뜀: {e}")
    return out


def read_depth_samples(symbol: str, socket_root: Path, onboard_ms: int, max_files: int = 0) -> list[Sample]:
    """`depth20` 기록에서 표본을 뽑는다."""
    files = sorted(glob.glob(os.path.join(str(socket_root), "depth20", symbol, "*.jsonl.gz")))
    if max_files:
        files = files[:max_files]

    def extract(d: dict):
        return parse_levels(d["b"]), parse_levels(d["a"])

    return [s for f in files for s in _samples_from_file(f, onboard_ms, extract)]


def read_bookticker_samples(symbol: str, socket_root: Path, onboard_ms: int, max_files: int = 0) -> list[Sample]:
    """`bookTicker`를 1레벨 호가창으로 읽는다.

    기존 58종목에는 depth20이 없으므로 파이프라인 검증·대조용으로만 쓴다.
    최우선 호가 물량을 넘는 주문은 여기서 체결 불가로 나온다.
    """
    files = sorted(glob.glob(os.path.join(str(socket_root), "bookTicker", symbol, "*.jsonl.gz")))
    files = files[:-1] if len(files) > 1 else files  # 마지막 파일은 쓰는 중일 수 있다
    if max_files:
        files = files[:max_files]

    def extract(d: dict):
        return [(float(d["b"]), float(d["B"]))], [(float(d["a"]), float(d["A"]))]

    return [s for f in files for s in _samples_from_file(f, onboard_ms, extract)]


def _in_window(samples: list[Sample]) -> list[Sample]:
    lo, hi = ENTRY_WINDOW_H
    return [s for s in samples if lo <= s[0] <= hi]


def hourly_profile(samples: list[Sample]) -> list[dict]:
    """시간대별 스프레드 분포. 진입 창 밖도 포함한다 — 프로파일 전체를 보기 위해서다."""
    buckets: dict[int, list[Sample]] = {}
    for s in samples:
        buckets.setdefault(int(s[0]), []).append(s)

    rows = []
    for hour in sorted(buckets):
        group = buckets[hour]
        bps = np.array([b for _, b, _ in group], dtype=float)
        rows.append(
            {
                "hour": hour,
                "n": len(group),
                "median_bp": float(np.median(bps)),
                "p25_bp": float(np.percentile(bps, 25)),
                "p75_bp": float(np.percentile(bps, 75)),
                "unfillable_frac": float(np.mean([not f for _, _, f in group])),
            }
        )
    return rows


def verdict(samples: list[Sample]) -> dict:
    """진입 창 표본으로 사전 등록된 판정을 낸다."""
    window = _in_window(samples)
    n = len(window)
    if n < MIN_SAMPLES:
        return {"n": n, "median_bp": float("nan"), "unfillable_frac": float("nan"), "decision": "표본부족"}

    median_bp = float(np.median([b for _, b, _ in window]))
    unfillable = float(np.mean([not f for _, _, f in window]))

    if unfillable > MAX_UNFILLABLE_FRAC:
        decision = "종료"  # 스프레드와 무관하다 — 체결이 안 되면 전략이 성립하지 않는다
    elif median_bp <= PASS_BP:
        decision = "진행"
    elif median_bp <= EXTEND_BP:
        decision = "재검토"
    else:
        decision = "종료"

    return {"n": n, "median_bp": median_bp, "unfillable_frac": unfillable, "decision": decision}
```

- [ ] **Step 4: 테스트가 통과하는지 확인한다**

```bash
uv run pytest tests/test_listing_analyze.py -v
uv run pytest tests/test_listing_analyze.py -m slow -v
```

기대: 빠른 테스트 7개 PASS, slow 1개 PASS.

- [ ] **Step 5: 커밋**

```bash
uv run ruff format quantbox/listing tests/test_listing_analyze.py
uv run ruff check quantbox/listing --fix
git add quantbox/listing/analyze.py tests/test_listing_analyze.py
git commit -m "feat(listing): 비용 분석기 + 사전 등록 판정 — 체결 불가가 스프레드를 덮어쓴다"
```

---

### Task 7: 판정 리포트 진입점과 운영 기동

**Files:**
- Create: `scripts/listing_report.py`
- Modify: `README.md` (구조 섹션에 `quantbox/listing/`, `quantbox/collect/` 추가)
- Modify: `AGENTS.md` (새 모듈 등재)

**Interfaces:**
- Consumes: `analyze.{hourly_profile, read_depth_samples, verdict}`, `events.EventLog`
- Produces: 사람이 읽는 표준 출력. 자동 판정 없음 — 사람이 보고 결정한다.

- [ ] **Step 1: 리포트 스크립트를 쓴다**

`scripts/listing_report.py`:

```python
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
```

- [ ] **Step 2: 실행 확인**

```bash
uv run python scripts/listing_report.py
```

기대: 이벤트가 없으므로 `기록된 이벤트가 없다...` 출력 후 종료 코드 1. 크래시하지 않아야 한다.

- [ ] **Step 3: 문서를 갱신한다**

CLAUDE.md의 수정 영향 매트릭스가 요구한다 — "Any new module → `README.md` structure, `__init__.py`, test file, `AGENTS.md`".

`README.md` 구조 섹션에 추가:

```
quantbox/collect/     소켓 수집 공용 부품 (RotatingWriter)
quantbox/listing/     신규 상장 비용 측정기 (연구 전용, 주문 없음)
```

`AGENTS.md`에 같은 두 줄과 한 줄 설명을 추가한다.

- [ ] **Step 4: 전체 테스트를 돌린다**

```bash
uv run pytest tests/ -v
uv run mypy
uv run ruff check quantbox/ scripts/
```

기대: 전부 통과. 기존 테스트가 깨지면 Task 3의 추출이 원인일 가능성이 높다.

- [ ] **Step 5: 커밋**

```bash
git add scripts/listing_report.py README.md AGENTS.md
git commit -m "feat(listing): 판정 리포트 진입점 + 문서 등재"
```

- [ ] **Step 6: 감시기를 기동한다**

기존 `collector` 세션은 **건드리지 않는다**.

```bash
tmux list-sessions                      # collector가 살아 있는지 먼저 확인
tmux new-session -d -s listing \
  "cd /home/user/Documents/git/quantbox && uv run python scripts/listing_probe.py 2>&1 | tee -a data/listing/probe.log"
sleep 5 && tmux capture-pane -p -t listing | tail -20
```

기대: `상장 감시 시작` + `첫 실행 — 기존 859심볼을 등록하고 시작한다` (또는 `--once`를 이미 돌렸으면 그 줄 없이 조용히 대기).

확인 후 사용자에게 보고한다: 감시 가동 중, 판정에는 크립토 상장 3건 필요, 실측 주당 1.09건이므로 약 3주 예상, 6주 안에 안 모이면 종료 조건 발동.

---

## Self-Review

**스펙 커버리지**

| 스펙 항목 | 담당 |
|---|---|
| ① 상장 감시기 (exchangeInfo 60초) | Task 2 + Task 5 |
| ② 소켓 기록기 (3스트림, 7일) | Task 4 |
| ③ 비용 분석기 | Task 6 |
| 대상 필터 (COIN/PERPETUAL/USDT) | Task 2 |
| 실효 스프레드 (24달러 체결) | Task 1 |
| 사전 등록 판정 (25/50bp, 체결불가 10%) | Task 6 |
| 첫 실행 폭주 방지 | Task 2 `bootstrap` |
| 알 수 없는 `underlyingType` 경보 | Task 2 `unknown_types` + Task 5 로그 |
| 디스크 상한 5GB | Task 4 `record(max_gb=)` |
| `PENDING_TRADING` 48h 타임아웃 | Task 5 |
| 재시작 중복 방지 | Task 2 `SeenStore` + Task 5 |
| 기존 수집기 미재시작 | Global Constraints + Task 3 Step 5 검증 |
| 저장 레이아웃 공유 | Task 3 (writer 추출) |
| 이벤트 빈도가 전략 상한이라는 사실 | Task 7 Step 6 보고 |

**미해결로 남기는 것 (의도됨)**

- 실효 스프레드 로직은 과거 데이터로 회귀 고정할 수 없다 — 기존 58종목에 `depth20`이 없다. 스펙이 이 한계를 명시했고, Task 1 Step 5와 Task 4의 실구독 테스트로 대신한다.
- 3건이 모이기 전에는 종합 판정이 나오지 않는다. Task 7의 리포트는 그 상태를 명시적으로 출력한다.
