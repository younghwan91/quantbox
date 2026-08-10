"""캐시 OHLCV 로딩 — 유일 구현.

research/ 전역에 `load_data`가 16벌 복제돼 있었고, 동작이 **6종으로 갈라져** 있었다.
그중 4벌은 `_offset`을 아예 설정하지 않았다 — `_offset`은 상장일이 달라 길이가 다른
심볼들을 공통 타임라인에 정렬하는 값이라, 빠지면 포트폴리오 sim이 심볼 간 봉이
어긋난 채로 돌아간다(= 조용히 틀린 백테스트). 최소 봉수도 500/300/무제한으로 제각각이었다.

여기서는 정렬을 **선택이 아니라 불변식**으로 만든다: `MarketData.load()`를 거친
데이터는 항상 `_offset`이 채워져 있고, 그 사실을 생성 시점에 검증한다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quantbox.backtest.trade_sim import build_resampled
from quantbox.research.harness.gitinfo import repo_root

logger = logging.getLogger("harness.data")

#: 15m 확장 캐시의 기본 위치/패턴 (research 스크립트들이 수렴해 온 사실상의 표준).
DEFAULT_CACHE_DIR = "data/cache_extended"
DEFAULT_PATTERN = "*_15m_2190d_*.npz"

#: 전 종목 유니버스 캐시 (폐지 종목 포함, 타임스탬프 보유).
UNIVERSE_CACHE_DIR = "data/cache_universe"
UNIVERSE_PATTERN = "*.npz"

#: 15m 봉 간격 (ms). 타임스탬프 기반 정렬의 격자 단위.
BAR_MS = 15 * 60 * 1000

#: 이보다 짧은 시계열은 워밍업(SMA200@4h 등) 이후 남는 봉이 없어 통계가 무의미하다.
MIN_BARS = 500

#: 시장 전체 게이트(BTC ATR 등)에 쓰이므로 화이트리스트에 없어도 기본 포함한다.
MARKET_REF_SYMBOL = "BTCUSDT"

SymbolData = dict[str, Any]


@dataclass(frozen=True)
class MarketData:
    """정렬 불변식이 보장된 심볼별 OHLCV 묶음.

    `data`는 기존 `collect_*` 함수들이 받는 것과 **동일한 형태**의 dict이다:
    `{symbol: {"close": arr, "high": arr, "low": arr, ..., "_offset": int}}`.
    따라서 그대로 넘길 수 있다 — 어댑터가 필요 없다.
    """

    data: dict[str, SymbolData]
    source: Path

    def __post_init__(self) -> None:
        for sym, d in self.data.items():
            if "_offset" not in d:
                raise ValueError(f"{sym}: _offset 누락 — 심볼 간 정렬이 깨진 데이터는 사용할 수 없다")

    @classmethod
    def load(
        cls,
        symbols: list[str] | None = None,
        *,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        pattern: str = DEFAULT_PATTERN,
        min_bars: int = MIN_BARS,
        include_market_ref: bool = True,
    ) -> MarketData:
        """캐시에서 15m OHLCV를 로드하고 공통 타임라인에 정렬한다.

        Args:
            symbols: 로드할 심볼. None이면 캐시에 있는 전부.
            cache_dir: 캐시 디렉토리 (레포 루트 기준 상대경로 허용).
            pattern: npz 글로브 패턴.
            min_bars: 이보다 짧은 심볼은 제외.
            include_market_ref: BTCUSDT를 항상 포함할지 (시장 게이트용).

        Returns:
            `_offset`이 채워진 MarketData.

        Raises:
            FileNotFoundError: 캐시 디렉토리가 없을 때.
        """
        root = Path(cache_dir)
        if not root.is_absolute():
            root = repo_root() / root
        if not root.is_dir():
            raise FileNotFoundError(f"캐시 디렉토리 없음: {root}")

        wanted: set[str] | None = None
        if symbols is not None:
            wanted = set(symbols)
            if include_market_ref:
                wanted.add(MARKET_REF_SYMBOL)

        # 같은 심볼의 npz가 여러 벌이면 가장 긴 것을 쓴다.
        best: dict[str, tuple[Path, int]] = {}
        for f in sorted(root.glob(pattern)):
            if "numtrades" in f.name:
                continue
            sym = f.name.split("_")[0].removesuffix(".npz")
            if wanted is not None and sym not in wanted:
                continue
            n_bars = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
            if sym not in best or n_bars > best[sym][1]:
                best[sym] = (f, n_bars)

        data: dict[str, SymbolData] = {}
        for sym, (fpath, _) in sorted(best.items()):
            npz = np.load(str(fpath), allow_pickle=False)
            if not {"close", "high", "low"}.issubset(set(npz.files)):
                continue
            if len(npz["close"]) < min_bars:
                continue
            data[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}

        cls._align(data)
        return cls(data=data, source=root)

    @staticmethod
    def _align(data: dict[str, SymbolData]) -> None:
        """공통 타임라인 정렬.

        타임스탬프가 있으면 **실제 시각**으로 정렬한다. 없으면 구 캐시 호환을 위해
        우측(최신) 정렬로 떨어지되 경고한다.

        우측 정렬은 "모든 심볼이 같은 시점에 끝난다"고 가정한다. 상장폐지 종목에는
        성립하지 않는다 — 2024년에 폐지된 종목이 현재까지 거래된 것처럼 배치되어
        BTC 게이트·슬롯 경합·일손실 경계가 전부 잘못된 시각으로 계산된다.
        조용히 틀리므로 반드시 타임스탬프를 쓴다.

        Args:
            data: 심볼별 OHLCV dict. 제자리에서 ``_offset``을 채운다.
        """
        if not data:
            return

        if all("timestamps" in d for d in data.values()):
            t0 = min(float(d["timestamps"][0]) for d in data.values())
            for sym, d in data.items():
                ts = d["timestamps"]
                off = (float(ts[0]) - t0) / BAR_MS
                if abs(off - round(off)) > 1e-6:
                    raise ValueError(f"{sym}: 시작 시각이 15m 격자에 맞지 않는다 (offset={off})")
                d["_offset"] = int(round(off))
            return

        logger.warning(
            "타임스탬프 없는 캐시 — 우측 정렬로 대체한다. "
            "폐지 종목이 섞이면 조용히 어긋나므로 cache_universe 사용을 권장한다."
        )
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])

    @staticmethod
    def _contiguity_report(data: dict[str, SymbolData]) -> dict[str, int]:
        """심볼별 내부 결측 구간 수.

        위치 기반 인덱싱(``_offset`` + 배열 위치)은 봉이 **연속**임을 전제한다.
        중간에 빠진 구간이 있으면 그 이후 전부가 시각과 어긋난다.

        Args:
            data: 심볼별 OHLCV dict.

        Returns:
            ``{symbol: 결측 구간 수}`` — 0이 아닌 항목만.
        """
        out: dict[str, int] = {}
        for sym, d in data.items():
            ts = d.get("timestamps")
            if ts is None or len(ts) < 2:
                continue
            n_bad = int(np.count_nonzero(np.diff(ts) != BAR_MS))
            if n_bad:
                out[sym] = n_bad
        return out

    def resample(self, ratio: int) -> dict[str, SymbolData]:
        """상위 TF로 리샘플. 완결된 봉만 사용 (프로덕션 `build_resampled` 재사용)."""
        return build_resampled(self.data, ratio)

    @property
    def symbols(self) -> list[str]:
        return sorted(self.data)

    def __getitem__(self, symbol: str) -> SymbolData:
        return self.data[symbol]

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.data

    def __len__(self) -> int:
        return len(self.data)
