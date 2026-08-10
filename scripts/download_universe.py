#!/usr/bin/env python3
"""Binance 공개 아카이브에서 USDT-M 무기한 15m 전 종목을 받는다.

기존 `download_extended_data.py`와 다른 점:

  1. **폐지 종목을 포함한다.** REST `exchangeInfo`는 현재 상장 종목만 반환하므로
     생존편향을 고칠 수 없다. 아카이브(`data.binance.vision`)에는 폐지 종목이
     남아 있다 — 2026-08-11 기준 USDT 페어 832개, 알려진 폐지 종목 20/20 보존.
  2. **타임스탬프를 저장한다.** 기존 캐시에는 없어서 정렬이 "모든 심볼이 같은
     시점에 끝난다"는 가정으로 위치 기반이었다. 폐지 종목은 그 가정을 깨므로
     타임스탬프 없이는 조용히 어긋난다.
  3. API 키가 필요 없고 레이트 리밋이 사실상 없다 (S3 정적 파일).

사용:
    uv run python scripts/download_universe.py                  # 전 종목
    uv run python scripts/download_universe.py --limit 20       # 앞 20종목만
    uv run python scripts/download_universe.py --symbols BTCUSDT,SRMUSDT
    uv run python scripts/download_universe.py --workers 12

이미 받은 심볼은 건너뛴다 (`--force`로 재수집).
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "cache_universe"
BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision"
PREFIX = "data/futures/um/monthly/klines/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
INTERVAL = "15m"

#: 사전 등록된 제외 규칙 (docs/research/hypotheses/20260811_universe_expansion.md)
MIN_BARS = 5_000
MAX_BAD_FRAC = 0.05

#: 아카이브 CSV 컬럼 (헤더 없음)
C_OPEN_TIME, C_OPEN, C_HIGH, C_LOW, C_CLOSE, C_VOLUME = 0, 1, 2, 3, 4, 5
C_TAKER_BUY = 9


def _get(url: str, timeout: int = 120) -> bytes:
    """단순 GET. 아카이브는 정적 파일이라 재시도만 있으면 충분하다."""
    last: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - 네트워크 오류는 재시도 대상
            last = e
    raise RuntimeError(f"GET 실패: {url} ({last})")


def list_symbols() -> list[str]:
    """아카이브에 존재하는 USDT-M 심볼 전체 (폐지 종목 포함).

    Returns:
        정렬된 USDT 페어 목록.
    """
    syms: list[str] = []
    marker = ""
    while True:
        url = f"{BASE}?delimiter=/&prefix={PREFIX}" + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(_get(url))
        got = [p.find(NS + "Prefix").text.split("/")[-2] for p in root.findall(NS + "CommonPrefixes")]
        syms += got
        if root.find(NS + "IsTruncated").text != "true":
            break
        marker = PREFIX + got[-1] + "/"
    return sorted(s for s in syms if s.endswith("USDT"))


def list_months(symbol: str) -> list[str]:
    """해당 심볼의 월별 파일 목록.

    Args:
        symbol: 예 ``BTCUSDT``.

    Returns:
        ``BTCUSDT-15m-2020-01.zip`` 형태의 파일명 목록 (시간순).
    """
    prefix = f"{PREFIX}{symbol}/{INTERVAL}/"
    names: list[str] = []
    marker = ""
    while True:
        url = f"{BASE}?prefix={prefix}" + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(_get(url))
        keys = [c.find(NS + "Key").text for c in root.findall(NS + "Contents")]
        names += [k.split("/")[-1] for k in keys if k.endswith(".zip")]
        if root.find(NS + "IsTruncated").text != "true":
            break
        marker = keys[-1]
    return sorted(names)


def fetch_month(symbol: str, fname: str) -> np.ndarray | None:
    """월별 zip 하나를 받아 float64 행렬로 파싱한다.

    Args:
        symbol: 심볼.
        fname: ``{SYMBOL}-15m-YYYY-MM.zip``.

    Returns:
        ``(n, 6)`` 배열 [open_time, open, high, low, close, volume, taker_buy] 중
        필요한 열만 담은 ``(n, 7)`` 배열. 파일이 없으면 None.
    """
    url = f"{DL}/{PREFIX}{symbol}/{INTERVAL}/{fname}"
    try:
        blob = _get(url)
    except RuntimeError:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", "replace")

    rows: list[tuple[float, ...]] = []
    for line in raw.splitlines():
        if not line or line[0] not in "0123456789":
            continue  # 일부 파일에 헤더가 있다
        p = line.split(",")
        try:
            rows.append(
                (
                    float(p[C_OPEN_TIME]),
                    float(p[C_OPEN]),
                    float(p[C_HIGH]),
                    float(p[C_LOW]),
                    float(p[C_CLOSE]),
                    float(p[C_VOLUME]),
                    float(p[C_TAKER_BUY]),
                )
            )
        except (ValueError, IndexError):
            continue
    return np.array(rows, dtype=np.float64) if rows else None


def save_symbol(symbol: str, force: bool = False) -> str:
    """심볼 하나를 받아 npz로 저장한다.

    Args:
        symbol: 심볼.
        force: 이미 있어도 다시 받을지.

    Returns:
        결과 문자열 (로그용).
    """
    out = OUT_DIR / f"{symbol}.npz"
    if out.exists() and not force:
        return f"skip {symbol}"

    months = list_months(symbol)
    if not months:
        return f"none {symbol}"

    parts = [m for m in (fetch_month(symbol, f) for f in months) if m is not None]
    if not parts:
        return f"empty {symbol}"

    a = np.concatenate(parts, axis=0)
    a = a[np.argsort(a[:, 0])]
    _, uniq = np.unique(a[:, 0], return_index=True)
    a = a[np.sort(uniq)]

    close = a[:, 4]
    bad = float(np.mean(~np.isfinite(close) | (close <= 0)))
    if len(a) < MIN_BARS or bad > MAX_BAD_FRAC:
        return f"drop {symbol} (bars={len(a)}, bad={bad:.1%})"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        timestamps=a[:, 0],
        open=a[:, 1],
        high=a[:, 2],
        low=a[:, 3],
        close=close,
        volume=a[:, 5],
        taker_buy=a[:, 6],
    )
    return f"ok   {symbol} ({len(a)} bars)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", help="쉼표 구분 심볼. 생략 시 아카이브 전체")
    ap.add_argument("--limit", type=int, help="앞에서 N개만")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    syms = args.symbols.split(",") if args.symbols else list_symbols()
    if args.limit:
        syms = syms[: args.limit]
    print(f"대상 {len(syms)}종목 → {OUT_DIR}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(save_symbol, s, args.force): s for s in syms}
        for fut in as_completed(futs):
            done += 1
            try:
                msg = fut.result()
            except Exception as e:  # noqa: BLE001 - 한 종목 실패가 전체를 막지 않게
                msg = f"FAIL {futs[fut]}: {e}"
            if done % 25 == 0 or not msg.startswith(("ok", "skip")):
                print(f"[{done}/{len(syms)}] {msg}", flush=True)

    n = len(list(OUT_DIR.glob("*.npz"))) if OUT_DIR.is_dir() else 0
    print(f"완료. 저장된 심볼 {n}개", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
