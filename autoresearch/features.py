"""진입 시점 피처 추출 — 룩어헤드 없음.

각 거래의 **진입 4h 봉까지의 정보만** 쓴다. 미래 봉을 한 개라도 보면 그 자리에서
결과가 무의미해지므로, 모든 창은 `[:i+1]` 로 닫는다.

피처는 "이 진입이 좋은 진입인가"를 가를 만한 것들로 고른다:
  시장 상태(BTC 변동성·추세) · 종목 상태(변동성·거래량·나이) · 신호 품질(모멘텀·클러스터 폭)
"""

from __future__ import annotations

import numpy as np

BARS_4H_PER_DAY = 6

FEATURE_NAMES = [
    "btc_vol_1d",
    "btc_vol_7d",
    "btc_trend",
    "sym_vol_1d",
    "vol_ratio",
    "mom_abs",
    "cluster_width",
    "dist_sma200",
    "vol_z",
    "sym_age",
    "ret_1d",
    "ret_7d",
    "direction",
    "dollar_vol_log",
]


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    """각 시점까지의 직전 w개 수익률 표준편차 (그 시점 포함, 미래 미사용)."""
    r = np.zeros(len(x))
    r[1:] = np.diff(x) / np.maximum(x[:-1], 1e-12)
    c1, c2 = np.cumsum(r), np.cumsum(r * r)
    out = np.full(len(x), np.nan)
    if len(x) > w:
        s1 = c1[w:] - c1[:-w]
        s2 = c2[w:] - c2[:-w]
        out[w:] = np.sqrt(np.maximum(s2 / w - (s1 / w) ** 2, 0.0))
    return out


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        c = np.concatenate([[0.0], np.cumsum(x)])
        out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def build_symbol_features(d4: dict) -> dict[str, np.ndarray]:
    """심볼의 4h 배열에서 봉별 피처를 만든다."""
    c, v = d4["close"], d4["volume"]
    n = len(c)
    sma20, sma50, sma100, sma200 = (_sma(c, p) for p in (20, 50, 100, 200))
    spread = np.abs(np.stack([sma20, sma50, sma100, sma200]) - sma20) / np.maximum(np.abs(sma20), 1e-12)
    width = np.nanmax(spread, axis=0)

    mom = np.full(n, np.nan)
    mom[8:] = c[8:] / np.maximum(c[:-8], 1e-12) - 1.0

    vol_sma = _sma(v, 100)
    vol_sd = _roll_std(np.maximum(v, 1e-12), 100)

    ret1d = np.full(n, np.nan)
    ret1d[BARS_4H_PER_DAY:] = c[BARS_4H_PER_DAY:] / np.maximum(c[:-BARS_4H_PER_DAY], 1e-12) - 1.0
    w7 = BARS_4H_PER_DAY * 7
    ret7d = np.full(n, np.nan)
    if n > w7:
        ret7d[w7:] = c[w7:] / np.maximum(c[:-w7], 1e-12) - 1.0

    return {
        "sym_vol_1d": _roll_std(c, BARS_4H_PER_DAY),
        "mom_abs": np.abs(mom),
        "cluster_width": width,
        "dist_sma200": (c - sma200) / np.maximum(np.abs(sma200), 1e-12),
        "vol_z": (v - vol_sma) / np.maximum(vol_sd, 1e-12),
        "ret_1d": ret1d,
        "ret_7d": ret7d,
        "dollar_vol_log": np.log1p(np.maximum(v, 0.0) * np.maximum(c, 0.0)),
    }


def build_btc_features(btc4: dict) -> dict[str, np.ndarray]:
    c = btc4["close"]
    sma50 = _sma(c, 50)
    return {
        "btc_vol_1d": _roll_std(c, BARS_4H_PER_DAY),
        "btc_vol_7d": _roll_std(c, BARS_4H_PER_DAY * 7),
        "btc_trend": (c - sma50) / np.maximum(np.abs(sma50), 1e-12),
    }


def extract(
    data15: dict, data4: dict, trades: list, dirs: list, ratio: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, entry_bar) — 거래별 피처 행렬.

    Args:
        data15: 15m OHLCV (`_offset` 포함).
        data4: 4h 리샘플.
        trades: `[(sym, entry_bar_15m, exit_bar_15m, pnl)]`.
        dirs: 거래별 방향.
        ratio: 15m→4h 비율.

    Returns:
        X (n, F), y (n,), entry_bar (n,). NaN 행은 제거되지 않는다 — 호출자가 처리.
    """
    btc4 = data4.get("BTCUSDT")
    btc_f = build_btc_features(btc4) if btc4 is not None else None
    btc_off = int(data15["BTCUSDT"]["_offset"]) if "BTCUSDT" in data15 else 0

    cache: dict[str, dict[str, np.ndarray]] = {}
    rows, ys, ebs = [], [], []

    for (sym, eb, _xb, pnl), dr in zip(trades, dirs, strict=True):
        d4 = data4.get(sym)
        if d4 is None:
            continue
        if sym not in cache:
            cache[sym] = build_symbol_features(d4)
        f = cache[sym]

        off = int(data15[sym].get("_offset", 0))
        i4 = (eb - off) // ratio
        if i4 < 0 or i4 >= len(d4["close"]):
            continue

        row = [
            btc_f["btc_vol_1d"][(eb - btc_off) // ratio] if btc_f is not None else np.nan,
            btc_f["btc_vol_7d"][(eb - btc_off) // ratio] if btc_f is not None else np.nan,
            btc_f["btc_trend"][(eb - btc_off) // ratio] if btc_f is not None else np.nan,
            f["sym_vol_1d"][i4],
            np.nan,  # vol_ratio — 아래에서 채움
            f["mom_abs"][i4],
            f["cluster_width"][i4],
            f["dist_sma200"][i4],
            f["vol_z"][i4],
            float(i4),  # sym_age: 상장 후 경과 4h 봉 수
            f["ret_1d"][i4],
            f["ret_7d"][i4],
            float(dr),
            f["dollar_vol_log"][i4],
        ]
        row[4] = row[3] / max(row[0], 1e-12) if np.isfinite(row[3]) and np.isfinite(row[0]) else np.nan
        rows.append(row)
        ys.append(pnl)
        ebs.append(eb)

    return np.asarray(rows, dtype=np.float64), np.asarray(ys), np.asarray(ebs)
