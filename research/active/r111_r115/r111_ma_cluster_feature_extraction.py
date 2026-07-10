#!/usr/bin/env python3
"""r111: MA Cluster Feature Extraction + MI Pre-Check.

Extracts 8 orthogonal features per MA Cluster trade entry for ML quality scoring.
Phase 1.5: Mutual information pre-check gates further ML development.

(ref: r92 data loading pattern, plan at .omc/plans/ma-cluster-ml-experiment.md)

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r111_r115/r111_ma_cluster_feature_extraction.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    build_resampled,
    compute_metrics,
    portfolio_sim,
    sim_ma_cluster_trailing,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig
from quantbox.strategy.vol_straddle.signals import compute_momentum

# ── Config ──────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

RATIO = 16  # 15m → 4h
WARMUP_4H = 200
DATA_YEARS = 6.0

MA_PARAMS = {
    "cluster_pct": cfg.ma_cluster_pct,
    "min_duration": cfg.ma_cluster_min_duration,
    "max_duration": cfg.ma_cluster_max_duration,
    "min_momentum": cfg.ma_cluster_min_momentum,
    "sl": cfg.ma_cluster_sl,
    "tp": cfg.ma_cluster_tp,
    "trail_activate": cfg.ma_cluster_trail_activate,
    "trail_distance": cfg.ma_cluster_trail_distance,
    "max_hold": cfg.ma_cluster_max_hold_4h,
    "cost": 0.0008,
    "warmup": WARMUP_4H,
}
BTC_GATE_PERIOD = cfg.ma_cluster_btc_ma_period_4h  # 50


# ── Data Loading (r92 패턴) ─────────────────────────────────────────────────

def load_15m_data(whitelist: list[str]) -> dict:
    """15m 캐시 데이터 로드. BTCUSDT는 항상 포함."""
    cache_dir = ROOT / "data" / "cache_extended"
    all_syms = set(whitelist) | {"BTCUSDT"}
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in all_syms:
            continue
        nb = int(np.load(str(f), allow_pickle=False)["close"].shape[0])
        if sym not in seen or nb > seen[sym][1]:
            seen[sym] = (f, nb)
    data: dict = {}
    for sym, (fpath, _) in sorted(seen.items()):
        npz = np.load(str(fpath), allow_pickle=False)
        if not {"close", "high", "low", "volume"}.issubset(set(npz.files)):
            continue
        c = npz["close"].astype(np.float64)
        if len(c) < 500:
            continue
        data[sym] = {k: npz[k].astype(np.float64) for k in npz.files if k != "timestamps"}
    if data:
        max_bars = max(len(d["close"]) for d in data.values())
        for d in data.values():
            d["_offset"] = max_bars - len(d["close"])
    return data


def load_numtrades_data(whitelist: list[str]) -> dict[str, np.ndarray]:
    """num_trades 15m 데이터 로드 (있는 심볼만)."""
    cache_dir = ROOT / "data" / "cache_extended"
    result: dict[str, np.ndarray] = {}
    for f in sorted(cache_dir.glob("*_15m_numtrades_2190d_*.npz")):
        sym = f.name.split("_")[0]
        if sym not in whitelist:
            continue
        npz = np.load(str(f), allow_pickle=False)
        if "num_trades" in npz.files:
            result[sym] = npz["num_trades"].astype(np.float64)
    return result


# ── Helper: SMA / ATR 계산 ──────────────────────────────────────────────────

def compute_sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling SMA (numba 안 쓰는 순수 numpy 버전)."""
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    s = float(np.sum(arr[:period]))
    out[period - 1] = s / period
    for i in range(period, len(arr)):
        s += arr[i] - arr[i - period]
        out[i] = s / period
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR(period) — True Range의 SMA."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return compute_sma(tr, period)


def resample_to_4h(arr_15m: np.ndarray, ratio: int, agg: str = "last") -> np.ndarray:
    """15m 배열을 4h로 리샘플링."""
    n4 = len(arr_15m) // ratio
    if agg == "last":
        return arr_15m[ratio - 1 : n4 * ratio : ratio].copy()
    elif agg == "sum":
        return np.array([np.sum(arr_15m[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    elif agg == "max":
        return np.array([np.max(arr_15m[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    elif agg == "min":
        return np.array([np.min(arr_15m[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    raise ValueError(f"Unknown agg: {agg}")


# ── Feature Extraction ──────────────────────────────────────────────────────

def extract_features_for_symbol(
    sym: str,
    d4: dict,
    entry_bars_4h: np.ndarray,
    pnl_arr: np.ndarray,
    btc_close_4h: np.ndarray,
    btc_sma50_4h: np.ndarray,
    btc_offset_4h: int,
    numtrades_4h: np.ndarray | None,
) -> list[dict]:
    """심볼의 각 트레이드 진입 시점에서 8개 피처 추출.

    모든 피처는 COMPLETED bar (entry bar 자체 또는 이전 bar)에서만 추출.
    entry_bars_4h: 4h bar 인덱스 (sim_ma_cluster_trailing 출력).
    """
    close = d4["close"]
    high = d4["high"]
    low = d4["low"]
    volume = d4["volume"]
    mom8 = d4["mom_8"]
    n4 = len(close)

    # 사전 계산: rolling 통계 (20-bar window)
    vol_sma20 = compute_sma(volume, 20)
    atr20 = compute_atr(high, low, close, 20)

    # BTC 정렬 (r92 패턴과 동일)
    if len(btc_close_4h) >= btc_offset_4h + n4:
        btc_c = btc_close_4h[btc_offset_4h : btc_offset_4h + n4]
        btc_m = btc_sma50_4h[btc_offset_4h : btc_offset_4h + n4]
    else:
        btc_c = np.full(n4, np.nan)
        btc_m = np.full(n4, np.nan)

    # numtrades rolling mean
    nt_sma20 = compute_sma(numtrades_4h, 20) if numtrades_4h is not None else None

    rows: list[dict] = []
    for k in range(len(entry_bars_4h)):
        i = int(entry_bars_4h[k])  # 4h bar index (진입 시점, completed bar)
        if i < 20 or i >= n4:
            continue

        pnl_val = float(pnl_arr[k])
        direction = 1 if mom8[i] > 0 else -1

        # ── Feature 1: volume_ratio_4h ──
        # entry bar의 거래량 / 20-bar 평균 거래량
        # bar i 사용 (completed bar, 캔들 종가 이후)
        vr4h = volume[i] / vol_sma20[i] if vol_sma20[i] > 0 and not np.isnan(vol_sma20[i]) else np.nan

        # ── Feature 2: taker_buy_ratio_4h ──
        # taker_buy_volume가 데이터에 없으므로, volume 대비 (high-close)/(high-low) proxy 사용
        # 실제 taker buy volume이 있으면 그것을 사용
        # 현재 데이터에 taker_buy_vol이 없으므로 NaN → 나중에 드롭 여부 결정
        taker_buy_ratio = np.nan  # 데이터 미보유 — 아래에서 proxy 대체
        # Proxy: (close - low) / (high - low) — 양봉 비율
        bar_range = high[i] - low[i]
        if bar_range > 0:
            taker_buy_ratio = (close[i] - low[i]) / bar_range
        else:
            taker_buy_ratio = 0.5  # doji

        # ── Feature 3: num_trades_ratio ──
        # entry bar num_trades / 20-bar 평균
        if numtrades_4h is not None and nt_sma20 is not None and i < len(numtrades_4h):
            nt_val = numtrades_4h[i]
            nt_mean = nt_sma20[i]
            num_trades_ratio = nt_val / nt_mean if nt_mean > 0 and not np.isnan(nt_mean) else np.nan
        else:
            num_trades_ratio = np.nan

        # ── Feature 4: range_ratio ──
        # (high - low) / ATR(20) at entry bar
        atr_val = atr20[i]
        range_ratio = bar_range / atr_val if atr_val > 0 and not np.isnan(atr_val) else np.nan

        # ── Feature 5: btc_ma50_distance ──
        # (BTC_close - BTC_MA50) / BTC_close — BTC gate의 continuous 버전
        if i < len(btc_c) and not np.isnan(btc_c[i]) and not np.isnan(btc_m[i]) and btc_c[i] > 0:
            btc_dist = (btc_c[i] - btc_m[i]) / btc_c[i]
        else:
            btc_dist = np.nan

        # ── Feature 6: atr_pct_20 ──
        # ATR(20) / close — 변동성 컨텍스트
        atr_pct = atr_val / close[i] if close[i] > 0 and not np.isnan(atr_val) else np.nan

        # ── Feature 7: pre_squeeze_volume_trend ──
        # 클러스터 형성 기간 동안 volume의 회귀 기울기 / 평균
        # 클러스터 기간 = min_duration bars (진입 직전까지)
        dur = cfg.ma_cluster_min_duration
        if i >= dur:
            vol_window = volume[i - dur : i]
            vol_mean = np.mean(vol_window)
            if vol_mean > 0 and len(vol_window) >= 2:
                # 단순 선형 회귀 기울기: slope = cov(x,y) / var(x)
                x = np.arange(dur, dtype=np.float64)
                x_mean = np.mean(x)
                y_mean = np.mean(vol_window)
                cov_xy = np.sum((x - x_mean) * (vol_window - y_mean))
                var_x = np.sum((x - x_mean) ** 2)
                slope = cov_xy / var_x if var_x > 0 else 0.0
                vol_trend = slope / vol_mean
            else:
                vol_trend = np.nan
        else:
            vol_trend = np.nan

        # ── Feature 8: breakout_direction ──
        # +1 (LONG) or -1 (SHORT) — momentum 방향
        breakout_dir = float(direction)

        rows.append({
            "symbol": sym,
            "entry_bar_4h": i,
            "pnl": pnl_val,
            "profitable": 1 if pnl_val > 0 else 0,
            "direction": direction,
            "volume_ratio_4h": vr4h,
            "taker_buy_ratio_4h": taker_buy_ratio,
            "num_trades_ratio": num_trades_ratio,
            "range_ratio": range_ratio,
            "btc_ma50_distance": btc_dist,
            "atr_pct_20": atr_pct,
            "pre_squeeze_volume_trend": vol_trend,
            "breakout_direction": breakout_dir,
        })

    return rows


# ── MI Pre-Check (scipy 기반) ───────────────────────────────────────────────

def histogram_mi(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """히스토그램 기반 MI 추정.

    MI(X;Y) = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    y는 binary label (0/1).
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask].astype(int)
    if len(x_clean) < 30:
        return 0.0

    # x를 등분위 bin으로 분할
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x_clean, percentiles)
    edges[-1] += 1e-10  # 마지막 bin 포함
    x_binned = np.digitize(x_clean, edges) - 1
    x_binned = np.clip(x_binned, 0, n_bins - 1)

    # Joint + marginal 확률
    n = len(x_clean)
    mi = 0.0
    for xi in range(n_bins):
        mask_x = x_binned == xi
        px = np.sum(mask_x) / n
        if px <= 0:
            continue
        for yi in range(2):
            mask_xy = mask_x & (y_clean == yi)
            pxy = np.sum(mask_xy) / n
            py = np.sum(y_clean == yi) / n
            if pxy > 0 and py > 0:
                mi += pxy * np.log(pxy / (px * py))
    return max(mi, 0.0)


def mi_permutation_test(
    x: np.ndarray, y: np.ndarray, n_perms: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """MI + 순열 검정. p-value = 순열 MI >= 관측 MI인 비율."""
    rng = np.random.default_rng(seed)
    observed_mi = histogram_mi(x, y)
    count = 0
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        if histogram_mi(x, y_perm) >= observed_mi:
            count += 1
    p_value = count / n_perms
    return observed_mi, p_value


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("r111: MA Cluster Feature Extraction + MI Pre-Check")
    print("=" * 65)
    print(f"Config: cluster_pct={MA_PARAMS['cluster_pct']} min_dur={MA_PARAMS['min_duration']} "
          f"max_dur={MA_PARAMS['max_duration']} sl={MA_PARAMS['sl']} "
          f"trail_act={MA_PARAMS['trail_activate']} trail_dist={MA_PARAMS['trail_distance']}")
    print()

    # ── 1. 데이터 로드 ──
    print("Loading 15m data...", end=" ", flush=True)
    data_15m = load_15m_data(cfg.symbol_whitelist)
    print(f"{len(data_15m)} symbols")

    if "BTCUSDT" not in data_15m:
        print("ERROR: BTCUSDT not found")
        return

    # numtrades 로드
    numtrades_15m = load_numtrades_data(cfg.symbol_whitelist)
    print(f"numtrades available: {len(numtrades_15m)}/{len(cfg.symbol_whitelist)} symbols")

    # ── 2. 4h 리샘플링 ──
    print("Resampling to 4h...", end=" ", flush=True)
    data_4h = build_resampled(data_15m, RATIO)
    print(f"{len(data_4h)} symbols")

    # SMA + momentum 사전 계산 (collect_ma_cluster_trades와 동일한 로직)
    for sym, d4 in data_4h.items():
        c = d4["close"]
        n = len(c)
        for period, key in ((20, "sma_20"), (50, "sma_50"), (100, "sma_100"), (200, "sma_200")):
            if key not in d4:
                d4[key] = compute_sma(c, period)
        if "mom_8" not in d4:
            d4["mom_8"] = compute_momentum(c, 8)

    # BTC 4h 준비
    btc_4h_close = data_4h["BTCUSDT"]["close"]
    btc_4h_sma50 = compute_sma(btc_4h_close, BTC_GATE_PERIOD)

    # numtrades 4h 리샘플링
    numtrades_4h: dict[str, np.ndarray] = {}
    for sym, nt_15m in numtrades_15m.items():
        if sym in data_15m:
            n4 = len(nt_15m) // RATIO
            if n4 > 0:
                numtrades_4h[sym] = np.array([
                    np.sum(nt_15m[i : i + RATIO]) for i in range(0, n4 * RATIO, RATIO)
                ])

    # ── 3. 트레이드 수집 + 피처 추출 ──
    print("\nCollecting trades & extracting features...", flush=True)
    all_rows: list[dict] = []
    syms = [s for s in cfg.symbol_whitelist if s in data_4h and s != "BTCUSDT"]

    for sym in syms:
        d4 = data_4h[sym]
        if "sma_200" not in d4 or "mom_8" not in d4:
            continue

        offset = int(data_15m[sym].get("_offset", 0))
        btc_offset_4h = offset // RATIO

        # sim_ma_cluster_trailing: 4h bar 인덱스로 반환
        pnl, ent_4h, ext_4h = sim_ma_cluster_trailing(
            d4["close"], d4["high"], d4["low"],
            d4["sma_20"], d4["sma_50"], d4["sma_100"], d4["sma_200"],
            d4["mom_8"],
            MA_PARAMS["cluster_pct"], MA_PARAMS["min_duration"],
            MA_PARAMS["max_duration"], MA_PARAMS["min_momentum"],
            MA_PARAMS["sl"], MA_PARAMS["tp"],
            MA_PARAMS["trail_activate"], MA_PARAMS["trail_distance"],
            MA_PARAMS["max_hold"], MA_PARAMS["cost"], MA_PARAMS["warmup"],
        )

        if len(pnl) == 0:
            continue

        # 볼륨 4h: build_resampled이 이미 처리했으므로 d4["volume"] 사용
        nt_4h = numtrades_4h.get(sym)

        rows = extract_features_for_symbol(
            sym, d4, ent_4h, pnl,
            btc_4h_close, btc_4h_sma50, btc_offset_4h,
            nt_4h,
        )
        all_rows.extend(rows)
        if len(rows) > 0:
            print(f"  {sym}: {len(rows)} trades", flush=True)

    if not all_rows:
        print("ERROR: No trades collected")
        return

    # ── 4. DataFrame 생성 ──
    df = pd.DataFrame(all_rows)
    feature_cols = [
        "volume_ratio_4h", "taker_buy_ratio_4h", "num_trades_ratio",
        "range_ratio", "btc_ma50_distance", "atr_pct_20",
        "pre_squeeze_volume_trend", "breakout_direction",
    ]

    # ── 5. Dataset Summary ──
    print("\n" + "=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    n_total = len(df)
    n_profitable = int(df["profitable"].sum())
    n_loss = n_total - n_profitable
    print(f"Total trades: {n_total}")
    print(f"Profitable:   {n_profitable} ({100 * n_profitable / n_total:.1f}%)")
    print(f"Loss:         {n_loss} ({100 * n_loss / n_total:.1f}%)")
    print(f"Symbols:      {df['symbol'].nunique()}")
    print(f"Mean PnL:     {df['pnl'].mean() * 10000:.1f} bp")
    print(f"Median PnL:   {df['pnl'].median() * 10000:.1f} bp")

    # NaN 비율 확인
    print(f"\n{'Feature':<30} {'NaN%':>6}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print("-" * 80)
    for col in feature_cols:
        nan_pct = 100 * df[col].isna().sum() / n_total
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"{col:<30} {nan_pct:5.1f}%  {vals.mean():10.4f}  {vals.std():10.4f}  {vals.min():10.4f}  {vals.max():10.4f}")
        else:
            print(f"{col:<30} {nan_pct:5.1f}%  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")

    # num_trades_ratio: >10% NaN이면 경고
    nt_nan_pct = 100 * df["num_trades_ratio"].isna().sum() / n_total
    if nt_nan_pct > 10:
        print(f"\n  WARNING: num_trades_ratio has {nt_nan_pct:.1f}% NaN (>10%) — consider dropping")

    # ── 6. Correlation Matrix ──
    print("\n" + "=" * 65)
    print("CORRELATION MATRIX (|r| > 0.85 flagged)")
    print("=" * 65)
    corr = df[feature_cols].corr()
    # 피처명 줄임
    short_names = {
        "volume_ratio_4h": "vol_r",
        "taker_buy_ratio_4h": "tkr_buy",
        "num_trades_ratio": "ntrade",
        "range_ratio": "rng_r",
        "btc_ma50_distance": "btc_d",
        "atr_pct_20": "atr_p",
        "pre_squeeze_volume_trend": "vol_trnd",
        "breakout_direction": "bkout_d",
    }
    header = f"{'':>10}" + "".join(f"{short_names[c]:>10}" for c in feature_cols)
    print(header)
    high_corr_pairs: list[tuple[str, str, float]] = []
    for i_row, row_col in enumerate(feature_cols):
        line = f"{short_names[row_col]:>10}"
        for j_col, col_col in enumerate(feature_cols):
            r = corr.loc[row_col, col_col]
            if np.isnan(r):
                line += f"{'NaN':>10}"
            else:
                line += f"{r:10.3f}"
            if i_row < j_col and abs(r) > 0.85:
                high_corr_pairs.append((row_col, col_col, r))
        print(line)

    if high_corr_pairs:
        print(f"\n  FLAGGED high-correlation pairs (|r| > 0.85):")
        for a, b, r in high_corr_pairs:
            print(f"    {a} <-> {b}: r={r:.3f}")
    else:
        print("\n  No high-correlation pairs found (all |r| <= 0.85)")

    # ── 7. MI Pre-Check ──
    print("\n" + "=" * 65)
    print("MUTUAL INFORMATION PRE-CHECK (permutation test, n=1000)")
    print("=" * 65)
    label = df["profitable"].values.astype(np.float64)
    mi_results: list[dict] = []

    print(f"{'Feature':<30} {'MI':>8} {'p-value':>8} {'Sig':>5}")
    print("-" * 55)
    for col in feature_cols:
        vals = df[col].values.astype(np.float64)
        # NaN이 많은 피처는 유효한 행만 사용
        mask = ~np.isnan(vals)
        if np.sum(mask) < 50:
            print(f"{col:<30} {'N/A':>8} {'N/A':>8} {'N/A':>5}")
            mi_results.append({"feature": col, "mi": np.nan, "p_value": np.nan, "significant": False})
            continue
        mi_val, p_val = mi_permutation_test(vals[mask], label[mask], n_perms=1000)
        sig = p_val < 0.05
        marker = " *" if sig else ""
        print(f"{col:<30} {mi_val:8.5f} {p_val:8.3f} {'YES' if sig else 'no':>5}{marker}")
        mi_results.append({"feature": col, "mi": mi_val, "p_value": p_val, "significant": sig})

    n_sig = sum(1 for r in mi_results if r["significant"])
    gate_pass = n_sig >= 2
    print(f"\nSignificant features (p<0.05): {n_sig}/8")
    print(f"MI Gate (>=2 significant):     {'PASS' if gate_pass else 'FAIL'}")

    # ── 8. Lookahead Audit ──
    print("\n" + "=" * 65)
    print("LOOKAHEAD AUDIT")
    print("=" * 65)
    audit = [
        ("volume_ratio_4h",            "bar[i] / SMA(bar[i-19:i+1])",      "i = entry bar (completed)"),
        ("taker_buy_ratio_4h",         "(close-low)/(high-low) at bar[i]",  "i = entry bar (completed)"),
        ("num_trades_ratio",           "bar[i] / SMA(bar[i-19:i+1])",      "i = entry bar (completed)"),
        ("range_ratio",                "(high-low)/ATR(20) at bar[i]",      "i = entry bar, ATR uses [i-19:i+1]"),
        ("btc_ma50_distance",          "(BTC_close-BTC_MA50)/BTC_close",    "i = entry bar (completed)"),
        ("atr_pct_20",                 "ATR(20)/close at bar[i]",           "i = entry bar, ATR uses [i-19:i+1]"),
        ("pre_squeeze_volume_trend",   "slope(vol[i-dur:i]) / mean",       "bar[i-dur] to bar[i-1] (strictly before entry)"),
        ("breakout_direction",         "sign(momentum[i])",                 "i = entry bar (completed)"),
    ]
    print(f"{'Feature':<30} {'Computation':<35} {'Bar Index Used'}")
    print("-" * 95)
    for feat, comp, bar_note in audit:
        print(f"{feat:<30} {comp:<35} {bar_note}")
    print("\nAll features use COMPLETED bars only. No future data accessed.")

    # ── 9. Save to Parquet ──
    out_dir = ROOT / "data" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ma_cluster_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows, {out_path.stat().st_size / 1024:.1f} KB)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
