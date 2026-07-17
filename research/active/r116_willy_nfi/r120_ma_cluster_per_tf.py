"""r120: MA Cluster Per-TF Breakdown — 15m / 30m / 1h / 4h 독립 성능 분석.

동기:
  모든 백테스트(r92, r115, r119)는 4h 전용 시뮬레이션.
  그러나 live는 enable_15m/30m/1h/4h 모두 ON.
  최근 라이브 저널에서 bars_held이 짧은 트레이드(37봉, 56봉)는
  15m/30m 진입으로 추정 — 이 TF들의 PF를 측정한 적이 없음.

질문:
  - 짧은 TF(15m, 30m, 1h) MA cluster가 4h만큼 수익인가?
  - 어떤 TF를 끄는 게 전체 시스템에 유리한가?

방법:
  각 TF별 독립 sim: 해당 TF의 bar를 사용, BTC MA50 gate는 4h 기준 유지.
  max_hold: 15m=192봉(48h), 30m=96봉(48h), 1h=48봉(48h), 4h=6봉(24h)
  공통 파라미터: cluster_pct, min_duration, max_duration, min_momentum,
                sl, trail_activate, trail_distance (config에서 읽기)

판단 기준:
  - PF >= 1.5 AND WR >= 55% AND MaxDD <= 30% → 유지
  - PF < 1.2 → 끄기 권장
  - 1.2 <= PF < 1.5 → 조건부 (추가 연구)

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r116_willy_nfi/r120_ma_cluster_per_tf.py
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import bootstrap_pf, compute_metrics, portfolio_sim
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config ───────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

SL             = cfg.ma_cluster_sl
TRAIL_ACTIVATE = cfg.ma_cluster_trail_activate
TRAIL_DISTANCE = cfg.ma_cluster_trail_distance
CLUSTER_PCT    = cfg.ma_cluster_pct
MIN_DUR        = cfg.ma_cluster_min_duration
MAX_DUR        = cfg.ma_cluster_max_duration
MIN_MOM        = cfg.ma_cluster_min_momentum
BTC_MA_PERIOD  = cfg.ma_cluster_btc_ma_period_4h   # 50
COST           = 0.0008
WARMUP         = 200
DATA_YEARS     = 6.0
MA_LN          = cfg.ma_cluster_leverage / cfg.max_concurrent  # 2.1

# TF definitions: (label, ratio_from_15m, max_hold_bars)
TF_CONFIGS = [
    ("15m", 1,  192),   # 192 × 15m = 48h
    ("30m", 2,  96),    # 96  × 30m = 48h
    ("1h",  4,  48),    # 48  × 1h  = 48h
    ("4h",  16, 6),     # 6   × 4h  = 24h
]


def get_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    all_syms = set(whitelist) | {"BTCUSDT"}
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
        if not {"close", "high", "low"}.issubset(set(npz.files)):
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


def resample(c15: np.ndarray, h15: np.ndarray, l15: np.ndarray, ratio: int) -> tuple:
    """Downsample 15m → target TF via ratio."""
    if ratio == 1:
        return c15.copy(), h15.copy(), l15.copy()
    n = len(c15) // ratio
    close = c15[ratio - 1 : n * ratio : ratio].copy()
    high  = np.array([np.max(h15[i : i + ratio]) for i in range(0, n * ratio, ratio)])
    low   = np.array([np.min(l15[i : i + ratio]) for i in range(0, n * ratio, ratio)])
    return close, high, low


def compute_sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.mean(arr[i - period + 1 : i + 1])
    return out


def momentum8(close: np.ndarray) -> np.ndarray:
    mom = np.full(len(close), np.nan)
    for i in range(8, len(close)):
        mom[i] = (close[i] - close[i - 8]) / close[i - 8]
    return mom


# ── Numba sim ─────────────────────────────────────────────────────────────────

@numba.jit(nopython=True, cache=True)
def sim_ma_tf(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    btc_gate_mask,   # 1.0=LONG OK, 0.0=blocked; already at TF resolution
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """MA cluster sim for a single TF. BTC gate pre-applied via mask array.

    SL-first fix applied. Cooldown dissolves when cluster spread > cluster_pct.
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    dir_arr = np.empty(cap, np.int64)
    cnt = 0
    i = warmup
    cluster_count = 0
    in_cooldown = 0

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            in_cooldown = 0
            i += 1
            continue

        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0.0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            cluster_count += 1
        else:
            cluster_count = 0
            in_cooldown = 0
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue
        if in_cooldown == 1:
            i += 1
            continue
        if ma_min <= close[i] <= ma_max:
            i += 1
            continue
        if abs(momentum[i]) < min_momentum:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1

        # BTC MA50 gate (LONG only, pre-computed mask)
        if d == 1 and i < len(btc_gate_mask) and btc_gate_mask[i] < 0.5:
            in_cooldown = 1
            i += 1
            continue

        in_cooldown = 1
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 + peak_gain - trail_distance)
                    if low[j] <= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                if high[j] >= entry * (1.0 + tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_active and low[j] <= entry * (1.0 - sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
            else:
                cur_gain = (entry - low[j]) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                if trailing_active:
                    trail_level = entry * (1.0 - peak_gain + trail_distance)
                    if high[j] >= trail_level:
                        trade_pnl = max(peak_gain - trail_distance, -sl)
                        eb = j
                        break
                if low[j] <= entry * (1.0 - tp):
                    trade_pnl = tp
                    eb = j
                    break
                if not trailing_active and high[j] >= entry * (1.0 + sl):
                    trade_pnl = -sl
                    eb = j
                    break
                if not trailing_active and peak_gain >= trail_activate:
                    trailing_active = True
        else:
            trade_pnl = ((close[eb] - entry) / entry) * d

        if cnt < cap:
            pnl_arr[cnt] = trade_pnl - cost
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            dir_arr[cnt] = d
            cnt += 1

        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], dir_arr[:cnt]


# ── Per-symbol runner ─────────────────────────────────────────────────────────

def run_symbol_tf(
    sym: str,
    d15: dict,
    ratio: int,
    max_hold: int,
    btc_close_4h: np.ndarray,
    btc_ma50_4h: np.ndarray,
) -> list[tuple[str, int, int, float]]:
    """Run MA cluster sim for one symbol at given TF. Returns trades in 15m bar space."""
    c15, h15, l15 = d15["close"], d15["high"], d15["low"]
    c, h, l = resample(c15, h15, l15, ratio)

    sma20  = compute_sma(c, 20)
    sma50  = compute_sma(c, 50)
    sma100 = compute_sma(c, 100)
    sma200 = compute_sma(c, 200)
    mom    = momentum8(c)

    offset_15m = d15.get("_offset", 0)
    offset_tf  = offset_15m // ratio

    n_tf = len(c)

    # Build BTC gate mask at TF resolution, accounting for symbol offset
    # BTC arrays start at bar 0 (full history). Symbol starts at offset_tf.
    # BTC 4h bar for symbol TF bar j = (j + offset_tf) // (16 // ratio)
    ratio_to_4h = max(1, 16 // ratio)
    mask = np.ones(n_tf, dtype=np.float64)
    for j in range(n_tf):
        idx_4h = (j + offset_tf) // ratio_to_4h
        if idx_4h < len(btc_ma50_4h) and not np.isnan(btc_ma50_4h[idx_4h]):
            mask[j] = 1.0 if btc_close_4h[idx_4h] >= btc_ma50_4h[idx_4h] else 0.0

    pnl, ent_tf, ext_tf, _ = sim_ma_tf(
        c, h, l, sma20, sma50, sma100, sma200, mom, mask,
        CLUSTER_PCT, MIN_DUR, MAX_DUR, MIN_MOM,
        SL, 0.20, TRAIL_ACTIVATE, TRAIL_DISTANCE, max_hold, COST, WARMUP,
    )

    # Convert TF bars → 15m bars
    ent15 = (ent_tf * ratio + offset_15m).astype(np.int64)
    ext15 = ((ext_tf + 1) * ratio - 1 + offset_15m).astype(np.int64)

    return [(sym, int(e), int(x), float(p)) for p, e, x in zip(pnl, ent15, ext15)]


def build_trades_tf(
    data: dict,
    syms: list[str],
    ratio: int,
    max_hold: int,
    btc_close_4h: np.ndarray,
    btc_ma50_4h: np.ndarray,
) -> list[tuple[str, int, int, float]]:
    trades = []
    for sym in syms:
        try:
            trades.extend(run_symbol_tf(sym, data[sym], ratio, max_hold,
                                        btc_close_4h, btc_ma50_4h))
        except Exception as e:
            print(f"  SKIP {sym}: {e}")
    return trades


# ── Metrics helpers ───────────────────────────────────────────────────────────

def metrics(trades: list) -> dict:
    if len(trades) < 5:
        return {}
    pnl = portfolio_sim(trades, max_concurrent=cfg.max_concurrent)
    return compute_metrics(pnl, DATA_YEARS)


def yearly(trades: list) -> dict[int, dict]:
    bars_per_year = 35064
    buckets: dict[int, list[float]] = {}
    for _, ent, _, pnl in trades:
        yr = 2020 + int(ent) // bars_per_year
        buckets.setdefault(yr, []).append(float(pnl))
    out = {}
    for yr, pnls in sorted(buckets.items()):
        arr = np.array(pnls)
        w = arr[arr > 0]; l = arr[arr < 0]
        pf = (w.sum() / abs(l.sum())) if len(l) > 0 and abs(l.sum()) > 0 else float("inf")
        out[yr] = {
            "N": len(pnls), "PF": pf,
            "WR": 100 * len(w) / max(len(pnls), 1),
            "EV": float(np.mean(pnls)) * 10000,
        }
    return out


def verdict(m: dict, label: str) -> str:
    if not m:
        return "❓"
    pf = m.get("PF", 0)
    wr = m.get("WR", 0)
    dd = m.get("max_dd_pct", 99)
    if pf >= 1.5 and wr >= 55 and dd <= 30:
        return "✅ 유지"
    elif pf < 1.2:
        return "🔴 끄기 권장"
    else:
        return "🟡 조건부"


# ── Numba warmup ─────────────────────────────────────────────────────────────

def warmup_numba() -> None:
    c = np.ones(500); nan = np.full(500, np.nan); nan[200:] = 1.0
    mask = np.ones(500, dtype=np.float64)
    sim_ma_tf(c, c, c, nan, nan, nan, nan, nan, mask,
              0.03, 6, 24, 0.01, 0.02, 0.20, 0.01, 0.005, 6, 0.0, 200)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    commit = get_commit()
    today  = datetime.now().strftime("%Y%m%d")

    print("r120: MA Cluster Per-TF Breakdown")
    print("=" * 70)
    print(f"SL={SL}  trail_act={TRAIL_ACTIVATE}  trail_dist={TRAIL_DISTANCE}")
    print(f"cluster_pct={CLUSTER_PCT}  min_dur={MIN_DUR}  min_mom={MIN_MOM}")
    print(f"BTC gate: MA{BTC_MA_PERIOD} 4h | MA_LN={MA_LN}")
    print()

    print("Loading data...")
    data = load_data(cfg.symbol_whitelist)
    print(f"  {len(data)} symbols loaded")
    if "BTCUSDT" not in data:
        print("ERROR: BTCUSDT missing"); return

    syms = [s for s in cfg.symbol_whitelist if s in data and s != "BTCUSDT"]
    print(f"  {len(syms)} symbols for sim\n")

    # BTC 4h arrays (full history, no offset)
    btc = data["BTCUSDT"]
    btc_c4, btc_h4, btc_l4 = resample(btc["close"], btc["high"], btc["low"], 16)
    btc_ma50_4h = compute_sma(btc_c4, BTC_MA_PERIOD)
    print(f"  BTC 4h bars: {len(btc_c4)}\n")

    print("Numba warmup... ", end="", flush=True)
    warmup_numba()
    print("done\n")

    # ── Per-TF simulation ────────────────────────────────────────────────────
    tf_trades:  dict[str, list] = {}
    tf_metrics: dict[str, dict] = {}
    tf_yearly:  dict[str, dict] = {}

    for tf_label, ratio, max_hold in TF_CONFIGS:
        print(f"Running {tf_label} (ratio={ratio}, max_hold={max_hold}봉×{tf_label}="
              f"{max_hold * ratio * 15 // 60}h)...", end=" ", flush=True)
        t = build_trades_tf(data, syms, ratio, max_hold, btc_c4, btc_ma50_4h)
        m = metrics(t)
        yr = yearly(t)
        tf_trades[tf_label]  = t
        tf_metrics[tf_label] = m
        tf_yearly[tf_label]  = yr
        n = m.get("N", 0) if m else 0
        pf = m.get("PF", 0) if m else 0
        print(f"N={n}, PF={pf:.3f}")

    # ── Combined (all 4 TFs together) ────────────────────────────────────────
    print("Running Combined (all TFs)...", end=" ", flush=True)
    all_trades = []
    for trades in tf_trades.values():
        all_trades.extend(trades)
    m_combined = metrics(all_trades)
    yr_combined = yearly(all_trades)
    print(f"N={m_combined.get('N',0)}, PF={m_combined.get('PF',0):.3f}\n")

    # ── Results table ────────────────────────────────────────────────────────
    HDR = f"  {'TF':<10} {'N':>5}  {'PF':>6}  {'WR':>5}  {'MaxDD':>6}  {'EV(bp)':>7}  {'판정':<14}"
    print("=" * 70)
    print("결과: TF별 MA Cluster 성능 (BTC MA50 gate ON)")
    print("=" * 70)
    print(HDR)
    print("  " + "-" * 65)
    for tf_label, _, _ in TF_CONFIGS:
        m = tf_metrics[tf_label]
        if not m:
            print(f"  {tf_label:<10} no data")
            continue
        v = verdict(m, tf_label)
        ev = m.get("EV_bp", 0)
        print(f"  {tf_label:<10} {m['N']:>5}  {m['PF']:>6.3f}  {m['WR']:>5.1f}%  "
              f"{m['max_dd_pct']:>6.1f}%  {ev:>7.1f}  {v}")

    print("  " + "-" * 65)
    ev_c = m_combined.get("EV_bp", 0) if m_combined else 0
    v_c  = verdict(m_combined, "Combined")
    if m_combined:
        print(f"  {'Combined':<10} {m_combined['N']:>5}  {m_combined['PF']:>6.3f}  "
              f"{m_combined['WR']:>5.1f}%  {m_combined['max_dd_pct']:>6.1f}%  "
              f"{ev_c:>7.1f}  {v_c}")

    # ── Yearly breakdown ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("연도별 PF")
    print("=" * 70)
    all_years = sorted(set().union(*[set(tf_yearly[tf]) for tf, _, _ in TF_CONFIGS]))

    # Header
    hdr = f"  {'Year':<6}"
    for tf_label, _, _ in TF_CONFIGS:
        hdr += f"  {tf_label:>12}"
    hdr += f"  {'Combined':>12}"
    print(hdr)
    print("  " + "-" * (6 + 14 * 5))

    for yr in all_years:
        row = f"  {yr:<6}"
        for tf_label, _, _ in TF_CONFIGS:
            yd = tf_yearly[tf_label].get(yr)
            if yd and yd["N"] >= 3:
                pf_s = f"{yd['PF']:.2f}({yd['N']})"
            else:
                pf_s = "-"
            row += f"  {pf_s:>12}"
        yd_c = yr_combined.get(yr)
        pf_s = f"{yd_c['PF']:.2f}({yd_c['N']})" if yd_c and yd_c["N"] >= 3 else "-"
        row += f"  {pf_s:>12}"
        print(row)

    # ── Bootstrap CI per TF ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Bootstrap PF (95% CI)")
    print("=" * 70)
    for tf_label, _, _ in TF_CONFIGS:
        t = tf_trades[tf_label]
        if len(t) >= 20:
            pnl_arr = portfolio_sim(t, max_concurrent=cfg.max_concurrent)
            boot = bootstrap_pf(pnl_arr)
            print(f"  {tf_label}: median={boot['median']:.3f}  "
                  f"CI=[{boot['CI_lo']:.3f}, {boot['CI_hi']:.3f}]  "
                  f"P(PF≥1)={boot['P_PF_ge_1']:.1f}%")

    # ── Decision ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("판단")
    print("=" * 70)
    for tf_label, _, _ in TF_CONFIGS:
        m = tf_metrics[tf_label]
        v = verdict(m, tf_label)
        pf = m.get("PF", 0) if m else 0
        wr = m.get("WR", 0) if m else 0
        dd = m.get("max_dd_pct", 0) if m else 0
        print(f"  {tf_label}: {v}  (PF={pf:.3f}, WR={wr:.1f}%, MaxDD={dd:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.0f}s")

    # ── Save report ──────────────────────────────────────────────────────────
    report_dir = ROOT / "docs" / "backtest_results" / "202604"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{today}_{commit}_r120_ma_cluster_per_tf.md"

    lines = [
        "# r120: MA Cluster Per-TF Breakdown",
        "",
        f"Date: {today}, Commit: {commit}",
        f"Config: SL={SL}, trail_act={TRAIL_ACTIVATE}, trail_dist={TRAIL_DISTANCE}, "
        f"cluster_pct={CLUSTER_PCT}, min_dur={MIN_DUR}",
        "",
        "## Per-TF Results",
        "",
        "| TF | N | PF | WR% | MaxDD% | EV(bp) | 판정 |",
        "|----|---|----|----|--------|--------|------|",
    ]
    for tf_label, _, _ in TF_CONFIGS:
        m = tf_metrics[tf_label]
        if m:
            v = verdict(m, tf_label)
            lines.append(f"| {tf_label} | {m['N']} | {m['PF']:.3f} | {m['WR']:.1f} | "
                         f"{m['max_dd_pct']:.1f} | {m.get('EV_bp',0):.1f} | {v} |")
    if m_combined:
        v = verdict(m_combined, "Combined")
        lines.append(f"| Combined | {m_combined['N']} | {m_combined['PF']:.3f} | "
                     f"{m_combined['WR']:.1f} | {m_combined['max_dd_pct']:.1f} | "
                     f"{m_combined.get('EV_bp',0):.1f} | {v} |")

    lines += ["", "## Yearly PF", "", "| Year | 15m | 30m | 1h | 4h | Combined |",
              "|------|-----|-----|----|----|----------|"]
    for yr in all_years:
        row_parts = [str(yr)]
        for tf_label, _, _ in TF_CONFIGS:
            yd = tf_yearly[tf_label].get(yr)
            row_parts.append(f"{yd['PF']:.2f}" if yd and yd["N"] >= 3 else "-")
        yd_c = yr_combined.get(yr)
        row_parts.append(f"{yd_c['PF']:.2f}" if yd_c and yd_c["N"] >= 3 else "-")
        lines.append("| " + " | ".join(row_parts) + " |")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
