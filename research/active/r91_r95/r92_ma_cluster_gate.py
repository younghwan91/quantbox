"""r92: MA Cluster — 4h BTC Gate effect on IS performance.

Goal:
  Quantify how much the new 4h BTC MA gate (50-bar, ~8.3 days) improves
  MA cluster performance vs the current G6 (15m MA50, 12.5h — too short).

Method:
  Run sim_live_style (SL-first, dissolve-cooldown) twice:
    - Gate OFF: baseline (current live state — no MA cluster gate)
    - Gate ON:  block MA cluster LONG signals when BTC 4h below MA50_4h

Key question: Does gate-ON remove the bad trades (March crash LONG entries)
without destroying overall PF?

Pass criteria (proceed to re-enable MA cluster):
  - Gate-ON PF >= 1.4
  - Gate-ON EV >= +50 bp
  - No loss year with gate ON
  - N >= 200 (sufficient sample)

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r91_r95/r92_ma_cluster_gate.py
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from quantbox.backtest.trade_sim import compute_metrics, portfolio_sim
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Config ──────────────────────────────────────────────────────────────────
cfg = VolStraddleConfig()

MA_PARAMS = {
    "cluster_pct":    cfg.ma_cluster_pct,
    "min_duration":   cfg.ma_cluster_min_duration,
    "max_duration":   cfg.ma_cluster_max_duration,
    "min_momentum":   cfg.ma_cluster_min_momentum,
    "sl":             cfg.ma_cluster_sl,
    "trail_activate": cfg.ma_cluster_trail_activate,
    "trail_distance": cfg.ma_cluster_trail_distance,
    "tp":             0.20,   # dead cap — trail exits first
    "max_hold":       cfg.ma_cluster_max_hold_4h,
    "cost":           0.0008,
    "warmup":         200,
}
BTC_GATE_PERIOD = cfg.ma_cluster_btc_ma_period_4h  # 50 × 4h = 200h ≈ 8.3 days
MA_LN = cfg.ma_cluster_leverage / cfg.max_concurrent  # 21/10 = 2.1
DATA_YEARS = 6.0
RATIO = 16  # 15m → 4h


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    all_syms = set(whitelist) | {"BTCUSDT"}  # always load BTC for gate
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


def build_resampled(c15: np.ndarray, h15: np.ndarray, l15: np.ndarray, ratio: int) -> tuple:
    n4 = len(c15) // ratio
    close = c15[ratio - 1 : n4 * ratio : ratio].copy()
    high = np.array([np.max(h15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    low = np.array([np.min(l15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
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


# ── Core sim (live-style + optional BTC gate) ────────────────────────────────

@numba.jit(nopython=True, cache=True)
def sim_live_style_gated(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    btc_close_4h, btc_ma50_4h, gate_enabled,
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """Live-style MA cluster sim with optional 4h BTC gate.

    Gate: if gate_enabled=1 and signal=LONG and btc_close_4h[i] < btc_ma50_4h[i]:
      - Set cooldown (matching live behavior: strategy fires before scalper gate)
      - Do NOT enter trade
    This correctly models the live system where strategy sets cooldown on signal,
    but scalper gate prevents the actual order.

    SL-first fix applied: SL checked before trail activation (intra-bar SL fires first).
    Cooldown: dissolves only when cluster spread > cluster_pct (live-style).
    """
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    dir_arr = np.empty(cap, np.int64)   # +1 LONG, -1 SHORT
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
            in_cooldown = 0  # cluster dissolved — reset cooldown
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

        # ── 4h BTC gate ── (applies to LONG only, matching live gate logic)
        if gate_enabled == 1 and d == 1 and i < len(btc_ma50_4h) and not np.isnan(btc_ma50_4h[i]):
            if btc_close_4h[i] < btc_ma50_4h[i]:
                in_cooldown = 1  # strategy sets cooldown even when gate blocks
                i += 1
                continue

        # Enter trade
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
                # SL before trail activation (SL-first fix)
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
            if d == 1:
                trade_pnl = (close[eb] - entry) / entry
            else:
                trade_pnl = (entry - close[eb]) / entry

        if cnt < cap:
            pnl_arr[cnt] = trade_pnl - cost
            ent_arr[cnt] = i
            ext_arr[cnt] = eb
            dir_arr[cnt] = d
            cnt += 1

        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], dir_arr[:cnt]


# ── Per-symbol runner ─────────────────────────────────────────────────────────

def run_symbol(sym: str, d15: dict, btc_4h: np.ndarray, btc_ma50: np.ndarray, p: dict) -> dict:
    c15, h15, l15 = d15["close"], d15["high"], d15["low"]
    c4, h4, l4 = build_resampled(c15, h15, l15, RATIO)
    sma20  = compute_sma(c4, 20)
    sma50  = compute_sma(c4, 50)
    sma100 = compute_sma(c4, 100)
    sma200 = compute_sma(c4, 200)
    mom    = momentum8(c4)
    offset = d15.get("_offset", 0)

    # Align BTC 4h to this symbol's 4h bars via offset
    btc_offset_4h = offset // RATIO
    n4 = len(c4)
    # BTC 4h arrays are from BTC's full resampled data; slice to align
    # BTC has offset=0 (full history); sym may have positive offset
    # So sym's bar 0 corresponds to btc bar btc_offset_4h
    # Truncate or pad BTC to match n4 length starting at btc_offset_4h
    if len(btc_4h) >= btc_offset_4h + n4:
        btc_close_aligned = btc_4h[btc_offset_4h : btc_offset_4h + n4]
        btc_ma50_aligned   = btc_ma50[btc_offset_4h : btc_offset_4h + n4]
    else:
        btc_close_aligned = np.full(n4, np.nan)
        btc_ma50_aligned   = np.full(n4, np.nan)

    base_args = (c4, h4, l4, sma20, sma50, sma100, sma200, mom,
                 btc_close_aligned, btc_ma50_aligned)
    sim_args  = (p["cluster_pct"], p["min_duration"], p["max_duration"], p["min_momentum"],
                 p["sl"], p["tp"], p["trail_activate"], p["trail_distance"],
                 p["max_hold"], p["cost"], p["warmup"])

    pnl_off, ent_off, ext_off, dir_off = sim_live_style_gated(*base_args, 0, *sim_args)
    pnl_on,  ent_on,  ext_on,  dir_on  = sim_live_style_gated(*base_args, 1, *sim_args)

    # Convert 4h bars → 15m bars for portfolio_sim
    def to_15m(ent4: np.ndarray, ext4: np.ndarray) -> tuple:
        return ((ent4 * RATIO + offset).astype(np.int64),
                ((ext4 + 1) * RATIO - 1 + offset).astype(np.int64))

    ent_off15, ext_off15 = to_15m(ent_off, ext_off)
    ent_on15,  ext_on15  = to_15m(ent_on,  ext_on)

    return {
        "sym": sym,
        "pnl_off": pnl_off, "ent_off": ent_off15, "ext_off": ext_off15, "dir_off": dir_off,
        "pnl_on":  pnl_on,  "ent_on":  ent_on15,  "ext_on":  ext_on15,  "dir_on":  dir_on,
        # 4h bar indices (unshifted) for yearly analysis
        "ent_off_4h": ent_off, "dir_off_4h": dir_off,
        "ent_on_4h":  ent_on,  "dir_on_4h":  dir_on,
        "pnl_off_raw": pnl_off, "pnl_on_raw": pnl_on,
        "n4h": n4,
        "offset_4h": offset // RATIO,
    }


# ── Yearly breakdown ─────────────────────────────────────────────────────────

def yearly_metrics(results: list[dict], data: dict, key: str, bars_per_year: int = 2190 * 16) -> dict:
    """Break down trades by entry year using 15m bar indices."""
    years: dict[int, list[float]] = {}
    for r in results:
        pnl = r[f"pnl_{key}"]
        ent = r[f"ent_{key}"]
        for p, e in zip(pnl, ent):
            # Approximate year from 15m bar index
            # 15m bars per year = 365.25 * 24 * 4 ≈ 35,064
            bars_per_year_15m = 35064
            # Offset from full data start (2020-01-01 approx)
            year = 2020 + int(e) // bars_per_year_15m
            if year not in years:
                years[year] = []
            years[year].append(float(p))
    result_by_year: dict[int, dict] = {}
    for yr in sorted(years):
        pnls = np.array(years[yr])
        wins  = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")
        wr = 100.0 * len(wins) / max(len(pnls), 1)
        ev = float(np.mean(pnls)) * 10000  # bp
        result_by_year[yr] = {"N": len(pnls), "PF": pf, "WR": wr, "EV": ev}
    return result_by_year


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("r92: MA Cluster — 4h BTC Gate Effect\n" + "=" * 60)
    print(f"Gate: BTC 4h MA{BTC_GATE_PERIOD} (={BTC_GATE_PERIOD * 4}h = {BTC_GATE_PERIOD * 4 / 24:.1f} days)")
    print(f"Config: cluster_pct={MA_PARAMS['cluster_pct']} min_dur={MA_PARAMS['min_duration']} "
          f"sl={MA_PARAMS['sl']} trail_act={MA_PARAMS['trail_activate']} "
          f"trail_dist={MA_PARAMS['trail_distance']}")
    print()

    data = load_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data)} symbols from cache_extended")
    if not data:
        print("ERROR: No data found in cache_extended")
        return

    if "BTCUSDT" not in data:
        print("ERROR: BTCUSDT not in cache — gate cannot be computed")
        return

    # Build BTC 4h arrays
    btc = data["BTCUSDT"]
    btc_close_4h, _, _ = build_resampled(btc["close"], btc["high"], btc["low"], RATIO)
    btc_ma50_4h = compute_sma(btc_close_4h, BTC_GATE_PERIOD)
    print(f"BTC 4h bars: {len(btc_close_4h)}")
    print()

    # Warm up numba
    print("Warming up numba JIT... ", end="", flush=True)
    _c = np.ones(500)
    _nan = np.full(500, np.nan)
    _nan[200:] = 1.0
    _btc = np.ones(500) * 90000.0
    _bma = np.full(500, np.nan)
    _bma[50:] = 88000.0
    for gate in (0, 1):
        sim_live_style_gated(_c, _c, _c, _nan, _nan, _nan, _nan, _nan,
                             _btc, _bma, gate,
                             0.02, 6, 24, 0.01, 0.02, 0.20, 0.02, 0.003, 48, 0.0, 200)
    print("done\n")

    results = []
    syms = [s for s in cfg.symbol_whitelist if s in data and s != "BTCUSDT"]
    for sym in syms:
        try:
            r = run_symbol(sym, data[sym], btc_close_4h, btc_ma50_4h, MA_PARAMS)
            results.append(r)
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    print(f"Processed {len(results)} symbols\n")

    # ── Aggregate ────────────────────────────────────────────────────────────
    def stack_trades(results: list[dict], key: str) -> list:
        trades = []
        for r in results:
            for p, e, x in zip(r[f"pnl_{key}"], r[f"ent_{key}"], r[f"ext_{key}"]):
                trades.append((r["sym"], int(e), int(x), float(p)))
        return trades

    trades_off = stack_trades(results, "off")
    trades_on  = stack_trades(results, "on")

    def portfolio_metrics(trades: list) -> dict:
        if len(trades) < 10:
            return {}
        pnl = portfolio_sim(trades, max_concurrent=cfg.max_concurrent)
        return compute_metrics(pnl, DATA_YEARS)

    m_off = portfolio_metrics(trades_off)
    m_on  = portfolio_metrics(trades_on)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("─" * 60)
    print("OVERALL RESULTS")
    print("─" * 60)
    fmt = "{:<12} {:>6} {:>7} {:>7} {:>7} {:>9} {:>11}"
    print(fmt.format("Variant", "N", "PF", "WR%", "MaxDD%", "EV(bp)", f"CAGR_LN{MA_LN:.1f}%"))
    print("-" * 60)

    def row(name: str, m: dict) -> None:
        if not m:
            print(f"  {name}: no data")
            return
        ev_bp = m.get("EV_bp", float("nan"))
        ann_1x = m.get("ann_ret_pct", float("nan"))
        ann_ln = (1 + ann_1x / 100) ** (1 / DATA_YEARS) - 1  # per-year
        # compound over DATA_YEARS at MA_LN
        cagr_ln = ((1 + ann_1x / 100 * MA_LN) ** (1 / DATA_YEARS) - 1) * 100
        print(fmt.format(
            name,
            m["N"],
            f"{m['PF']:.3f}",
            f"{m['WR']:.1f}",
            f"{m['max_dd_pct']:.1f}",
            f"{ev_bp:.1f}",
            f"{cagr_ln:.1f}",
        ))

    row("Gate OFF", m_off)
    row("Gate ON",  m_on)

    if m_off and m_on:
        n_diff  = m_on["N"] - m_off["N"]
        pf_diff = m_on["PF"] - m_off["PF"]
        print(f"\n  ΔN={n_diff:+d}  ΔPF={pf_diff:+.3f}")

    # ── Yearly breakdown ─────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("YEARLY BREAKDOWN")
    print("─" * 60)
    yr_off = yearly_metrics(results, data, "off")
    yr_on  = yearly_metrics(results, data, "on")
    all_years = sorted(set(yr_off) | set(yr_on))
    fmt2 = "{:<6} {:>6} {:>6} {:>6}  {:>6} {:>6} {:>6}   {:>8}"
    print(fmt2.format("Year", "N_off", "PF_off", "WR_off", "N_on", "PF_on", "WR_on", "ΔN"))
    print("-" * 60)
    for yr in all_years:
        o = yr_off.get(yr, {})
        n = yr_on.get(yr, {})
        o_pf = f"{o.get('PF', 0):.2f}" if o else "-"
        n_pf = f"{n.get('PF', 0):.2f}" if n else "-"
        o_wr = f"{o.get('WR', 0):.0f}" if o else "-"
        n_wr = f"{n.get('WR', 0):.0f}" if n else "-"
        on_n  = n.get("N", 0)
        off_n = o.get("N", 0)
        delta_n = on_n - off_n
        print(fmt2.format(yr, off_n, o_pf, o_wr, on_n, n_pf, n_wr, f"{delta_n:+d}"))

    # ── Gate statistics ──────────────────────────────────────────────────────
    total_off = sum(len(r["pnl_off"]) for r in results)
    total_on  = sum(len(r["pnl_on"])  for r in results)
    long_off  = sum(int(np.sum(r["dir_off"] == 1)) for r in results)
    long_on   = sum(int(np.sum(r["dir_on"] == 1))  for r in results)
    blocked   = long_off - long_on

    print(f"\n  Gate OFF: {total_off} trades ({long_off} LONG)")
    print(f"  Gate ON:  {total_on} trades ({long_on} LONG)")
    print(f"  Blocked by 4h gate: {blocked} LONG entries ({100*blocked/max(long_off,1):.1f}%)")

    # ── Decision ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("DECISION")
    print("─" * 60)
    if m_on:
        ev_on = m_on.get("EV_bp", 0)
        pf_on = m_on.get("PF", 0)
        n_on  = m_on.get("N", 0)
        pass_n  = n_on >= 200
        pass_pf = pf_on >= 1.4
        pass_ev = ev_on >= 50
        pass_yr = all(yr_on.get(yr, {}).get("PF", 0) >= 1.0 for yr in all_years if yr_on.get(yr, {}).get("N", 0) >= 20)
        verdict = "✅ PASS — proceed to re-enable MA cluster" if all([pass_n, pass_pf, pass_ev, pass_yr]) else "❌ FAIL — do not re-enable"
        print(f"  N≥200:  {'✅' if pass_n else '❌'} ({n_on})")
        print(f"  PF≥1.4: {'✅' if pass_pf else '❌'} ({pf_on:.3f})")
        print(f"  EV≥50bp:{'✅' if pass_ev else '❌'} ({ev_on:.1f}bp)")
        print(f"  No loss year: {'✅' if pass_yr else '❌'}")
        print(f"\n  → {verdict}")
    else:
        print("  → FAIL: insufficient data")


if __name__ == "__main__":
    main()
