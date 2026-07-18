"""r91: MA cluster backtest-vs-live parity root cause analysis.

Hypotheses being tested:
  H1 — Cooldown parity: backtest resets cluster_count=0 after exit and allows
       re-entry after min_duration bars within the same cluster episode.
       Live blocks re-entry until the cluster fully dissolves (spread>cluster_pct).
       → Inflates backtest N. Does it also inflate PF?

  H2 — Regime failure (OOS): March 2026 = BTC crash, no direction filter on MA cluster.
       Yearly breakdown to show which years drove PF=2.4.

  H3 — Exit resolution: 4h intra-bar sim vs server-SL + 15m trail (live).
       SL-first fix already applied to straddle; r75 sim predates this.
       Check whether trail-before-SL bias inflates r75 MA results.

Usage:
    cd /home/user/Documents/git/quantbox
    uv run python research/active/r91_r95/r91_ma_cluster_parity.py
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
    "cluster_pct": cfg.ma_cluster_pct,          # 0.02
    "min_duration": cfg.ma_cluster_min_duration, # 6
    "max_duration": cfg.ma_cluster_max_duration, # 24
    "min_momentum": cfg.ma_cluster_min_momentum, # 0.01
    "sl": cfg.ma_cluster_sl,                     # 0.02
    "trail_activate": cfg.ma_cluster_trail_activate,  # 0.02
    "trail_distance": cfg.ma_cluster_trail_distance,  # 0.003
    "tp": 0.20,                                  # dead cap (trail exits first)
    "max_hold": 48,                              # bars (4h) = 192h = 8 days
    "cost": 0.0008,
    "warmup": 200,
}

DATA_YEARS = 6.0
RATIO = 16  # 15m → 4h


# ── Data loading (same as r75) ───────────────────────────────────────────────

def load_data(whitelist: list[str]) -> dict:
    cache_dir = ROOT / "data" / "cache_extended"
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(cache_dir.glob("*_15m_2190d_*.npz")):
        if "numtrades" in f.name:
            continue
        sym = f.name.split("_")[0]
        if sym not in set(whitelist):
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
    """Resample 15m → 4h arrays (close, high, low)."""
    n4 = len(c15) // ratio
    close = c15[ratio - 1 : n4 * ratio : ratio].copy()
    high = np.array([np.max(h15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    low = np.array([np.min(l15[i : i + ratio]) for i in range(0, n4 * ratio, ratio)])
    return close, high, low


from quantbox.research.harness import compute_sma  # 이주(P2): 재구현 제거 (라이브 SMA 정렬)


def momentum8(close: np.ndarray) -> np.ndarray:
    mom = np.full(len(close), np.nan)
    for i in range(8, len(close)):
        mom[i] = (close[i] - close[i - 8]) / close[i - 8]
    return mom


# ── H1: Two variants of the sim ─────────────────────────────────────────────

@numba.jit(nopython=True, cache=True)
def sim_backtest_style(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """Backtest style: reset cluster_count=0 after exit (allows re-entry in same episode).
    Also: trail checked BEFORE SL (pre-SL-first-fix bug, as in r75)."""
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
    # also track: was trade part of same episode as prev trade?
    multi_arr = np.zeros(cap, np.int64)   # 1 = re-entry within same episode
    cnt = 0
    i = warmup
    cluster_count = 0
    last_cluster_start = -1
    current_episode_start = -1

    while i < n - 1:
        if np.isnan(sma200[i]):
            cluster_count = 0
            i += 1
            continue
        ma_max = max(sma20[i], max(sma50[i], max(sma100[i], sma200[i])))
        ma_min = min(sma20[i], min(sma50[i], min(sma100[i], sma200[i])))
        if close[i] <= 0.0:
            i += 1
            continue
        spread = (ma_max - ma_min) / close[i]

        if spread <= cluster_pct:
            if cluster_count == 0:
                current_episode_start = i
            cluster_count += 1
        else:
            cluster_count = 0
            current_episode_start = -1
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue
        if ma_min <= close[i] <= ma_max:
            i += 1
            continue
        if abs(momentum[i]) < min_momentum:
            i += 1
            continue

        d = 1 if momentum[i] > 0 else -1
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False

        is_reentry = 1 if last_cluster_start == current_episode_start and cnt > 0 else 0
        last_cluster_start = current_episode_start

        for j in range(i + 1, end):
            if d == 1:
                cur_gain = (high[j] - entry) / entry
                if cur_gain > peak_gain:
                    peak_gain = cur_gain
                # Trail check BEFORE SL (r75 original — pre-fix bug)
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
                if low[j] <= entry * (1.0 - sl):
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
                if high[j] >= entry * (1.0 + sl):
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
            multi_arr[cnt] = is_reentry
            cnt += 1

        # r75-style reset: cluster_count=0 → re-entry allowed after min_duration bars
        cluster_count = 0
        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt], multi_arr[:cnt]


@numba.jit(nopython=True, cache=True)
def sim_live_style(
    close, high, low, sma20, sma50, sma100, sma200, momentum,
    cluster_pct, min_duration, max_duration, min_momentum,
    sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
):
    """Live style: in_cooldown stays True until cluster fully dissolves (spread>cluster_pct).
    Also: SL checked BEFORE trail-activation (SL-first fix, as in live code)."""
    n = len(close)
    cap = n // 4 + 1
    pnl_arr = np.empty(cap, np.float64)
    ent_arr = np.empty(cap, np.int64)
    ext_arr = np.empty(cap, np.int64)
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
            in_cooldown = 0  # Cluster dissolved — cooldown reset
            i += 1
            continue

        if cluster_count < min_duration:
            i += 1
            continue
        if max_duration > 0 and cluster_count > max_duration:
            i += 1
            continue

        # Cooldown: skip until cluster dissolves
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
        entry = close[i]
        end = min(i + 1 + max_hold, n)
        eb = end - 1
        trade_pnl = 0.0
        peak_gain = 0.0
        trailing_active = False

        # Mark cooldown immediately
        in_cooldown = 1

        for j in range(i + 1, end):
            # Must continue scanning forward; cooldown state independent of exit
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
                # SL BEFORE trail activation (SL-first fix)
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
            cnt += 1

        # After exit: i continues from eb+1 BUT in_cooldown stays 1 until spread>cluster_pct
        # The outer while loop continues from eb+1, checking spread each bar
        i = eb + 1

    return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]


# ── Per-symbol runner ─────────────────────────────────────────────────────────

def run_symbol(sym: str, d15: dict, p: dict) -> dict:
    c15, h15, l15 = d15["close"], d15["high"], d15["low"]
    c4, h4, l4 = build_resampled(c15, h15, l15, RATIO)
    sma20 = compute_sma(c4, 20)
    sma50 = compute_sma(c4, 50)
    sma100 = compute_sma(c4, 100)
    sma200 = compute_sma(c4, 200)
    mom = momentum8(c4)
    offset = d15.get("_offset", 0)

    args = (c4, h4, l4, sma20, sma50, sma100, sma200, mom,
            p["cluster_pct"], p["min_duration"], p["max_duration"], p["min_momentum"],
            p["sl"], p["tp"], p["trail_activate"], p["trail_distance"],
            p["max_hold"], p["cost"], p["warmup"])

    pnl_bt, ent_bt, ext_bt, multi_bt = sim_backtest_style(*args)
    pnl_lv, ent_lv, ext_lv = sim_live_style(*args)

    # Convert 4h bars → 15m bars for portfolio_sim
    def to_15m(ent4, ext4):
        return (ent4 * RATIO + offset).astype(np.int64), ((ext4 + 1) * RATIO - 1 + offset).astype(np.int64)

    ent_bt15, ext_bt15 = to_15m(ent_bt, ext_bt)
    ent_lv15, ext_lv15 = to_15m(ent_lv, ext_lv)

    return {
        "sym": sym,
        "pnl_bt": pnl_bt, "ent_bt": ent_bt15, "ext_bt": ext_bt15, "multi_bt": multi_bt,
        "pnl_lv": pnl_lv, "ent_lv": ent_lv15, "ext_lv": ext_lv15,
        "n4h": len(c4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("r91: MA Cluster Parity Analysis\n" + "=" * 60)
    print(f"Config: cluster_pct={MA_PARAMS['cluster_pct']} min_dur={MA_PARAMS['min_duration']} "
          f"max_dur={MA_PARAMS['max_duration']} sl={MA_PARAMS['sl']} "
          f"trail_act={MA_PARAMS['trail_activate']} trail_dist={MA_PARAMS['trail_distance']}")
    print()

    data = load_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data)} symbols from cache_extended")
    if not data:
        print("ERROR: No data found in cache_extended")
        return

    # Warm up numba
    print("Warming up numba JIT... ", end="", flush=True)
    _dummy = np.ones(500)
    _nan = np.full(500, np.nan)
    _nan[200:] = _dummy[200:]
    sim_backtest_style(_dummy, _dummy, _dummy, _nan, _nan, _nan, _nan, _nan,
                       0.02, 6, 24, 0.01, 0.02, 0.20, 0.02, 0.003, 48, 0.0008, 200)
    sim_live_style(_dummy, _dummy, _dummy, _nan, _nan, _nan, _nan, _nan,
                   0.02, 6, 24, 0.01, 0.02, 0.20, 0.02, 0.003, 48, 0.0008, 200)
    print("done")

    results = []
    for sym, d15 in data.items():
        try:
            r = run_symbol(sym, d15, MA_PARAMS)
            results.append(r)
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    print(f"Processed {len(results)} symbols\n")

    # ── Aggregate backtest-style ─────────────────────────────────────────────
    all_pnl_bt, all_ent_bt, all_ext_bt = [], [], []
    all_pnl_lv, all_ent_lv, all_ext_lv = [], [], []
    multi_count = 0
    total_bt = 0

    for r in results:
        all_pnl_bt.append(r["pnl_bt"])
        all_ent_bt.append(r["ent_bt"])
        all_ext_bt.append(r["ext_bt"])
        all_pnl_lv.append(r["pnl_lv"])
        all_ent_lv.append(r["ent_lv"])
        all_ext_lv.append(r["ext_lv"])
        multi_count += int(np.sum(r["multi_bt"]))
        total_bt += len(r["pnl_bt"])

    def stack_trades(pnl_list, ent_list, ext_list, sym_list):
        trades = []
        for pnl, ent, ext, sym in zip(pnl_list, ent_list, ext_list, sym_list):
            for p, e, x in zip(pnl, ent, ext):
                trades.append((sym, int(e), int(x), float(p)))
        return trades

    syms = [r["sym"] for r in results]
    trades_bt = stack_trades(all_pnl_bt, all_ent_bt, all_ext_bt, syms)
    trades_lv = stack_trades(all_pnl_lv, all_ent_lv, all_ext_lv, syms)

    def to_portfolio(trades):
        if not trades:
            return np.array([])
        # portfolio_sim expects list[tuple[str, int, int, float]]
        return portfolio_sim(trades, max_concurrent=cfg.max_concurrent)

    port_bt = to_portfolio(trades_bt)
    port_lv = to_portfolio(trades_lv)

    m_bt = compute_metrics(port_bt, DATA_YEARS) if len(port_bt) > 10 else {}
    m_lv = compute_metrics(port_lv, DATA_YEARS) if len(port_lv) > 10 else {}

    ln = cfg.ma_cluster_leverage / cfg.max_concurrent  # 2.1

    print("─" * 60)
    print("H1: COOLDOWN PARITY IMPACT")
    print("─" * 60)
    print(f"  BACKTEST STYLE (cluster_count reset, trail-before-SL):")
    if m_bt:
        print(f"    N={m_bt['N']}  PF={m_bt['PF']:.3f}  WR={m_bt['WR']:.1f}%  "
              f"MaxDD={m_bt['max_dd_pct']:.1f}%  CAGR_1x={m_bt['ann_ret_pct']:.1f}%")
    print(f"    Re-entries in same cluster episode: {multi_count}/{total_bt} "
          f"({100*multi_count/max(total_bt,1):.1f}%)")

    print(f"\n  LIVE STYLE (dissolve cooldown, SL-first):")
    if m_lv:
        print(f"    N={m_lv['N']}  PF={m_lv['PF']:.3f}  WR={m_lv['WR']:.1f}%  "
              f"MaxDD={m_lv['max_dd_pct']:.1f}%  CAGR_1x={m_lv['ann_ret_pct']:.1f}%")

    if m_bt and m_lv:
        n_ratio = m_bt["N"] / max(m_lv["N"], 1)
        pf_ratio = m_bt["PF"] / max(m_lv["PF"], 0.001)
        print(f"\n  N inflation (backtest/live): {n_ratio:.2f}×")
        print(f"  PF inflation (backtest/live): {pf_ratio:.2f}×")

    # ── H2: Yearly breakdown (live-style) ────────────────────────────────────
    print("\n" + "─" * 60)
    print("H2: YEARLY BREAKDOWN (live-style sim)")
    print("─" * 60)

    # Map bars → approximate year using 15m bar count
    # 2190d = 6yr, bar 0 ≈ Jan 2020, bar N ≈ Feb 2026
    # 1 year ≈ 365*96 = 35,040 bars
    BARS_PER_YEAR = 365 * 96
    max_bars_15m = max(r["n4h"] * RATIO + r.get("offset", 0) for r in results if r.get("n4h"))

    def year_of_bar(bar: int, max_bar: int) -> int:
        frac = bar / max_bar  # 0=2020, 1=Feb2026
        return 2020 + int(frac * 6.17)

    yearly: dict[int, list] = {}
    for pnl, ent in zip(all_pnl_lv, all_ent_lv):
        for p, e in zip(pnl, ent):
            yr = year_of_bar(int(e), max_bars_15m)
            yearly.setdefault(yr, []).append(float(p))

    print(f"  {'Year':>5} {'N':>5} {'WR%':>6} {'PF':>6}  {'EV_bp':>7}")
    for yr in sorted(yearly.keys()):
        trades_yr = yearly[yr]
        if len(trades_yr) < 5:
            continue
        wins = [t for t in trades_yr if t > 0]
        losses = [t for t in trades_yr if t <= 0]
        wr = 100 * len(wins) / len(trades_yr)
        pf = sum(wins) / abs(sum(losses)) if losses and wins else 0.0
        ev = np.mean(trades_yr) * 10000  # bp
        print(f"  {yr:>5} {len(trades_yr):>5} {wr:>6.1f} {pf:>6.3f}  {ev:>+7.1f}bp")

    # ── H3: SL-first fix impact ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("H3: TRAIL-BEFORE-SL BIAS (r75 original vs SL-first)")
    print("─" * 60)

    # Collect backtest-style WITHOUT cooldown fix but WITH SL-first
    # = Pure test of SL order effect
    print("  Running SL-first-only variant... ", end="", flush=True)

    @numba.jit(nopython=True, cache=True)
    def sim_slfirst_only(
        close, high, low, sma20, sma50, sma100, sma200, momentum,
        cluster_pct, min_duration, max_duration, min_momentum,
        sl, tp, trail_activate, trail_distance, max_hold, cost, warmup,
    ):
        """Same as backtest-style but with SL checked before trail activation."""
        n = len(close)
        cap = n // 4 + 1
        pnl_arr = np.empty(cap, np.float64)
        ent_arr = np.empty(cap, np.int64)
        ext_arr = np.empty(cap, np.int64)
        cnt = 0
        i = warmup
        cluster_count = 0

        while i < n - 1:
            if np.isnan(sma200[i]):
                cluster_count = 0
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
                i += 1
                continue
            if cluster_count < min_duration:
                i += 1
                continue
            if max_duration > 0 and cluster_count > max_duration:
                i += 1
                continue
            if ma_min <= close[i] <= ma_max:
                i += 1
                continue
            if abs(momentum[i]) < min_momentum:
                i += 1
                continue

            d = 1 if momentum[i] > 0 else -1
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
                    # SL BEFORE trail activation
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
                cnt += 1

            cluster_count = 0
            i = eb + 1

        return pnl_arr[:cnt], ent_arr[:cnt], ext_arr[:cnt]

    all_pnl_sf, all_ent_sf, all_ext_sf = [], [], []
    for r in results:
        d15 = data[r["sym"]]
        c15, h15, l15 = d15["close"], d15["high"], d15["low"]
        c4, h4, l4 = build_resampled(c15, h15, l15, RATIO)
        sma20 = compute_sma(c4, 20)
        sma50 = compute_sma(c4, 50)
        sma100 = compute_sma(c4, 100)
        sma200 = compute_sma(c4, 200)
        mom = momentum8(c4)
        offset = d15.get("_offset", 0)
        p = MA_PARAMS
        try:
            pnl_sf, ent_sf, ext_sf = sim_slfirst_only(
                c4, h4, l4, sma20, sma50, sma100, sma200, mom,
                p["cluster_pct"], p["min_duration"], p["max_duration"], p["min_momentum"],
                p["sl"], p["tp"], p["trail_activate"], p["trail_distance"],
                p["max_hold"], p["cost"], p["warmup"])
            all_pnl_sf.append(pnl_sf)
            all_ent_sf.append((ent_sf * RATIO + offset).astype(np.int64))
            all_ext_sf.append(((ext_sf + 1) * RATIO - 1 + offset).astype(np.int64))
        except Exception as e:
            all_pnl_sf.append(np.array([]))
            all_ent_sf.append(np.array([]))
            all_ext_sf.append(np.array([]))

    trades_sf = stack_trades(all_pnl_sf, all_ent_sf, all_ext_sf, syms)
    port_sf = to_portfolio(trades_sf)
    m_sf = compute_metrics(port_sf, DATA_YEARS) if len(port_sf) > 10 else {}
    print("done")

    if m_sf:
        print(f"  Backtest-style + SL-first (cooldown NOT fixed):")
        print(f"    N={m_sf['N']}  PF={m_sf['PF']:.3f}  WR={m_sf['WR']:.1f}%")
        if m_bt:
            print(f"    SL-order effect on PF: {m_sf['PF']:.3f} vs {m_bt['PF']:.3f} "
                  f"(Δ={m_sf['PF']-m_bt['PF']:+.3f})")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = [
        ("r75 reported (IS 6yr)", 1233, 2.403, 40.2, "r75 report"),
        ("Backtest-style repro", m_bt.get("N", 0), m_bt.get("PF", 0), m_bt.get("WR", 0),
         f"cluster_count=0 reset, trail-before-SL"),
        ("SL-first only (cooldown unchanged)", m_sf.get("N", 0), m_sf.get("PF", 0), m_sf.get("WR", 0),
         "SL before trail-activation"),
        ("Live-style sim", m_lv.get("N", 0), m_lv.get("PF", 0), m_lv.get("WR", 0),
         "dissolve cooldown + SL-first"),
        ("LIVE actual (Mar 4-13)", 16, 0.587, 31.2, "N=16, regime hostile"),
    ]
    print(f"  {'Scenario':<45} {'N':>5} {'PF':>6} {'WR%':>6}")
    print(f"  {'-'*45} {'-'*5} {'-'*6} {'-'*6}")
    for name, n, pf, wr, note in rows:
        print(f"  {name:<45} {n:>5} {pf:>6.3f} {wr:>6.1f}  ← {note}")

    print("\nConclusion:")
    if m_bt and m_lv:
        pf_gap = m_bt["PF"] / max(m_lv["PF"], 0.001)
        n_gap = m_bt["N"] / max(m_lv["N"], 1)
        print(f"  • H1 Cooldown: N gap={n_gap:.2f}×, PF gap={pf_gap:.2f}× between backtest and live-style sim")
    if m_sf and m_bt:
        print(f"  • H3 SL-order: PF change = {m_sf['PF']-m_bt['PF']:+.3f} (SL-first vs trail-first)")
    print(f"  • H2 Regime: See yearly breakdown above — check if 2025/2026 years show EV decay")
    print(f"  • Live actual: N=16 trades vs live-style sim N={m_lv.get('N','?')} → regime gap explains remainder")

    print("\nDocs: docs/backtest_results/202603/YYYYMMDD_COMMIT_r91_ma_parity.md")


if __name__ == "__main__":
    main()
