#!/usr/bin/env python3
"""r116-prep: Baseline measurement for D_mom8, MA Cluster, and Donchian.

Window: 2024-01-01T00:00:00+00:00 → 2026-04-12T16:18:00+00:00
Engine: TradeEventSim (canonical — NEVER reimplement collect_trades/portfolio_sim)
Symbols: cfg.symbol_whitelist from VolStraddleConfig

Usage:
    uv run python -m research.active.r116_willy_nfi.r116_prep_baseline --strategy vr_directional
    uv run python -m research.active.r116_willy_nfi.r116_prep_baseline --strategy ma_cluster
    uv run python -m research.active.r116_willy_nfi.r116_prep_baseline --strategy donchian

(ref: r116 plan Section 0)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantbox.backtest.trade_sim import (
    G_4TF,
    build_resampled,
    collect_trades,
    compute_metrics,
    portfolio_sim_with_details,
    precompute,
    precompute_all_vr,
)
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

# ── Constants ────────────────────────────────────────────────────────────────
# Cutoff: session-start timestamp to avoid data leak from current live session
WINDOW_START = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_END = datetime(2026, 4, 12, 16, 18, 0, tzinfo=timezone.utc)
DATA_YEARS = (WINDOW_END - WINDOW_START).days / 365.25  # ~2.28 years
CACHE_DIR = ROOT / "data" / "cache_extended"
STATE_DIR = ROOT / ".omc" / "state" / "r116_prep"

BARS_PER_YEAR_15M = 365 * 24 * 4  # 35040 bars/year at 15m


# ── Data Loading ─────────────────────────────────────────────────────────────


def load_15m_data(symbols: list[str]) -> dict:
    """Load 15m OHLCV arrays from cache_extended for given symbols + BTCUSDT."""
    all_syms = set(symbols) | {"BTCUSDT"}
    seen: dict[str, tuple[Path, int]] = {}
    for f in sorted(CACHE_DIR.glob("*_15m_2190d_*.npz")):
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
        data[sym] = {
            "close": c,
            "high": npz["high"].astype(np.float64),
            "low": npz["low"].astype(np.float64),
            "volume": npz["volume"].astype(np.float64),
            "bars": len(c),
            "days": len(c) / (4 * 24),
        }

    # Trim to window: last N bars corresponding to WINDOW_START→WINDOW_END
    # Cache files are 2190d (6yr) ending ~2026-04. We slice to DATA_YEARS from end.
    window_bars = int(DATA_YEARS * BARS_PER_YEAR_15M)
    for sym, d in data.items():
        n = len(d["close"])
        if n > window_bars:
            for k in ("close", "high", "low", "volume"):
                d[k] = d[k][n - window_bars:]
            d["bars"] = len(d["close"])
            d["days"] = len(d["close"]) / (4 * 24)

    return data


# ── MFE computation for losing trades ────────────────────────────────────────


def compute_mfe_losing(
    accepted_trades: list[tuple[str, int, int, float]],
    data_15m: dict,
) -> float:
    """Compute average bar-close MFE (%) for losing trades only.

    MFE = max favorable excursion measured at bar closes (not intrabar high/low).
    For LONG: max((close[j] - entry) / entry) over j in [entry+1, exit].
    For SHORT: max((entry - close[j]) / entry) over j in [entry+1, exit].

    Direction is inferred from sign of (close[exit] - close[entry]) vs PnL sign.
    We use the trade PnL sign to determine direction: negative PnL → losing trade.
    """
    mfe_values: list[float] = []
    for sym, ent_bar, ext_bar, pnl in accepted_trades:
        if pnl >= 0:
            continue  # only losing trades
        d = data_15m.get(sym)
        if d is None:
            continue
        closes = d["close"]
        n = len(closes)
        if ent_bar >= n or ext_bar >= n or ent_bar >= ext_bar:
            continue
        entry_price = closes[ent_bar]
        if entry_price <= 0:
            continue
        # Infer direction from entry→exit close movement vs PnL
        # If close[exit] > close[entry] → trade was SHORT (loss on upward move)
        # If close[exit] < close[entry] → trade was LONG (loss on downward move)
        exit_price = closes[ext_bar]
        is_long = exit_price <= entry_price  # lost because price fell

        hold_closes = closes[ent_bar + 1 : ext_bar + 1]
        if len(hold_closes) == 0:
            mfe_values.append(0.0)
            continue

        if is_long:
            excursions = (hold_closes - entry_price) / entry_price
        else:
            excursions = (entry_price - hold_closes) / entry_price

        mfe = float(np.max(excursions)) * 100.0
        mfe_values.append(max(mfe, 0.0))

    return float(np.mean(mfe_values)) if mfe_values else 0.0


# ── Per-symbol breakdown for D_mom8 ─────────────────────────────────────────


def compute_per_symbol_metrics(
    accepted_trades: list[tuple[str, int, int, float]],
    data_15m: dict,
) -> list[dict]:
    from collections import defaultdict

    by_sym: dict[str, list[tuple[str, int, int, float]]] = defaultdict(list)
    for t in accepted_trades:
        by_sym[t[0]].append(t)

    rows: list[dict] = []
    for sym in sorted(by_sym.keys()):
        trades = by_sym[sym]
        pnl_arr = np.array([t[3] for t in trades])
        n = len(pnl_arr)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr < 0]
        wr = len(wins) / n * 100 if n > 0 else 0.0
        gross_win = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 1e-9
        pf = gross_win / gross_loss if gross_loss > 0 else 999.0
        mfe = compute_mfe_losing(trades, data_15m)
        rows.append({
            "symbol": sym,
            "N": n,
            "WR_pct": round(wr, 1),
            "PF": round(pf, 3),
            "mfe_losing_avg_pct": round(mfe, 3),
        })
    return rows


# ── D_mom8 Baseline ──────────────────────────────────────────────────────────


def run_vr_directional(cfg: VolStraddleConfig) -> dict:
    print("\n=== D_mom8 (vr_directional) baseline ===")
    t0 = time.time()

    data_15m = load_15m_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data_15m)} symbols in {time.time()-t0:.1f}s")

    # Precompute VR + momentum for all symbols at all TFs
    sp, lp = cfg.vr_short, cfg.vr_long
    precompute_all_vr([data_15m], [(sp, lp)])
    precompute(data_15m, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute_all_vr([data_1h, data_4h], [(sp, lp)])
    for ds in (data_1h, data_4h):
        precompute(ds, vr_sp=sp, vr_lp=lp, mom_lb=cfg.momentum_period)

    # Build cfg dict for collect_trades — values from config only
    ct_cfg = {
        "sp": cfg.vr_short,
        "lp": cfg.vr_long,
        "d_vr": cfg.d_vr_threshold,
        "d_sl": cfg.d_sl,
        "tp": cfg.d_tp,
        "d_trail_activate": cfg.d_trail_activate,
        "d_trail_distance": cfg.d_trail_distance,
    }

    all_trades = collect_trades(ct_cfg, data_15m, data_1h, data_4h, "D_mom8", tf_group=G_4TF)
    print(f"Raw trades (all symbols, all TFs): {len(all_trades)}")

    pnl_arr, accepted = portfolio_sim_with_details(all_trades, cfg.max_concurrent)
    print(f"Accepted after portfolio_sim: {len(accepted)}")

    metrics = compute_metrics(pnl_arr, DATA_YEARS, portfolio_size=cfg.max_concurrent)
    mfe_losing = compute_mfe_losing(accepted, data_15m)
    per_symbol = compute_per_symbol_metrics(accepted, data_15m)

    n = metrics["N"]
    low_n = n < 100
    result = {
        "strategy": "vr_directional",
        "label": "D_mom8",
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "data_years": round(DATA_YEARS, 3),
        "N": n,
        "WR_pct": metrics["WR"],
        "PF": metrics["PF"],
        "MDD_pct": metrics["max_dd_pct"],
        "avg_win_pct": metrics["avg_win_pct"],
        "avg_loss_pct": metrics["avg_loss_pct"],
        "sharpe": _compute_sharpe(pnl_arr),
        "calmar": metrics["calmar"],
        "mfe_losing_avg_pct": round(mfe_losing, 3),
        "ann_ret_pct": metrics["ann_ret_pct"],
        "low_N": low_n,
        "tolerance_bands": {"WR": 0.05, "PF": 0.10} if low_n else None,
        "per_symbol": per_symbol,
        "elapsed_s": round(time.time() - t0, 1),
    }

    _print_summary(result)
    # Fixed 2026-04-17 (US-R116B-1): compute_metrics() now accepts portfolio_size param.
    # Pass cfg.max_concurrent (=10) so Calmar/ann_ret reflect realistic per-slot compounding.
    return result


def _compute_sharpe(pnl: np.ndarray) -> float:
    """Annualized Sharpe from per-trade PnL (trade-count-based annualization)."""
    if len(pnl) < 2:
        return 0.0
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    if std <= 0:
        return 0.0
    # Annualize: trades per year × mean / std × sqrt(trades_per_year)
    tpy = len(pnl) / DATA_YEARS
    return round(mean / std * np.sqrt(tpy), 3)


def _print_summary(r: dict) -> None:
    print(f"\n--- {r['label']} Summary ---")
    print(f"  N       = {r['N']}")
    print(f"  WR      = {r['WR_pct']}%")
    print(f"  PF      = {r['PF']}")
    print(f"  MDD     = {r['MDD_pct']}%")
    print(f"  Sharpe  = {r['sharpe']}")
    print(f"  Calmar  = {r['calmar']}")
    print(f"  AvgWin  = {r['avg_win_pct']}%  AvgLoss = {r['avg_loss_pct']}%")
    print(f"  MFE(losing avg) = {r['mfe_losing_avg_pct']}%")
    if r["low_N"]:
        print(f"  [WARNING] N < 100 — low statistical power. Tolerance bands: {r['tolerance_bands']}")
    print(f"  Elapsed = {r['elapsed_s']}s")


# ── MA Cluster Baseline ───────────────────────────────────────────────────────


def run_ma_cluster(cfg: VolStraddleConfig) -> dict:
    """MA Cluster baseline — reuses canonical sim from trade_sim."""
    print("\n=== MA Cluster baseline ===")
    t0 = time.time()

    from quantbox.backtest.trade_sim import (
        sim_ma_cluster_trailing,
        portfolio_sim_with_details as _pswd,
    )
    from quantbox.strategy.vol_straddle.signals import compute_momentum as _cmom

    data_15m = load_15m_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data_15m)} symbols")

    # Resample to 4h for MA Cluster
    data_4h = build_resampled(data_15m, 16)

    def sma_vec(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.full(len(arr), np.nan)
        for i in range(p - 1, len(arr)):
            out[i] = arr[i - p + 1 : i + 1].mean()
        return out

    periods = cfg.ma_cluster_sma_periods  # [20, 50, 100, 200]
    all_trades: list[tuple[str, int, int, float]] = []

    for sym in sorted(data_4h.keys()):
        if sym == "BTCUSDT":
            continue
        d = data_4h[sym]
        c, h, lo = d["close"], d["high"], d["low"]

        sma20 = sma_vec(c, periods[0])
        sma50 = sma_vec(c, periods[1])
        sma100 = sma_vec(c, periods[2])
        sma200 = sma_vec(c, periods[3])
        mom = _cmom(c, cfg.momentum_period)

        pnl_arr, ent_arr, ext_arr = sim_ma_cluster_trailing(
            c, h, lo, sma20, sma50, sma100, sma200, mom,
            cfg.ma_cluster_pct,
            cfg.ma_cluster_min_duration,
            cfg.ma_cluster_max_duration,
            cfg.ma_cluster_min_momentum,
            cfg.ma_cluster_sl,
            cfg.ma_cluster_tp,
            cfg.ma_cluster_trail_activate,
            cfg.ma_cluster_trail_distance,
            cfg.ma_cluster_max_hold_4h,
            cfg.round_trip_cost,
            200,  # warmup (200 4h bars ≈ 33 days)
        )

        # Convert 4h bar indices to 15m base (ratio=16)
        ratio = 16
        for i in range(len(pnl_arr)):
            ent_15 = int(ent_arr[i]) * ratio
            ext_15 = int(ext_arr[i]) * ratio
            all_trades.append((sym, ent_15, ext_15, float(pnl_arr[i])))

    print(f"Raw MA Cluster trades: {len(all_trades)}")
    pnl_port, accepted = _pswd(all_trades, cfg.max_concurrent)
    print(f"Accepted after portfolio_sim: {len(accepted)}")

    metrics = compute_metrics(pnl_port, DATA_YEARS, portfolio_size=cfg.max_concurrent)
    mfe_losing = compute_mfe_losing(accepted, data_15m)

    n = metrics["N"]
    low_n = n < 100
    result = {
        "strategy": "ma_cluster",
        "label": "MA Cluster",
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "data_years": round(DATA_YEARS, 3),
        "N": n,
        "WR_pct": metrics["WR"],
        "PF": metrics["PF"],
        "MDD_pct": metrics["max_dd_pct"],
        "avg_win_pct": metrics["avg_win_pct"],
        "avg_loss_pct": metrics["avg_loss_pct"],
        "sharpe": _compute_sharpe(pnl_port),
        "calmar": metrics["calmar"],
        "mfe_losing_avg_pct": round(mfe_losing, 3),
        "ann_ret_pct": metrics["ann_ret_pct"],
        "low_N": low_n,
        "tolerance_bands": {"WR": 0.05, "PF": 0.10} if low_n else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _print_summary(result)
    return result


# ── Donchian Baseline ─────────────────────────────────────────────────────────


def run_donchian(cfg: VolStraddleConfig) -> dict:
    """Donchian channel baseline — uses canonical sim from trade_sim."""
    print("\n=== Donchian (DC_vol50) baseline ===")
    t0 = time.time()

    from quantbox.backtest.trade_sim import (
        sim_donchian_trailing,
        portfolio_sim_with_details as _pswd,
    )

    data_15m = load_15m_data(cfg.symbol_whitelist)
    print(f"Loaded {len(data_15m)} symbols")

    data_4h = build_resampled(data_15m, 16)

    all_trades: list[tuple[str, int, int, float]] = []

    warmup = cfg.donchian_lookback + cfg.donchian_vol_sma

    for sym in sorted(data_4h.keys()):
        if sym == "BTCUSDT":
            continue
        d = data_4h[sym]
        c, h, lo, vol = d["close"], d["high"], d["low"], d["volume"]

        pnl_arr, ent_arr, ext_arr = sim_donchian_trailing(
            c, h, lo, vol,
            cfg.donchian_lookback,
            cfg.donchian_vol_sma,
            cfg.donchian_vol_mult,
            cfg.donchian_breakout_pct,
            cfg.donchian_cooldown_bars,
            cfg.donchian_sl,
            cfg.donchian_tp,
            cfg.donchian_trail_activate,
            cfg.donchian_trail_distance,
            cfg.donchian_max_hold_4h,
            cfg.round_trip_cost,
            warmup,
        )

        ratio = 16
        for i in range(len(pnl_arr)):
            ent_15 = int(ent_arr[i]) * ratio
            ext_15 = int(ext_arr[i]) * ratio
            all_trades.append((sym, ent_15, ext_15, float(pnl_arr[i])))

    print(f"Raw Donchian trades: {len(all_trades)}")
    pnl_port, accepted = _pswd(all_trades, cfg.max_concurrent)
    print(f"Accepted after portfolio_sim: {len(accepted)}")

    metrics = compute_metrics(pnl_port, DATA_YEARS, portfolio_size=cfg.max_concurrent)
    mfe_losing = compute_mfe_losing(accepted, data_15m)

    n = metrics["N"]
    low_n = n < 100
    result = {
        "strategy": "donchian",
        "label": "Donchian (DC_vol50)",
        "window_start": WINDOW_START.isoformat(),
        "window_end": WINDOW_END.isoformat(),
        "data_years": round(DATA_YEARS, 3),
        "N": n,
        "WR_pct": metrics["WR"],
        "PF": metrics["PF"],
        "MDD_pct": metrics["max_dd_pct"],
        "avg_win_pct": metrics["avg_win_pct"],
        "avg_loss_pct": metrics["avg_loss_pct"],
        "sharpe": _compute_sharpe(pnl_port),
        "calmar": metrics["calmar"],
        "mfe_losing_avg_pct": round(mfe_losing, 3),
        "ann_ret_pct": metrics["ann_ret_pct"],
        "low_N": low_n,
        "tolerance_bands": {"WR": 0.05, "PF": 0.10} if low_n else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _print_summary(result)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="r116-prep baseline measurement")
    parser.add_argument(
        "--strategy",
        choices=["vr_directional", "ma_cluster", "donchian"],
        required=True,
        help="Strategy to run baseline for",
    )
    args = parser.parse_args()

    cfg = VolStraddleConfig()

    strategy_map = {
        "vr_directional": run_vr_directional,
        "ma_cluster": run_ma_cluster,
        "donchian": run_donchian,
    }

    result = strategy_map[args.strategy](cfg)

    # Write JSON output
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATE_DIR / f"{args.strategy}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
