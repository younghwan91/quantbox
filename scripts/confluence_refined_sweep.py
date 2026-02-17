#!/usr/bin/env python3
"""
v8.0 Phase 2: Refined sweep of Dynamic Target + Confidence + Leverage.

Phase 1 results:
  - RSI Divergence: REJECTED (blocks good trades)
  - Volatility Squeeze: NEUTRAL (always fires, redundant with threshold)
  - Dynamic Target: WINNER (ride confident trades longer)
  - Reversal Confirmation: REJECTED (snap-back is immediate)

This script sweeps the dynamic target parameters and leverage to find the
sweet spot between return amplification and MDD control.

Key principle: "if uncertain, do not trade. if certain, use leverage."
"""

import sys, os, time, gzip, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.confluence_ab_test import (
    compute_signals_confluence,
    load_data, load_btc, TOP10,
    COST_PER_SIDE, FUNDING_15M, MTF_SMA_PERIOD, BASE_PARAMS,
)
from quantbox.strategy.rubberband import (
    apply_mtf_confirmation, apply_mtf_leverage, compute_signals_confidence,
)
from quantbox.backtest.vectorized_backtest import _backtest_core


def portfolio_backtest(symbols_data, close_4h_data, btc_15m, **overrides):
    """Run portfolio backtest with given overrides."""
    equities = []
    all_results = []

    # Separate MTF leverage from signal params
    mtf_lev = overrides.pop("mtf_leverage_mult", 2.0) if "mtf_leverage_mult" in overrides else 2.0
    overrides_clean = {k: v for k, v in overrides.items() if k != "mtf_leverage_mult"}

    for sym in TOP10:
        d = symbols_data[sym]
        c4h = close_4h_data[sym]

        if overrides_clean.get("use_dynamic_target", False):
            # Use new confluence function
            params = dict(
                close=d["close"], high=d["high"], low=d["low"],
                volume=d["volume"], taker_buy=d["taker_buy"], btc_close=btc_15m,
                **BASE_PARAMS,
                use_rsi_divergence=False, divergence_lookback=24,
                use_squeeze=False, squeeze_ratio=0.7,
                atr_period=14, atr_sma_period=50,
                use_dynamic_target=False,
                target_conf3=0.5, target_conf4=0.65, target_conf5=0.8,
                use_reversal_confirmation=False,
            )
            params.update(overrides_clean)
            sig = compute_signals_confluence(**params)
        else:
            sig = compute_signals_confidence(
                close=d["close"], volume=d["volume"],
                taker_buy=d["taker_buy"], btc_close=btc_15m,
                **{k: v for k, v in {**BASE_PARAMS, **overrides_clean}.items()
                   if k in BASE_PARAMS},
            )

        sig = apply_mtf_confirmation(sig, c4h, MTF_SMA_PERIOD, ratio=16)
        sig = apply_mtf_leverage(sig, mtf_lev)

        result = _backtest_core(d["close"], sig, COST_PER_SIDE, FUNDING_15M)
        equities.append(result[0])
        all_results.append({
            "ret": result[1], "sharpe": result[2], "mdd": result[3],
            "win_rate": result[4], "n_trades": result[5],
        })

    # Portfolio
    min_len = min(len(e) for e in equities)
    combined = np.zeros(min_len)
    for eq in equities:
        combined += eq[:min_len]
    combined /= len(equities)

    total_ret = combined[-1] / combined[0] - 1.0
    peak = combined[0]
    max_dd = 0.0
    for i in range(len(combined)):
        if combined[i] > peak:
            peak = combined[i]
        dd = (peak - combined[i]) / peak
        if dd > max_dd:
            max_dd = dd
    rets = np.diff(combined) / combined[:-1]
    mean_r = np.mean(rets)
    std_r = np.std(rets)
    sharpe = mean_r / std_r * np.sqrt(96 * 365.25) if std_r > 0 else 0.0
    calmar = total_ret / max_dd if max_dd > 0 else 0.0

    avg_trades = sum(r["n_trades"] for r in all_results) / len(all_results)
    avg_wr = sum(r["win_rate"] for r in all_results) / len(all_results)

    return {
        "ret": total_ret, "mdd": max_dd, "sharpe": sharpe,
        "calmar": calmar, "trades": avg_trades, "win_rate": avg_wr,
        "equity": combined,
    }


def walk_forward(symbols_data, close_4h_data, btc_15m, **overrides):
    """Quarterly walk-forward validation."""
    mtf_lev = overrides.pop("mtf_leverage_mult", 2.0) if "mtf_leverage_mult" in overrides else 2.0
    overrides_clean = {k: v for k, v in overrides.items() if k != "mtf_leverage_mult"}

    n_bars = len(btc_15m)
    q_size = n_bars // 4
    results = []

    for q_idx in range(4):
        start = q_idx * q_size
        end = min((q_idx + 1) * q_size, n_bars)
        equities = []

        for sym in TOP10:
            d = symbols_data[sym]
            d_q = {k: v[start:end] for k, v in d.items()}
            btc_q = btc_15m[start:end]
            n_4h = len(close_4h_data[sym])
            q4h_s = start * n_4h // n_bars
            q4h_e = end * n_4h // n_bars
            c4h_q = close_4h_data[sym][q4h_s:q4h_e]

            if len(d_q["close"]) < 500 or len(c4h_q) < 20:
                continue

            if overrides_clean.get("use_dynamic_target", False):
                params = dict(
                    close=d_q["close"], high=d_q["high"], low=d_q["low"],
                    volume=d_q["volume"], taker_buy=d_q["taker_buy"], btc_close=btc_q,
                    **BASE_PARAMS,
                    use_rsi_divergence=False, divergence_lookback=24,
                    use_squeeze=False, squeeze_ratio=0.7,
                    atr_period=14, atr_sma_period=50,
                    use_dynamic_target=False,
                    target_conf3=0.5, target_conf4=0.65, target_conf5=0.8,
                    use_reversal_confirmation=False,
                )
                params.update(overrides_clean)
                sig = compute_signals_confluence(**params)
            else:
                sig = compute_signals_confidence(
                    close=d_q["close"], volume=d_q["volume"],
                    taker_buy=d_q["taker_buy"], btc_close=btc_q,
                    **{k: v for k, v in {**BASE_PARAMS, **overrides_clean}.items()
                       if k in BASE_PARAMS},
                )

            sig = apply_mtf_confirmation(sig, c4h_q, MTF_SMA_PERIOD, ratio=16)
            sig = apply_mtf_leverage(sig, mtf_lev)
            result = _backtest_core(d_q["close"], sig, COST_PER_SIDE, FUNDING_15M)
            equities.append(result[0])

        if equities:
            min_len = min(len(e) for e in equities)
            combined = np.zeros(min_len)
            for eq in equities:
                combined += eq[:min_len]
            combined /= len(equities)
            total_ret = combined[-1] / combined[0] - 1.0
            peak_v = combined[0]
            max_dd = 0.0
            for v in combined:
                if v > peak_v:
                    peak_v = v
                dd = (peak_v - v) / peak_v
                if dd > max_dd:
                    max_dd = dd
            results.append({"ret": total_ret, "mdd": max_dd})
        else:
            results.append({"ret": 0.0, "mdd": 0.0})

    return results


def main():
    t0 = time.time()
    print("=" * 110)
    print("  v8.0 REFINED SWEEP: Dynamic Target + Confidence + Leverage")
    print("  Goal: maximize Calmar while keeping portfolio MDD < 7%")
    print("=" * 110)

    btc_15m = load_btc("15m")
    symbols_data = {}
    close_4h_data = {}
    for sym in TOP10:
        symbols_data[sym] = load_data(sym, "15m")
        close_4h_data[sym] = load_data(sym, "4h")["close"]

    # ──────────────────────────────────────────────────────
    # Sweep 1: Dynamic Target parameters at fixed leverage
    # ──────────────────────────────────────────────────────
    print("\n── Sweep 1: Dynamic Target (conf3/conf4/conf5), mc=2, flat 1.5x, MTF 2.0x ──")
    print(f"  {'Config':40s} | {'Ret':>8s} {'DD':>6s} {'Calmar':>8s} {'Sharpe':>7s} {'WR':>5s} {'Trades':>7s}")
    print("  " + "-" * 90)

    # Baseline
    bl = portfolio_backtest(symbols_data, close_4h_data, btc_15m)
    print(f"  {'v7.0 BASELINE (no dyn target)':40s} | {bl['ret']:+7.1%} {bl['mdd']:5.1%} {bl['calmar']:8.2f} {bl['sharpe']:7.2f} {bl['win_rate']:4.0f}% {bl['trades']:6.0f}")

    target_configs = [
        (0.4, 0.45, 0.5, "t: 0.4/0.45/0.5 (gentle)"),
        (0.4, 0.5, 0.6, "t: 0.4/0.5/0.6 (mild)"),
        (0.4, 0.55, 0.7, "t: 0.4/0.55/0.7 (moderate)"),
        (0.5, 0.6, 0.7, "t: 0.5/0.6/0.7 (confident)"),
        (0.5, 0.65, 0.8, "t: 0.5/0.65/0.8 (aggressive)"),
        (0.5, 0.7, 0.9, "t: 0.5/0.7/0.9 (full ride)"),
        (0.45, 0.5, 0.55, "t: 0.45/0.5/0.55 (subtle)"),
        (0.4, 0.6, 0.8, "t: 0.4/0.6/0.8 (wide spread)"),
    ]

    best_target_calmar = bl["calmar"]
    best_target_cfg = None

    for c3, c4, c5, label in target_configs:
        r = portfolio_backtest(symbols_data, close_4h_data, btc_15m,
                               use_dynamic_target=True,
                               target_conf3=c3, target_conf4=c4, target_conf5=c5)
        marker = " <<<" if r["calmar"] > best_target_calmar and r["mdd"] < 0.07 else ""
        print(f"  {label:40s} | {r['ret']:+7.1%} {r['mdd']:5.1%} {r['calmar']:8.2f} {r['sharpe']:7.2f} {r['win_rate']:4.0f}% {r['trades']:6.0f}{marker}")
        if r["calmar"] > best_target_calmar and r["mdd"] < 0.07:
            best_target_calmar = r["calmar"]
            best_target_cfg = (c3, c4, c5)

    if best_target_cfg:
        print(f"\n  >>> Best dynamic target: conf3={best_target_cfg[0]} conf4={best_target_cfg[1]} conf5={best_target_cfg[2]}")
    else:
        print(f"\n  >>> No dynamic target config beat baseline with MDD < 7%. Using baseline targets.")
        best_target_cfg = (0.4, 0.4, 0.4)  # same as fixed 40%

    # ──────────────────────────────────────────────────────
    # Sweep 2: min_confidence + leverage with best target
    # ──────────────────────────────────────────────────────
    print(f"\n── Sweep 2: min_confidence + leverage (best target: {best_target_cfg}) ──")
    print(f"  {'Config':45s} | {'Ret':>8s} {'DD':>6s} {'Calmar':>8s} {'Sharpe':>7s} {'WR':>5s} {'Trades':>7s}")
    print("  " + "-" * 95)

    lev_configs = [
        # (mc, base_lev, conf_step, max_lev, mtf_mult, label)
        (2, 1.5, 0.0, 1.5, 2.0, "mc=2 flat1.5x mtf2.0x"),
        (2, 1.5, 0.0, 1.5, 1.5, "mc=2 flat1.5x mtf1.5x"),
        (2, 1.5, 0.0, 1.5, 2.5, "mc=2 flat1.5x mtf2.5x"),
        (2, 1.5, 0.25, 2.5, 2.0, "mc=2 lev1.5-2.5 mtf2.0x"),
        (2, 1.5, 0.5, 3.0, 1.5, "mc=2 lev1.5-3.0 mtf1.5x"),
        (2, 1.0, 0.5, 2.5, 2.0, "mc=2 lev1.0-2.5 mtf2.0x"),
        (3, 1.5, 0.0, 1.5, 2.0, "mc=3 flat1.5x mtf2.0x"),
        (3, 2.0, 0.0, 2.0, 2.0, "mc=3 flat2.0x mtf2.0x"),
        (3, 1.5, 0.0, 1.5, 2.5, "mc=3 flat1.5x mtf2.5x"),
        (3, 1.5, 0.5, 2.5, 2.0, "mc=3 lev1.5-2.5 mtf2.0x"),
        (3, 2.0, 0.5, 3.0, 2.0, "mc=3 lev2.0-3.0 mtf2.0x"),
        (3, 1.5, 0.5, 2.5, 1.5, "mc=3 lev1.5-2.5 mtf1.5x"),
        (3, 2.0, 0.0, 2.0, 1.5, "mc=3 flat2.0x mtf1.5x"),
        (3, 2.0, 0.5, 3.0, 1.5, "mc=3 lev2.0-3.0 mtf1.5x"),
        (2, 2.0, 0.0, 2.0, 2.0, "mc=2 flat2.0x mtf2.0x"),
        (2, 2.0, 0.0, 2.0, 1.5, "mc=2 flat2.0x mtf1.5x"),
    ]

    best_calmar = 0
    best_config = None
    all_configs = []

    for mc, blv, cs, ml, mtf_m, label in lev_configs:
        use_dt = best_target_cfg != (0.4, 0.4, 0.4)
        r = portfolio_backtest(
            symbols_data, close_4h_data, btc_15m,
            min_confidence=mc, base_leverage=blv,
            confidence_leverage_step=cs, max_leverage=ml,
            mtf_leverage_mult=mtf_m,
            use_dynamic_target=use_dt,
            target_conf3=best_target_cfg[0],
            target_conf4=best_target_cfg[1],
            target_conf5=best_target_cfg[2],
        )
        dd_ok = r["mdd"] < 0.07
        marker = " <<<" if r["calmar"] > best_calmar and dd_ok else ""
        if not dd_ok:
            marker = " [DD>7%]"
        print(f"  {label:45s} | {r['ret']:+7.1%} {r['mdd']:5.1%} {r['calmar']:8.2f} {r['sharpe']:7.2f} {r['win_rate']:4.0f}% {r['trades']:6.0f}{marker}")

        all_configs.append((label, r, mc, blv, cs, ml, mtf_m))
        if r["calmar"] > best_calmar and dd_ok:
            best_calmar = r["calmar"]
            best_config = dict(
                min_confidence=mc, base_leverage=blv,
                confidence_leverage_step=cs, max_leverage=ml,
                mtf_leverage_mult=mtf_m,
                use_dynamic_target=use_dt,
                target_conf3=best_target_cfg[0],
                target_conf4=best_target_cfg[1],
                target_conf5=best_target_cfg[2],
            )

    if not best_config:
        # Fallback to best overall ignoring MDD
        print("\n  No config with DD < 7%. Selecting best overall.")
        best_calmar = 0
        for label, r, mc, blv, cs, ml, mtf_m in all_configs:
            if r["calmar"] > best_calmar:
                best_calmar = r["calmar"]
                best_config = dict(
                    min_confidence=mc, base_leverage=blv,
                    confidence_leverage_step=cs, max_leverage=ml,
                    mtf_leverage_mult=mtf_m,
                    use_dynamic_target=best_target_cfg != (0.4, 0.4, 0.4),
                    target_conf3=best_target_cfg[0],
                    target_conf4=best_target_cfg[1],
                    target_conf5=best_target_cfg[2],
                )

    print(f"\n  >>> BEST CONFIG (DD < 7%): Calmar={best_calmar:.2f}")
    for k, v in best_config.items():
        print(f"      {k} = {v}")

    # ──────────────────────────────────────────────────────
    # Walk-Forward Validation
    # ──────────────────────────────────────────────────────
    print(f"\n── Walk-Forward Validation ──")
    baseline_wf = walk_forward(symbols_data, close_4h_data, btc_15m)
    best_wf = walk_forward(symbols_data, close_4h_data, btc_15m, **best_config)

    print(f"  {'Quarter':6s} | {'Baseline Ret':>12s} {'DD':>6s} | {'v8.0 Ret':>10s} {'DD':>6s} {'DD Change':>10s}")
    print("  " + "-" * 65)
    for i, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        bl_r, bl_dd = baseline_wf[i]["ret"], baseline_wf[i]["mdd"]
        v8_r, v8_dd = best_wf[i]["ret"], best_wf[i]["mdd"]
        dd_chg = (v8_dd - bl_dd) / bl_dd * 100 if bl_dd > 0 else 0
        print(f"  {q_name:6s} | {bl_r:+11.1%} {bl_dd:5.1%} | {v8_r:+9.1%} {v8_dd:5.1%} {dd_chg:+9.0f}%")

    # ──────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────
    print(f"\n── FINAL COMPARISON ──")
    bl_full = portfolio_backtest(symbols_data, close_4h_data, btc_15m)
    v8_full = portfolio_backtest(symbols_data, close_4h_data, btc_15m, **best_config)

    print(f"  {'Metric':15s} | {'v7.0':>10s} | {'v8.0':>10s} | {'Change':>10s}")
    print("  " + "-" * 55)
    for key, fmt in [("ret", "+.1%"), ("mdd", ".1%"), ("calmar", ".2f"), ("sharpe", ".2f")]:
        v7 = bl_full[key]
        v8 = v8_full[key]
        if key == "mdd":
            chg = (v8 - v7) / v7 * 100 if v7 > 0 else 0
        else:
            chg = (v8 - v7) / abs(v7) * 100 if v7 != 0 else 0
        fmt_str = f"{{:{fmt}}}"
        print(f"  {key:15s} | {fmt_str.format(v7):>10s} | {fmt_str.format(v8):>10s} | {chg:+9.1f}%")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
