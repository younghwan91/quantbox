#!/usr/bin/env python3
"""
v9.0 Refined Sweep: Fine-tune MACD + Fib + Stochastic parameters
================================================================
Phase 1 winners:
  - MACD Histogram: +7.6% Calmar
  - Fib Exits 0.5/0.786: +6.4% Calmar, -3.5% DD (less DD is gold)
  - Fib0.5+Stoch combo: +9.9% Calmar, -1.3% DD (best combo)

This sweep explores:
  1. MACD alone: different params (fast/slow/signal)
  2. Fib stop variations: 0.7-0.9 range (with fixed 0.5 target)
  3. Fib target variations: 0.4-0.6 range (with fixed 0.786 stop)
  4. Dynamic Fib target: scale Fib target by confidence (merge v8 + v9)
  5. MACD + Fib combined: best of both
  6. Triple: MACD + Fib + Stoch
  7. Walk-forward validation of best config
"""

import sys, os, time, gzip, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.v9_srm_ab_test import (
    load_data, load_btc, run_baseline, run_v9,
    portfolio_metrics, TOP10, DATA_DIR,
)

def main():
    print("=" * 80)
    print("v9.0 Refined Sweep: MACD + Fib + Stochastic Parameter Tuning")
    print("=" * 80)

    # Load data
    btc_15m = load_btc("15m")
    btc_4h = load_btc("4h")
    all_data = {sym: load_data(sym, "15m") for sym in TOP10}
    print(f"Loaded {len(all_data)} symbols")

    # JIT warm-up
    _ = run_baseline(all_data[TOP10[0]], btc_15m, btc_4h)
    _ = run_v9(all_data[TOP10[0]], btc_15m, btc_4h, {
        "stoch_enabled": True, "macd_enabled": True,
        "fib_exits": True, "fib_target": 0.5, "fib_stop": 0.786,
    })

    # Baseline
    base_eq, base_res = [], []
    for sym in TOP10:
        r = run_baseline(all_data[sym], btc_15m, btc_4h)
        base_eq.append(r[0])
        base_res.append(r)
    baseline = portfolio_metrics(base_eq, base_res)
    print(f"\nBaseline (v8.0): Ret={baseline['ret']:.1f}%, DD={baseline['dd']:.1f}%, "
          f"Calmar={baseline['calmar']:.2f}, Trades={baseline['trades']}")

    def sweep(name, ov):
        eq, res = [], []
        for sym in TOP10:
            r = run_v9(all_data[sym], btc_15m, btc_4h, ov)
            eq.append(r[0]); res.append(r)
        m = portfolio_metrics(eq, res)
        cd = ((m['calmar'] / baseline['calmar']) - 1) * 100 if baseline['calmar'] > 0 else 0
        dd_d = ((m['dd'] / baseline['dd']) - 1) * 100 if baseline['dd'] > 0 else 0
        print(f"  {name:42s}: Ret={m['ret']:6.1f}%  DD={m['dd']:5.1f}%  "
              f"Calmar={m['calmar']:6.2f}  CΔ={cd:+6.1f}%  DDΔ={dd_d:+5.1f}%  "
              f"T={m['trades']:3d}  WR={m['wr']:4.1f}%")
        return m

    # ═══════════════════ Sweep 1: Fib Stop Variations ═══════════════════
    print("\n" + "=" * 80)
    print("SWEEP 1: Fib Stop Variations (target=0.5, vary stop)")
    print("=" * 80)
    best_fib_stop = 0.786
    best_calmar = 0
    for stop in [0.6, 0.65, 0.7, 0.75, 0.786, 0.8, 0.85, 0.9, 1.0]:
        m = sweep(f"stop={stop:.3f}", {"fib_exits": True, "fib_target": 0.5, "fib_stop": stop})
        if m['calmar'] > best_calmar:
            best_calmar = m['calmar']
            best_fib_stop = stop

    print(f"\n  >> Best fib stop: {best_fib_stop}")

    # ═══════════════════ Sweep 2: Fib Target Variations ═══════════════════
    print("\n" + "=" * 80)
    print(f"SWEEP 2: Fib Target Variations (stop={best_fib_stop}, vary target)")
    print("=" * 80)
    best_fib_target = 0.5
    best_calmar = 0
    for target in [0.35, 0.382, 0.4, 0.45, 0.5, 0.55, 0.6, 0.618, 0.7]:
        m = sweep(f"target={target:.3f}", {"fib_exits": True, "fib_target": target, "fib_stop": best_fib_stop})
        if m['calmar'] > best_calmar:
            best_calmar = m['calmar']
            best_fib_target = target

    print(f"\n  >> Best fib target: {best_fib_target}, best fib stop: {best_fib_stop}")

    # ═══════════════════ Sweep 3: MACD Params ═══════════════════
    print("\n" + "=" * 80)
    print("SWEEP 3: MACD Parameter Variations")
    print("=" * 80)
    macd_configs = [
        (12, 26, 9, "standard"),
        (8, 21, 5, "fast"),
        (6, 13, 5, "very_fast"),
        (12, 26, 5, "fast_signal"),
        (12, 26, 14, "slow_signal"),
        (8, 17, 9, "medium"),
    ]
    best_macd = (12, 26, 9)
    best_calmar = 0
    for fast, slow, signal, label in macd_configs:
        m = sweep(f"MACD({fast},{slow},{signal}) {label}", {
            "macd_enabled": True, "macd_fast": fast, "macd_slow": slow, "macd_signal": signal,
        })
        if m['calmar'] > best_calmar:
            best_calmar = m['calmar']
            best_macd = (fast, slow, signal)

    print(f"\n  >> Best MACD: {best_macd}")

    # ═══════════════════ Sweep 4: Stochastic Params ═══════════════════
    print("\n" + "=" * 80)
    print("SWEEP 4: Stochastic Parameter Variations")
    print("=" * 80)
    stoch_configs = [
        (14, 3, "standard"),
        (10, 3, "fast"),
        (5, 3, "very_fast"),
        (14, 5, "slow_d"),
        (21, 5, "slow"),
    ]
    best_stoch = (14, 3)
    best_calmar = 0
    for k, d, label in stoch_configs:
        m = sweep(f"Stoch({k},{d}) {label}", {
            "stoch_enabled": True, "stoch_k_period": k, "stoch_d_period": d,
        })
        if m['calmar'] > best_calmar:
            best_calmar = m['calmar']
            best_stoch = (k, d)

    print(f"\n  >> Best Stoch: K={best_stoch[0]}, D={best_stoch[1]}")

    # ═══════════════════ Sweep 5: MACD + Fib Combined ═══════════════════
    print("\n" + "=" * 80)
    print("SWEEP 5: MACD + Fib Combined (best params)")
    print("=" * 80)

    combos = [
        ("MACD only", {"macd_enabled": True, "macd_fast": best_macd[0],
                        "macd_slow": best_macd[1], "macd_signal": best_macd[2]}),
        ("Fib only", {"fib_exits": True, "fib_target": best_fib_target,
                       "fib_stop": best_fib_stop}),
        ("MACD+Fib", {"macd_enabled": True, "macd_fast": best_macd[0],
                       "macd_slow": best_macd[1], "macd_signal": best_macd[2],
                       "fib_exits": True, "fib_target": best_fib_target,
                       "fib_stop": best_fib_stop}),
        ("Stoch+Fib", {"stoch_enabled": True, "stoch_k_period": best_stoch[0],
                        "stoch_d_period": best_stoch[1],
                        "fib_exits": True, "fib_target": best_fib_target,
                        "fib_stop": best_fib_stop}),
        ("MACD+Stoch+Fib", {"macd_enabled": True, "macd_fast": best_macd[0],
                             "macd_slow": best_macd[1], "macd_signal": best_macd[2],
                             "stoch_enabled": True, "stoch_k_period": best_stoch[0],
                             "stoch_d_period": best_stoch[1],
                             "fib_exits": True, "fib_target": best_fib_target,
                             "fib_stop": best_fib_stop}),
        ("MACD+Stoch(no Fib)", {"macd_enabled": True, "macd_fast": best_macd[0],
                                 "macd_slow": best_macd[1], "macd_signal": best_macd[2],
                                 "stoch_enabled": True, "stoch_k_period": best_stoch[0],
                                 "stoch_d_period": best_stoch[1]}),
    ]

    best_combo_name = ""
    best_combo_ov = {}
    best_calmar = 0
    for name, ov in combos:
        m = sweep(name, ov)
        if m['calmar'] > best_calmar:
            best_calmar = m['calmar']
            best_combo_name = name
            best_combo_ov = ov

    print(f"\n  >> Best combo: {best_combo_name}")

    # ═══════════════════ Sweep 6: Dynamic Fib Target (Fib × Confidence) ═══════════════════
    print("\n" + "=" * 80)
    print("SWEEP 6: Dynamic Fib Target (v8 confidence scaling + Fib stop)")
    print("=" * 80)
    print("  Using dynamic_target with Fib stop (not overriding target via fib_exits)")

    # Instead of fib_exits=True (overrides target), use the Fib stop directly
    # Need to test: keep dynamic target scaling but use Fib stop
    dyn_fib_configs = [
        # Stop=0.786 with existing dynamic target
        ("DynTarget+stop=0.786", {"stop_mult": 0.786}),
        ("DynTarget+stop=0.75", {"stop_mult": 0.75}),
        ("DynTarget+stop=0.7", {"stop_mult": 0.7}),
        # Wider dynamic targets with Fib stop
        ("DynTarget(0.45/0.5/0.55)+stop=0.786",
         {"stop_mult": 0.786,
          "target_conf3": 0.45, "target_conf4": 0.5, "target_conf5": 0.55}),
        ("DynTarget(0.5/0.55/0.618)+stop=0.786",
         {"stop_mult": 0.786,
          "target_conf3": 0.5, "target_conf4": 0.55, "target_conf5": 0.618}),
        ("DynTarget(0.5/0.618/0.786)+stop=0.786",
         {"stop_mult": 0.786,
          "target_conf3": 0.5, "target_conf4": 0.618, "target_conf5": 0.786}),
        # With MACD
        ("DynTarget+stop=0.786+MACD",
         {"stop_mult": 0.786, "macd_enabled": True,
          "macd_fast": best_macd[0], "macd_slow": best_macd[1], "macd_signal": best_macd[2]}),
        ("DynTarget(0.5/0.55/0.618)+stop=0.786+MACD",
         {"stop_mult": 0.786,
          "target_conf3": 0.5, "target_conf4": 0.55, "target_conf5": 0.618,
          "macd_enabled": True,
          "macd_fast": best_macd[0], "macd_slow": best_macd[1], "macd_signal": best_macd[2]}),
    ]

    for name, ov in dyn_fib_configs:
        sweep(name, ov)

    # ═══════════════════ Sweep 7: Walk-Forward of Top 3 ═══════════════════
    print("\n" + "=" * 80)
    print("SWEEP 7: Walk-Forward Quarterly Validation (Top configs)")
    print("=" * 80)

    # Top configs to validate
    top_configs = [
        ("Baseline v8.0", {}),
        ("Best combo", best_combo_ov),
        ("MACD only (best)", {"macd_enabled": True, "macd_fast": best_macd[0],
                               "macd_slow": best_macd[1], "macd_signal": best_macd[2]}),
        ("Fib only (best)", {"fib_exits": True, "fib_target": best_fib_target,
                              "fib_stop": best_fib_stop}),
    ]

    bars_per_quarter = 96 * 91
    n_bars = len(btc_15m)
    n_quarters = min(n_bars // bars_per_quarter, 4)

    for config_name, ov in top_configs:
        print(f"\n  {config_name}:")
        for q in range(n_quarters):
            start = q * bars_per_quarter
            end = min((q + 1) * bars_per_quarter, n_bars)
            q_eq, q_res = [], []
            for sym in TOP10:
                data = all_data[sym]
                q_data = {k: v[start:end] for k, v in data.items()}
                q_btc = btc_15m[start:end]
                q_btc4h = btc_4h[start // 16: end // 16]
                if len(q_data["close"]) < 500:
                    continue
                try:
                    if ov:
                        r = run_v9(q_data, q_btc, q_btc4h, ov)
                    else:
                        r = run_baseline(q_data, q_btc, q_btc4h)
                    q_eq.append(r[0])
                    q_res.append(r)
                except Exception:
                    continue

            if q_eq:
                qm = portfolio_metrics(q_eq, q_res)
                print(f"    Q{q+1}: Ret={qm['ret']:+6.1f}%  DD={qm['dd']:5.1f}%  "
                      f"Calmar={qm['calmar']:6.2f}  T={qm['trades']:3d}")


if __name__ == "__main__":
    main()
