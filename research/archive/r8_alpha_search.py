#!/usr/bin/env python3
"""
Alpha Search R8 — STRESS TEST: Extreme vs Balanced Exit Configs
================================================================
MOTIVATION: R7 found T62_S79_PBT (balanced, R:R=1.22) and T100_S38_PBT
  (extreme, R:R=1.66). User prefers extreme because "WR will drop live."
  This test validates: does extreme survive when conditions degrade?

Tests:
  A. BREAKEVEN WR — mathematical safety margin
  B. FEE/SLIPPAGE SENSITIVITY — re-simulate with 0.04% → 0.06/0.08/0.10%
  C. REGIME-SPLIT — BTC bull/bear/chop segmented performance
  D. WR DEGRADATION — randomly block X% of entries, measure PF collapse

Candidates:
  1. REF (T38_S79_Std) — current production
  2. T62_S79_PBT — Tier B balanced
  3. T62_S100_PBT — Tier A conservative
  4. T100_S38_PBT — Tier C extreme (user's pick)
  5. T38_S100_PBT — Tier B alt (highest R:R with DD<9%)

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/alpha_search_r8.py
"""

import sys
import time
import json
import subprocess
import numba
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

# ═══ Reuse R7 infrastructure ═══
from scripts.alpha_search_r7 import (
    R7Config, simulate_r7, scan_entries, apply_mtf_filter,
    load_15m, load_4h, compute_atr_pct,
    LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
    ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
    STOCH_ENABLED, STOCH_K, STOCH_D,
    MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP, MTF_SMA_PERIOD,
    DATA_DAYS, CACHE_DIR,
)

SYMBOLS_50 = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'BNBUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'XLMUSDT', 'DOTUSDT', '1000PEPEUSDT',
    'ARBUSDT', 'FETUSDT', '1000SHIBUSDT', 'OPUSDT', 'ATOMUSDT', 'HBARUSDT',
    'ICPUSDT', 'FILUSDT', 'TRXUSDT', 'SUIUSDT', 'TONUSDT', 'LTCUSDT',
    'BCHUSDT', 'UNIUSDT', 'NEARUSDT', 'APTUSDT', 'AAVEUSDT', 'INJUSDT',
    'ETCUSDT', 'MATICUSDT', 'GRTUSDT', 'STXUSDT', 'FTMUSDT', 'EGLDUSDT',
    'AXSUSDT', 'SANDUSDT', 'THETAUSDT', 'ALGOUSDT', 'RUNEUSDT', 'DYDXUSDT',
    'CRVUSDT', 'LDOUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'VETUSDT',
    'EOSUSDT', 'APEUSDT',
]


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


# ═══════════════════════════════════════════════════════════════════
# CANDIDATE CONFIGS
# ═══════════════════════════════════════════════════════════════════

def make_candidates() -> list[R7Config]:
    """5 candidates spanning conservative → extreme."""
    return [
        R7Config(
            name='REF-Prod',
            desc='Production: TP=0.382 SL=0.786 Standard',
            target_retrace=0.382, stop_mult=0.786,
            exit_mode='standard',
        ),
        R7Config(
            name='TierA-T62S100',
            desc='Tier A conservative: TP=0.618 SL=1.000 PBT',
            target_retrace=0.618, stop_mult=1.0,
            exit_mode='partial_be', breakeven_after_tp1=True,
        ),
        R7Config(
            name='TierB-T62S79',
            desc='Tier B balanced: TP=0.618 SL=0.786 PBT',
            target_retrace=0.618, stop_mult=0.786,
            exit_mode='partial_be', breakeven_after_tp1=True,
        ),
        R7Config(
            name='TierB-T38S100',
            desc='Tier B alt: TP=0.382 SL=1.000 PBT (high R:R, low DD)',
            target_retrace=0.382, stop_mult=1.0,
            exit_mode='partial_be', breakeven_after_tp1=True,
        ),
        R7Config(
            name='TierC-T100S38',
            desc='Tier C extreme: TP=1.000 SL=0.382 PBT',
            target_retrace=1.0, stop_mult=0.382,
            exit_mode='partial_be', breakeven_after_tp1=True,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
# TEST A: BREAKEVEN WR (mathematical)
# ═══════════════════════════════════════════════════════════════════

def test_breakeven_wr(candidates: list[R7Config], baseline_results: dict[str, dict]):
    """Calculate breakeven WR and safety margin for each candidate."""
    print("\n" + "=" * 110)
    print("  TEST A: BREAKEVEN WR ANALYSIS")
    print("=" * 110)
    print(f"\n  Formula: Breakeven WR = 1 / (1 + R:R)")
    print(f"  Safety Margin = Actual WR - Breakeven WR")
    print(f"  Higher margin = survives larger WR drops before going unprofitable\n")

    print(f"  {'Name':<18} {'R:R':>5} {'BE_WR':>7} {'Actual_WR':>10} {'Margin':>8} "
          f"{'WR@PF=1.1':>10} {'WR@PF=0.9':>10} {'Verdict':>12}")
    print(f"  {'─' * 95}")

    for cfg in candidates:
        r = baseline_results[cfg.name]
        rr = r['actual_rr']
        wr = r['win_rate'] / 100.0
        be_wr = 1.0 / (1.0 + rr) if rr > 0 else 1.0
        margin = wr - be_wr

        # WR needed for PF=1.1 and PF=0.9
        # PF = (WR × AvgW) / ((1-WR) × AvgL) = (WR × R:R) / (1 - WR)
        # WR_target = PF_target / (PF_target + R:R)
        wr_pf11 = 1.1 / (1.1 + rr) if rr > 0 else 1.0
        wr_pf09 = 0.9 / (0.9 + rr) if rr > 0 else 1.0

        verdict = '✅ SAFE' if margin > 0.15 else ('⚠️ TIGHT' if margin > 0.10 else '❌ RISKY')

        print(f"  {cfg.name:<18} {rr:>5.2f} {be_wr*100:>6.1f}% {r['win_rate']:>8.1f}%  "
              f"{margin*100:>+6.1f}pp {wr_pf11*100:>8.1f}%  {wr_pf09*100:>8.1f}%  {verdict}")

    print(f"\n  📊 Interpretation:")
    print(f"    BE_WR    = WR where PF=1.0 (breakeven, no profit no loss)")
    print(f"    Margin   = how far WR can drop before strategy dies")
    print(f"    WR@PF=1.1 = WR needed to still earn 10% excess over losses")
    print(f"    WR@PF=0.9 = WR where you start losing money (PF<1.0)")


# ═══════════════════════════════════════════════════════════════════
# TEST B: FEE/SLIPPAGE SENSITIVITY
# ═══════════════════════════════════════════════════════════════════

def test_fee_sensitivity(
    candidates: list[R7Config],
    sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr,
):
    """Re-simulate each candidate with increasing fee levels."""
    print("\n" + "=" * 110)
    print("  TEST B: FEE/SLIPPAGE SENSITIVITY")
    print("=" * 110)
    print(f"\n  Baseline fee: 0.04%/side. Testing: 0.06%, 0.08%, 0.10% (simulates slippage)")
    print(f"  Tight stops are more sensitive — each stopped-out trade pays fees twice.\n")

    fee_levels = [0.0004, 0.0006, 0.0008, 0.0010]
    fee_labels = ['0.04%', '0.06%', '0.08%', '0.10%']

    # header
    print(f"  {'Name':<18}", end='')
    for fl in fee_labels:
        print(f"  {'PF@'+fl:>9} {'DD@'+fl:>8} {'Sh@'+fl:>8}", end='')
    print(f"  {'PF_decay':>9}")
    print(f"  {'─' * (18 + len(fee_levels) * 28 + 10)}")

    all_fee_results = {}

    for cfg in candidates:
        pfs = []
        dds = []
        sharpes = []

        for fee in fee_levels:
            t1 = time.time()
            result = simulate_r7(
                sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves,
                cfg=cfg, cost_per_side=fee, sym_atr=sym_atr,
            )
            elapsed = time.time() - t1
            pfs.append(result['profit_factor'])
            dds.append(result['max_dd_pct'])
            sharpes.append(result['sharpe'])

        # PF decay = (PF@0.10% - PF@0.04%) / PF@0.04%
        pf_decay = (pfs[-1] - pfs[0]) / pfs[0] * 100 if pfs[0] > 0 else -999

        print(f"  {cfg.name:<18}", end='')
        for j in range(len(fee_levels)):
            print(f"  {pfs[j]:>8.2f}  {dds[j]:>6.1f}%  {sharpes[j]:>6.2f}", end='')
        print(f"  {pf_decay:>+7.1f}%")

        all_fee_results[cfg.name] = {
            'fees': fee_labels,
            'pf': pfs,
            'dd': dds,
            'sharpe': sharpes,
            'pf_decay_pct': round(pf_decay, 1),
        }

    print(f"\n  📊 PF_decay = (PF@0.10% - PF@0.04%) / PF@0.04%")
    print(f"    Lower decay = more robust to execution costs")
    print(f"    Tight-stop configs trade more → pay more fees → higher decay")

    return all_fee_results


# ═══════════════════════════════════════════════════════════════════
# TEST C: REGIME-SPLIT PERFORMANCE
# ═══════════════════════════════════════════════════════════════════

def classify_regime(btc_close: np.ndarray, window: int = 96 * 30) -> np.ndarray:
    """Classify each bar as bull(1), bear(-1), or chop(0) based on BTC 30-day return.

    Bull: BTC 30d return > +5%
    Bear: BTC 30d return < -5%
    Chop: -5% to +5%
    """
    n = len(btc_close)
    regime = np.zeros(n, dtype=np.int32)

    for i in range(window, n):
        ret = btc_close[i] / btc_close[i - window] - 1.0
        if ret > 0.05:
            regime[i] = 1   # bull
        elif ret < -0.05:
            regime[i] = -1  # bear
        else:
            regime[i] = 0   # chop

    return regime


def test_regime_split(
    candidates: list[R7Config],
    sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr,
    btc_close: np.ndarray,
):
    """Run each candidate, then split trade results by BTC regime at entry time."""
    print("\n" + "=" * 110)
    print("  TEST C: REGIME-SPLIT PERFORMANCE")
    print("=" * 110)

    regime = classify_regime(btc_close)
    n = len(regime)
    n_bull = np.sum(regime == 1)
    n_bear = np.sum(regime == -1)
    n_chop = np.sum(regime == 0)
    print(f"\n  BTC Regime classification (30d rolling return):")
    print(f"    Bull (>+5%):  {n_bull:>8,} bars ({n_bull/n*100:.1f}%)")
    print(f"    Chop (±5%):   {n_chop:>8,} bars ({n_chop/n*100:.1f}%)")
    print(f"    Bear (<-5%):  {n_bear:>8,} bars ({n_bear/n*100:.1f}%)")
    print(f"\n  Key question: Does tight-stop extreme config collapse in bear/chop?\n")

    regime_labels = ['BULL', 'CHOP', 'BEAR']
    regime_vals = [1, 0, -1]

    # For regime-split, we need to modify simulation to tag trades with entry regime
    # Simpler approach: mask entries by regime and re-simulate

    all_regime_results = {}

    for rl, rv in zip(regime_labels, regime_vals):
        print(f"\n  ── {rl} regime only ──")
        print(f"  {'Name':<18} {'WR':>6} {'R:R':>5} {'DD':>6} {'Sharpe':>7} "
              f"{'PF':>6} {'Trades':>7} {'Ret':>10}")
        print(f"  {'─' * 80}")

        for cfg in candidates:
            # Mask entries to only this regime
            masked_dirs = {}
            masked_confs = {}
            masked_moves = {}

            for sym in sym_dirs:
                d = sym_dirs[sym].copy()
                c = sym_confs[sym].copy()
                m = sym_moves[sym].copy()

                for i in range(len(d)):
                    if i < len(regime) and regime[i] != rv:
                        d[i] = 0.0
                        c[i] = 0.0
                        m[i] = 0.0

                masked_dirs[sym] = d
                masked_confs[sym] = c
                masked_moves[sym] = m

            result = simulate_r7(
                sym_close, sym_high, sym_low,
                masked_dirs, masked_confs, masked_moves,
                cfg=cfg, sym_atr=sym_atr,
            )

            key = f"{cfg.name}_{rl}"
            all_regime_results[key] = result

            print(f"  {cfg.name:<18} {result['win_rate']:>5.1f}% {result['actual_rr']:>5.2f} "
                  f"{result['max_dd_pct']:>5.1f}% {result['sharpe']:>6.2f} "
                  f"{result['profit_factor']:>5.2f} {result['n_trades']:>6,} "
                  f"{result['total_return_pct']:>+9,.1f}%")

    # Summary comparison table
    print(f"\n\n  {'─' * 110}")
    print(f"  📊 REGIME SUMMARY (PF comparison):")
    print(f"  {'Name':<18} {'PF_BULL':>8} {'PF_CHOP':>8} {'PF_BEAR':>8} "
          f"{'CHOP/BULL':>10} {'BEAR/BULL':>10} {'Worst_PF':>9} {'Robust':>8}")
    print(f"  {'─' * 95}")

    for cfg in candidates:
        pf_bull = all_regime_results.get(f"{cfg.name}_BULL", {}).get('profit_factor', 0)
        pf_chop = all_regime_results.get(f"{cfg.name}_CHOP", {}).get('profit_factor', 0)
        pf_bear = all_regime_results.get(f"{cfg.name}_BEAR", {}).get('profit_factor', 0)

        chop_ratio = pf_chop / pf_bull if pf_bull > 0 else 0
        bear_ratio = pf_bear / pf_bull if pf_bull > 0 else 0
        worst_pf = min(pf_bull, pf_chop, pf_bear)

        robust = '✅' if worst_pf >= 1.10 else ('⚠️' if worst_pf >= 1.00 else '❌')

        print(f"  {cfg.name:<18} {pf_bull:>7.2f}  {pf_chop:>7.2f}  {pf_bear:>7.2f}  "
              f"{chop_ratio:>8.1%}   {bear_ratio:>8.1%}   {worst_pf:>7.2f}   {robust}")

    print(f"\n  📊 REGIME SUMMARY (R:R comparison):")
    print(f"  {'Name':<18} {'RR_BULL':>8} {'RR_CHOP':>8} {'RR_BEAR':>8} "
          f"{'WR_BULL':>8} {'WR_CHOP':>8} {'WR_BEAR':>8}")
    print(f"  {'─' * 80}")

    for cfg in candidates:
        rr_b = all_regime_results.get(f"{cfg.name}_BULL", {}).get('actual_rr', 0)
        rr_c = all_regime_results.get(f"{cfg.name}_CHOP", {}).get('actual_rr', 0)
        rr_r = all_regime_results.get(f"{cfg.name}_BEAR", {}).get('actual_rr', 0)
        wr_b = all_regime_results.get(f"{cfg.name}_BULL", {}).get('win_rate', 0)
        wr_c = all_regime_results.get(f"{cfg.name}_CHOP", {}).get('win_rate', 0)
        wr_r = all_regime_results.get(f"{cfg.name}_BEAR", {}).get('win_rate', 0)

        print(f"  {cfg.name:<18} {rr_b:>7.2f}  {rr_c:>7.2f}  {rr_r:>7.2f}  "
              f"{wr_b:>6.1f}%  {wr_c:>6.1f}%  {wr_r:>6.1f}%")

    return all_regime_results


# ═══════════════════════════════════════════════════════════════════
# TEST D: WR DEGRADATION (Monte Carlo entry removal)
# ═══════════════════════════════════════════════════════════════════

def test_wr_degradation(
    candidates: list[R7Config],
    sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr,
    n_seeds: int = 5,
):
    """Randomly remove X% of entries and measure how each candidate degrades.

    This simulates "what if our entry signal degrades and we get fewer winning setups."
    We remove entries randomly (simulating general signal decay), not selectively
    removing winners — the actual WR change depends on which entries happen to be removed.
    """
    print("\n" + "=" * 110)
    print("  TEST D: WR DEGRADATION (MONTE CARLO ENTRY REMOVAL)")
    print("=" * 110)
    print(f"\n  Method: Randomly block X% of all entries, re-simulate.")
    print(f"  This tests: 'What if our signal generates fewer entries (alpha decay)?'")
    print(f"  {n_seeds} random seeds per degradation level, averaged.\n")

    drop_pcts = [0, 10, 20, 30, 40, 50]

    # header
    print(f"  {'Name':<18}", end='')
    for dp in drop_pcts:
        print(f"  {'PF@-'+str(dp)+'%':>9} {'WR@-'+str(dp)+'%':>9}", end='')
    print(f"  {'PF_surv':>8}")
    print(f"  {'─' * (18 + len(drop_pcts) * 20 + 9)}")

    all_degrade_results = {}

    for cfg in candidates:
        row_pfs = []
        row_wrs = []

        for dp in drop_pcts:
            if dp == 0:
                # Baseline — no degradation
                result = simulate_r7(
                    sym_close, sym_high, sym_low,
                    sym_dirs, sym_confs, sym_moves,
                    cfg=cfg, sym_atr=sym_atr,
                )
                row_pfs.append(result['profit_factor'])
                row_wrs.append(result['win_rate'])
            else:
                seed_pfs = []
                seed_wrs = []

                for seed in range(n_seeds):
                    rng = np.random.RandomState(42 + seed)

                    masked_dirs = {}
                    masked_confs = {}
                    masked_moves = {}

                    for sym in sym_dirs:
                        d = sym_dirs[sym].copy()
                        c = sym_confs[sym].copy()
                        m = sym_moves[sym].copy()

                        entry_idxs = np.nonzero(d)[0]
                        n_drop = int(len(entry_idxs) * dp / 100)
                        if n_drop > 0:
                            drop_idxs = rng.choice(entry_idxs, size=n_drop, replace=False)
                            d[drop_idxs] = 0.0
                            c[drop_idxs] = 0.0
                            m[drop_idxs] = 0.0

                        masked_dirs[sym] = d
                        masked_confs[sym] = c
                        masked_moves[sym] = m

                    result = simulate_r7(
                        sym_close, sym_high, sym_low,
                        masked_dirs, masked_confs, masked_moves,
                        cfg=cfg, sym_atr=sym_atr,
                    )
                    seed_pfs.append(result['profit_factor'])
                    seed_wrs.append(result['win_rate'])

                row_pfs.append(float(np.mean(seed_pfs)))
                row_wrs.append(float(np.mean(seed_wrs)))

        # PF survival = PF at -50% entries / PF at 0%
        pf_surv = row_pfs[-1] / row_pfs[0] * 100 if row_pfs[0] > 0 else 0

        print(f"  {cfg.name:<18}", end='')
        for j in range(len(drop_pcts)):
            print(f"  {row_pfs[j]:>8.2f}  {row_wrs[j]:>7.1f}%", end='')
        flag = '✅' if pf_surv > 90 else ('⚠️' if pf_surv > 80 else '❌')
        print(f"  {pf_surv:>6.1f}% {flag}")

        all_degrade_results[cfg.name] = {
            'drop_pcts': drop_pcts,
            'pf': row_pfs,
            'wr': row_wrs,
            'pf_survival_pct': round(pf_surv, 1),
        }

    print(f"\n  📊 PF_surv = PF@-50% / PF@0% — higher = more robust to signal decay")
    print(f"    ✅ >90%: Excellent survivability")
    print(f"    ⚠️ 80-90%: Moderate — needs monitoring")
    print(f"    ❌ <80%: Fragile — alpha decay will kill this config")

    return all_degrade_results


# ═══════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════

def print_verdict(
    candidates, baseline_results, fee_results, regime_results, degrade_results,
):
    """Synthesize all tests into a final recommendation."""
    print("\n" + "=" * 110)
    print("  R8 FINAL VERDICT")
    print("=" * 110)

    print(f"\n  {'Name':<18} {'R:R':>5} {'BE_margin':>10} {'Fee_decay':>10} "
          f"{'Worst_PF':>9} {'PF_surv':>8} {'Score':>7} {'Grade':>6}")
    print(f"  {'─' * 85}")

    scores = []

    for cfg in candidates:
        r = baseline_results[cfg.name]
        rr = r['actual_rr']
        wr = r['win_rate'] / 100.0
        be_wr = 1.0 / (1.0 + rr) if rr > 0 else 1.0
        be_margin = wr - be_wr

        fee_decay = abs(fee_results.get(cfg.name, {}).get('pf_decay_pct', 0))

        # Worst regime PF
        pf_bull = regime_results.get(f"{cfg.name}_BULL", {}).get('profit_factor', 0)
        pf_chop = regime_results.get(f"{cfg.name}_CHOP", {}).get('profit_factor', 0)
        pf_bear = regime_results.get(f"{cfg.name}_BEAR", {}).get('profit_factor', 0)
        worst_pf = min(pf_bull, pf_chop, pf_bear)

        pf_surv = degrade_results.get(cfg.name, {}).get('pf_survival_pct', 0)

        # Composite robustness score
        # Higher = more robust. Components:
        #   BE margin (0-30pp) → normalized to 0-1
        #   Fee robustness (0-10% decay) → 1 - decay/10, clamped
        #   Worst regime PF (0-2) → normalized
        #   PF survival (0-100%) → /100
        s_margin = min(be_margin / 0.30, 1.0)
        s_fee = max(1.0 - fee_decay / 10.0, 0.0)
        s_regime = min(worst_pf / 1.5, 1.0)
        s_decay = pf_surv / 100.0

        # Weighted composite (BE margin most important for user's concern)
        composite = (s_margin * 0.35 + s_fee * 0.15 + s_regime * 0.30 + s_decay * 0.20) * 10

        grade = 'S' if composite > 7.5 else ('A' if composite > 6.5 else ('B' if composite > 5.5 else ('C' if composite > 4.5 else 'D')))

        scores.append((cfg.name, composite, grade))

        print(f"  {cfg.name:<18} {rr:>5.2f} {be_margin*100:>+8.1f}pp "
              f"{fee_decay:>8.1f}%  {worst_pf:>8.2f} {pf_surv:>7.1f}% "
              f"{composite:>6.1f}  {'★' if grade in ('S','A') else ' '}{grade}")

    scores.sort(key=lambda x: x[1], reverse=True)
    winner = scores[0]

    print(f"\n  🏆 ROBUSTNESS CHAMPION: {winner[0]} (Score={winner[1]:.1f}, Grade={winner[2]})")
    print(f"\n  📊 Score weights: BE Margin 35% | Worst Regime PF 30% | PF Survival 20% | Fee Robustness 15%")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    commit = _get_git_commit()

    print("=" * 110)
    print("  ALPHA SEARCH R8 — STRESS TEST: EXTREME vs BALANCED")
    print(f"  Commit: {commit}")
    print(f"  Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Data: {DATA_DAYS}d (6yr, crash-era)")
    print(f"  Tests: A=Breakeven WR, B=Fee Sensitivity, C=Regime Split, D=WR Degradation")
    print(f"  Goal: Validate if extreme config (T100_S38_PBT) survives real-world stress")
    print("=" * 110)

    # ── Load data ──
    print(f"\n📊 Loading data...")
    btc_data = load_15m('BTCUSDT')
    if btc_data is None:
        print("ERROR: BTC data not found")
        sys.exit(1)
    btc_close = btc_data['close']
    print(f"  BTC: {len(btc_close):,} bars ({len(btc_close)/96:.0f}d)")

    all_data: dict[str, dict] = {}
    all_4h: dict[str, np.ndarray] = {}
    for sym in SYMBOLS_50:
        d = load_15m(sym)
        if d is None:
            continue
        all_data[sym] = d
        h4 = load_4h(sym)
        if h4 is not None:
            all_4h[sym] = h4

    available = [s for s in SYMBOLS_50 if s in all_data]
    print(f"  Loaded: {len(available)} symbols")

    # ── Scan entries ──
    print(f"\n🔍 Scanning entries...")
    t0 = time.time()

    sym_dirs = {}
    sym_confs = {}
    sym_moves = {}
    sym_close = {}
    sym_high = {}
    sym_low = {}

    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        c, h, l, v, t = d['close'][:n], d['high'][:n], d['low'][:n], d['volume'][:n], d['taker_buy'][:n]
        b = btc_close[:n]

        dirs, confs, moves = scan_entries(
            c, h, l, v, t, b,
            LOOKBACK, THRESHOLD, BTC_MOM_PERIOD, BTC_REGIME_PERIOD,
            ASSET_TREND_PERIOD, MIN_CONFIDENCE, RSI_PERIOD, VOL_PERIOD,
            STOCH_ENABLED, STOCH_K, STOCH_D,
            MACD_ENABLED, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            ATR_PERIOD, ATR_MULT, THR_FLOOR, THR_CAP,
        )

        close_4h = all_4h.get(sym)
        if close_4h is not None:
            dirs, confs, moves = apply_mtf_filter(
                dirs, confs, moves, close_4h, MTF_SMA_PERIOD, ratio=16)

        sym_dirs[sym] = dirs
        sym_confs[sym] = confs
        sym_moves[sym] = moves
        sym_close[sym] = c
        sym_high[sym] = h
        sym_low[sym] = l

    # Pre-compute ATR
    sym_atr = {}
    for sym in available:
        d = all_data[sym]
        n = min(len(d['close']), len(btc_close))
        atr = compute_atr_pct(d['high'][:n], d['low'][:n], d['close'][:n], ATR_PERIOD)
        sym_atr[sym] = atr

    # Left-pad to max length
    max_bars = max(len(sym_close[s]) for s in sym_close)
    for sym in list(sym_close.keys()):
        for d_ref, fill in [
            (sym_close, 'first'), (sym_high, 'first'), (sym_low, 'first'),
            (sym_dirs, 'zero'), (sym_confs, 'zero'), (sym_moves, 'zero'),
            (sym_atr, 'nan'),
        ]:
            arr = d_ref[sym]
            if len(arr) < max_bars:
                pad_n = max_bars - len(arr)
                if fill == 'first':
                    fv = arr[0]
                elif fill == 'nan':
                    fv = np.nan
                else:
                    fv = 0.0
                d_ref[sym] = np.concatenate([np.full(pad_n, fv), arr])

    # Also pad btc_close for regime classification
    if len(btc_close) < max_bars:
        pad_n = max_bars - len(btc_close)
        btc_close = np.concatenate([np.full(pad_n, btc_close[0]), btc_close])

    total_entries = sum(np.count_nonzero(sym_dirs[s]) for s in sym_dirs)
    print(f"  Entries: {total_entries:,} ({time.time()-t0:.1f}s)")
    print(f"  Window: {max_bars:,} bars ({max_bars/96:.0f}d)")

    # ── Define candidates ──
    candidates = make_candidates()

    # ── Baseline run (0.04% fees) ──
    print(f"\n🧪 Running baseline simulations...")
    baseline_results = {}
    for cfg in candidates:
        t1 = time.time()
        result = simulate_r7(
            sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves,
            cfg=cfg, sym_atr=sym_atr,
        )
        elapsed = time.time() - t1
        baseline_results[cfg.name] = result
        print(f"  {cfg.name:<18} WR={result['win_rate']:5.1f}% R:R={result['actual_rr']:.2f} "
              f"DD={result['max_dd_pct']:5.1f}% Sharpe={result['sharpe']:.2f} "
              f"PF={result['profit_factor']:.2f} Ret={result['total_return_pct']:+8,.1f}% [{elapsed:.1f}s]")

    # ══════════════════════════════════════════════════════════════
    # Run all 4 tests
    # ══════════════════════════════════════════════════════════════

    # TEST A: Breakeven WR
    test_breakeven_wr(candidates, baseline_results)

    # TEST B: Fee/Slippage sensitivity
    fee_results = test_fee_sensitivity(
        candidates, sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr)

    # TEST C: Regime-split
    regime_results = test_regime_split(
        candidates, sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr,
        btc_close)

    # TEST D: WR Degradation
    degrade_results = test_wr_degradation(
        candidates, sym_close, sym_high, sym_low, sym_dirs, sym_confs, sym_moves, sym_atr,
        n_seeds=5)

    # ── Final verdict ──
    print_verdict(candidates, baseline_results, fee_results, regime_results, degrade_results)

    # ── Save results ──
    output_path = Path(__file__).parent.parent / 'logs' / 'alpha_search_r8_results.json'
    save_data = {
        'baseline': {k: {kk: vv for kk, vv in v.items() if kk != 'equity_curve'}
                     for k, v in baseline_results.items()},
        'fee_sensitivity': fee_results,
        'regime_split': {k: {kk: vv for kk, vv in v.items() if kk != 'equity_curve'}
                         for k, v in regime_results.items()},
        'wr_degradation': degrade_results,
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    print(f"\n{'=' * 110}")
    print("  DONE — R8 STRESS TEST COMPLETE")
    print(f"{'=' * 110}")


if __name__ == '__main__':
    main()
