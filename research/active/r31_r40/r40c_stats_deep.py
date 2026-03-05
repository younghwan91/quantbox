"""R40 Champion — Deep Statistical Analysis Report.

VR8_56_SL1_TP5_D4 @3x leverage, 15 slots 의 성능을 다차원 통계 분석.
PnL 분포, 연속 승률/패배, Drawdown 해부, Bootstrap CI,
VaR/CVaR, 자기상관, 레버리지별 비교, 심볼별 분석 등.

Usage:
    PYTHONUNBUFFERED=1 uv run python research/active/stats_r40_deep.py
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from scipy import stats as sp_stats

from research.active.r31_r40.r31_mtf_vol_straddle import (
    load_15m, build_resampled, precompute,
    bootstrap_pf, pf_from_pnl,
)
from research.active.r31_r40.scaling_r38 import (
    compute_metrics, precompute_all_vr,
    COST_DIRECTIONAL, COST_STRADDLE, G_3TF,
)
from research.active.r31_r40.r40d_tournament import (
    collect_trades_custom,
    portfolio_sim_levered,
    portfolio_sim_levered_with_details,
    yearly_equity_levered,
    BARS_PER_YEAR,
)

# ══════════════════════════════════════════════════════════════════════════════
#  Config: R40 Champion
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    "sp": 8, "lp": 56,
    "d_vr": 0.50, "s_vr": 0.55, "s_dur": 4,
    "d_sl": 0.01, "s_sl": 0.015, "tp": 0.05,
}
LEV_CHAMPION = 3.0
SLOTS_CHAMPION = 15
NAME = "VR8_56_SL1_TP5_D4"

# Comparison leverages
LEV_COMPARE = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
SLOT_COMPARE = [5, 10, 15, 20, 30]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def equity_curve(pnl: np.ndarray) -> np.ndarray:
    """Cumulative equity curve (start=1)."""
    eq = np.empty(len(pnl) + 1)
    eq[0] = 1.0
    for i, p in enumerate(pnl):
        eq[i + 1] = eq[i] * (1.0 + p)
    return eq


def drawdown_series(eq: np.ndarray) -> np.ndarray:
    """Drawdown % series from equity curve."""
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    return dd


def drawdown_periods(eq: np.ndarray) -> list[dict]:
    """Identify each drawdown period: start, trough, recovery, depth, duration."""
    dd = drawdown_series(eq)
    periods = []
    in_dd = False
    start = 0
    max_dd = 0.0
    trough_idx = 0

    for i in range(len(dd)):
        if dd[i] > 0 and not in_dd:
            in_dd = True
            start = i
            max_dd = dd[i]
            trough_idx = i
        elif dd[i] > 0 and in_dd:
            if dd[i] > max_dd:
                max_dd = dd[i]
                trough_idx = i
        elif dd[i] == 0 and in_dd:
            in_dd = False
            periods.append({
                "start": start,
                "trough": trough_idx,
                "recovery": i,
                "depth_pct": max_dd * 100,
                "duration_trades": i - start,
            })
            max_dd = 0.0

    # Still in drawdown
    if in_dd:
        periods.append({
            "start": start,
            "trough": trough_idx,
            "recovery": None,
            "depth_pct": max_dd * 100,
            "duration_trades": len(dd) - start,
        })

    return periods


def streaks(pnl: np.ndarray) -> tuple[list[int], list[int]]:
    """Win streaks and loss streaks."""
    wins, losses = [], []
    curr_w, curr_l = 0, 0
    for p in pnl:
        if p > 0:
            curr_w += 1
            if curr_l > 0:
                losses.append(curr_l)
                curr_l = 0
        else:
            curr_l += 1
            if curr_w > 0:
                wins.append(curr_w)
                curr_w = 0
    if curr_w > 0:
        wins.append(curr_w)
    if curr_l > 0:
        losses.append(curr_l)
    return wins, losses


def autocorrelation(pnl: np.ndarray, max_lag: int = 10) -> list[float]:
    """Autocorrelation of PnL at lags 1..max_lag."""
    n = len(pnl)
    if n < max_lag + 2:
        return [0.0] * max_lag
    x = pnl - np.mean(pnl)
    var = np.var(pnl)
    if var < 1e-18:
        return [0.0] * max_lag
    acf = []
    for lag in range(1, max_lag + 1):
        c = np.mean(x[lag:] * x[:-lag]) / var
        acf.append(float(c))
    return acf


def bootstrap_metric(pnl: np.ndarray, metric_fn, n_boot: int = 5000,
                     ci: float = 0.95) -> dict:
    """Bootstrap any scalar metric."""
    rng = np.random.default_rng(42)
    n = len(pnl)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[b] = metric_fn(pnl[idx])
    alpha = (1 - ci) / 2
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_lo": float(np.percentile(values, alpha * 100)),
        "ci_hi": float(np.percentile(values, (1 - alpha) * 100)),
        "std": float(np.std(values)),
    }


def sharpe_ratio(pnl: np.ndarray, annual_trades: float) -> float:
    """Annualized Sharpe (assumes 0 risk-free)."""
    if len(pnl) < 2:
        return 0.0
    mu = np.mean(pnl)
    sigma = np.std(pnl, ddof=1)
    if sigma < 1e-18:
        return 0.0
    return float(mu / sigma * np.sqrt(annual_trades))


def sortino_ratio(pnl: np.ndarray, annual_trades: float) -> float:
    """Annualized Sortino (downside deviation only)."""
    if len(pnl) < 2:
        return 0.0
    mu = np.mean(pnl)
    downside = pnl[pnl < 0]
    if len(downside) < 2:
        return 999.0
    dd_std = np.std(downside, ddof=1)
    if dd_std < 1e-18:
        return 999.0
    return float(mu / dd_std * np.sqrt(annual_trades))


def value_at_risk(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """VaR at given confidence level (negative = loss)."""
    return float(np.percentile(pnl, (1 - confidence) * 100))


def cvar_expected_shortfall(pnl: np.ndarray, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall."""
    var = value_at_risk(pnl, confidence)
    tail = pnl[pnl <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def max_consecutive_losses_pnl(pnl: np.ndarray) -> float:
    """Max cumulative loss from a consecutive losing streak (compound)."""
    max_loss = 0.0
    cum = 1.0
    in_streak = False
    for p in pnl:
        if p < 0:
            if not in_streak:
                cum = 1.0
                in_streak = True
            cum *= (1 + p)
            loss = 1.0 - cum
            if loss > max_loss:
                max_loss = loss
        else:
            in_streak = False
            cum = 1.0
    return max_loss * 100


# ══════════════════════════════════════════════════════════════════════════════
#  Main Analysis
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()

    # ── Data Load ──
    data_15m = load_15m()
    data_years = max(d.get("days", 0) for d in data_15m.values()) / 365
    n_years = int(data_years)
    data_1h = build_resampled(data_15m, 4)
    data_4h = build_resampled(data_15m, 16)
    precompute(data_15m); precompute(data_1h); precompute(data_4h)
    precompute_all_vr([data_15m, data_1h, data_4h], [(CFG["sp"], CFG["lp"])])

    # ── Collect Trades ──
    d_trades = collect_trades_custom(CFG, data_15m, data_1h, data_4h, "D_mom8")
    s_trades = collect_trades_custom(CFG, data_15m, data_1h, data_4h, "S_d8")
    all_raw_trades = d_trades + s_trades

    # Champion config: 3x, 15 slots
    pnl_3x, accepted_3x = portfolio_sim_levered_with_details(
        all_raw_trades, SLOTS_CHAMPION, LEV_CHAMPION,
    )
    pnl_1x, accepted_1x = portfolio_sim_levered_with_details(
        all_raw_trades, SLOTS_CHAMPION, 1.0,
    )
    pnl_5x, accepted_5x = portfolio_sim_levered_with_details(
        all_raw_trades, SLOTS_CHAMPION, 5.0,
    )

    eq_3x = equity_curve(pnl_3x)
    eq_1x = equity_curve(pnl_1x)
    annual_trades = len(pnl_3x) / data_years

    print("=" * 120)
    print(f"  R40 DEEP STATISTICAL ANALYSIS — {NAME} @{LEV_CHAMPION}x, {SLOTS_CHAMPION} slots")
    print(f"  Data: {len(data_15m)} symbols × {data_years:.1f} years | Trades: {len(pnl_3x)} accepted")
    print("=" * 120)

    # ═══════════════════════════════════════════════════════
    #  1. CORE METRICS SUMMARY
    # ═══════════════════════════════════════════════════════
    m = compute_metrics(pnl_3x, data_years)
    m1x = compute_metrics(pnl_1x, data_years)

    print(f"\n{'─'*80}")
    print(f"  1. CORE METRICS (3x vs 1x)")
    print(f"{'─'*80}")
    print(f"  {'Metric':<28} {'3x @15 slots':>18} {'1x @15 slots':>18}")
    print(f"  {'─'*64}")
    for key in ["N", "PF", "EV_bp", "t", "WR", "RR", "avg_win_pct", "avg_loss_pct",
                "ann_ret_pct", "max_dd_pct", "calmar", "tpd"]:
        v3 = m.get(key, 0)
        v1 = m1x.get(key, 0)
        if isinstance(v3, int):
            print(f"  {key:<28} {v3:>18} {v1:>18}")
        else:
            print(f"  {key:<28} {v3:>18.3f} {v1:>18.3f}")

    s3 = sharpe_ratio(pnl_3x, annual_trades)
    s1 = sharpe_ratio(pnl_1x, annual_trades)
    so3 = sortino_ratio(pnl_3x, annual_trades)
    so1 = sortino_ratio(pnl_1x, annual_trades)
    print(f"  {'Sharpe (annualized)':<28} {s3:>18.3f} {s1:>18.3f}")
    print(f"  {'Sortino (annualized)':<28} {so3:>18.3f} {so1:>18.3f}")

    # ═══════════════════════════════════════════════════════
    #  2. PnL DISTRIBUTION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  2. PnL DISTRIBUTION (3x leverage)")
    print(f"{'─'*80}")

    pnl_pct = pnl_3x * 100  # convert to %
    print(f"  Mean:          {np.mean(pnl_pct):>+8.3f}%")
    print(f"  Median:        {np.median(pnl_pct):>+8.3f}%")
    print(f"  Std Dev:       {np.std(pnl_pct):>8.3f}%")
    print(f"  Skewness:      {float(sp_stats.skew(pnl_pct)):>8.3f}")
    print(f"  Kurtosis:      {float(sp_stats.kurtosis(pnl_pct)):>8.3f}")
    print(f"  Min:           {np.min(pnl_pct):>+8.3f}%")
    print(f"  Max:           {np.max(pnl_pct):>+8.3f}%")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        val = np.percentile(pnl_pct, p)
        print(f"    P{p:>2}: {val:>+8.3f}%")

    # PnL histogram bins
    bins = [(-100, -10), (-10, -5), (-5, -3), (-3, -1), (-1, 0),
            (0, 1), (1, 3), (3, 5), (5, 10), (10, 20)]
    print(f"\n  PnL Distribution (bin counts):")
    print(f"  {'Range':>15}  {'Count':>6}  {'Pct':>6}  {'Bar'}")
    for lo, hi in bins:
        mask = (pnl_pct >= lo) & (pnl_pct < hi)
        cnt = np.sum(mask)
        pct = cnt / len(pnl_pct) * 100
        bar = "█" * int(pct)
        print(f"  [{lo:>+5.0f},{hi:>+5.0f})  {cnt:>6}  {pct:>5.1f}%  {bar}")

    # ═══════════════════════════════════════════════════════
    #  3. WIN/LOSS STREAK ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  3. WIN/LOSS STREAK ANALYSIS")
    print(f"{'─'*80}")

    w_streaks, l_streaks = streaks(pnl_3x)
    if w_streaks:
        print(f"  Win streaks:   max={max(w_streaks)}, avg={np.mean(w_streaks):.1f}, "
              f"median={np.median(w_streaks):.0f}, count={len(w_streaks)}")
    if l_streaks:
        print(f"  Loss streaks:  max={max(l_streaks)}, avg={np.mean(l_streaks):.1f}, "
              f"median={np.median(l_streaks):.0f}, count={len(l_streaks)}")

    # Streak distribution
    print(f"\n  Win Streak Length Distribution:")
    if w_streaks:
        for length in range(1, max(w_streaks) + 1):
            cnt = sum(1 for s in w_streaks if s == length)
            if cnt > 0:
                bar = "█" * min(cnt, 40)
                print(f"    {length:>2} wins: {cnt:>4}  {bar}")

    print(f"\n  Loss Streak Length Distribution:")
    if l_streaks:
        for length in range(1, min(max(l_streaks) + 1, 20)):
            cnt = sum(1 for s in l_streaks if s == length)
            if cnt > 0:
                bar = "█" * min(cnt, 40)
                print(f"    {length:>2} losses: {cnt:>4}  {bar}")

    max_consec_loss = max_consecutive_losses_pnl(pnl_3x)
    print(f"\n  Max consecutive loss (compound): {max_consec_loss:.2f}%")

    # ═══════════════════════════════════════════════════════
    #  4. DRAWDOWN ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  4. DRAWDOWN ANALYSIS")
    print(f"{'─'*80}")

    dd_series = drawdown_series(eq_3x)
    dd_periods = drawdown_periods(eq_3x)
    dd_periods.sort(key=lambda x: x["depth_pct"], reverse=True)

    print(f"  Max Drawdown:       {np.max(dd_series) * 100:.2f}%")
    print(f"  Avg Drawdown:       {np.mean(dd_series) * 100:.2f}%")
    print(f"  Median Drawdown:    {np.median(dd_series) * 100:.2f}%")
    print(f"  Time in Drawdown:   {np.mean(dd_series > 0) * 100:.1f}%")
    print(f"  Total DD Periods:   {len(dd_periods)}")

    # Top 10 drawdowns
    print(f"\n  Top 10 Drawdowns:")
    print(f"  {'#':>3} {'Depth%':>8} {'Duration':>10} {'Start':>8} {'Trough':>8} {'Recovery':>10}")
    print(f"  {'─'*50}")
    for rank, dd in enumerate(dd_periods[:10], 1):
        recov = str(dd["recovery"]) if dd["recovery"] is not None else "ongoing"
        print(f"  {rank:>3} {dd['depth_pct']:>8.2f} {dd['duration_trades']:>10} "
              f"{dd['start']:>8} {dd['trough']:>8} {recov:>10}")

    # Drawdown duration statistics
    dd_durs = [d["duration_trades"] for d in dd_periods if d["recovery"] is not None]
    if dd_durs:
        print(f"\n  Recovery Duration Stats (trades):")
        print(f"    Mean:    {np.mean(dd_durs):.1f}")
        print(f"    Median:  {np.median(dd_durs):.0f}")
        print(f"    Max:     {max(dd_durs)}")
        print(f"    P90:     {np.percentile(dd_durs, 90):.0f}")

    # ═══════════════════════════════════════════════════════
    #  5. RISK METRICS (VaR, CVaR)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  5. RISK METRICS")
    print(f"{'─'*80}")

    for conf in [0.90, 0.95, 0.99]:
        var = value_at_risk(pnl_pct, conf)
        cvar = cvar_expected_shortfall(pnl_pct, conf)
        print(f"  VaR  {conf*100:.0f}%: {var:>+8.3f}%   |   CVaR {conf*100:.0f}%: {cvar:>+8.3f}%")

    # Tail risk
    big_wins = pnl_3x[pnl_3x > 0.03]  # > 3%
    big_loss = pnl_3x[pnl_3x < -0.03]  # < -3%
    print(f"\n  Tail Events (>3% magnitude):")
    print(f"    Big wins  (>+3%): {len(big_wins):>5}  ({len(big_wins)/len(pnl_3x)*100:.1f}%)")
    print(f"    Big losses(<-3%): {len(big_loss):>5}  ({len(big_loss)/len(pnl_3x)*100:.1f}%)")
    if len(big_wins) > 0:
        print(f"    Avg big win:  {np.mean(big_wins)*100:>+8.3f}%")
    if len(big_loss) > 0:
        print(f"    Avg big loss: {np.mean(big_loss)*100:>+8.3f}%")

    # ═══════════════════════════════════════════════════════
    #  6. BOOTSTRAP CONFIDENCE INTERVALS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  6. BOOTSTRAP CONFIDENCE INTERVALS (5,000 samples)")
    print(f"{'─'*80}")

    # PF
    bp = bootstrap_pf(pnl_3x, 5000)
    print(f"  Profit Factor:")
    print(f"    Median: {bp['median']:.3f}  |  95% CI: [{bp['CI_lo']:.3f}, {bp['CI_hi']:.3f}]")
    print(f"    P(PF ≥ 1.0): {bp['P_PF_ge_1']:.1f}%")

    # WR
    bwr = bootstrap_metric(pnl_3x, lambda p: np.mean(p > 0) * 100, 5000)
    print(f"  Win Rate:")
    print(f"    Median: {bwr['median']:.1f}%  |  95% CI: [{bwr['ci_lo']:.1f}%, {bwr['ci_hi']:.1f}%]")

    # EV
    bev = bootstrap_metric(pnl_3x, lambda p: np.mean(p) * 10000, 5000)
    print(f"  Expected Value:")
    print(f"    Median: {bev['median']:+.1f}bp  |  95% CI: [{bev['ci_lo']:+.1f}bp, {bev['ci_hi']:+.1f}bp]")

    # Sharpe
    def _sharpe(p):
        mu = np.mean(p)
        s = np.std(p, ddof=1)
        return (mu / s * np.sqrt(annual_trades)) if s > 1e-18 else 0.0
    bshp = bootstrap_metric(pnl_3x, _sharpe, 5000)
    print(f"  Sharpe Ratio:")
    print(f"    Median: {bshp['median']:.3f}  |  95% CI: [{bshp['ci_lo']:.3f}, {bshp['ci_hi']:.3f}]")

    # Max DD
    def _max_dd(p):
        eq = np.empty(len(p) + 1); eq[0] = 1.0
        for i in range(len(p)): eq[i+1] = eq[i] * (1 + p[i])
        pk = np.maximum.accumulate(eq)
        return np.max((pk - eq) / pk) * 100
    bdd = bootstrap_metric(pnl_3x, _max_dd, 5000)
    print(f"  Max Drawdown:")
    print(f"    Median: {bdd['median']:.1f}%  |  95% CI: [{bdd['ci_lo']:.1f}%, {bdd['ci_hi']:.1f}%]")

    # ═══════════════════════════════════════════════════════
    #  7. AUTOCORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  7. AUTOCORRELATION OF RETURNS")
    print(f"{'─'*80}")

    acf = autocorrelation(pnl_3x, 15)
    # Ljung-Box Q statistic
    n = len(pnl_3x)
    q_stat = 0.0
    for lag, r in enumerate(acf[:10], 1):
        q_stat += r**2 / (n - lag)
    q_stat *= n * (n + 2)
    q_pval = 1 - sp_stats.chi2.cdf(q_stat, 10)

    print(f"  {'Lag':>4} {'ACF':>8}  {'Significance'}")
    print(f"  {'─'*35}")
    sig_threshold = 1.96 / np.sqrt(n)
    for lag, r in enumerate(acf, 1):
        sig = "★ significant" if abs(r) > sig_threshold else ""
        bar_len = int(abs(r) * 50)
        bar_dir = "+" if r > 0 else "-"
        bar = bar_dir * bar_len
        print(f"  {lag:>4} {r:>+8.4f}  {bar:>15} {sig}")
    print(f"\n  Ljung-Box Q(10) = {q_stat:.2f},  p-value = {q_pval:.4f}")
    print(f"  → {'Significant serial dependence' if q_pval < 0.05 else 'No significant serial dependence (iid-like)'}")

    # ═══════════════════════════════════════════════════════
    #  8. YEAR-BY-YEAR DETAILED STATISTICS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  8. YEAR-BY-YEAR ANALYSIS")
    print(f"{'─'*80}")

    yr_data_3x = yearly_equity_levered(all_raw_trades, SLOTS_CHAMPION, LEV_CHAMPION, n_years)

    # Detailed per-year metrics
    print(f"\n  {'Year':>6} {'N':>5} {'Ret%':>10} {'DD%':>7} {'PF':>6} "
          f"{'Sharpe':>7} {'Sortino':>8} {'WR%':>5}")
    print(f"  {'─'*60}")

    year_start = 2020
    for yr_info in yr_data_3x:
        yr_idx = yr_info["year"]
        yr_start_bar = yr_idx * BARS_PER_YEAR
        yr_end_bar = (yr_idx + 1) * BARS_PER_YEAR

        yr_pnl = np.array([
            pnl for _, entry, _, pnl in accepted_3x
            if yr_start_bar <= entry < yr_end_bar
        ])
        yr_n = len(yr_pnl)
        if yr_n < 2:
            print(f"  {year_start + yr_idx:>6} {yr_n:>5} {'---':>10}")
            continue

        yr_wr = np.mean(yr_pnl > 0) * 100
        yr_sharpe = sharpe_ratio(yr_pnl, yr_n)  # already annualized-ish
        yr_sortino = sortino_ratio(yr_pnl, yr_n)

        print(f"  {year_start + yr_idx:>6} {yr_n:>5} {yr_info['ret_pct']:>+10.1f} "
              f"{yr_info['max_dd_pct']:>7.1f} {yr_info['pf']:>6.2f} "
              f"{yr_sharpe:>7.2f} {yr_sortino:>8.2f} {yr_wr:>5.1f}")

    # ═══════════════════════════════════════════════════════
    #  9. MONTHLY RETURN HEATMAP
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  9. MONTHLY RETURN APPROXIMATION (15m bars → months)")
    print(f"{'─'*80}")

    # 15m bars per month ≈ 30 * 24 * 4 = 2880
    BARS_PER_MONTH = 2880
    n_months = int(data_years * 12)
    month_rets = []

    for mi in range(n_months):
        m_start = mi * BARS_PER_MONTH
        m_end = (mi + 1) * BARS_PER_MONTH
        m_pnl = [pnl for _, entry, _, pnl in accepted_3x
                 if m_start <= entry < m_end]
        if not m_pnl:
            month_rets.append(0.0)
            continue
        eq = 1.0
        for p in m_pnl:
            eq *= (1 + p)
        month_rets.append((eq - 1.0) * 100)

    month_rets_arr = np.array(month_rets)
    pos_months = np.sum(month_rets_arr > 0)
    neg_months = np.sum(month_rets_arr < 0)
    zero_months = np.sum(month_rets_arr == 0)

    print(f"  Total Months:   {n_months}")
    print(f"  Positive:       {pos_months} ({pos_months/n_months*100:.1f}%)")
    print(f"  Negative:       {neg_months} ({neg_months/n_months*100:.1f}%)")
    print(f"  Zero/No Trade:  {zero_months}")
    print(f"  Best Month:     {np.max(month_rets_arr):>+8.1f}%")
    print(f"  Worst Month:    {np.min(month_rets_arr):>+8.1f}%")
    print(f"  Mean Month:     {np.mean(month_rets_arr):>+8.1f}%")
    print(f"  Median Month:   {np.median(month_rets_arr):>+8.1f}%")
    print(f"  Std Month:      {np.std(month_rets_arr):>8.1f}%")

    # Year × Month heatmap
    print(f"\n  Monthly Returns Heatmap (% per month, Year × Month):")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    print(f"  {'Year':>6} " + " ".join(f"{m:>7}" for m in month_labels) + f" {'Total':>8}")
    print(f"  {'─'*110}")

    for yr_idx in range(n_years):
        row = month_rets[yr_idx*12 : (yr_idx+1)*12]
        yr_total = sum(row)
        row_strs = []
        for v in row:
            if v == 0.0:
                row_strs.append(f"{'---':>7}")
            else:
                row_strs.append(f"{v:>+7.1f}")
        print(f"  {year_start + yr_idx:>6} " + " ".join(row_strs) + f" {yr_total:>+8.1f}")

    # ═══════════════════════════════════════════════════════
    #  10. TRADE DURATION DISTRIBUTION
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  10. TRADE DURATION DISTRIBUTION")
    print(f"{'─'*80}")

    durations = np.array([exit_b - entry for _, entry, exit_b, _ in accepted_3x])
    dur_hours = durations / 4.0  # 15m bars → hours

    print(f"  Mean duration:    {np.mean(dur_hours):>8.1f} hours")
    print(f"  Median duration:  {np.median(dur_hours):>8.1f} hours")
    print(f"  Min duration:     {np.min(dur_hours):>8.1f} hours")
    print(f"  Max duration:     {np.max(dur_hours):>8.1f} hours")
    print(f"  Std duration:     {np.std(dur_hours):>8.1f} hours")

    dur_bins_h = [(0, 1), (1, 4), (4, 12), (12, 24), (24, 48),
                  (48, 96), (96, 192), (192, 500)]
    print(f"\n  Duration Distribution:")
    print(f"  {'Range (hours)':>18}  {'Count':>6}  {'Pct':>6}  {'WR%':>5}")
    print(f"  {'─'*45}")
    for lo, hi in dur_bins_h:
        mask = (dur_hours >= lo) & (dur_hours < hi)
        cnt = np.sum(mask)
        pct = cnt / len(dur_hours) * 100 if len(dur_hours) > 0 else 0
        # WR for this bucket
        bucket_pnl = pnl_3x[mask]
        wr = np.mean(bucket_pnl > 0) * 100 if len(bucket_pnl) > 0 else 0
        print(f"  [{lo:>5.0f},{hi:>5.0f})h  {cnt:>6}  {pct:>5.1f}%  {wr:>5.1f}")

    # Duration vs outcome
    winners_dur = dur_hours[pnl_3x > 0]
    losers_dur = dur_hours[pnl_3x <= 0]
    print(f"\n  Winner avg duration:  {np.mean(winners_dur):.1f}h  "
          f"(median {np.median(winners_dur):.1f}h)")
    print(f"  Loser avg duration:   {np.mean(losers_dur):.1f}h  "
          f"(median {np.median(losers_dur):.1f}h)")

    # ═══════════════════════════════════════════════════════
    #  11. SYMBOL-LEVEL ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  11. SYMBOL-LEVEL ANALYSIS")
    print(f"{'─'*80}")

    sym_stats: dict[str, list[float]] = {}
    for sym, entry, exit_b, pnl_val in accepted_3x:
        sym_stats.setdefault(sym, []).append(pnl_val)

    sym_metrics = []
    for sym, pnls in sym_stats.items():
        arr = np.array(pnls)
        n_trades = len(arr)
        wr = np.mean(arr > 0) * 100
        pf = pf_from_pnl(arr)
        ev_bp = np.mean(arr) * 10000
        eq = 1.0
        for p in arr:
            eq *= (1 + p)
        total_ret = (eq - 1.0) * 100
        sym_metrics.append({
            "sym": sym, "N": n_trades, "WR": wr, "PF": pf,
            "EV_bp": ev_bp, "ret_pct": total_ret,
        })

    sym_metrics.sort(key=lambda x: x["ret_pct"], reverse=True)

    # Top/Bottom symbols
    print(f"  Total active symbols: {len(sym_metrics)}")
    print(f"\n  Top 15 Symbols by Compound Return:")
    print(f"  {'Symbol':<18} {'N':>5} {'WR%':>5} {'PF':>6} {'EV_bp':>8} {'Ret%':>10}")
    print(f"  {'─'*55}")
    for s in sym_metrics[:15]:
        print(f"  {s['sym']:<18} {s['N']:>5} {s['WR']:>5.1f} {s['PF']:>6.2f} "
              f"{s['EV_bp']:>+8.1f} {s['ret_pct']:>+10.1f}")

    print(f"\n  Bottom 10 Symbols by Compound Return:")
    print(f"  {'Symbol':<18} {'N':>5} {'WR%':>5} {'PF':>6} {'EV_bp':>8} {'Ret%':>10}")
    print(f"  {'─'*55}")
    for s in sym_metrics[-10:]:
        print(f"  {s['sym']:<18} {s['N']:>5} {s['WR']:>5.1f} {s['PF']:>6.2f} "
              f"{s['EV_bp']:>+8.1f} {s['ret_pct']:>+10.1f}")

    # Symbol contribution concentration
    positive_syms = [s for s in sym_metrics if s["ret_pct"] > 0]
    negative_syms = [s for s in sym_metrics if s["ret_pct"] <= 0]
    print(f"\n  Profitable symbols:   {len(positive_syms)} / {len(sym_metrics)} "
          f"({len(positive_syms)/len(sym_metrics)*100:.1f}%)")
    print(f"  Unprofitable symbols: {len(negative_syms)} / {len(sym_metrics)}")

    # ═══════════════════════════════════════════════════════
    #  12. STRATEGY TYPE BREAKDOWN (Directional vs Straddle)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  12. STRATEGY TYPE BREAKDOWN")
    print(f"{'─'*80}")

    # Re-collect separately for analysis
    pnl_d_3x, _ = portfolio_sim_levered_with_details(d_trades, SLOTS_CHAMPION, LEV_CHAMPION)
    pnl_s_3x, _ = portfolio_sim_levered_with_details(s_trades, SLOTS_CHAMPION, LEV_CHAMPION)
    md = compute_metrics(pnl_d_3x, data_years) if len(pnl_d_3x) > 0 else {}
    ms = compute_metrics(pnl_s_3x, data_years) if len(pnl_s_3x) > 0 else {}

    print(f"  {'Metric':<28} {'Directional':>15} {'Straddle':>15} {'Combined':>15}")
    print(f"  {'─'*75}")
    for key in ["N", "PF", "EV_bp", "WR", "RR", "ann_ret_pct", "max_dd_pct", "calmar"]:
        vd = md.get(key, 0)
        vs = ms.get(key, 0)
        vc = m.get(key, 0)
        if isinstance(vc, int):
            print(f"  {key:<28} {vd:>15} {vs:>15} {vc:>15}")
        else:
            print(f"  {key:<28} {vd:>15.3f} {vs:>15.3f} {vc:>15.3f}")

    print(f"  {'Raw trades':<28} {len(d_trades):>15} {len(s_trades):>15} "
          f"{len(all_raw_trades):>15}")

    # ═══════════════════════════════════════════════════════
    #  13. LEVERAGE SENSITIVITY TABLE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  13. LEVERAGE SENSITIVITY (15 slots fixed)")
    print(f"{'─'*80}")

    print(f"  {'Lev':>5} {'N':>5} {'AnnRet%':>10} {'DD%':>7} {'Calmar':>7} "
          f"{'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>5} {'LoseYr':>7}")
    print(f"  {'─'*80}")
    for lev in LEV_COMPARE:
        pnl_l = portfolio_sim_levered(all_raw_trades, SLOTS_CHAMPION, lev)
        ml = compute_metrics(pnl_l, data_years)
        at = len(pnl_l) / data_years
        sh = sharpe_ratio(pnl_l, at)
        so = sortino_ratio(pnl_l, at)
        yr = yearly_equity_levered(all_raw_trades, SLOTS_CHAMPION, lev, n_years)
        lose_yr = sum(1 for y in yr if y["ret_pct"] < 0)
        marker = " ◄" if lev == LEV_CHAMPION else ""
        print(f"  {lev:>5.1f} {ml['N']:>5} {ml['ann_ret_pct']:>+10.1f} {ml['max_dd_pct']:>7.1f} "
              f"{ml['calmar']:>7.1f} {sh:>7.2f} {so:>8.2f} {ml['PF']:>6.3f} "
              f"{ml['WR']:>5.1f} {lose_yr:>7}{marker}")

    # ═══════════════════════════════════════════════════════
    #  14. SLOT SENSITIVITY TABLE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  14. SLOT SENSITIVITY (3x leverage fixed)")
    print(f"{'─'*80}")

    print(f"  {'Slots':>5} {'N':>5} {'AnnRet%':>10} {'DD%':>7} {'Calmar':>7} "
          f"{'PF':>6} {'WR%':>5} {'LoseYr':>7}")
    print(f"  {'─'*60}")
    for slots in SLOT_COMPARE:
        pnl_s = portfolio_sim_levered(all_raw_trades, slots, LEV_CHAMPION)
        ms_ = compute_metrics(pnl_s, data_years)
        yr = yearly_equity_levered(all_raw_trades, slots, LEV_CHAMPION, n_years)
        lose_yr = sum(1 for y in yr if y["ret_pct"] < 0)
        marker = " ◄" if slots == SLOTS_CHAMPION else ""
        print(f"  {slots:>5} {ms_['N']:>5} {ms_['ann_ret_pct']:>+10.1f} {ms_['max_dd_pct']:>7.1f} "
              f"{ms_['calmar']:>7.1f} {ms_['PF']:>6.3f} {ms_['WR']:>5.1f} "
              f"{lose_yr:>7}{marker}")

    # ═══════════════════════════════════════════════════════
    #  15. KELLY CRITERION DEEP ANALYSIS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  15. KELLY CRITERION & OPTIMAL SIZING")
    print(f"{'─'*80}")

    # Kelly per year
    print(f"  Year-by-Year Kelly Fractions (1x PnL base):")
    for yr_idx in range(n_years):
        yr_start_bar = yr_idx * BARS_PER_YEAR
        yr_end_bar = (yr_idx + 1) * BARS_PER_YEAR
        yr_pnl_1x = np.array([
            pnl for _, entry, _, pnl in accepted_1x
            if yr_start_bar <= entry < yr_end_bar
        ])
        if len(yr_pnl_1x) < 10:
            print(f"    {year_start + yr_idx}: insufficient data")
            continue
        wr = np.mean(yr_pnl_1x > 0)
        avg_w = np.mean(yr_pnl_1x[yr_pnl_1x > 0]) if np.any(yr_pnl_1x > 0) else 0
        avg_l = abs(np.mean(yr_pnl_1x[yr_pnl_1x < 0])) if np.any(yr_pnl_1x < 0) else 1e-12
        rr = avg_w / avg_l
        kelly = wr - (1 - wr) / rr
        opt_lev = kelly / avg_l if avg_l > 0 else 1.0
        print(f"    {year_start + yr_idx}: WR={wr:.3f}  R:R={rr:.2f}  "
              f"Kelly={kelly:.3f}  OptLev={opt_lev:.1f}x  HalfKelly={opt_lev/2:.1f}x")

    # Overall Kelly
    overall_pnl_1x = pnl_1x
    wr_ov = np.mean(overall_pnl_1x > 0)
    avg_w_ov = np.mean(overall_pnl_1x[overall_pnl_1x > 0])
    avg_l_ov = abs(np.mean(overall_pnl_1x[overall_pnl_1x < 0]))
    rr_ov = avg_w_ov / avg_l_ov
    kelly_ov = wr_ov - (1 - wr_ov) / rr_ov
    opt_lev_ov = kelly_ov / avg_l_ov

    print(f"\n  Overall: Kelly={kelly_ov:.3f}  OptLev={opt_lev_ov:.1f}x  "
          f"HalfKelly={opt_lev_ov/2:.1f}x")
    print(f"  Current leverage: {LEV_CHAMPION}x = "
          f"{LEV_CHAMPION/opt_lev_ov*100:.0f}% of optimal Kelly")

    # ═══════════════════════════════════════════════════════
    #  16. STATISTICAL SIGNIFICANCE TESTS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  16. STATISTICAL SIGNIFICANCE")
    print(f"{'─'*80}")

    # t-test: is mean PnL > 0?
    t_stat, t_pval = sp_stats.ttest_1samp(pnl_3x, 0)
    print(f"  H₀: Mean PnL = 0  →  t={t_stat:.3f},  p={t_pval:.6f}")
    print(f"  → {'Reject H₀ (edge exists)' if t_pval < 0.05 else 'Cannot reject H₀'}")

    # Wilcoxon signed-rank (non-parametric)
    w_stat, w_pval = sp_stats.wilcoxon(pnl_3x)
    print(f"\n  Wilcoxon signed-rank: W={w_stat:.0f},  p={w_pval:.6f}")
    print(f"  → {'Reject H₀ (significant)' if w_pval < 0.05 else 'Cannot reject H₀'}")

    # Runs test for randomness
    median_pnl = np.median(pnl_3x)
    above = pnl_3x > median_pnl
    runs = 1
    for i in range(1, len(above)):
        if above[i] != above[i-1]:
            runs += 1
    n_above = np.sum(above)
    n_below = len(above) - n_above
    expected_runs = 1 + 2 * n_above * n_below / len(above)
    var_runs = (2 * n_above * n_below * (2 * n_above * n_below - len(above))) / \
               (len(above)**2 * (len(above) - 1)) if len(above) > 1 else 1
    z_runs = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    runs_pval = 2 * (1 - sp_stats.norm.cdf(abs(z_runs)))
    print(f"\n  Runs test (randomness): runs={runs}, expected={expected_runs:.0f}, "
          f"z={z_runs:.2f}, p={runs_pval:.4f}")
    print(f"  → {'Non-random pattern detected' if runs_pval < 0.05 else 'Consistent with random (no exploitable pattern in sequence)'}")

    # Effect size (Cohen's d)
    cohens_d = np.mean(pnl_3x) / np.std(pnl_3x, ddof=1)
    effect_label = "negligible" if abs(cohens_d) < 0.2 else \
                   "small" if abs(cohens_d) < 0.5 else \
                   "medium" if abs(cohens_d) < 0.8 else "large"
    print(f"\n  Cohen's d = {cohens_d:.4f} ({effect_label} effect)")

    # ═══════════════════════════════════════════════════════
    #  17. ROLLING PERFORMANCE WINDOWS
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*80}")
    print(f"  17. ROLLING WINDOW ANALYSIS")
    print(f"{'─'*80}")

    for window in [100, 200, 500]:
        if len(pnl_3x) < window:
            continue
        rolling_wr = []
        rolling_pf = []
        rolling_sharpe = []
        for i in range(len(pnl_3x) - window + 1):
            chunk = pnl_3x[i:i+window]
            rolling_wr.append(np.mean(chunk > 0) * 100)
            rolling_pf.append(pf_from_pnl(chunk))
            mu = np.mean(chunk); s = np.std(chunk, ddof=1)
            rolling_sharpe.append(mu/s*np.sqrt(annual_trades) if s > 1e-18 else 0)

        rwr = np.array(rolling_wr)
        rpf = np.array(rolling_pf)
        rsh = np.array(rolling_sharpe)

        print(f"\n  Rolling {window}-trade window:")
        print(f"    WR:     min={np.min(rwr):.1f}%  max={np.max(rwr):.1f}%  "
              f"mean={np.mean(rwr):.1f}%  std={np.std(rwr):.1f}%")
        print(f"    PF:     min={np.min(rpf):.3f}  max={np.max(rpf):.3f}  "
              f"mean={np.mean(rpf):.3f}  std={np.std(rpf):.3f}")
        print(f"    Sharpe: min={np.min(rsh):.2f}  max={np.max(rsh):.2f}  "
              f"mean={np.mean(rsh):.2f}  std={np.std(rsh):.2f}")
        print(f"    P(WR<40%): {np.mean(rwr < 40)*100:.1f}%  "
              f"P(PF<1.0): {np.mean(rpf < 1.0)*100:.1f}%")

    # ═══════════════════════════════════════════════════════
    #  18. FINAL SUMMARY TABLE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print(f"  R40 CHAMPION COMPREHENSIVE SUMMARY: {NAME} @{LEV_CHAMPION}x, {SLOTS_CHAMPION} slots")
    print(f"{'='*120}")

    final_eq = eq_3x[-1]
    total_ret = (final_eq - 1) * 100

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  RETURNS                                                              │
  │    6yr Compound Return:  {total_ret:>+12,.0f}%                                    │
  │    Annualized Return:    {m['ann_ret_pct']:>+12.1f}%                                    │
  │    Monthly Win Rate:     {pos_months}/{n_months} ({pos_months/n_months*100:.0f}%)                                         │
  │    Best/Worst Month:     {np.max(month_rets_arr):>+.0f}% / {np.min(month_rets_arr):>+.0f}%                                      │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  RISK                                                                 │
  │    Max Drawdown:         {m['max_dd_pct']:>12.1f}%                                    │
  │    VaR 95%:              {value_at_risk(pnl_pct, 0.95):>+12.3f}%                                  │
  │    CVaR 95%:             {cvar_expected_shortfall(pnl_pct, 0.95):>+12.3f}%                                  │
  │    Time in Drawdown:     {np.mean(dd_series > 0)*100:>12.1f}%                                    │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  QUALITY                                                              │
  │    Profit Factor:        {m['PF']:>12.3f}                                          │
  │    Sharpe Ratio:         {s3:>12.3f}                                          │
  │    Sortino Ratio:        {so3:>12.3f}                                          │
  │    Calmar Ratio:         {m['calmar']:>12.1f}                                          │
  │    t-statistic:          {m['t']:>12.2f} (p={t_pval:.6f})                        │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  TRADES                                                               │
  │    Total Trades:         {m['N']:>12,}                                          │
  │    Win Rate:             {m['WR']:>12.1f}%                                    │
  │    Risk:Reward:          {m['RR']:>12.2f}                                          │
  │    Avg Win:              {m['avg_win_pct']:>+12.3f}%                                  │
  │    Avg Loss:             {-m['avg_loss_pct']:>+12.3f}%                                  │
  │    EV per trade:         {m['EV_bp']:>+12.1f}bp                                     │
  │    Trades/Day:           {m['tpd']:>12.2f}                                          │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  RELIABILITY                                                          │
  │    Losing Years:         {sum(1 for y in yr_data_3x if y['ret_pct'] < 0):>12}  / {n_years}                                      │
  │    Bootstrap P(PF≥1):    {bp['P_PF_ge_1']:>12.1f}%                                    │
  │    PF 95% CI:            [{bp['CI_lo']:.3f}, {bp['CI_hi']:.3f}]                                   │
  │    Kelly Fraction:       {kelly_ov:>12.3f}                                          │
  │    Kelly Utilization:    {LEV_CHAMPION/opt_lev_ov*100:>12.0f}% of optimal                             │
  └─────────────────────────────────────────────────────────────────────────┘
""")

    total_time = time.time() - t0
    print(f"  Analysis completed in {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
