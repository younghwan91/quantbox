"""QuantBox v13.0 Strategy Report — 25x / 10 slots (L/N=2.5)

Single comprehensive PDF for the currently deployed v13.0 strategy.
All metrics are either v13.0-specific or leverage-independent.

Usage:
    uv run python scripts/generate_combined_report.py
"""
from __future__ import annotations

import base64
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np

# ── Font ──────────────────────────────────────────────────────────────────────
_KR_TTF = Path.home() / ".local" / "share" / "fonts" / "NotoSansKR.ttf"
_TTC    = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_font_path = _KR_TTF if _KR_TTF.exists() else _TTC
if _font_path.exists():
    fm.fontManager.addfont(str(_font_path))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False

OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "docs" / "backtest_results"
    / "quantbox_v13_strategy_report.pdf"
)

# ════════════════════════════════════════════════════════════════════════════════
# DATA — v13.0 (25x leverage, 10 slots, L/N=2.5, Model B compound)
# ════════════════════════════════════════════════════════════════════════════════

# ── v13.0 aggregate ──
V13_ANN_RET   = 687     # % annual return (Model B compound)
V13_MAX_DD    = 67.9    # %
V13_CALMAR    = 10.12
V13_TRADES    = 2265
V13_TPD       = 1.03    # trades per day
V13_LEV       = 25
V13_SLOTS     = 10
V13_LN        = 2.5
V13_BALANCE   = 1983    # $

# ── Per-trade edge (leverage-independent) ──
PF       = 1.285
WR       = 43.4   # %
RR       = 1.67   # avg_win / avg_loss
EV_1X    = 33     # bp per trade at 1x
T_STAT   = 5.42
SHARPE   = 2.21
SORTINO  = 5.99
COHENS_D = 0.114

# Per-trade equity impact at L/N=2.5
AVG_LOSS_EQ = 5.2   # % portfolio per losing trade
AVG_WIN_EQ  = 8.7   # % portfolio per winning trade
EV_EQ       = 0.82  # % portfolio per trade (33bp × 2.5)

# ── Yearly stats (PF/WR/Sharpe = exact, leverage-independent) ──
YEARLY = {
    2020: {"trades": 630, "pf": 1.29, "sharpe": 2.85, "wr": 39.4, "note": "코로나 변동성"},
    2021: {"trades": 202, "pf": 1.41, "sharpe": 2.30, "wr": 46.5, "note": "Bull run"},
    2022: {"trades": 414, "pf": 1.36, "sharpe": 2.82, "wr": 45.7, "note": "LUNA/FTX"},
    2023: {"trades": 374, "pf": 1.13, "sharpe": 1.04, "wr": 41.2, "note": "저변동"},
    2024: {"trades": 330, "pf": 1.41, "sharpe": 2.89, "wr": 48.5, "note": "Halving"},
    2025: {"trades": 330, "pf": 1.16, "sharpe": 1.24, "wr": 43.9, "note": "혼합"},
}

# ── L/N grid (25×25 = 625 combinations) ──
LN_GRID = {
    "ln":      [0.20, 0.50, 1.00, 1.50, 2.00, 2.50, 2.78, 3.50, 4.00, 4.50],
    "ann_ret": [27.4, 80.0, 162,  270,  425,  687,  740,  998,  1200, 1350],
    "max_dd":  [8.0,  16.0, 29.6, 41.2, 53.8, 67.9, 73.1, 79.2, 84.7, 89.1],
    "calmar":  [3.42, 5.0,  5.48, 6.55, 7.90, 10.12,10.13,12.60,14.16,15.15],
}

# ── Risk tiers ──
TIERS = {
    "names":   ["보수적\n9x/10s", "중도\n16x/10s", "공격적\n25x/10s\n(v13.0)"],
    "ann_ret": [162, 367, 687],
    "max_dd":  [29.6, 48.0, 67.9],
    "calmar":  [5.48, 7.65, 10.12],
}

# ── Streak (leverage-independent) ──
WIN_STREAK  = {1:284,2:128,3:63,4:28,5:10,6:4,7:7,8:2,10:1}
LOSS_STREAK = {1:226,2:115,3:82,4:37,5:32,6:13,7:5,8:8,9:3,10:5,12:1,14:1}

# ── Duration (leverage-independent) ──
DURATION = {
    "buckets": ["0-1h","1-4h","4-12h","12-24h","24-48h","48-96h","96h+"],
    "pct":     [3.3,14.9,27.1,17.9,20.7,12.0,4.1],
    "wr":      [10.7,31.6,38.2,52.3,48.2,57.1,44.1],
}

# ── Bootstrap (leverage-independent) ──
BS_PF_MED = 1.284;  BS_PF_CI = (1.173, 1.405)
BS_WR_CI  = (41.4, 45.4)
BS_EV_MED = 97.8;   BS_EV_CI = (62.0, 133.1)
BS_SH_CI  = (1.41, 3.00)
BS_P_GE1  = 100.0   # P(PF >= 1.0)

# ── Top/Bottom symbols (PF-based, leverage-independent) ──
TOP_SYM = [
    ("ETHUSDT",      127, 46.5, 1.51, 148),
    ("DOGEUSDT",     120, 44.2, 1.47, 152),
    ("1000SHIBUSDT", 100, 46.0, 1.50, 172),
    ("FETUSDT",       17, 76.5, 5.49, 672),
    ("OPUSDT",        29, 58.6, 2.63, 395),
    ("UNIUSDT",       35, 51.4, 2.31, 323),
    ("VETUSDT",       34, 55.9, 1.98, 284),
    ("LTCUSDT",       45, 53.3, 1.73, 225),
    ("WIFUSDT",       14, 71.4, 3.61, 591),
    ("1000PEPEUSDT",  37, 54.1, 1.74, 246),
]
BOT_SYM = [
    ("HBARUSDT",  40, 37.5, 0.89, -49),
    ("BCHUSDT",   67, 35.8, 0.91, -38),
    ("XLMUSDT",   65, 38.5, 0.81, -81),
    ("ATOMUSDT",  40, 25.0, 0.64, -160),
    ("COMPUSDT",  42, 23.8, 0.46, -313),
]

# ── Strategy type (leverage-independent, raw pre-filter) ──
STRAT_D_WR = 25.9;  STRAT_D_RR = 3.75;  STRAT_D_PF = 1.31
STRAT_S_WR = 53.9;  STRAT_S_RR = 1.08;  STRAT_S_PF = 1.27

# Colors
C = {"green":"#27ae60","red":"#e74c3c","blue":"#2980b9","gold":"#f39c12",
     "purple":"#8e44ad","gray":"#7f8c8d","dark":"#2c3e50","accent":"#1abc9c"}


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _png_b64(fig, dpi=130):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    return base64.b64encode(data).decode()

def _ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

def _sim_monthly(start, rate, months, trigger=150, keep=70):
    comp, wb, wt = start, start, 0.0
    cs, ws, wts = [], [], []
    for _ in range(months):
        comp *= (1 + rate); wb *= (1 + rate)
        if wb >= trigger:
            wt += wb - keep; wb = keep
        cs.append(comp); ws.append(wb); wts.append(wb + wt)
    return cs, ws, wts


# ════════════════════════════════════════════════════════════════════════════════
# CHARTS (7)
# ════════════════════════════════════════════════════════════════════════════════

def chart_ln():
    """L/N ratio grid — core of v13.0 optimization."""
    d = LN_GRID
    fig, ax1 = plt.subplots(figsize=(9, 4.5)); ax2 = ax1.twinx()
    ax1.plot(d["ln"], d["ann_ret"], "o-", color=C["blue"], linewidth=2.5,
             markersize=7, label="Annual Return %")
    ax2.plot(d["ln"], d["max_dd"], "s--", color=C["red"], linewidth=2,
             markersize=6, label="Max DD %")
    ax2.plot(d["ln"], d["calmar"], "^:", color=C["green"], linewidth=2,
             markersize=6, label="Calmar")
    ax1.axvline(2.5, color=C["gold"], linewidth=2.2, linestyle="--")
    ax1.text(2.52, 900, "v13.0 선택점\nL/N=2.5\nLosing Year=0",
             fontsize=7.8, color=C["gold"], fontweight="bold")
    ax1.axvline(4.5, color=C["red"], linewidth=1.5, linestyle=":", alpha=0.7)
    ax1.text(4.52, 200, "Losing Year\n등장", fontsize=7.5, color=C["red"])
    ax1.set_xlabel("L/N Ratio (Leverage / Slots)")
    ax1.set_ylabel("Annual Return %", color=C["blue"])
    ax2.set_ylabel("Max DD % / Calmar"); _ax(ax1)
    ax1.set_title("25x25 Fine Grid (625개 조합) — L/N ratio가 수익을 결정한다",
                  fontsize=10, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout(); return _png_b64(fig)


def chart_tiers():
    """Risk tier comparison — conservative / moderate / aggressive."""
    d = TIERS; x = np.arange(len(d["names"])); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w, [r / 10 for r in d["ann_ret"]], w,
                label="Return /10", color=C["blue"], alpha=0.82)
    b2 = ax.bar(x, d["max_dd"], w,
                label="Max DD %", color=C["red"], alpha=0.82)
    b3 = ax.bar(x + w, [c * 10 for c in d["calmar"]], w,
                label="Calmar x10", color=C["green"], alpha=0.82)
    for bars, vals, fmt in [
        (b1, d["ann_ret"], "+{:.0f}%"),
        (b2, d["max_dd"],  "{:.0f}%"),
        (b3, d["calmar"],  "{:.2f}"),
    ]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    fmt.format(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n", " ") for n in d["names"]], fontsize=8.5)
    ax.set_title("리스크 티어별 성과 비교 (S=10 slots 고정)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); _ax(ax); fig.tight_layout(); return _png_b64(fig)


def chart_yearly():
    """Yearly PF & WR — proves edge stability over 6 years."""
    years = sorted(YEARLY.keys())
    pfs = [YEARLY[y]["pf"] for y in years]
    wrs = [YEARLY[y]["wr"] for y in years]
    bar_colors = [C["green"] if pf >= 1.3 else (C["gold"] if pf >= 1.15 else C["red"])
                  for pf in pfs]
    fig, ax1 = plt.subplots(figsize=(9, 4.5)); ax2 = ax1.twinx()
    bars = ax1.bar([str(y) for y in years], pfs, color=bar_colors,
                   edgecolor="white", width=0.5, alpha=0.85, label="Profit Factor")
    ax2.plot([str(y) for y in years], wrs, "o-", color=C["blue"],
             linewidth=2.2, markersize=8, label="Win Rate %")
    ax1.axhline(1.0, color=C["red"], linewidth=1.2, linestyle="--", alpha=0.6)
    ax1.axhline(PF, color=C["gray"], linewidth=1, linestyle=":", alpha=0.5)
    ax1.text(5.3, PF + 0.01, f"6yr avg {PF}", fontsize=7, color=C["gray"])
    for bar, pf in zip(bars, pfs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{pf:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Profit Factor"); ax2.set_ylabel("Win Rate (%)", color=C["blue"])
    ax1.set_ylim(0.8, 1.6); ax2.set_ylim(30, 55)
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_title("6년 연도별 PF & WR — 전년 PF > 1.0 유지",
                  fontsize=11, fontweight="bold")
    _ax(ax1); fig.tight_layout(); return _png_b64(fig)


def chart_rr():
    """R:R structure — fewer wins but bigger wins."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Win Rate donut
    wedges, texts, autotexts = ax1.pie(
        [WR, 100 - WR], labels=["승리", "패배"],
        colors=[C["green"], C["red"]], autopct="%1.1f%%",
        startangle=90, pctdistance=0.55,
        wedgeprops=dict(width=0.35, edgecolor="white"),
        textprops={"fontsize": 11})
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold")
    ax1.set_title("승률 = 43.4%", fontsize=12, fontweight="bold")

    # Right: Per-trade equity impact at L/N=2.5
    cats = ["평균 손실", "평균 수익", "기대값 (EV)"]
    vals = [-AVG_LOSS_EQ, AVG_WIN_EQ, EV_EQ]
    colors_bar = [C["red"], C["green"], C["gold"]]
    bars = ax2.bar(cats, vals, color=colors_bar, edgecolor="white", width=0.45)
    for bar, v in zip(bars, vals):
        y = v + 0.4 if v > 0 else v - 0.6
        ax2.text(bar.get_x() + bar.get_width() / 2, y,
                 f"{v:+.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("포트폴리오 영향 (%)")
    ax2.set_title("Per-Trade Equity Impact (L/N=2.5)",
                  fontsize=10, fontweight="bold")

    fig.suptitle("R:R = 1.67 — 적게 이기되 크게 이긴다",
                 fontsize=13, fontweight="bold")
    _ax(ax2); fig.tight_layout(); return _png_b64(fig)


def chart_streak():
    """Win/loss streak distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ws_k = [str(k) for k in WIN_STREAK]; ws_v = list(WIN_STREAK.values())
    ax1.bar(ws_k, ws_v, color=C["green"], edgecolor="white", width=0.6)
    ax1.set_title("연승 분포 (최대 10연승)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("연승 길이"); ax1.set_ylabel("횟수"); _ax(ax1)
    for x, v in zip(ws_k, ws_v):
        ax1.text(x, v + 3, str(v), ha="center", fontsize=7.5)

    ls_k = [str(k) for k in LOSS_STREAK]; ls_v = list(LOSS_STREAK.values())
    bars = ax2.bar(ls_k, ls_v, color=C["red"], edgecolor="white", width=0.6)
    bars[-1].set_color(C["purple"])
    ax2.set_title("연패 분포 (최대 14연패 → compound -56%)",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("연패 길이"); ax2.set_ylabel("횟수"); _ax(ax2)
    for x, v in zip(ls_k, ls_v):
        ax2.text(x, v + 2, str(v), ha="center", fontsize=7.5)

    fig.suptitle("승/패 연속 분포 (레버리지 무관)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(); return _png_b64(fig)


def chart_duration():
    """Duration vs Win Rate — longer holds win more."""
    d = DURATION; x = np.arange(len(d["buckets"]))
    fig, ax1 = plt.subplots(figsize=(9, 4)); ax2 = ax1.twinx()
    ax1.bar(x, d["pct"], color=C["blue"], alpha=0.65, label="비중 %", width=0.5)
    ax2.plot(x, d["wr"], "o-", color=C["red"], linewidth=2.2,
             markersize=7, label="Win Rate %")
    ax2.axhline(WR, color=C["gray"], linewidth=1, linestyle="--", alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(d["buckets"], fontsize=9)
    ax1.set_ylabel("비중 (%)", color=C["blue"])
    ax2.set_ylabel("Win Rate (%)", color=C["red"]); ax2.set_ylim(0, 75)
    ax1.set_title("Duration별 Win Rate — 장기 보유일수록 WR 상승",
                  fontsize=10, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper right", fontsize=8)
    _ax(ax1); fig.tight_layout(); return _png_b64(fig)


def chart_compound():
    """Withdraw vs full-compound simulation."""
    months = list(range(1, 15))
    comp, wb, wt = _sim_monthly(50, 0.145, 14, 150, 70)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(months, comp, "o-", color=C["blue"], linewidth=2.5,
            markersize=6, label="올복리 잔고")
    ax.plot(months, wb, "s-", color=C["green"], linewidth=2.5,
            markersize=6, label="인출전략 잔고")
    ax.plot(months, wt, "^--", color=C["accent"], linewidth=1.8,
            markersize=5, alpha=0.9, label="인출전략 총자산(잔고+현금)")
    ax.axhline(150, color=C["red"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(0.5, 153, "인출 트리거 ₩1.5억", fontsize=8,
            color=C["red"], va="bottom")
    ax.axhline(70, color=C["gray"], linewidth=1, linestyle=":", alpha=0.6)
    ax.text(0.5, 72, "유지 잔고 ₩7,000만", fontsize=7.5,
            color=C["gray"], va="bottom")
    ax.set_xlabel("월수"); ax.set_ylabel("금액 (백만원)")
    ax.set_title("복리 vs 인출전략 — 5천만원 시작 / 월 +14.5% 추정 (백테스트 60% 감쇄)",
                 fontsize=9.5, fontweight="bold")
    ax.set_xticks(months); ax.legend(fontsize=8); _ax(ax)
    fig.tight_layout(); return _png_b64(fig)


# ════════════════════════════════════════════════════════════════════════════════
# HTML
# ════════════════════════════════════════════════════════════════════════════════

def build_html(charts: dict[str, str], font_b64: str) -> str:
    def img(key):
        return (f'<div class="chart"><img src="data:image/png;base64,'
                f'{charts[key]}" alt="{key}"></div>')

    font_face = ""
    if font_b64:
        font_face = (f"@font-face{{font-family:'NotoSansKR';"
                     f"src:url('data:font/truetype;base64,{font_b64}') "
                     f"format('truetype');}}")

    css = f"""
{font_face}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'NotoSansKR','Noto Sans CJK KR',sans-serif;font-size:9.5pt;
      line-height:1.65;color:#222;word-wrap:break-word;overflow-wrap:break-word}}
@page{{size:A4;margin:14mm 13mm 16mm 13mm}}
.cover{{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        color:#fff;padding:32pt 28pt;margin-bottom:20pt;border-radius:6pt}}
.cover h1{{font-size:22pt;font-weight:900;letter-spacing:-0.5pt;margin-bottom:6pt;
           color:#e8f4f8}}
.cover .subtitle{{font-size:11pt;color:#a8d8ea;margin-bottom:18pt}}
.cover .meta{{font-size:8pt;color:#78a0b0;border-top:1pt solid rgba(255,255,255,0.15);
              padding-top:8pt;display:flex;gap:24pt;flex-wrap:wrap}}
.cover .meta span{{display:flex;align-items:center;gap:4pt}}
.cover .badge{{background:rgba(255,255,255,0.12);padding:2pt 7pt;border-radius:12pt;
               font-size:7.5pt;font-weight:bold;color:#e0f0ff}}
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8pt;margin:14pt 0}}
.kpi{{background:#f0f7ff;border:1pt solid #c5dcf0;border-radius:5pt;
      padding:10pt 12pt;text-align:center}}
.kpi .val{{font-size:17pt;font-weight:900;color:#1a4a7a;line-height:1.1}}
.kpi .lbl{{font-size:7.5pt;color:#557;margin-top:3pt;font-weight:600}}
.kpi .sub{{font-size:7pt;color:#889;margin-top:2pt}}
.kpi.highlight{{background:#e8f9f0;border-color:#52c788}}
.kpi.highlight .val{{color:#1a6a3a}}
.kpi.warn{{background:#fff3e0;border-color:#f0a050}}
.kpi.warn .val{{color:#8a4a00}}
h2{{font-size:13pt;color:#16355a;margin:14pt 0 6pt;border-left:4pt solid #2980b9;
    padding-left:9pt;font-weight:700}}
.tw{{overflow-x:auto;margin:8pt 0}}
table{{border-collapse:collapse;width:100%;font-size:8pt;table-layout:fixed}}
th,td{{border:0.5pt solid #bbb;padding:3pt 5pt;text-align:left;
       word-wrap:break-word;overflow-wrap:break-word}}
th{{background:#e0e8f5;font-weight:bold;font-size:7.8pt}}
tr:nth-child(even) td{{background:#f8f9fc}}
td.r{{text-align:right}} td.c{{text-align:center}}
td.bold{{font-weight:bold;color:#1a4060}}
td.green{{color:#1a6a3a;font-weight:bold}}
td.red{{color:#8a1a1a;font-weight:bold}}
td.gold{{color:#8a5a00;font-weight:bold}}
.chart{{margin:10pt 0;page-break-inside:avoid;text-align:center}}
.chart img{{width:100%;height:auto;border:0.5pt solid #dde;border-radius:3pt}}
.callout{{background:#fff9e6;border-left:4pt solid #f0a800;padding:8pt 12pt;
          margin:8pt 0;border-radius:0 4pt 4pt 0;font-size:8.5pt}}
.callout strong{{color:#6a4a00}}
.callout.green{{background:#eaf7ef;border-color:#27ae60}}
.callout.green strong{{color:#1a5a30}}
.callout.red{{background:#fdecea;border-color:#e74c3c}}
.callout.red strong{{color:#6a1a1a}}
.callout.blue{{background:#e8f4fc;border-color:#2980b9}}
.callout.blue strong{{color:#1a3a5a}}
.verdict{{background:linear-gradient(135deg,#0f3460,#1a4a7a);color:#fff;
          padding:18pt 20pt;border-radius:6pt;margin:16pt 0;page-break-inside:avoid}}
.verdict h3{{font-size:12pt;color:#7dd;margin-bottom:8pt;font-weight:800}}
.verdict p{{font-size:9pt;color:#cde;line-height:1.7}}
.verdict ul{{margin-left:14pt;color:#cde;font-size:9pt}}
.verdict li{{margin:3pt 0}}
.footer{{margin-top:20pt;padding-top:10pt;border-top:1pt solid #ddd;
         font-size:7.5pt;color:#999;text-align:center}}
.section-break{{page-break-before:always}}
@media print{{.chart{{page-break-inside:avoid}}.verdict{{page-break-inside:avoid}}}}
"""

    # ── Yearly stats rows ──
    yr_rows = ""
    for y in sorted(YEARLY.keys()):
        d = YEARLY[y]
        pf_cls = "gold" if d["pf"] < 1.2 else ""
        yr_rows += (
            f'<tr><td{" class=" + repr(pf_cls) if pf_cls else ""}>{y}</td>'
            f'<td class="r">{d["trades"]}</td>'
            f'<td class="r{" bold" if d["pf"]>=1.3 else ""}">{d["pf"]:.2f}</td>'
            f'<td class="r">{d["sharpe"]:.2f}</td>'
            f'<td class="r">{d["wr"]}%</td>'
            f'<td>{d["note"]}</td></tr>\n'
        )

    # ── Top/Bottom symbol rows ──
    top_rows = ""
    for sym, tr, wr, pf, ev in TOP_SYM:
        top_rows += (f'<tr><td>{sym}</td><td class="r">{tr}</td>'
                     f'<td class="r">{wr}%</td><td class="r bold">{pf:.2f}</td>'
                     f'<td class="r green">{ev:+d}</td></tr>\n')
    bot_rows = ""
    for sym, tr, wr, pf, ev in BOT_SYM:
        bot_rows += (f'<tr><td>{sym}</td><td class="r">{tr}</td>'
                     f'<td class="r">{wr}%</td><td class="r">{pf:.2f}</td>'
                     f'<td class="r red">{ev:+d}</td></tr>\n')

    body = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8">
<title>QuantBox v13.0 전략 보고서</title>
<style>{css}</style>
</head><body>

<!-- COVER -->
<div class="cover">
  <div class="badge">CONFIDENTIAL — INTERNAL USE ONLY</div>
  <h1 style="margin-top:10pt">QuantBox v13.0 전략 보고서</h1>
  <div class="subtitle">VR8_56_SL1_TP5_D4 @25x leverage / 10 slots (L/N=2.5)</div>
  <div class="meta">
    <span>📅 2026-03-03</span>
    <span>🔖 Commit: e4b5831 (tag v13.0)</span>
    <span>📊 50 symbols × 6.0년 (2020~2025)</span>
    <span>⚡ Binance USDT-M Futures · 15m봉</span>
    <span>💰 Balance: ${V13_BALANCE:,}</span>
  </div>
</div>

<!-- KPI -->
<div class="kpi-grid">
  <div class="kpi highlight">
    <div class="val">+{V13_ANN_RET}%</div>
    <div class="lbl">연간 수익률</div>
    <div class="sub">Model B compound (L/N=2.5)</div>
  </div>
  <div class="kpi highlight">
    <div class="val">{V13_CALMAR}</div>
    <div class="lbl">Calmar Ratio</div>
    <div class="sub">AnnRet / MaxDD</div>
  </div>
  <div class="kpi highlight">
    <div class="val">0회</div>
    <div class="lbl">Losing Year (6년)</div>
    <div class="sub">2020~2025 전년 흑자</div>
  </div>
  <div class="kpi warn">
    <div class="val">{V13_MAX_DD}%</div>
    <div class="lbl">Max Drawdown</div>
    <div class="sub">${V13_BALANCE:,} → ${int(V13_BALANCE * (1 - V13_MAX_DD/100)):,}</div>
  </div>
  <div class="kpi">
    <div class="val">{PF}</div>
    <div class="lbl">Profit Factor</div>
    <div class="sub">CI [{BS_PF_CI[0]}, {BS_PF_CI[1]}]</div>
  </div>
  <div class="kpi">
    <div class="val">{BS_P_GE1:.0f}%</div>
    <div class="lbl">P(PF ≥ 1.0)</div>
    <div class="sub">5,000회 Bootstrap</div>
  </div>
  <div class="kpi">
    <div class="val">+{EV_EQ:.2f}%</div>
    <div class="lbl">EV per trade (자산)</div>
    <div class="sub">1x 기준 +{EV_1X}bp × L/N={V13_LN}</div>
  </div>
  <div class="kpi">
    <div class="val">{V13_TRADES:,}</div>
    <div class="lbl">Total Trades</div>
    <div class="sub">{V13_TPD}/day × 6년</div>
  </div>
</div>

<!-- 1. SETTINGS -->
<h2>1. 현재 운용 설정</h2>
<div class="tw"><table>
<tr><th>Setting</th><th class="c">값</th><th>설명</th></tr>
<tr><td>VS_BASE_LEVERAGE</td><td class="c bold">25.0x</td><td>25×25 grid 최적화 결과</td></tr>
<tr><td>VS_MAX_LEVERAGE</td><td class="c">25.0x</td><td>Hard cap (base와 일치)</td></tr>
<tr><td>VS_MAX_CONCURRENT</td><td class="c bold">10</td><td>L/N=2.5 유지</td></tr>
<tr><td>L/N Ratio</td><td class="c green bold">2.50</td><td>Calmar 최대화 경계점</td></tr>
<tr><td>Per-slot Notional</td><td class="c">${int(V13_BALANCE / V13_SLOTS * V13_LEV):,}</td><td>잔고/슬롯 × 레버리지</td></tr>
<tr><td>VS_STRADDLE_ENABLED</td><td class="c">true</td><td>S_d8 양방향 스트래들</td></tr>
<tr><td>VS_HEDGE_MODE</td><td class="c">true</td><td>Binance dualSidePosition</td></tr>
<tr><td>VR Threshold (D)</td><td class="c">0.50</td><td>VR = ATR(8)/ATR(56) &lt; 0.50</td></tr>
<tr><td>VR Threshold (S)</td><td class="c">0.55</td><td>S squeeze VR &lt; 0.55 × 4바</td></tr>
<tr><td>SL / TP</td><td class="c">1% / 5%</td><td>R:R = 1:5 고정</td></tr>
<tr><td>Max Hold</td><td class="c">48h</td><td>15m=192바, 1h=48바, 4h=12바</td></tr>
<tr><td>MTF</td><td class="c">15m / 1h / 4h</td><td>높은 TF 우선 (4h &gt; 1h &gt; 15m)</td></tr>
</table></div>
<div class="callout blue"><strong>Model B Compound:</strong> <code>eq *= (1 + pnl × L/N)</code> — 각 트레이드는 포트폴리오의 1/N 배분. 전체 자본 compound(Model A)와 달리 실제 운용과 동일한 수학 모델.</div>

<!-- 2. L/N GRID -->
<div class="section-break"></div>
<h2>2. 왜 L/N = 2.5인가 — 25×25 Grid 최적화</h2>
{img("ln")}
<div class="tw"><table>
<tr><th>L/N Ratio</th><th class="r">AnnRet</th><th class="r">Max DD</th><th class="r">Calmar</th><th class="c">Losing Years</th></tr>
<tr><td>0.20 (구 v12.9)</td><td class="r">+27.4%</td><td class="r">8.0%</td><td class="r">3.42</td><td class="c">0</td></tr>
<tr><td>1.00</td><td class="r">+162%</td><td class="r">29.6%</td><td class="r">5.48</td><td class="c">0</td></tr>
<tr><td>2.00</td><td class="r">+425%</td><td class="r">53.8%</td><td class="r">7.90</td><td class="c">0</td></tr>
<tr><td class="bold">2.50 (v13.0) ★</td><td class="r green bold">+687%</td><td class="r">67.9%</td><td class="r bold">10.12</td><td class="c green bold">0</td></tr>
<tr><td>2.78</td><td class="r">+740%</td><td class="r">73.1%</td><td class="r">10.13</td><td class="c green">0</td></tr>
<tr><td>4.50</td><td class="r">+1,350%</td><td class="r red">89.1%</td><td class="r">15.15</td><td class="c red">1</td></tr>
<tr><td>5.00+</td><td class="r red">급감</td><td class="r red">95%+</td><td class="r red">—</td><td class="c red">2+</td></tr>
</table></div>
<div class="callout green"><strong>L/N = 2.5 선택 근거:</strong> 625개 조합 중 Losing Year = 0을 유지하면서 Calmar가 최대화되는 경계점. 같은 L/N이면 (S=5, L=12.5)과 (S=10, L=25)이 거의 동일 — 슬롯과 레버리지가 아니라 <strong>비율</strong>이 중요하다는 핵심 발견.</div>

<!-- 3. TIERS & ADAPTIVE -->
<h2>3. 리스크 티어별 선택지</h2>
{img("tiers")}
<div class="tw"><table>
<tr><th>Tier</th><th>구성</th><th class="r">AnnRet</th><th class="r">Max DD</th><th class="r">Calmar</th><th class="c">Losing Yr</th><th class="r">슬롯 노셔널</th></tr>
<tr><td>보수적</td><td>S=10, L=9x</td><td class="r">+162%</td><td class="r">29.6%</td><td class="r">5.48</td><td class="c">0</td><td class="r">$1,785</td></tr>
<tr><td>중도</td><td>S=10, L=16x</td><td class="r">+367%</td><td class="r">48.0%</td><td class="r">7.65</td><td class="c">0</td><td class="r">$3,173</td></tr>
<tr><td class="bold">공격적 (v13.0) ★</td><td class="bold">S=10, L=25x</td><td class="r green bold">+687%</td><td class="r warn">67.9%</td><td class="r bold">10.12</td><td class="c green bold">0</td><td class="r">$4,957</td></tr>
<tr><td>YOLO</td><td>S=9, L=25x</td><td class="r">+740%</td><td class="r red">73.1%</td><td class="r">10.13</td><td class="c green">0</td><td class="r">$5,508</td></tr>
</table></div>

<h2 style="margin-top:16pt">Adaptive vs Fixed 레버리지 — 4개 모델 토너먼트</h2>
<div class="tw"><table>
<tr><th>Model</th><th>최적 구성</th><th class="r">Calmar</th><th class="r">AnnRet</th><th class="r">Max DD</th><th>결론</th></tr>
<tr><td class="bold">Fixed L/N ★</td><td>S=9, L=25x</td><td class="r green bold">10.13</td><td class="r">+740%</td><td class="r">73.1%</td><td class="green bold">승자</td></tr>
<tr><td>Adaptive-DD</td><td>S=10, L=25x</td><td class="r">5.06</td><td class="r">+344%</td><td class="r">67.9%</td><td>수익 절반</td></tr>
<tr><td>Target L/N=2.0</td><td>S=10, L=25x</td><td class="r">7.65</td><td class="r">+367%</td><td class="r">48.0%</td><td>고정 대비 열위</td></tr>
<tr><td>Adaptive-Conc.</td><td>—</td><td class="r red bold">BLOWUP</td><td class="r red">—</td><td class="r red">100%</td><td class="red">실패</td></tr>
</table></div>
<div class="callout"><strong>결론:</strong> 적응형 레버리지는 고정보다 나쁘다. DD 구간에서 레버리지를 줄이면 복구 기회를 놓친다. 복잡성 추가가 성과를 악화시킨다 (KISS 원칙).</div>

<!-- 4. EDGE VERIFICATION -->
<div class="section-break"></div>
<h2>4. 엣지 구조 — R:R = {RR}</h2>
{img("rr")}

<div class="tw"><table>
<tr><th>Metric</th><th class="r">값</th><th>해석</th></tr>
<tr><td>Win Rate</td><td class="r">{WR}%</td><td>10번 중 4.3번 승리</td></tr>
<tr><td>Risk:Reward</td><td class="r bold">{RR}</td><td>승리 시 평균 수익이 손실의 1.67배</td></tr>
<tr><td>Profit Factor</td><td class="r bold">{PF}</td><td>총 수익 / 총 손실</td></tr>
<tr><td>EV per trade (raw)</td><td class="r">+{EV_1X}bp</td><td>1x leverage 기준</td></tr>
<tr><td>EV per trade (equity)</td><td class="r green bold">+{EV_EQ:.1f}%</td><td>L/N=2.5 기준 포트폴리오 영향</td></tr>
<tr><td>Sharpe / Sortino</td><td class="r">{SHARPE} / {SORTINO}</td><td>연간 위험조정 수익</td></tr>
<tr><td>Cohen's d</td><td class="r">{COHENS_D}</td><td>개별 효과 작음 → L/N으로 증폭</td></tr>
</table></div>

<h2 style="margin-top:16pt">전략 유형 분해 (레버리지 무관)</h2>
<div class="tw"><table>
<tr><th>Type</th><th class="r">WR</th><th class="r">R:R</th><th class="r">PF</th><th>역할</th></tr>
<tr><td>Directional (D)</td><td class="r">25.9%</td><td class="r bold">3.75</td><td class="r">1.31</td><td>추세추종 — 낮은 WR, 높은 R:R</td></tr>
<tr><td>Straddle (S)</td><td class="r bold">53.9%</td><td class="r">1.08</td><td class="r">1.27</td><td>변동성 포착 — 높은 WR, 수익 ~80% 기여</td></tr>
<tr><td class="bold">Combined</td><td class="r">{WR}%</td><td class="r">{RR}</td><td class="r bold">{PF}</td><td>D+S 시너지</td></tr>
</table></div>

<h2 style="margin-top:16pt">통계적 유의성 & Bootstrap</h2>
<div class="tw"><table>
<tr><th>검정</th><th class="r">통계량</th><th class="r">p-value</th><th>결론</th></tr>
<tr><td>t-test (mean &gt; 0)</td><td class="r">t = {T_STAT}</td><td class="r green bold">&lt; 0.000001</td><td>엣지 존재 확실</td></tr>
<tr><td>Wilcoxon signed-rank</td><td class="r">W = 928,145</td><td class="r green bold">&lt; 0.000001</td><td>비모수적으로도 유의</td></tr>
<tr><td>200-trade Rolling PF&lt;1</td><td class="r">—</td><td class="r green bold">0%</td><td>장기 안정적 엣지</td></tr>
</table></div>
<div class="tw"><table>
<tr><th>Bootstrap Metric</th><th class="r">Median</th><th>95% CI</th><th>비고</th></tr>
<tr><td>Profit Factor</td><td class="r bold">{BS_PF_MED}</td><td>[{BS_PF_CI[0]}, {BS_PF_CI[1]}]</td><td class="green bold">P(PF≥1.0) = {BS_P_GE1:.0f}%</td></tr>
<tr><td>Win Rate</td><td class="r">{WR}%</td><td>[{BS_WR_CI[0]}%, {BS_WR_CI[1]}%]</td><td></td></tr>
<tr><td>EV (bp)</td><td class="r green">{BS_EV_MED:+.1f}</td><td>[{BS_EV_CI[0]:+.1f}, {BS_EV_CI[1]:+.1f}]</td><td>CI 하한도 양수</td></tr>
<tr><td>Sharpe</td><td class="r">{SHARPE}</td><td>[{BS_SH_CI[0]}, {BS_SH_CI[1]}]</td><td></td></tr>
</table></div>

<h2 style="margin-top:16pt">Rolling Window 안정성</h2>
<div class="tw"><table>
<tr><th>Window</th><th>WR range</th><th>PF range</th><th>Sharpe range</th><th class="r">P(PF&lt;1.0)</th></tr>
<tr><td>100 trades</td><td>30~55%</td><td>0.794~2.172</td><td>-1.94~6.94</td><td class="r">9.1%</td></tr>
<tr><td>200 trades</td><td>36~52%</td><td>1.004~1.745</td><td>0.03~4.82</td><td class="r green bold">0.0%</td></tr>
<tr><td>500 trades</td><td>39~48%</td><td>1.091~1.400</td><td>0.75~2.95</td><td class="r green bold">0.0%</td></tr>
</table></div>
<div class="callout green"><strong>200-trade 이상 윈도우에서 PF &lt; 1.0 구간 0%.</strong> 약 200일(~7개월) 이상 운용하면 장기적으로 엣지가 실현된다.</div>

<!-- 5. YEARLY PERFORMANCE -->
<div class="section-break"></div>
<h2>5. 6년 연도별 성과 (PF · WR — 레버리지 무관)</h2>
{img("yearly")}
<div class="tw"><table>
<tr><th>Year</th><th class="r">Trades</th><th class="r">PF</th><th class="r">Sharpe</th><th class="r">WR</th><th>특징</th></tr>
{yr_rows}
</table></div>
<div class="callout"><strong>2023·2025 취약 구간:</strong> PF 1.13~1.16으로 가장 약하지만 여전히 흑자. 저변동 시장에서 전략이 &quot;간신히 버텨도 흑자&quot;라는 점이 중요. 더 나쁜 해가 올 수 있음을 인지해야 한다.</div>

<!-- 6. RISK ANALYSIS -->
<h2>6. 승/패 연속 & Duration 분석</h2>
{img("streak")}
{img("duration")}
<div class="tw"><table>
<tr><th></th><th class="r">Win Streak</th><th class="r">Loss Streak</th></tr>
<tr><td>최대</td><td class="r green bold">10연승</td><td class="r red bold">14연패</td></tr>
<tr><td>평균</td><td class="r">1.9</td><td class="r">2.4</td></tr>
<tr><td>14연패 포트폴리오 손실 (L/N=2.5)</td><td class="r">—</td><td class="r red bold">-56.2%</td></tr>
</table></div>

<h2 style="margin-top:16pt">리스크 시나리오 (L/N=2.5 기준)</h2>
<div class="tw"><table>
<tr><th>시나리오</th><th>영향</th><th class="c">6년 빈도</th></tr>
<tr><td>14연패 (최대 기록)</td><td class="red">compound -56.2% → ${V13_BALANCE:,} → ${int(V13_BALANCE * 0.438):,}</td><td class="c">1회</td></tr>
<tr><td>10연패</td><td>compound -37.8% → ${V13_BALANCE:,} → ${int(V13_BALANCE * 0.622):,}</td><td class="c">~2회</td></tr>
<tr><td>DD &gt; 50%</td><td>연 4~5회 예상</td><td class="c">빈번</td></tr>
<tr><td>2023형 저변동 연도</td><td>PF 1.13 — 간신히 흑자</td><td class="c">~2회/6년</td></tr>
</table></div>
<div class="callout red"><strong>14연패 시나리오:</strong> ${V13_BALANCE:,} → ${int(V13_BALANCE * 0.438):,}. 6년에 딱 1회 발생. 이 구간을 버텨야 복구된다.</div>

<!-- 7. SYMBOLS -->
<h2>7. 심볼별 성과 (PF · EV 기준, 레버리지 무관)</h2>
<p style="font-size:8.5pt;color:#555;margin-bottom:6pt">50개 심볼 중 37개 수익 (74%). 아래는 PF 기준 상/하위 심볼.</p>
<div class="tw"><table>
<tr><th>Top 10</th><th class="r">Trades</th><th class="r">WR</th><th class="r">PF</th><th class="r">EV (bp)</th></tr>
{top_rows}
</table></div>
<div class="tw"><table>
<tr><th>Bottom 5</th><th class="r">Trades</th><th class="r">WR</th><th class="r">PF</th><th class="r">EV (bp)</th></tr>
{bot_rows}
</table></div>
<div class="callout"><strong>COMPUSDT PF=0.46:</strong> 6년간 가장 나쁜 심볼. 그래도 전체 포트폴리오에서 42 trades / 2,265 = 1.9% 비중이라 전체 PF에 미미한 영향.</div>

<!-- 8. OPS PLAN -->
<div class="section-break"></div>
<h2>8. 모니터링 & 퇴각 계획</h2>
<div class="tw"><table>
<tr><th>항목</th><th>모니터링 기준</th><th>액션</th></tr>
<tr><td>DD</td><td>&gt; 70%</td><td>긴급 점검</td></tr>
<tr><td>30일 PF</td><td>&lt; 0.9</td><td>경고 — 60일까지 관찰</td></tr>
<tr><td>60일 PF</td><td>&lt; 1.0</td><td>레버리지 축소 (→ 중도 Tier)</td></tr>
<tr><td>Trade freq</td><td>&lt; 0.5/day (30일 평균)</td><td>시장 구조 변화 의심</td></tr>
</table></div>

<h2 style="margin-top:16pt">퇴각 3단계</h2>
<div class="tw"><table>
<tr><th>단계</th><th>트리거</th><th>설정 변경</th><th class="r">AnnRet</th><th class="r">Max DD</th></tr>
<tr><td class="bold">1단계</td><td>DD &gt; 70% 또는 60일 PF &lt; 0.95</td><td>L=16x, S=10 (L/N=1.6)</td><td class="r">~+270%</td><td class="r">~41%</td></tr>
<tr><td class="bold">2단계</td><td>DD &gt; 50% (축소 후에도)</td><td>L=9x, S=10 (L/N=0.9)</td><td class="r">~+162%</td><td class="r">~30%</td></tr>
<tr><td class="bold">3단계</td><td>근본적 의심</td><td>L=3x, S=15 (L/N=0.2)</td><td class="r">~+27%</td><td class="r">~8%</td></tr>
</table></div>

<h2 style="margin-top:16pt">복리 vs 인출전략 — 용량 한계</h2>
{img("compound")}
<div class="tw"><table>
<tr><th>잔고</th><th class="r">슬롯 노셔널</th><th>하위 심볼 영향</th><th class="c">판정</th></tr>
<tr><td>₩5,000만 ($35K)</td><td class="r">$86K</td><td>전혀 없음</td><td class="c green bold">안전</td></tr>
<tr><td><strong>₩1.5억 ($100K)</strong></td><td class="r"><strong>$258K</strong></td><td>경계선</td><td class="c gold bold">인출 트리거</td></tr>
<tr><td>₩2억 ($140K)</td><td class="r">$345K</td><td>슬리피지 시작</td><td class="c red bold">위험</td></tr>
</table></div>
<div class="callout green"><strong>최적 인출 전략 (₩5,000만 시작):</strong> ₩1.5억 도달 시 ₩8,000만 인출 → ₩7,000만 유지. 1년 기준 총 ~₩2.4억 인출 + ₩7,000만 잔고 = <strong>₩3.1억</strong> (원금 6.2배). 올복리(₩2.56억) 대비 총자산이 높고 심리적 부담도 낮다.</div>

<h2 style="margin-top:16pt">원금별 1년 수익 예상 (현실적 60% 감쇄)</h2>
<div class="tw"><table>
<tr><th>원금</th><th class="r">1년 후 (현실적)</th><th class="r">DD 최저점</th><th class="r">슬롯 노셔널</th></tr>
<tr><td>₩1,000만</td><td class="r">₩5,100만</td><td class="r red">₩320만</td><td class="r">$17K</td></tr>
<tr><td>₩5,000만</td><td class="r bold">₩2.56억</td><td class="r red">₩1,600만</td><td class="r">$86K</td></tr>
<tr><td>₩1억</td><td class="r bold">₩5.1억</td><td class="r red">₩3,200만</td><td class="r">$172K</td></tr>
</table></div>

<!-- VERDICT -->
<div class="verdict">
  <h3>최종 판단 — 돌려야 하는가?</h3>
  <p><strong style="color:#7ff;font-size:11pt">YES — 조건부</strong></p>
  <ul>
    <li>엣지는 통계적으로 확실 (t={T_STAT}, P(PF≥1.0)={BS_P_GE1:.0f}%). 노이즈가 아닌 진짜 비효율.</li>
    <li>6년 연속 흑자 — 코로나·LUNA·FTX·저변동 모두 통과. Regime-independent edge.</li>
    <li>L/N=2.5는 625개 조합에서 과학적으로 선택한 Calmar 최대화 경계점.</li>
    <li>${V13_BALANCE:,}이 전 재산이 아닐 것. DD {V13_MAX_DD}% (${int(V13_BALANCE * (1 - V13_MAX_DD/100)):,})에서도 시스템을 끄지 않을 것.</li>
    <li>최소 3개월(~90거래일) 관찰 후 평가. 기대치는 +687%가 아니라 <strong style="color:#ffe">+300~400%</strong>.</li>
  </ul>
  <p style="margin-top:8pt;color:#fdd"><strong>핵심 리스크:</strong> 엣지가 아니라 레버리지와 심리. DD 60%에서 &quot;끄면 안 된다&quot;는 것을 미리 각오해야 한다.</p>
</div>

<div class="footer">
  QuantBox v13.0 전략 보고서 — Commit: e4b5831 · Tag: v13.0 · Generated: 2026-03-03<br>
  VR8_56_SL1_TP5_D4 @25x leverage / 10 slots (L/N=2.5) · 50 symbols × 6yr · Binance USDT-M Futures
</div>
</body></html>"""
    return body


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("Generating QuantBox v13.0 Strategy Report…")
    c = {}
    for name, fn in [
        ("ln",       chart_ln),
        ("tiers",    chart_tiers),
        ("yearly",   chart_yearly),
        ("rr",       chart_rr),
        ("streak",   chart_streak),
        ("duration", chart_duration),
        ("compound", chart_compound),
    ]:
        print(f"  chart: {name}…"); c[name] = fn()
    print(f"  {len(c)} charts embedded")

    font_b64 = ""
    if _font_path.exists():
        font_b64 = base64.b64encode(_font_path.read_bytes()).decode()
        print(f"  font: {_font_path.name}")

    html = build_html(c, font_b64)

    with tempfile.NamedTemporaryFile(
        suffix=".html", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(html); html_path = f.name

    print("  Chrome headless → PDF…")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([
        "google-chrome", "--headless", "--no-sandbox", "--disable-gpu",
        "--disable-software-rasterizer",
        f"--print-to-pdf={OUTPUT}",
        "--print-to-pdf-no-header",
        f"file://{html_path}",
    ], capture_output=True, text=True, timeout=180)
    Path(html_path).unlink(missing_ok=True)

    if not OUTPUT.exists():
        print("Error:", result.stderr[-600:]); sys.exit(1)

    size = OUTPUT.stat().st_size / 1024
    print(f"\n✓ PDF: {OUTPUT}")
    print(f"  Size: {size:.1f} KB")


if __name__ == "__main__":
    main()
