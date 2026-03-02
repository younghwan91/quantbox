"""Generate PDF from markdown report using fpdf2 with embedded charts.

Usage:
    uv run python scripts/generate_report_pdf.py [markdown_file]
    # defaults to v13.0 deployment report
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np
from fpdf import FPDF, XPos, YPos

# ─── Font Setup ──────────────────────────────────────────────────────────────
_FONT_DIR = Path.home() / ".local" / "share" / "fonts"
KR_FONT = str(_FONT_DIR / "NotoSansKR.ttf")
_TTC_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_kr_font_path = KR_FONT if Path(KR_FONT).exists() else _TTC_FONT

if Path(_kr_font_path).exists():
    fm.fontManager.addfont(_kr_font_path)
    _fp = fm.FontProperties(fname=_kr_font_path)
    plt.rcParams["font.family"] = _fp.get_name()
plt.rcParams["axes.unicode_minus"] = False

MARGIN = 10
PAGE_W = 210

DEFAULT_MD = (
    Path(__file__).resolve().parent.parent
    / "docs" / "backtest_results"
    / "e4b5831_20260302_v13_deployment.md"
)

# ─── R40 Chart Data ──────────────────────────────────────────────────────────
R40_MONTHLY = {
    2020: [957, -37,   5,  63,  70,  40,  83,  57, -34,   6,  89,  -46],
    2021: [-38,  83,  58,   1,  37,   0, 102,  92, -19, -36,  -8,   70],
    2022: [ 45, -23,  16, 179,  14, -37,   5,  16,   0,   3,  13,  364],
    2023: [ 13,  74,  58,  12,  42, -47,   7, -25,  45,  17, -26,   98],
    2024: [-33,  46,   9, -13,  62, -45, -30, 386, -32, -15,  93,  122],
    2025: [100, -23,  65,  81, -42, 127,  16,  20, -36,  21, -57,   -8],
}
R40_ANNUAL = {2020: 4838, 2021: 749, 2022: 3036, 2023: 54, 2024: 2606, 2025: 109}
R40_LEVERAGE = {
    "lev":     [1,    2,    3,    5,   7.5,   10],
    "ann_ret": [195, 540,  923,  960,  104,  -94],
    "max_dd":  [35.3, 60.0, 77.2, 95.8, 99.9, 100],
    "calmar":  [5.5,  9.0,  12.0, 10.0,  1.0, -0.9],
}
R40_DURATION = {
    "buckets": ["0-1h","1-4h","4-12h","12-24h","24-48h","48-96h","96h+"],
    "pct":     [  3.3,  14.9,  27.1,   17.9,    20.7,    12.0,   4.1],
    "wr":      [ 10.7,  31.6,  38.2,   52.3,    48.2,    57.1,  44.1],
}
R40_TOP_SYM = {
    "sym": ["ETH","DOGE","1000SHIB","FET","OP","UNI","VET","LTC","WIF","1000PEPE"],
    "ret": [325, 287, 266, 189, 178, 170, 129, 128, 112, 111],
}
R40_BOT_SYM = {
    "sym": ["HBAR","BCH","XLM","ATOM","COMP"],
    "ret": [-29, -39, -53, -54, -77],
}
R40_WIN_STREAK  = {1:284, 2:128, 3:63, 4:28, 5:10, 6:4, 7:7, 8:2, 10:1}
R40_LOSS_STREAK = {1:226, 2:115, 3:82, 4:37, 5:32, 6:13, 7:5, 8:8, 9:3, 10:5, 12:1, 14:1}

# ─── V13 Chart Data ──────────────────────────────────────────────────────────
V13_LN = {
    "ln":      [0.20, 0.50, 1.00, 1.50, 2.00, 2.50, 2.78, 3.50, 4.00, 4.50],
    "ann_ret": [27.4, 80.0, 162., 270., 425., 687., 740., 998., 1200, 1350],
    "max_dd":  [ 8.0, 16.0, 29.6, 41.2, 53.8, 67.9, 73.1, 79.2, 84.7, 89.1],
    "calmar":  [3.42,  5.0, 5.48, 6.55, 7.90,10.12,10.13,12.60,14.16,15.15],
}
V13_TIERS = {
    "names":   ["보수적\n9x/10slots","중도\n16x/10slots","공격적\n25x/10slots\n(v13.0)"],
    "ann_ret": [162,  367,  687],
    "max_dd":  [29.6, 48.0, 67.9],
    "calmar":  [5.48, 7.65, 10.12],
}

STYLE = {
    "green": "#27ae60", "red": "#e74c3c", "blue": "#2980b9",
    "gold": "#f39c12", "purple": "#8e44ad", "gray": "#7f8c8d",
}


def _sim_monthly(start, monthly_rate, months, trigger=150, keep=70):
    comp, wb, wt = start, start, 0.0
    cs, ws, wts = [], [], []
    for _ in range(months):
        comp *= (1 + monthly_rate)
        wb *= (1 + monthly_rate)
        if wb >= trigger:
            wt += wb - keep
            wb = keep
        cs.append(comp); ws.append(wb); wts.append(wb + wt)
    return cs, ws, wts


def _savefig(fig, dpi=150):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def _style_ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


# ─── R40 Charts ──────────────────────────────────────────────────────────────
def chart_monthly_heatmap():
    years = sorted(R40_MONTHLY.keys())
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data = np.array([R40_MONTHLY[y] for y in years], dtype=float)
    fig, ax = plt.subplots(figsize=(13, 3.8))
    im = ax.imshow(np.clip(data, -100, 200), cmap="RdYlGn", aspect="auto", vmin=-100, vmax=200)
    ax.set_xticks(range(12)); ax.set_xticklabels(months, fontsize=8.5)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years, fontsize=9)
    for yi, year in enumerate(years):
        for mi, val in enumerate(R40_MONTHLY[year]):
            clr = "white" if abs(val) > 80 else "black"
            fw = "bold" if abs(val) >= 50 else "normal"
            ax.text(mi, yi, f"{val:+.0f}%", ha="center", va="center", fontsize=6.5, color=clr, fontweight=fw)
    plt.colorbar(im, ax=ax, shrink=0.85, label="월수익률 (%)")
    ax.set_title("월별 수익률 히트맵 — 6년 72개월  (3x lev / 15 slots)", fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout(); return _savefig(fig)


def chart_annual_returns():
    years = sorted(R40_ANNUAL.keys())
    rets = [R40_ANNUAL[y] for y in years]
    colors = [STYLE["green"] if r > 0 else STYLE["red"] for r in rets]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar([str(y) for y in years], rets, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, rets):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+60,
                f"+{val:,}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title("연도별 수익률 (3x 레버리지, 15 슬롯)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual Return (%)", fontsize=9)
    ax.set_ylim(0, max(rets)*1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}%"))
    _style_ax(ax); fig.tight_layout(); return _savefig(fig)


def chart_streak_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ws_k = [str(k) for k in R40_WIN_STREAK]; ws_v = list(R40_WIN_STREAK.values())
    ax1.bar(ws_k, ws_v, color=STYLE["green"], edgecolor="white", width=0.6)
    ax1.set_title("연승 분포  (최대 10연승)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("연승 길이", fontsize=9); ax1.set_ylabel("횟수", fontsize=9)
    _style_ax(ax1)
    for x, v in zip(ws_k, ws_v): ax1.text(x, v+3, str(v), ha="center", fontsize=7.5)

    ls_k = [str(k) for k in R40_LOSS_STREAK]; ls_v = list(R40_LOSS_STREAK.values())
    bars = ax2.bar(ls_k, ls_v, color=STYLE["red"], edgecolor="white", width=0.6)
    bars[-1].set_color(STYLE["purple"])
    ax2.set_title("연패 분포  (최대 14연패 — compound -56%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("연패 길이", fontsize=9); ax2.set_ylabel("횟수", fontsize=9)
    _style_ax(ax2)
    for x, v in zip(ls_k, ls_v): ax2.text(x, v+2, str(v), ha="center", fontsize=7.5)

    fig.suptitle("승/패 연속 분포 분석", fontsize=11, fontweight="bold")
    fig.tight_layout(); return _savefig(fig)


def chart_duration_wr():
    d = R40_DURATION; x = np.arange(len(d["buckets"]))
    fig, ax1 = plt.subplots(figsize=(9, 3.8)); ax2 = ax1.twinx()
    ax1.bar(x, d["pct"], color=STYLE["blue"], alpha=0.65, label="비중 %", width=0.5)
    ax2.plot(x, d["wr"], "o-", color=STYLE["red"], linewidth=2.2, markersize=7, label="Win Rate %", zorder=5)
    ax2.axhline(43.4, color=STYLE["gray"], linewidth=1, linestyle="--", alpha=0.7)
    ax2.text(len(x)-0.9, 45, "전체 WR 43.4%", fontsize=7.5, color=STYLE["gray"])
    ax1.set_xticks(x); ax1.set_xticklabels(d["buckets"], fontsize=9)
    ax1.set_ylabel("비중 (%)", color=STYLE["blue"], fontsize=9)
    ax2.set_ylabel("Win Rate (%)", color=STYLE["red"], fontsize=9)
    ax1.tick_params(axis="y", labelcolor=STYLE["blue"]); ax2.set_ylim(0, 72)
    ax1.set_title("트레이드 Duration별 비중 & Win Rate\n(장기 보유일수록 WR 상승)", fontsize=10, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="upper right", fontsize=8)
    _style_ax(ax1); fig.tight_layout(); return _savefig(fig)


def chart_symbol_performance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    syms_t = R40_TOP_SYM["sym"]; rets_t = R40_TOP_SYM["ret"]
    y1 = np.arange(len(syms_t))
    colors_t = plt.cm.YlGn(np.linspace(0.4, 0.9, len(syms_t)))[::-1]
    ax1.barh(y1, rets_t, color=colors_t, edgecolor="white")
    ax1.set_yticks(y1); ax1.set_yticklabels(syms_t, fontsize=8.5)
    ax1.set_xlabel("6년 Return %", fontsize=9)
    ax1.set_title("Top 10 수익 심볼", fontsize=10, fontweight="bold")
    for i, v in enumerate(rets_t): ax1.text(v+4, i, f"+{v}%", va="center", fontsize=7.5)
    ax1.spines[["top","right"]].set_visible(False); ax1.grid(axis="x", alpha=0.25)

    syms_b = R40_BOT_SYM["sym"]; rets_b = R40_BOT_SYM["ret"]
    y2 = np.arange(len(syms_b))
    colors_b = plt.cm.OrRd(np.linspace(0.3, 0.8, len(syms_b)))
    ax2.barh(y2, rets_b, color=colors_b, edgecolor="white")
    ax2.set_yticks(y2); ax2.set_yticklabels(syms_b, fontsize=8.5)
    ax2.set_xlabel("6년 Return %", fontsize=9)
    ax2.set_title("Bottom 5 손실 심볼", fontsize=10, fontweight="bold")
    for i, v in enumerate(rets_b): ax2.text(v-2, i, f"{v}%", va="center", ha="right", fontsize=7.5)
    ax2.spines[["top","right"]].set_visible(False); ax2.grid(axis="x", alpha=0.25)

    fig.suptitle("심볼별 성과 분석 — 50 symbols × 6년", fontsize=11, fontweight="bold")
    fig.tight_layout(); return _savefig(fig)


def chart_leverage_sensitivity():
    d = R40_LEVERAGE
    fig, ax1 = plt.subplots(figsize=(8, 3.8)); ax2 = ax1.twinx()
    ax1.plot(d["lev"], d["ann_ret"], "o-", color=STYLE["blue"], linewidth=2.5, markersize=7, label="Annual Return %")
    ax2.plot(d["lev"], d["max_dd"],  "s--", color=STYLE["red"], linewidth=2, markersize=7, label="Max DD %")
    ax2.plot(d["lev"], d["calmar"],  "^:", color=STYLE["green"], linewidth=2, markersize=7, label="Calmar")
    ax1.axvline(3, color=STYLE["gold"], linewidth=2, linestyle="--", alpha=0.85)
    ax1.text(3.08, 700, "3x\noptimal", fontsize=8, color=STYLE["gold"], fontweight="bold")
    ax1.set_xlabel("Leverage", fontsize=9); ax1.set_ylabel("Annual Return %", color=STYLE["blue"], fontsize=9)
    ax2.set_ylabel("Max DD % / Calmar", fontsize=9); ax1.tick_params(axis="y", labelcolor=STYLE["blue"])
    ax1.set_xticks(d["lev"]); _style_ax(ax1)
    ax1.set_title("레버리지 민감도  (5x+ 부터 복리 붕괴)", fontsize=11, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="upper right", fontsize=8)
    fig.tight_layout(); return _savefig(fig)


# ─── V13 Charts ──────────────────────────────────────────────────────────────
def chart_ln_ratio():
    d = V13_LN
    fig, ax1 = plt.subplots(figsize=(9, 4.2)); ax2 = ax1.twinx()
    ax1.plot(d["ln"], d["ann_ret"], "o-", color=STYLE["blue"], linewidth=2.5, markersize=7, label="Annual Return %")
    ax2.plot(d["ln"], d["max_dd"], "s--", color=STYLE["red"], linewidth=2, markersize=6, label="Max DD %")
    ax2.plot(d["ln"], d["calmar"], "^:", color=STYLE["green"], linewidth=2, markersize=6, label="Calmar")
    ax1.axvline(2.5, color=STYLE["gold"], linewidth=2.2, linestyle="--")
    ax1.text(2.52, 900, "v13.0  L/N=2.5\nLosing Year=0 경계", fontsize=7.8, color=STYLE["gold"], fontweight="bold")
    ax1.axvline(4.5, color=STYLE["red"], linewidth=1.5, linestyle=":", alpha=0.7)
    ax1.text(4.52, 200, "Losing Year\n시작 ->", fontsize=7.5, color=STYLE["red"])
    ax1.set_xlabel("L/N Ratio  (Leverage / Slots)", fontsize=9)
    ax1.set_ylabel("Annual Return %", color=STYLE["blue"], fontsize=9)
    ax2.set_ylabel("Max DD % / Calmar", fontsize=9); ax1.tick_params(axis="y", labelcolor=STYLE["blue"])
    ax1.set_title("L/N Ratio vs 성과 지표  (25x25 Fine Grid - 625 combinations)", fontsize=10, fontweight="bold")
    _style_ax(ax1)
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="upper left", fontsize=8)
    fig.tight_layout(); return _savefig(fig)


def chart_tier_comparison():
    d = V13_TIERS; x = np.arange(len(d["names"])); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.2))
    b1 = ax.bar(x-w, [r/10 for r in d["ann_ret"]], w, label="Annual Return /10", color=STYLE["blue"], alpha=0.82)
    b2 = ax.bar(x,   d["max_dd"],                  w, label="Max DD %", color=STYLE["red"], alpha=0.82)
    b3 = ax.bar(x+w, [c*10 for c in d["calmar"]],  w, label="Calmar x10", color=STYLE["green"], alpha=0.82)
    for bars, vals, fmt in [(b1, d["ann_ret"], "+{:.0f}%"), (b2, d["max_dd"], "{:.0f}%"), (b3, d["calmar"], "{:.2f}")]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    fmt.format(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(d["names"], fontsize=9)
    ax.set_title("운용 Tier별 성과 비교  (S=10 slots 고정)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); _style_ax(ax)
    ax.set_ylim(0, max([r/10 for r in d["ann_ret"]] + d["max_dd"] + [c*10 for c in d["calmar"]]) * 1.18)
    fig.tight_layout(); return _savefig(fig)


def chart_compound_projection():
    months = list(range(1, 15))
    comp, wb, wt = _sim_monthly(50, 0.15, 14, 150, 70)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(months, comp, "o-", color=STYLE["blue"], linewidth=2.5, markersize=6, label="올복리 잔고")
    ax.plot(months, wb, "s-", color=STYLE["green"], linewidth=2.5, markersize=6, label="인출전략 잔고")
    ax.plot(months, wt, "^--", color=STYLE["green"], linewidth=1.8, markersize=5, alpha=0.75, label="인출전략 총자산(잔고+현금)")
    ax.axhline(150, color=STYLE["red"], linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(0.5, 153, "인출 트리거 1.5억", fontsize=8, color=STYLE["red"], va="bottom")
    ax.axhline(70, color=STYLE["gray"], linewidth=1, linestyle=":", alpha=0.6)
    ax.text(0.5, 72, "유지 잔고 7000만", fontsize=7.5, color=STYLE["gray"], va="bottom")
    ax.set_xlabel("월수", fontsize=9); ax.set_ylabel("금액 (백만원)", fontsize=9)
    ax.set_title("복리 vs 인출전략 시뮬레이션 (5천만 시작 / 월+15% / 인출 시 총자산이 더 높음)", fontsize=9.5, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    ax.set_xticks(months); ax.legend(fontsize=8); _style_ax(ax)
    fig.tight_layout(); return _savefig(fig)


# ─── Chart map ───────────────────────────────────────────────────────────────
def build_chart_map(md_stem: str) -> dict[str, str]:
    if "r40_deep_stats" in md_stem:
        print("  Generating R40 charts...")
        return {
            "1. 핵심 성과 지표":   chart_annual_returns(),
            "4. 월별 수익률":      chart_monthly_heatmap(),
            "7. 승/패 연속":       chart_streak_distribution(),
            "8. 트레이드 Duration": chart_duration_wr(),
            "10. 심볼별 분석":     chart_symbol_performance(),
            "11. 레버리지 민감도": chart_leverage_sensitivity(),
        }
    elif "v13_deployment" in md_stem:
        print("  Generating V13 charts...")
        return {
            "2.1 Fine Grid":   chart_ln_ratio(),
            "3. Tier별 대안":  chart_tier_comparison(),
            "11. 원금별 복리": chart_compound_projection(),
        }
    return {}


# ─── PDF Class ───────────────────────────────────────────────────────────────
class ReportPDF(FPDF):
    def __init__(self, header_text="QuantBox Report"):
        super().__init__()
        self._header_text = header_text

    def header(self):
        self.set_font("NotoSans", "B", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, self._header_text, align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("NotoSans", "", 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def parse_table(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        rows.append(cells)
    return rows


def safe_write(pdf, h, txt):
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    if avail < 5:
        pdf.ln(h); return
    try:
        pdf.multi_cell(0, h, txt)
    except Exception:
        pdf.cell(0, h, txt[:80], new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_table(pdf, rows):
    if not rows: return
    n_cols = max(len(r) for r in rows)
    col_w = (PAGE_W - 2*MARGIN) / n_cols
    for idx, row in enumerate(rows):
        while len(row) < n_cols: row.append("")
        if idx == 0:
            pdf.set_font("NotoSans", "B", 6.5); pdf.set_fill_color(230, 230, 245)
        else:
            pdf.set_font("NotoSans", "", 6.5)
            pdf.set_fill_color(248, 248, 252) if idx % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        for cell in row:
            pdf.cell(col_w, 4.5, cell[:36], border=1, fill=True)
        pdf.ln()
    pdf.ln(2)


def embed_chart(pdf, path):
    if not path or not Path(path).exists(): return
    pdf.ln(2)
    if pdf.get_y() > pdf.h - pdf.b_margin - 55:
        pdf.add_page()
    pdf.image(path, x=MARGIN, w=PAGE_W - 2*MARGIN)
    pdf.ln(4)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    if not md_path.exists():
        print(f"Error: {md_path} not found"); sys.exit(1)
    output_pdf = md_path.with_suffix(".pdf")

    print(f"Building charts for: {md_path.stem} ...")
    chart_map = build_chart_map(md_path.stem)
    print(f"  -> {len(chart_map)} chart(s) prepared")

    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    title = "QuantBox Report"
    for line in lines:
        if line.strip().startswith("# ") and not line.strip().startswith("## "):
            title = line.strip().lstrip("# "); break

    pdf = ReportPDF(header_text=title)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_left_margin(MARGIN); pdf.set_right_margin(MARGIN)

    _ttc_kw = {} if _kr_font_path == KR_FONT else {"collection_font_number": 1}
    pdf.add_font("NotoSans", "",  _kr_font_path, **_ttc_kw)
    pdf.add_font("NotoSans", "B", _kr_font_path, **_ttc_kw)
    pdf.add_page()

    i = 0; in_table = False; table_lines = []

    def _check_chart(htxt):
        for key, path in chart_map.items():
            if key in htxt:
                embed_chart(pdf, path); break

    while i < len(lines):
        line = lines[i]; stripped = line.strip()

        if stripped.startswith("# ") and not stripped.startswith("## "):
            pdf.set_font("NotoSans", "B", 16); pdf.set_text_color(0, 0, 0)
            pdf.ln(4); safe_write(pdf, 9, stripped.lstrip("# ")); pdf.ln(3)
            i += 1; continue

        if stripped.startswith("## "):
            htxt = stripped.lstrip("# ")
            pdf.set_font("NotoSans", "B", 12); pdf.set_text_color(25, 25, 100)
            pdf.ln(5); safe_write(pdf, 7, htxt); pdf.ln(1)
            _check_chart(htxt); i += 1; continue

        if stripped.startswith("### "):
            htxt = stripped.lstrip("# ")
            pdf.set_font("NotoSans", "B", 10); pdf.set_text_color(50, 50, 50)
            pdf.ln(2); safe_write(pdf, 6, htxt); pdf.ln(1)
            _check_chart(htxt); i += 1; continue

        if stripped == "---":
            y = pdf.get_y(); pdf.set_draw_color(200, 200, 200)
            pdf.line(MARGIN, y+2, PAGE_W-MARGIN, y+2); pdf.ln(5)
            i += 1; continue

        if stripped.startswith("|"):
            if not in_table: in_table = True; table_lines = []
            table_lines.append(stripped); i += 1
            if i >= len(lines) or not lines[i].strip().startswith("|"):
                in_table = False; render_table(pdf, parse_table(table_lines)); table_lines = []
            continue

        if stripped.startswith("> "):
            pdf.set_font("NotoSans", "", 8); pdf.set_text_color(80, 80, 80)
            safe_write(pdf, 4.5, "  > " + stripped.lstrip("> ").replace("**", ""))
            pdf.set_text_color(0, 0, 0); i += 1; continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("NotoSans", "", 8); pdf.set_text_color(0, 0, 0)
            safe_write(pdf, 4.5, "  * " + stripped[2:].replace("**", "").replace("`", ""))
            i += 1; continue

        num_m = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if num_m:
            pdf.set_font("NotoSans", "", 8); pdf.set_text_color(0, 0, 0)
            safe_write(pdf, 4.5, f"  {num_m.group(1)}. " + num_m.group(2).replace("**","").replace("`",""))
            i += 1; continue

        if stripped.startswith("**") and ":**" in stripped:
            pdf.set_font("NotoSans", "", 8); pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 4.5, stripped.replace("**","").replace("`",""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            i += 1; continue

        if stripped.startswith("```"):
            i += 1
            pdf.set_font("NotoSans", "", 7); pdf.set_text_color(60, 60, 60)
            while i < len(lines) and not lines[i].strip().startswith("```"):
                safe_write(pdf, 3.5, lines[i]); i += 1
            pdf.set_text_color(0, 0, 0)
            if i < len(lines): i += 1
            continue

        if not stripped:
            pdf.ln(1.5); i += 1; continue

        pdf.set_font("NotoSans", "", 8); pdf.set_text_color(0, 0, 0)
        safe_write(pdf, 4.5, stripped.replace("**","").replace("`","").replace("*",""))
        i += 1

    pdf.output(str(output_pdf))
    print(f"PDF saved : {output_pdf}")
    print(f"Pages     : {pdf.pages_count}")
    print(f"Size      : {output_pdf.stat().st_size / 1024:.1f} KB")

    for path in chart_map.values():
        try: Path(path).unlink()
        except Exception: pass


if __name__ == "__main__":
    main()
