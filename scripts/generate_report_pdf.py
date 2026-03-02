"""Generate PDF from markdown report using Chrome headless.

Methodology:
    1. Parse markdown → build HTML with embedded base64 charts
    2. google-chrome --headless --print-to-pdf

Usage:
    uv run python scripts/generate_report_pdf.py [markdown_file]
"""
from __future__ import annotations

import base64
import re
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

# ─── Font ────────────────────────────────────────────────────────────────────
_KR_TTF = Path.home() / ".local" / "share" / "fonts" / "NotoSansKR.ttf"
_TTC     = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_font_path = _KR_TTF if _KR_TTF.exists() else _TTC

if _font_path.exists():
    fm.fontManager.addfont(str(_font_path))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_font_path)).get_name()
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_MD = (
    Path(__file__).resolve().parent.parent
    / "docs" / "backtest_results"
    / "e4b5831_20260302_v13_deployment.md"
)

# ─── Chart data ──────────────────────────────────────────────────────────────
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


def _sim_monthly(start, rate, months, trigger=150, keep=70):
    comp, wb, wt, cs, ws, wts = start, start, 0.0, [], [], []
    for _ in range(months):
        comp *= (1+rate); wb *= (1+rate)
        if wb >= trigger: wt += wb - keep; wb = keep
        cs.append(comp); ws.append(wb); wts.append(wb+wt)
    return cs, ws, wts


def _png_b64(fig: plt.Figure, dpi: int = 130) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    return base64.b64encode(data).decode()


def _style_ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


# ─── R40 Charts ──────────────────────────────────────────────────────────────
def chart_monthly_heatmap() -> str:
    years = sorted(R40_MONTHLY.keys())
    months_lbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data = np.array([R40_MONTHLY[y] for y in years], dtype=float)
    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(np.clip(data, -100, 200), cmap="RdYlGn", aspect="auto", vmin=-100, vmax=200)
    ax.set_xticks(range(12)); ax.set_xticklabels(months_lbl, fontsize=8)
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years, fontsize=9)
    for yi, year in enumerate(years):
        for mi, val in enumerate(R40_MONTHLY[year]):
            c = "white" if abs(val) > 80 else "black"
            fw = "bold" if abs(val) >= 50 else "normal"
            ax.text(mi, yi, f"{val:+.0f}%", ha="center", va="center", fontsize=6.5, color=c, fontweight=fw)
    plt.colorbar(im, ax=ax, shrink=0.8, label="월수익률 (%)")
    ax.set_title("월별 수익률 히트맵 — 6년 72개월 (3x lev / 15 slots)", fontsize=11, fontweight="bold")
    fig.tight_layout(); return _png_b64(fig)


def chart_annual_returns() -> str:
    years = sorted(R40_ANNUAL.keys()); rets = [R40_ANNUAL[y] for y in years]
    colors = [STYLE["green"] if r > 0 else STYLE["red"] for r in rets]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([str(y) for y in years], rets, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, rets):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+60,
                f"+{val:,}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title("연도별 수익률 (3x 레버리지, 15 슬롯)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual Return (%)"); ax.set_ylim(0, max(rets)*1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}%"))
    _style_ax(ax); fig.tight_layout(); return _png_b64(fig)


def chart_streak_distribution() -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ws_k = [str(k) for k in R40_WIN_STREAK]; ws_v = list(R40_WIN_STREAK.values())
    ax1.bar(ws_k, ws_v, color=STYLE["green"], edgecolor="white", width=0.6)
    ax1.set_title("연승 분포 (최대 10연승)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("연승 길이"); ax1.set_ylabel("횟수"); _style_ax(ax1)
    for x, v in zip(ws_k, ws_v): ax1.text(x, v+3, str(v), ha="center", fontsize=7.5)
    ls_k = [str(k) for k in R40_LOSS_STREAK]; ls_v = list(R40_LOSS_STREAK.values())
    bars = ax2.bar(ls_k, ls_v, color=STYLE["red"], edgecolor="white", width=0.6)
    bars[-1].set_color(STYLE["purple"])
    ax2.set_title("연패 분포 (최대 14연패 ← compound -56%)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("연패 길이"); ax2.set_ylabel("횟수"); _style_ax(ax2)
    for x, v in zip(ls_k, ls_v): ax2.text(x, v+2, str(v), ha="center", fontsize=7.5)
    fig.suptitle("승/패 연속 분포 분석", fontsize=11, fontweight="bold")
    fig.tight_layout(); return _png_b64(fig)


def chart_duration_wr() -> str:
    d = R40_DURATION; x = np.arange(len(d["buckets"]))
    fig, ax1 = plt.subplots(figsize=(9, 4)); ax2 = ax1.twinx()
    ax1.bar(x, d["pct"], color=STYLE["blue"], alpha=0.65, label="비중 %", width=0.5)
    ax2.plot(x, d["wr"], "o-", color=STYLE["red"], linewidth=2.2, markersize=7, label="Win Rate %")
    ax2.axhline(43.4, color=STYLE["gray"], linewidth=1, linestyle="--", alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(d["buckets"], fontsize=9)
    ax1.set_ylabel("비중 (%)", color=STYLE["blue"]); ax2.set_ylabel("Win Rate (%)", color=STYLE["red"])
    ax2.set_ylim(0, 75)
    ax1.set_title("Duration별 비중 & Win Rate (장기 보유일수록 WR 상승)", fontsize=10, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="upper right", fontsize=8)
    _style_ax(ax1); fig.tight_layout(); return _png_b64(fig)


def chart_symbol_performance() -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    sy = R40_TOP_SYM["sym"]; ry = R40_TOP_SYM["ret"]; y1 = np.arange(len(sy))
    ax1.barh(y1, ry, color=plt.cm.YlGn(np.linspace(0.4,0.9,len(sy)))[::-1], edgecolor="white")
    ax1.set_yticks(y1); ax1.set_yticklabels(sy, fontsize=8.5)
    ax1.set_xlabel("6년 Return %"); ax1.set_title("Top 10 수익 심볼", fontsize=10, fontweight="bold")
    for i, v in enumerate(ry): ax1.text(v+4, i, f"+{v}%", va="center", fontsize=7.5)
    ax1.spines[["top","right"]].set_visible(False); ax1.grid(axis="x", alpha=0.25)
    sb = R40_BOT_SYM["sym"]; rb = R40_BOT_SYM["ret"]; y2 = np.arange(len(sb))
    ax2.barh(y2, rb, color=plt.cm.OrRd(np.linspace(0.3,0.8,len(sb))), edgecolor="white")
    ax2.set_yticks(y2); ax2.set_yticklabels(sb, fontsize=8.5)
    ax2.set_xlabel("6년 Return %"); ax2.set_title("Bottom 5 손실 심볼", fontsize=10, fontweight="bold")
    for i, v in enumerate(rb): ax2.text(v-2, i, f"{v}%", va="center", ha="right", fontsize=7.5)
    ax2.spines[["top","right"]].set_visible(False); ax2.grid(axis="x", alpha=0.25)
    fig.suptitle("심볼별 성과 분석 — 50 symbols × 6년", fontsize=11, fontweight="bold")
    fig.tight_layout(); return _png_b64(fig)


def chart_leverage_sensitivity() -> str:
    d = R40_LEVERAGE
    fig, ax1 = plt.subplots(figsize=(8, 4)); ax2 = ax1.twinx()
    ax1.plot(d["lev"], d["ann_ret"], "o-", color=STYLE["blue"], linewidth=2.5, markersize=7, label="Annual Return %")
    ax2.plot(d["lev"], d["max_dd"],  "s--", color=STYLE["red"],   linewidth=2, markersize=7, label="Max DD %")
    ax2.plot(d["lev"], d["calmar"],  "^:",  color=STYLE["green"], linewidth=2, markersize=7, label="Calmar")
    ax1.axvline(3, color=STYLE["gold"], linewidth=2, linestyle="--", alpha=0.85)
    ax1.text(3.1, 700, "3x optimal", fontsize=8, color=STYLE["gold"], fontweight="bold")
    ax1.set_xlabel("Leverage"); ax1.set_ylabel("Annual Return %", color=STYLE["blue"])
    ax2.set_ylabel("Max DD % / Calmar"); ax1.set_xticks(d["lev"]); _style_ax(ax1)
    ax1.set_title("레버리지 민감도 (5x+ 부터 복리 붕괴)", fontsize=11, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="upper right", fontsize=8)
    fig.tight_layout(); return _png_b64(fig)


# ─── V13 Charts ──────────────────────────────────────────────────────────────
def chart_ln_ratio() -> str:
    d = V13_LN
    fig, ax1 = plt.subplots(figsize=(9, 4.5)); ax2 = ax1.twinx()
    ax1.plot(d["ln"], d["ann_ret"], "o-", color=STYLE["blue"], linewidth=2.5, markersize=7, label="Annual Return %")
    ax2.plot(d["ln"], d["max_dd"],  "s--", color=STYLE["red"],   linewidth=2, markersize=6, label="Max DD %")
    ax2.plot(d["ln"], d["calmar"],  "^:",  color=STYLE["green"], linewidth=2, markersize=6, label="Calmar")
    ax1.axvline(2.5, color=STYLE["gold"], linewidth=2.2, linestyle="--")
    ax1.text(2.52, 900, "v13.0  L/N=2.5\nLosing Year=0 경계", fontsize=7.8, color=STYLE["gold"], fontweight="bold")
    ax1.axvline(4.5, color=STYLE["red"], linewidth=1.5, linestyle=":", alpha=0.7)
    ax1.text(4.52, 200, "Losing Year 시작", fontsize=7.5, color=STYLE["red"])
    ax1.set_xlabel("L/N Ratio (Leverage / Slots)"); ax1.set_ylabel("Annual Return %", color=STYLE["blue"])
    ax2.set_ylabel("Max DD % / Calmar"); _style_ax(ax1)
    ax1.set_title("L/N Ratio vs 성과 지표 (25x25 Fine Grid — 625 combinations)", fontsize=10, fontweight="bold")
    lines1, l1 = ax1.get_legend_handles_labels(); lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, loc="upper left", fontsize=8)
    fig.tight_layout(); return _png_b64(fig)


def chart_tier_comparison() -> str:
    d = V13_TIERS; x = np.arange(len(d["names"])); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x-w, [r/10 for r in d["ann_ret"]], w, label="Return /10", color=STYLE["blue"], alpha=0.82)
    b2 = ax.bar(x,   d["max_dd"],                  w, label="Max DD %",   color=STYLE["red"],  alpha=0.82)
    b3 = ax.bar(x+w, [c*10 for c in d["calmar"]],  w, label="Calmar x10", color=STYLE["green"],alpha=0.82)
    for bars, vals, fmt in [(b1,d["ann_ret"],"+{:.0f}%"),(b2,d["max_dd"],"{:.0f}%"),(b3,d["calmar"],"{:.2f}")]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, fmt.format(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("\n"," ") for n in d["names"]], fontsize=8.5)
    ax.set_title("운용 Tier별 성과 비교 (S=10 slots 고정)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); _style_ax(ax)
    fig.tight_layout(); return _png_b64(fig)


def chart_compound_projection() -> str:
    months = list(range(1, 15))
    comp, wb, wt = _sim_monthly(50, 0.15, 14, 150, 70)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(months, comp, "o-", color=STYLE["blue"],  linewidth=2.5, markersize=6, label="올복리 잔고")
    ax.plot(months, wb,   "s-", color=STYLE["green"], linewidth=2.5, markersize=6, label="인출전략 잔고")
    ax.plot(months, wt,   "^--",color=STYLE["green"], linewidth=1.8, markersize=5, alpha=0.75, label="인출전략 총자산(잔고+현금)")
    ax.axhline(150, color=STYLE["red"],  linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(0.5, 153, "인출 트리거 1.5억", fontsize=8, color=STYLE["red"], va="bottom")
    ax.axhline(70,  color=STYLE["gray"], linewidth=1,   linestyle=":", alpha=0.6)
    ax.text(0.5, 72, "유지 잔고 7000만", fontsize=7.5, color=STYLE["gray"], va="bottom")
    ax.set_xlabel("월수"); ax.set_ylabel("금액 (백만원)")
    ax.set_title("복리 vs 인출전략 시뮬레이션 (5천만 시작 / 월+15%)", fontsize=10, fontweight="bold")
    ax.set_xticks(months); ax.legend(fontsize=8); _style_ax(ax)
    fig.tight_layout(); return _png_b64(fig)


# ─── Build chart map ─────────────────────────────────────────────────────────
def build_charts(md_stem: str) -> dict[str, str]:
    if "r40_deep_stats" in md_stem:
        print("  Generating R40 charts (6)...")
        return {
            "1. 핵심 성과 지표":    chart_annual_returns(),
            "4. 월별 수익률":       chart_monthly_heatmap(),
            "7. 승/패 연속":        chart_streak_distribution(),
            "8. 트레이드 Duration": chart_duration_wr(),
            "10. 심볼별 분석":      chart_symbol_performance(),
            "11. 레버리지 민감도":  chart_leverage_sensitivity(),
        }
    elif "v13_deployment" in md_stem:
        print("  Generating V13 charts (3)...")
        return {
            "2.1 Fine Grid":   chart_ln_ratio(),
            "3. Tier별 대안":  chart_tier_comparison(),
            "11. 원금별 복리": chart_compound_projection(),
        }
    return {}


# ─── Markdown → HTML ─────────────────────────────────────────────────────────
def md_to_html(md_text: str, title: str, charts: dict[str, str],
               font_b64: str) -> str:
    """Convert markdown to complete HTML with embedded font and charts."""

    def esc(s: str) -> str:
        return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def inline_fmt(s: str) -> str:
        s = esc(s)
        # bold
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        # inline code
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return s

    lines = md_text.split("\n")
    html_parts: list[str] = []

    def maybe_chart(heading_text: str) -> str:
        for key, b64 in charts.items():
            if key in heading_text:
                return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="chart"></div>'
        return ""

    i = 0
    in_table = False
    table_rows: list[list[str]] = []
    in_code = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Code block
        if stripped.startswith("```"):
            if not in_code:
                lang = stripped[3:].strip() or ""
                html_parts.append(f'<pre><code class="{esc(lang)}">')
                in_code = True
            else:
                html_parts.append("</code></pre>")
                in_code = False
            i += 1; continue

        if in_code:
            html_parts.append(esc(line))
            i += 1; continue

        # Table accumulation
        if stripped.startswith("|"):
            if not in_table: in_table = True; table_rows = []
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not all(re.match(r"^[-:]+$", c) for c in cells):
                table_rows.append(cells)
            i += 1
            if i < len(lines) and not lines[i].strip().startswith("|"):
                in_table = False
                html_parts.append('<div class="table-wrap"><table>')
                for ri, row in enumerate(table_rows):
                    tag = "th" if ri == 0 else "td"
                    html_parts.append("<tr>" + "".join(f"<{tag}>{esc(c)}</{tag}>" for c in row) + "</tr>")
                html_parts.append("</table></div>")
            continue

        # HR
        if stripped == "---":
            html_parts.append("<hr>"); i += 1; continue

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            tag = f"h{level}"
            chart_html = maybe_chart(text)
            html_parts.append(f"<{tag}>{inline_fmt(text)}</{tag}>")
            if chart_html:
                html_parts.append(chart_html)
            i += 1; continue

        # Blockquote
        if stripped.startswith("> "):
            html_parts.append(f'<blockquote>{inline_fmt(stripped[2:])}</blockquote>')
            i += 1; continue

        # Bullet
        if stripped.startswith("- ") or stripped.startswith("* "):
            html_parts.append(f'<ul><li>{inline_fmt(stripped[2:])}</li></ul>')
            i += 1; continue

        # Numbered list
        nm = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if nm:
            html_parts.append(f'<ol start="{nm.group(1)}"><li>{inline_fmt(nm.group(2))}</li></ol>')
            i += 1; continue

        # Empty line
        if not stripped:
            html_parts.append("<br>"); i += 1; continue

        # Paragraph
        html_parts.append(f"<p>{inline_fmt(stripped)}</p>"); i += 1

    body = "\n".join(html_parts)

    # Build font-face CSS from base64
    font_face = ""
    if font_b64:
        font_face = f"""
        @font-face {{
            font-family: 'NotoSansKR';
            src: url('data:font/truetype;base64,{font_b64}') format('truetype');
        }}"""

    css = f"""
    {font_face}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: 'NotoSansKR', 'Noto Sans CJK KR', sans-serif;
        font-size: 10pt;
        line-height: 1.6;
        color: #222;
        padding: 20mm 18mm 20mm 18mm;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}
    h1 {{ font-size: 20pt; color: #111; margin: 16pt 0 8pt; border-bottom: 2pt solid #2980b9; padding-bottom: 4pt; }}
    h2 {{ font-size: 14pt; color: #19195a; margin: 14pt 0 6pt; border-left: 4pt solid #2980b9; padding-left: 8pt; }}
    h3 {{ font-size: 12pt; color: #333; margin: 10pt 0 4pt; }}
    h4 {{ font-size: 10pt; color: #555; margin: 8pt 0 3pt; }}
    p  {{ margin: 4pt 0; }}
    hr {{ border: none; border-top: 1pt solid #ccc; margin: 10pt 0; }}
    code {{ background: #f4f4f8; padding: 1pt 3pt; border-radius: 2pt; font-size: 8.5pt; font-family: monospace; word-break: break-all; }}
    pre  {{ background: #f4f4f8; border: 1pt solid #ddd; padding: 8pt; border-radius: 4pt; margin: 6pt 0;
            overflow: hidden; font-size: 8pt; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }}
    pre code {{ background: none; padding: 0; }}
    blockquote {{ border-left: 3pt solid #aaa; margin: 4pt 0; padding: 2pt 8pt; color: #555; font-size: 9pt; }}
    ul, ol {{ margin: 3pt 0 3pt 18pt; }}
    li {{ margin: 1pt 0; }}
    .table-wrap {{ overflow-x: auto; margin: 8pt 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt; table-layout: fixed; }}
    th, td {{ border: 0.5pt solid #bbb; padding: 3pt 5pt; text-align: left;
               word-wrap: break-word; overflow-wrap: break-word; }}
    th {{ background: #e8e8f5; font-weight: bold; }}
    tr:nth-child(even) td {{ background: #f8f8fc; }}
    .chart {{ margin: 10pt 0; page-break-inside: avoid; }}
    .chart img {{ width: 100%; height: auto; }}
    strong {{ font-weight: bold; }}
    @media print {{
        body {{ padding: 0; }}
        .chart {{ page-break-inside: avoid; }}
    }}
    """

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    if not md_path.exists():
        print(f"Error: {md_path} not found"); sys.exit(1)
    output_pdf = md_path.with_suffix(".pdf")

    print(f"Generating PDF: {md_path.name}")
    charts = build_charts(md_path.stem)
    print(f"  -> {len(charts)} chart(s) embedded")

    # Read font as base64
    font_b64 = ""
    if _font_path.exists():
        font_b64 = base64.b64encode(_font_path.read_bytes()).decode()
        print(f"  -> Font embedded: {_font_path.name}")

    title = md_path.stem
    for line in md_path.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if s.startswith("# ") and not s.startswith("## "):
            title = s.lstrip("# "); break

    html = md_to_html(md_path.read_text(encoding="utf-8"), title, charts, font_b64)

    # Write HTML to temp file
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", encoding="utf-8", delete=False) as f:
        f.write(html); html_path = f.name

    print("  -> Converting via Chrome headless...")
    result = subprocess.run([
        "google-chrome", "--headless", "--no-sandbox", "--disable-gpu",
        "--disable-software-rasterizer",
        f"--print-to-pdf={output_pdf}",
        "--print-to-pdf-no-header",
        f"file://{html_path}",
    ], capture_output=True, text=True, timeout=120)

    Path(html_path).unlink(missing_ok=True)

    if not output_pdf.exists():
        print("Chrome error:", result.stderr[-500:])
        sys.exit(1)

    size = output_pdf.stat().st_size / 1024
    print(f"PDF saved : {output_pdf}")
    print(f"Size      : {size:.1f} KB")


if __name__ == "__main__":
    main()
