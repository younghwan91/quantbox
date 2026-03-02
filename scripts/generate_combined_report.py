"""Combined QuantBox R40 × v13.0 Strategy Report — single impactful PDF.

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
    / "quantbox_r40_v13_combined.pdf"
)

# ── Chart data ─────────────────────────────────────────────────────────────────
R40_MONTHLY = {
    2020: [957,-37,5,63,70,40,83,57,-34,6,89,-46],
    2021: [-38,83,58,1,37,0,102,92,-19,-36,-8,70],
    2022: [45,-23,16,179,14,-37,5,16,0,3,13,364],
    2023: [13,74,58,12,42,-47,7,-25,45,17,-26,98],
    2024: [-33,46,9,-13,62,-45,-30,386,-32,-15,93,122],
    2025: [100,-23,65,81,-42,127,16,20,-36,21,-57,-8],
}
R40_ANNUAL   = {2020:4838,2021:749,2022:3036,2023:54,2024:2606,2025:109}
R40_LEVERAGE = {
    "lev":     [1,2,3,5,7.5,10],
    "ann_ret": [195,540,923,960,104,-94],
    "max_dd":  [35.3,60.0,77.2,95.8,99.9,100],
    "calmar":  [5.5,9.0,12.0,10.0,1.0,-0.9],
}
R40_DURATION = {
    "buckets": ["0-1h","1-4h","4-12h","12-24h","24-48h","48-96h","96h+"],
    "pct":     [3.3,14.9,27.1,17.9,20.7,12.0,4.1],
    "wr":      [10.7,31.6,38.2,52.3,48.2,57.1,44.1],
}
R40_TOP_SYM = {
    "sym": ["ETH","DOGE","1000SHIB","FET","OP","UNI","VET","LTC","WIF","1000PEPE"],
    "ret": [325,287,266,189,178,170,129,128,112,111],
}
R40_BOT_SYM = {
    "sym": ["HBAR","BCH","XLM","ATOM","COMP"],
    "ret": [-29,-39,-53,-54,-77],
}
R40_WIN_STREAK  = {1:284,2:128,3:63,4:28,5:10,6:4,7:7,8:2,10:1}
R40_LOSS_STREAK = {1:226,2:115,3:82,4:37,5:32,6:13,7:5,8:8,9:3,10:5,12:1,14:1}
V13_LN = {
    "ln":      [0.20,0.50,1.00,1.50,2.00,2.50,2.78,3.50,4.00,4.50],
    "ann_ret": [27.4,80.0,162,270,425,687,740,998,1200,1350],
    "max_dd":  [8.0,16.0,29.6,41.2,53.8,67.9,73.1,79.2,84.7,89.1],
    "calmar":  [3.42,5.0,5.48,6.55,7.90,10.12,10.13,12.60,14.16,15.15],
}
V13_TIERS = {
    "names":   ["보수적\n9x/10slots","중도\n16x/10slots","공격적\n25x/10slots\n(v13.0)"],
    "ann_ret": [162,367,687],
    "max_dd":  [29.6,48.0,67.9],
    "calmar":  [5.48,7.65,10.12],
}
ST = {"green":"#27ae60","red":"#e74c3c","blue":"#2980b9","gold":"#f39c12",
      "purple":"#8e44ad","gray":"#7f8c8d","dark":"#2c3e50","accent":"#1abc9c"}


def _sim_monthly(start,rate,months,trigger=150,keep=70):
    comp,wb,wt,cs,ws,wts=start,start,0.0,[],[],[]
    for _ in range(months):
        comp*=(1+rate);wb*=(1+rate)
        if wb>=trigger:wt+=wb-keep;wb=keep
        cs.append(comp);ws.append(wb);wts.append(wb+wt)
    return cs,ws,wts

def _png_b64(fig,dpi=130):
    tmp=tempfile.NamedTemporaryFile(suffix=".png",delete=False)
    fig.savefig(tmp.name,dpi=dpi,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    data=Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    return base64.b64encode(data).decode()

def _ax(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=0.25,linewidth=0.6)

# ── R40 Charts ─────────────────────────────────────────────────────────────────
def chart_annual():
    years=sorted(R40_ANNUAL.keys());rets=[R40_ANNUAL[y] for y in years]
    colors=[ST["green"] if r>0 else ST["red"] for r in rets]
    fig,ax=plt.subplots(figsize=(8,4))
    bars=ax.bar([str(y) for y in years],rets,color=colors,edgecolor="white",width=0.6)
    for bar,val in zip(bars,rets):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+60,
                f"+{val:,}%",ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax.set_title("연도별 수익률 (3x Lev / 15 Slots)",fontsize=11,fontweight="bold")
    ax.set_ylabel("Annual Return (%)");ax.set_ylim(0,max(rets)*1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"{x:,.0f}%"))
    _ax(ax);fig.tight_layout();return _png_b64(fig)

def chart_monthly():
    years=sorted(R40_MONTHLY.keys())
    months_lbl=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data=np.array([R40_MONTHLY[y] for y in years],dtype=float)
    fig,ax=plt.subplots(figsize=(13,4))
    im=ax.imshow(np.clip(data,-100,200),cmap="RdYlGn",aspect="auto",vmin=-100,vmax=200)
    ax.set_xticks(range(12));ax.set_xticklabels(months_lbl,fontsize=8)
    ax.set_yticks(range(len(years)));ax.set_yticklabels(years,fontsize=9)
    for yi,year in enumerate(years):
        for mi,val in enumerate(R40_MONTHLY[year]):
            c="white" if abs(val)>80 else "black"
            fw="bold" if abs(val)>=50 else "normal"
            ax.text(mi,yi,f"{val:+.0f}%",ha="center",va="center",fontsize=6.5,color=c,fontweight=fw)
    plt.colorbar(im,ax=ax,shrink=0.8,label="월수익률 (%)")
    ax.set_title("72개월 수익률 히트맵 — 6년 연속 흑자",fontsize=11,fontweight="bold")
    fig.tight_layout();return _png_b64(fig)

def chart_streak():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4))
    ws_k=[str(k) for k in R40_WIN_STREAK];ws_v=list(R40_WIN_STREAK.values())
    ax1.bar(ws_k,ws_v,color=ST["green"],edgecolor="white",width=0.6)
    ax1.set_title("연승 분포 (최대 10연승)",fontsize=10,fontweight="bold")
    ax1.set_xlabel("연승 길이");ax1.set_ylabel("횟수");_ax(ax1)
    for x,v in zip(ws_k,ws_v):ax1.text(x,v+3,str(v),ha="center",fontsize=7.5)
    ls_k=[str(k) for k in R40_LOSS_STREAK];ls_v=list(R40_LOSS_STREAK.values())
    bars=ax2.bar(ls_k,ls_v,color=ST["red"],edgecolor="white",width=0.6)
    bars[-1].set_color(ST["purple"])
    ax2.set_title("연패 분포 (최대 14연패 → compound -56%)",fontsize=10,fontweight="bold")
    ax2.set_xlabel("연패 길이");ax2.set_ylabel("횟수");_ax(ax2)
    for x,v in zip(ls_k,ls_v):ax2.text(x,v+2,str(v),ha="center",fontsize=7.5)
    fig.suptitle("승/패 연속 분포",fontsize=11,fontweight="bold")
    fig.tight_layout();return _png_b64(fig)

def chart_duration():
    d=R40_DURATION;x=np.arange(len(d["buckets"]))
    fig,ax1=plt.subplots(figsize=(9,4));ax2=ax1.twinx()
    ax1.bar(x,d["pct"],color=ST["blue"],alpha=0.65,label="비중 %",width=0.5)
    ax2.plot(x,d["wr"],"o-",color=ST["red"],linewidth=2.2,markersize=7,label="Win Rate %")
    ax2.axhline(43.4,color=ST["gray"],linewidth=1,linestyle="--",alpha=0.7)
    ax1.set_xticks(x);ax1.set_xticklabels(d["buckets"],fontsize=9)
    ax1.set_ylabel("비중 (%)",color=ST["blue"]);ax2.set_ylabel("Win Rate (%)",color=ST["red"])
    ax2.set_ylim(0,75)
    ax1.set_title("Duration별 Win Rate — 장기 보유일수록 WR 상승",fontsize=10,fontweight="bold")
    lines1,l1=ax1.get_legend_handles_labels();lines2,l2=ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2,l1+l2,loc="upper right",fontsize=8)
    _ax(ax1);fig.tight_layout();return _png_b64(fig)

def chart_symbols():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
    sy=R40_TOP_SYM["sym"];ry=R40_TOP_SYM["ret"];y1=np.arange(len(sy))
    ax1.barh(y1,ry,color=plt.cm.YlGn(np.linspace(0.4,0.9,len(sy)))[::-1],edgecolor="white")
    ax1.set_yticks(y1);ax1.set_yticklabels(sy,fontsize=8.5)
    ax1.set_xlabel("6년 Return %");ax1.set_title("Top 10 수익 심볼",fontsize=10,fontweight="bold")
    for i,v in enumerate(ry):ax1.text(v+4,i,f"+{v}%",va="center",fontsize=7.5)
    ax1.spines[["top","right"]].set_visible(False);ax1.grid(axis="x",alpha=0.25)
    sb=R40_BOT_SYM["sym"];rb=R40_BOT_SYM["ret"];y2=np.arange(len(sb))
    ax2.barh(y2,rb,color=plt.cm.OrRd(np.linspace(0.3,0.8,len(sb))),edgecolor="white")
    ax2.set_yticks(y2);ax2.set_yticklabels(sb,fontsize=8.5)
    ax2.set_xlabel("6년 Return %");ax2.set_title("Bottom 5 손실 심볼",fontsize=10,fontweight="bold")
    for i,v in enumerate(rb):ax2.text(v-2,i,f"{v}%",va="center",ha="right",fontsize=7.5)
    ax2.spines[["top","right"]].set_visible(False);ax2.grid(axis="x",alpha=0.25)
    fig.suptitle("심볼별 성과 — 50 symbols × 6년 (37/50 수익)",fontsize=11,fontweight="bold")
    fig.tight_layout();return _png_b64(fig)

def chart_leverage():
    d=R40_LEVERAGE
    fig,ax1=plt.subplots(figsize=(8,4));ax2=ax1.twinx()
    ax1.plot(d["lev"],d["ann_ret"],"o-",color=ST["blue"],linewidth=2.5,markersize=7,label="Annual Return %")
    ax2.plot(d["lev"],d["max_dd"],"s--",color=ST["red"],linewidth=2,markersize=7,label="Max DD %")
    ax2.plot(d["lev"],d["calmar"],"^:",color=ST["green"],linewidth=2,markersize=7,label="Calmar")
    ax1.axvline(3,color=ST["gold"],linewidth=2.2,linestyle="--",alpha=0.85)
    ax1.text(3.1,700,"3x 최적",fontsize=9,color=ST["gold"],fontweight="bold")
    ax1.set_xlabel("Leverage");ax1.set_ylabel("Annual Return %",color=ST["blue"])
    ax2.set_ylabel("Max DD % / Calmar");ax1.set_xticks(d["lev"]);_ax(ax1)
    ax1.set_title("레버리지 민감도 — 5x+부터 복리 붕괴",fontsize=11,fontweight="bold")
    lines1,l1=ax1.get_legend_handles_labels();lines2,l2=ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2,l1+l2,loc="upper right",fontsize=8)
    fig.tight_layout();return _png_b64(fig)

# ── V13 Charts ─────────────────────────────────────────────────────────────────
def chart_ln():
    d=V13_LN
    fig,ax1=plt.subplots(figsize=(9,4.5));ax2=ax1.twinx()
    ax1.plot(d["ln"],d["ann_ret"],"o-",color=ST["blue"],linewidth=2.5,markersize=7,label="Annual Return %")
    ax2.plot(d["ln"],d["max_dd"],"s--",color=ST["red"],linewidth=2,markersize=6,label="Max DD %")
    ax2.plot(d["ln"],d["calmar"],"^:",color=ST["green"],linewidth=2,markersize=6,label="Calmar")
    ax1.axvline(2.5,color=ST["gold"],linewidth=2.2,linestyle="--")
    ax1.text(2.52,900,"v13.0 선택점\nL/N=2.5\nLosing Year=0",fontsize=7.8,color=ST["gold"],fontweight="bold")
    ax1.axvline(4.5,color=ST["red"],linewidth=1.5,linestyle=":",alpha=0.7)
    ax1.text(4.52,200,"Losing Year\n등장",fontsize=7.5,color=ST["red"])
    ax1.set_xlabel("L/N Ratio (Leverage ÷ Slots)");ax1.set_ylabel("Annual Return %",color=ST["blue"])
    ax2.set_ylabel("Max DD % / Calmar");_ax(ax1)
    ax1.set_title("25×25 Fine Grid (625개 조합) — L/N ratio가 수익을 결정한다",fontsize=10,fontweight="bold")
    lines1,l1=ax1.get_legend_handles_labels();lines2,l2=ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2,l1+l2,loc="upper left",fontsize=8)
    fig.tight_layout();return _png_b64(fig)

def chart_tiers():
    d=V13_TIERS;x=np.arange(len(d["names"]));w=0.25
    fig,ax=plt.subplots(figsize=(9,4.5))
    b1=ax.bar(x-w,[r/10 for r in d["ann_ret"]],w,label="Return /10",color=ST["blue"],alpha=0.82)
    b2=ax.bar(x,d["max_dd"],w,label="Max DD %",color=ST["red"],alpha=0.82)
    b3=ax.bar(x+w,[c*10 for c in d["calmar"]],w,label="Calmar x10",color=ST["green"],alpha=0.82)
    for bars,vals,fmt in [(b1,d["ann_ret"],"+{:.0f}%"),(b2,d["max_dd"],"{:.0f}%"),(b3,d["calmar"],"{:.2f}")]:
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,fmt.format(v),ha="center",va="bottom",fontsize=7.5)
    ax.set_xticks(x);ax.set_xticklabels([n.replace("\n"," ") for n in d["names"]],fontsize=8.5)
    ax.set_title("리스크 티어별 성과 비교 (S=10 slots 고정)",fontsize=11,fontweight="bold")
    ax.legend(fontsize=8);_ax(ax);fig.tight_layout();return _png_b64(fig)

def chart_compound():
    months=list(range(1,15))
    comp,wb,wt=_sim_monthly(50,0.15,14,150,70)
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(months,comp,"o-",color=ST["blue"],linewidth=2.5,markersize=6,label="올복리 잔고")
    ax.plot(months,wb,"s-",color=ST["green"],linewidth=2.5,markersize=6,label="인출전략 잔고")
    ax.plot(months,wt,"^--",color=ST["accent"],linewidth=1.8,markersize=5,alpha=0.9,label="인출전략 총자산(잔고+현금)")
    ax.axhline(150,color=ST["red"],linewidth=1.2,linestyle="--",alpha=0.7)
    ax.text(0.5,153,"인출 트리거 ₩1.5억",fontsize=8,color=ST["red"],va="bottom")
    ax.axhline(70,color=ST["gray"],linewidth=1,linestyle=":",alpha=0.6)
    ax.text(0.5,72,"유지 잔고 ₩7,000만",fontsize=7.5,color=ST["gray"],va="bottom")
    ax.set_xlabel("월수");ax.set_ylabel("금액 (백만원)")
    ax.set_title("복리 vs 인출전략 — 5천만원 시작 / 월+15% 가정",fontsize=10,fontweight="bold")
    ax.set_xticks(months);ax.legend(fontsize=8);_ax(ax)
    fig.tight_layout();return _png_b64(fig)


def build_html(charts: dict[str,str], font_b64: str) -> str:
    def img(key):
        b64=charts[key]
        return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>'

    font_face=""
    if font_b64:
        font_face=f"@font-face{{font-family:'NotoSansKR';src:url('data:font/truetype;base64,{font_b64}') format('truetype');}}"

    css=f"""
{font_face}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'NotoSansKR','Noto Sans CJK KR',sans-serif;font-size:9.5pt;line-height:1.65;color:#222;
      word-wrap:break-word;overflow-wrap:break-word}}
@page{{size:A4;margin:14mm 13mm 16mm 13mm}}

/* ── Header ── */
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

/* ── KPI boxes ── */
.kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8pt;margin:14pt 0}}
.kpi{{background:#f0f7ff;border:1pt solid #c5dcf0;border-radius:5pt;padding:10pt 12pt;text-align:center}}
.kpi .val{{font-size:17pt;font-weight:900;color:#1a4a7a;line-height:1.1}}
.kpi .lbl{{font-size:7.5pt;color:#557;margin-top:3pt;font-weight:600}}
.kpi .sub{{font-size:7pt;color:#889;margin-top:2pt}}
.kpi.highlight{{background:#e8f9f0;border-color:#52c788}}
.kpi.highlight .val{{color:#1a6a3a}}
.kpi.warn{{background:#fff3e0;border-color:#f0a050}}
.kpi.warn .val{{color:#8a4a00}}

/* ── Evidence boxes ── */
.evidence-row{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8pt;margin:12pt 0}}
.ev-box{{background:#1e3a5f;color:#e8f4ff;border-radius:5pt;padding:10pt 13pt}}
.ev-box .num{{font-size:18pt;font-weight:900;color:#5dd3f7;line-height:1.1}}
.ev-box .desc{{font-size:7.5pt;color:#acd;margin-top:3pt}}

/* ── Sections ── */
.part-header{{background:#2c3e50;color:#fff;padding:10pt 16pt;margin:20pt 0 12pt;
              border-radius:4pt;page-break-before:always}}
.part-header h2{{font-size:14pt;font-weight:800;letter-spacing:0.2pt}}
.part-header .desc{{font-size:8.5pt;color:#aac;margin-top:3pt}}
.part-header:first-of-type{{page-break-before:avoid}}

h2{{font-size:13pt;color:#16355a;margin:14pt 0 6pt;border-left:4pt solid #2980b9;
    padding-left:9pt;font-weight:700}}
h3{{font-size:10.5pt;color:#333;margin:11pt 0 5pt;font-weight:700}}

/* ── Tables ── */
.tw{{overflow-x:auto;margin:8pt 0}}
table{{border-collapse:collapse;width:100%;font-size:8pt;table-layout:fixed}}
th,td{{border:0.5pt solid #bbb;padding:3pt 5pt;text-align:left;
        word-wrap:break-word;overflow-wrap:break-word}}
th{{background:#e0e8f5;font-weight:bold;font-size:7.8pt}}
tr:nth-child(even) td{{background:#f8f9fc}}
td.r{{text-align:right}}
td.c{{text-align:center}}
td.bold{{font-weight:bold;color:#1a4060}}
td.green{{color:#1a6a3a;font-weight:bold}}
td.red{{color:#8a1a1a;font-weight:bold}}
td.gold{{color:#8a5a00;font-weight:bold}}

/* ── Chart ── */
.chart{{margin:10pt 0;page-break-inside:avoid;text-align:center}}
.chart img{{width:100%;height:auto;border:0.5pt solid #dde;border-radius:3pt}}

/* ── Callout ── */
.callout{{background:#fff9e6;border-left:4pt solid #f0a800;padding:8pt 12pt;
           margin:8pt 0;border-radius:0 4pt 4pt 0;font-size:8.5pt}}
.callout strong{{color:#6a4a00}}
.callout.green{{background:#eaf7ef;border-color:#27ae60}}
.callout.green strong{{color:#1a5a30}}
.callout.red{{background:#fdecea;border-color:#e74c3c}}
.callout.red strong{{color:#6a1a1a}}
.callout.blue{{background:#e8f4fc;border-color:#2980b9}}
.callout.blue strong{{color:#1a3a5a}}

/* ── Verdict ── */
.verdict{{background:linear-gradient(135deg,#0f3460,#1a4a7a);color:#fff;
           padding:18pt 20pt;border-radius:6pt;margin:16pt 0;page-break-inside:avoid}}
.verdict h3{{font-size:12pt;color:#7dd;margin-bottom:8pt;font-weight:800}}
.verdict p{{font-size:9pt;color:#cde;line-height:1.7}}
.verdict ul{{margin-left:14pt;color:#cde;font-size:9pt}}
.verdict li{{margin:3pt 0}}

/* ── Footer ── */
.footer{{margin-top:20pt;padding-top:10pt;border-top:1pt solid #ddd;
          font-size:7.5pt;color:#999;text-align:center}}

/* ── Print ── */
@media print{{
  .part-header{{page-break-before:always}}
  .chart{{page-break-inside:avoid}}
  .verdict{{page-break-inside:avoid}}
}}
"""

    body = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8">
<title>QuantBox R40 × v13.0 종합 보고서</title>
<style>{css}</style>
</head><body>

<!-- ═══════════════════ COVER ═══════════════════ -->
<div class="cover">
  <div class="badge">CONFIDENTIAL — INTERNAL USE ONLY</div>
  <h1 style="margin-top:10pt">QuantBox 종합 전략 보고서</h1>
  <div class="subtitle">R40 Champion 심층 통계 × v13.0 Aggressive Deployment</div>
  <div class="meta">
    <span>📅 2026-03-02</span>
    <span>🔖 R40: 748630b &nbsp;|&nbsp; v13.0: e4b5831</span>
    <span>📊 50 symbols × 6.0년 (2020~2025)</span>
    <span>⚡ VR8_56_SL1_TP5_D4 @15m봉</span>
    <span>🏦 Binance USDT-M Futures</span>
  </div>
</div>

<!-- KPI -->
<div class="kpi-grid">
  <div class="kpi highlight">
    <div class="val">100.0%</div>
    <div class="lbl">P(PF ≥ 1.0)</div>
    <div class="sub">5,000회 Bootstrap</div>
  </div>
  <div class="kpi highlight">
    <div class="val">0회</div>
    <div class="lbl">Losing Year (6년)</div>
    <div class="sub">2020~2025 전략 흑자</div>
  </div>
  <div class="kpi">
    <div class="val">p&lt;0.000001</div>
    <div class="lbl">통계적 유의성</div>
    <div class="sub">t = 5.42</div>
  </div>
  <div class="kpi warn">
    <div class="val">67.9%</div>
    <div class="lbl">v13.0 Max Drawdown</div>
    <div class="sub">$1,983 → $636</div>
  </div>
  <div class="kpi">
    <div class="val">+687%</div>
    <div class="lbl">v13.0 연간 수익률</div>
    <div class="sub">25x / 10 slots</div>
  </div>
  <div class="kpi">
    <div class="val">10.12</div>
    <div class="lbl">Calmar Ratio</div>
    <div class="sub">헤지펀드 기준 우수</div>
  </div>
  <div class="kpi">
    <div class="val">1.285</div>
    <div class="lbl">Profit Factor</div>
    <div class="sub">CI [1.173, 1.405]</div>
  </div>
  <div class="kpi">
    <div class="val">+27.4%</div>
    <div class="lbl">실제 연간 수익 (1/N 배분)</div>
    <div class="sub">과대 계산 교정 후</div>
  </div>
</div>

<!-- Evidence -->
<div class="evidence-row">
  <div class="ev-box">
    <div class="num">t = 5.42</div>
    <div class="desc">양측 t검정 p &lt; 0.000001<br>Wilcoxon도 비모수 유의</div>
  </div>
  <div class="ev-box">
    <div class="num">PF 1.285</div>
    <div class="desc">5,000회 Bootstrap<br>PF &lt; 1.0 = 단 0번</div>
  </div>
  <div class="ev-box">
    <div class="num">2,280</div>
    <div class="desc">총 트레이드 (6년)<br>200-trade 윈도우 PF&lt;1 = 0%</div>
  </div>
</div>

<!-- ═══════════════════ PART 1 ═══════════════════ -->
<div class="part-header">
  <h2>Part 1 — R40 전략 심층 분석</h2>
  <div class="desc">VR8_56_SL1_TP5_D4 @3x Leverage / 15 Slots · 백테스트 기반 엣지 검증</div>
</div>

<h2>1. 핵심 성과 지표</h2>
{img("annual")}
<div class="tw"><table>
<tr><th>Metric</th><th class="r">3x @15 slots</th><th class="r">1x @15 slots</th></tr>
<tr><td>연간 수익률</td><td class="r bold green">+922.5%</td><td class="r">+195.4%</td></tr>
<tr><td>Max Drawdown</td><td class="r warn">77.2%</td><td class="r">35.3%</td></tr>
<tr><td>Calmar Ratio</td><td class="r bold">12.0</td><td class="r">5.5</td></tr>
<tr><td>Sharpe / Sortino</td><td class="r">2.21 / 5.99</td><td class="r">2.21 / 5.99</td></tr>
<tr><td>Profit Factor</td><td class="r bold">1.285</td><td class="r">1.285</td></tr>
<tr><td>Win Rate</td><td class="r">43.4%</td><td class="r">43.4%</td></tr>
<tr><td>Risk:Reward</td><td class="r">1.67</td><td class="r">1.67</td></tr>
<tr><td>EV per trade</td><td class="r bold green">+98.0bp</td><td class="r">+32.7bp</td></tr>
<tr><td>t-statistic</td><td class="r">5.42</td><td class="r">5.42</td></tr>
<tr><td>Total Trades</td><td class="r">2,280</td><td class="r">2,280</td></tr>
</table></div>
<div class="callout blue"><strong>참고:</strong> Sharpe/Sortino/PF/WR은 레버리지와 무관(PnL 비례 스케일). 레버리지는 compound effect를 통해 연간 수익률에만 폭발적으로 작용.</div>

<h2>2. 6년 × 72개월 수익률 히트맵</h2>
{img("monthly")}
<div class="tw"><table>
<tr><th>Year</th><th class="r">최고월</th><th class="r">최저월</th><th class="r">연간 합계</th><th>특징</th></tr>
<tr><td>2020</td><td class="r green">+957% (1월)</td><td class="r red">-46% (12월)</td><td class="r bold">+4,838%</td><td>코로나 변동성 폭발</td></tr>
<tr><td>2021</td><td class="r green">+102% (7월)</td><td class="r red">-38% (1월)</td><td class="r bold">+749%</td><td>Bull run</td></tr>
<tr><td>2022</td><td class="r green">+364% (12월)</td><td class="r red">-37% (6월)</td><td class="r bold">+3,036%</td><td>LUNA/FTX 변동성</td></tr>
<tr><td>2023</td><td class="r green">+98% (12월)</td><td class="r red">-47% (6월)</td><td class="r">+54%</td><td class="gold">취약 구간 — 저변동</td></tr>
<tr><td>2024</td><td class="r green">+386% (8월)</td><td class="r red">-45% (6월)</td><td class="r bold">+2,606%</td><td>BTC Halving rally</td></tr>
<tr><td>2025</td><td class="r green">+127% (6월)</td><td class="r red">-57% (11월)</td><td class="r">+109%</td><td class="gold">취약 구간</td></tr>
</table></div>

<h2>3. 연도별 상세 통계</h2>
<div class="tw"><table>
<tr><th>Year</th><th class="r">Trades</th><th class="r">Return</th><th class="r">Max DD</th><th class="r">PF</th><th class="r">Sharpe</th><th class="r">WR</th></tr>
<tr><td>2020</td><td class="r">630</td><td class="r green bold">+4,838%</td><td class="r">70.7%</td><td class="r">1.29</td><td class="r">2.85</td><td class="r">39.4%</td></tr>
<tr><td>2021</td><td class="r">202</td><td class="r">+749%</td><td class="r">62.0%</td><td class="r">1.41</td><td class="r">2.30</td><td class="r">46.5%</td></tr>
<tr><td>2022</td><td class="r">414</td><td class="r green bold">+3,036%</td><td class="r">65.9%</td><td class="r">1.36</td><td class="r">2.82</td><td class="r">45.7%</td></tr>
<tr><td class="gold">2023</td><td class="r">374</td><td class="r">+54%</td><td class="r">66.3%</td><td class="r gold">1.13 ⚠</td><td class="r">1.04</td><td class="r">41.2%</td></tr>
<tr><td>2024</td><td class="r">330</td><td class="r green bold">+2,606%</td><td class="r">77.2%</td><td class="r">1.41</td><td class="r">2.89</td><td class="r">48.5%</td></tr>
<tr><td class="gold">2025</td><td class="r">330</td><td class="r">+109%</td><td class="r">74.2%</td><td class="r gold">1.16 ⚠</td><td class="r">1.24</td><td class="r">43.9%</td></tr>
</table></div>
<div class="callout"><strong>취약 구간:</strong> 2023·2025는 PF 1.13~1.16 — "저변동성 연도". 더 나쁜 해가 올 수 있다는 점을 인지해야 한다.</div>

<h2>4. 승/패 연속 & Duration 분석</h2>
{img("streak")}
{img("duration")}
<div class="tw"><table>
<tr><th></th><th class="r">Win Streak</th><th class="r">Loss Streak</th></tr>
<tr><td>최대</td><td class="r green bold">10연승</td><td class="r red bold">14연패</td></tr>
<tr><td>평균</td><td class="r">1.9</td><td class="r">2.4</td></tr>
<tr><td>최대 연속 손실 (compound)</td><td class="r">—</td><td class="r red bold">-56.2%</td></tr>
</table></div>
<div class="callout red"><strong>14연패 시나리오:</strong> compound -56.2% 손실. $1,983 기준 → $869. 그래도 6년에 딱 1회 발생.</div>

<h2>5. 심볼별 성과</h2>
{img("symbols")}

<h2>6. 레버리지 민감도 — 왜 3x인가</h2>
{img("leverage")}
<div class="tw"><table>
<tr><th>Leverage</th><th class="r">AnnRet</th><th class="r">Max DD</th><th class="r">Calmar</th><th class="c">Losing Years</th></tr>
<tr><td>1x</td><td class="r">+195%</td><td class="r">35.3%</td><td class="r">5.5</td><td class="c">0</td></tr>
<tr><td>2x</td><td class="r">+540%</td><td class="r">60.0%</td><td class="r">9.0</td><td class="c">0</td></tr>
<tr><td class="bold">3x ★</td><td class="r green bold">+923%</td><td class="r">77.2%</td><td class="r bold">12.0</td><td class="c green bold">0</td></tr>
<tr><td>5x</td><td class="r">+960%</td><td class="r red">95.8%</td><td class="r">10.0</td><td class="c red">2</td></tr>
<tr><td>7.5x</td><td class="r">+104%</td><td class="r red">99.9%</td><td class="r red">1.0</td><td class="c red">2</td></tr>
<tr><td>10x</td><td class="r red">-94%</td><td class="r red">100%</td><td class="r red">-0.9</td><td class="c red">4</td></tr>
</table></div>

<h2>7. Bootstrap 신뢰구간 & 통계적 유의성</h2>
<div class="tw"><table>
<tr><th>Metric</th><th class="r">Median</th><th>95% CI</th><th>비고</th></tr>
<tr><td>Profit Factor</td><td class="r bold">1.284</td><td>[1.173, 1.405]</td><td class="green bold">P(PF≥1.0) = 100.0%</td></tr>
<tr><td>Win Rate</td><td class="r">43.4%</td><td>[41.4%, 45.4%]</td><td></td></tr>
<tr><td>EV (bp)</td><td class="r green">+97.8</td><td>[+62.0, +133.1]</td><td>CI 하한도 양수</td></tr>
<tr><td>Sharpe</td><td class="r">2.21</td><td>[1.41, 3.00]</td><td></td></tr>
<tr><td>Max DD</td><td class="r">87.5%</td><td>[71.9%, 98.0%]</td><td>백테스트 기준</td></tr>
</table></div>
<div class="tw"><table>
<tr><th>통계 검정</th><th class="r">통계량</th><th class="r">p-value</th><th>결론</th></tr>
<tr><td>t-test (mean > 0)</td><td class="r">t = 5.42</td><td class="r green bold">&lt; 0.000001</td><td>엣지 존재 확실</td></tr>
<tr><td>Wilcoxon signed-rank</td><td class="r">W = 928,145</td><td class="r green bold">&lt; 0.000001</td><td>비모수적으로도 유의</td></tr>
<tr><td>200-trade Rolling PF&lt;1</td><td class="r">—</td><td class="r green bold">0%</td><td>장기 안정적 엣지</td></tr>
</table></div>

<h2>8. Lookahead Bias 감사 결과</h2>
<div class="tw"><table>
<tr><th>#</th><th>항목</th><th class="c">결과</th><th>상세</th></tr>
<tr><td>1</td><td>Signal Truncation (300 checkpoints)</td><td class="c green bold">CLEAN</td><td>VR/Momentum/Signal 모두 backward-looking</td></tr>
<tr><td>2</td><td>Entry Price 체크</td><td class="c green bold">CLEAN</td><td>close[i] 진입, SL/TP는 i+1부터</td></tr>
<tr><td>3</td><td>Intra-bar resolution</td><td class="c green bold">CLEAN</td><td>SL 우선 체크 (보수적)</td></tr>
<tr><td>4</td><td>Compound Equity Model</td><td class="c red bold">MAJOR</td><td>전체 자본 compound → 과대 계산</td></tr>
</table></div>
<div class="callout red"><strong>⚠ Compound Model 교정:</strong> 백테스트 +922.5%는 전체 자본에 각 trade PnL을 적용한 값. 실제 1/15 고정배분 시 <strong>연 +27.4%</strong>가 현실적 기대치. Per-trade 지표(PF, WR, t-stat)는 교정 불필요.</div>


<!-- ═══════════════════ PART 2 ═══════════════════ -->
<div class="part-header">
  <h2>Part 2 — v13.0 Aggressive Deployment</h2>
  <div class="desc">25x Leverage / 10 Slots (L/N=2.5) · 25×25 Fine Grid 최적화 결과 · 운용 결정</div>
</div>

<h2>1. v12.9 → v13.0 설정 변경</h2>
<div class="tw"><table>
<tr><th>Setting</th><th class="c">v12.9 (이전)</th><th class="c">v13.0 (현재)</th><th>변경 이유</th></tr>
<tr><td>Leverage</td><td class="c">3.0x</td><td class="c bold">25.0x</td><td>25×25 grid 최적화 결과</td></tr>
<tr><td>Max Slots</td><td class="c">15</td><td class="c bold">10</td><td>L/N=2.5 최적 비율</td></tr>
<tr><td>L/N Ratio</td><td class="c">0.20</td><td class="c green bold">2.50</td><td>Calmar 최대화 zone</td></tr>
<tr><td>Per-slot Notional ($1,983)</td><td class="c">$396</td><td class="c bold">$4,957</td><td>잔고/슬롯 × 레버리지</td></tr>
</table></div>

<h2>2. 25×25 Fine Grid — L/N Ratio가 수익을 결정한다</h2>
{img("ln")}
<div class="tw"><table>
<tr><th>L/N Ratio</th><th class="r">AnnRet</th><th class="r">Max DD</th><th class="r">Calmar</th><th class="c">Losing Years</th></tr>
<tr><td>0.20 (v12.9)</td><td class="r">+27.4%</td><td class="r">8.0%</td><td class="r">3.42</td><td class="c">0</td></tr>
<tr><td>1.00</td><td class="r">+162%</td><td class="r">29.6%</td><td class="r">5.48</td><td class="c">0</td></tr>
<tr><td>2.00</td><td class="r">+425%</td><td class="r">53.8%</td><td class="r">7.90</td><td class="c">0</td></tr>
<tr><td class="bold">2.50 (v13.0) ★</td><td class="r green bold">+687%</td><td class="r">67.9%</td><td class="r bold">10.12</td><td class="c green bold">0</td></tr>
<tr><td>2.78</td><td class="r">+740%</td><td class="r">73.1%</td><td class="r">10.13</td><td class="c green">0</td></tr>
<tr><td>4.50</td><td class="r">+1,350%</td><td class="r red">89.1%</td><td class="r">15.15</td><td class="c red">1</td></tr>
<tr><td>5.00+</td><td class="r red">급감</td><td class="r red">95%+</td><td class="r red">—</td><td class="c red">2+</td></tr>
</table></div>
<div class="callout green"><strong>L/N = 2.5 선택 근거:</strong> 625개 조합 중 Losing Year = 0을 유지하면서 Calmar가 최대화되는 경계점. 같은 L/N이면 (S=5,L=10)과 (S=10,L=20)이 거의 동일 — 슬롯과 레버리지가 아니라 비율이 중요하다는 핵심 발견.</div>

<h2>3. Adaptive vs Fixed — 4개 모델 토너먼트</h2>
<div class="tw"><table>
<tr><th>Model</th><th>최적 구성</th><th class="r">Calmar</th><th class="r">AnnRet</th><th class="r">Max DD</th><th>결론</th></tr>
<tr><td class="bold">Fixed L/N ★</td><td>S=9, L=25x</td><td class="r green bold">10.13</td><td class="r">+740%</td><td class="r">73.1%</td><td class="green bold">승자</td></tr>
<tr><td>Adaptive-Concurrency</td><td>—</td><td class="r red">BLOWUP</td><td class="r red">—</td><td class="r red">100%</td><td class="red">실패</td></tr>
<tr><td>Adaptive-DD</td><td>S=10, L=25x</td><td class="r">5.06</td><td class="r">+344%</td><td class="r">67.9%</td><td>수익 절반</td></tr>
<tr><td>Target L/N=2.0</td><td>S=10, L=25x</td><td class="r">7.65</td><td class="r">+367%</td><td class="r">48.0%</td><td>고정 대비 열위</td></tr>
</table></div>
<div class="callout"><strong>결론:</strong> 적응형 레버리지는 고정보다 나쁘다. DD 구간에서 레버리지를 줄이면 복구 기회를 놓친다. 복잡성 추가가 오히려 성과를 악화시킨다 (KISS 원칙).</div>

<h2>4. 리스크 티어별 선택지</h2>
{img("tiers")}
<div class="tw"><table>
<tr><th>Tier</th><th>구성</th><th class="r">AnnRet</th><th class="r">Max DD</th><th class="r">Calmar</th><th class="c">Losing Yr</th><th class="r">슬롯 노셔널</th></tr>
<tr><td>보수적</td><td>S=10, L=9x</td><td class="r">+162%</td><td class="r">29.6%</td><td class="r">5.48</td><td class="c">0</td><td class="r">$1,785</td></tr>
<tr><td>중도</td><td>S=10, L=16x</td><td class="r">+367%</td><td class="r">48.0%</td><td class="r">7.65</td><td class="c">0</td><td class="r">$3,173</td></tr>
<tr><td class="bold">공격적 (v13.0) ★</td><td class="bold">S=10, L=25x</td><td class="r green bold">+687%</td><td class="r warn">67.9%</td><td class="r bold">10.12</td><td class="c green bold">0</td><td class="r">$4,957</td></tr>
<tr><td>YOLO</td><td>S=9, L=25x</td><td class="r">+740%</td><td class="r red">73.1%</td><td class="r">10.13</td><td class="c green">0</td><td class="r">$5,508</td></tr>
</table></div>

<h2>5. 리스크 시나리오</h2>
<div class="tw"><table>
<tr><th>시나리오</th><th>영향 (L/N=2.5)</th><th class="c">6년 빈도</th></tr>
<tr><td>14연패 (최대 기록)</td><td class="red">compound -56.2% → $1,983 → $869</td><td class="c">1회</td></tr>
<tr><td>10연패</td><td>compound -37.8% → $1,983 → $1,233</td><td class="c">~2회</td></tr>
<tr><td>DD > 50%</td><td>연 4~5회 예상</td><td class="c">빈번</td></tr>
<tr><td>2023형 저변동 연도</td><td>연수익 ~80% (PF 1.13)</td><td class="c">~2회/6년</td></tr>
</table></div>
<div class="tw" style="margin-top:8pt"><table>
<tr><th>항목</th><th>모니터링 기준</th><th>액션</th></tr>
<tr><td>DD</td><td>&gt; 70%</td><td>긴급 점검</td></tr>
<tr><td>30일 PF</td><td>&lt; 0.9</td><td>경고 — 60일까지 관찰</td></tr>
<tr><td>60일 PF</td><td>&lt; 1.0</td><td>레버리지 축소 (→ 중도 Tier)</td></tr>
<tr><td>Trade freq</td><td>&lt; 0.5/day (30일 평균)</td><td>시장 구조 변화 의심</td></tr>
</table></div>

<h2>6. 퇴각 3단계 계획</h2>
<div class="tw"><table>
<tr><th>단계</th><th>트리거</th><th>설정 변경</th><th class="r">AnnRet</th><th class="r">Max DD</th></tr>
<tr><td class="bold">1단계</td><td>DD &gt; 70% 또는 60일 PF &lt; 0.95</td><td>L=16x, S=10</td><td class="r">~+270%</td><td class="r">~41%</td></tr>
<tr><td class="bold">2단계</td><td>DD &gt; 50% (축소 후에도)</td><td>L=9x, S=10</td><td class="r">~+162%</td><td class="r">~30%</td></tr>
<tr><td class="bold">3단계</td><td>근본적 의심</td><td>L=3x, S=15 (v12.9 복귀)</td><td class="r">~+27%</td><td class="r">~8%</td></tr>
</table></div>

<h2>7. 복리 vs 인출전략 — 용량 한계와 최적 운용</h2>
{img("compound")}
<div class="tw"><table>
<tr><th>잔고</th><th class="r">슬롯 노셔널</th><th>하위 심볼 영향</th><th class="c">판정</th></tr>
<tr><td>₩5,000만 ($35K)</td><td class="r">$86K</td><td>전혀 없음</td><td class="c green bold">안전</td></tr>
<tr><td><strong>₩1.5억 ($100K)</strong></td><td class="r"><strong>$258K</strong></td><td>경계선</td><td class="c gold bold">인출 트리거</td></tr>
<tr><td>₩2억 ($140K)</td><td class="r">$345K</td><td>슬리피지 시작</td><td class="c red bold">위험</td></tr>
</table></div>
<div class="callout green"><strong>최적 인출 전략 (₩5,000만 시작):</strong> ₩1.5억 도달 시 ₩8,000만 인출 → ₩7,000만 유지. 1년 기준 총 ~₩2.4억 인출 + ₩7,000만 잔고 = <strong>₩3.1억</strong> (원금 6.2배). 올복리(₩2.56억) 대비 총자산이 높고 심리적 부담도 낮다.</div>

<h2>8. 원금별 1년 수익 예상 (현실적 60% 시나리오)</h2>
<div class="tw"><table>
<tr><th>원금</th><th class="r">1년 후 (현실적)</th><th class="r">DD 최저점</th><th class="r">슬롯 노셔널</th></tr>
<tr><td>₩1,000만</td><td class="r">₩5,100만</td><td class="r red">₩320만</td><td class="r">$17K</td></tr>
<tr><td>₩5,000만</td><td class="r bold">₩2.56억</td><td class="r red">₩1,600만</td><td class="r">$86K</td></tr>
<tr><td>₩1억</td><td class="r bold">₩5.1억</td><td class="r red">₩3,200만</td><td class="r">$172K</td></tr>
</table></div>

<!-- ═══════════════════ VERDICT ═══════════════════ -->
<div class="verdict">
  <h3>최종 판단 — 돌려야 하는가?</h3>
  <p><strong style="color:#7ff;font-size:11pt">YES — 조건부</strong></p>
  <ul>
    <li>엣지는 통계적으로 확실하다 (t=5.42, Bootstrap P(PF≥1.0)=100.0%). 노이즈가 아닌 진짜 비효율.</li>
    <li>6년 연속 흑자 — 코로나·LUNA·FTX·저변동 모두 통과. Regime-independent edge.</li>
    <li>L/N=2.5는 625개 조합에서 과학적으로 선택한 Calmar 최대화 경계점.</li>
    <li>$1,983이 전 재산이 아닐 것. DD 68%($634)에서도 시스템을 끄지 않을 것.</li>
    <li>최소 3개월(~90거래일) 관찰 후 평가. 수익 기대치는 +687%가 아니라 <strong style="color:#ffe">+300~400%</strong>.</li>
  </ul>
  <p style="margin-top:8pt;color:#fdd"><strong>핵심 리스크:</strong> 엣지가 아니라 레버리지와 심리. DD 60%에서 "끄면 안 된다"는 것을 미리 각오해야 한다.</p>
</div>

<div class="footer">
  QuantBox 종합 보고서 — R40: 748630b · v13.0: e4b5831 · Generated: 2026-03-02<br>
  VR8_56_SL1_TP5_D4 @25x leverage / 10 slots · 50 symbols × 6yr · Binance USDT-M Futures
</div>
</body></html>"""
    return body


def main():
    print("Generating QuantBox Combined Report…")
    c = {}
    print("  chart: annual…");   c["annual"]   = chart_annual()
    print("  chart: monthly…");  c["monthly"]  = chart_monthly()
    print("  chart: streak…");   c["streak"]   = chart_streak()
    print("  chart: duration…"); c["duration"] = chart_duration()
    print("  chart: symbols…");  c["symbols"]  = chart_symbols()
    print("  chart: leverage…"); c["leverage"] = chart_leverage()
    print("  chart: ln…");       c["ln"]       = chart_ln()
    print("  chart: tiers…");    c["tiers"]    = chart_tiers()
    print("  chart: compound…"); c["compound"] = chart_compound()
    print(f"  {len(c)} charts embedded")

    font_b64 = ""
    if _font_path.exists():
        font_b64 = base64.b64encode(_font_path.read_bytes()).decode()
        print(f"  font: {_font_path.name}")

    html = build_html(c, font_b64)

    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", encoding="utf-8", delete=False) as f:
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
