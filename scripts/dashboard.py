"""QuantBox Terminal — Bloomberg-style live dashboard.

Usage:
    uv run streamlit run scripts/dashboard.py --server.headless true
Access:
    http://localhost:8501
"""
import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

DB = Path(__file__).parent.parent / "data" / "journal.sqlite"
STATE = Path(__file__).parent.parent / "data" / "state" / "scalper_state.json"
REAL = "ABS(pnl_pct) < 10 AND symbol NOT LIKE 'TEST%' AND symbol NOT LIKE '%:%'"
REFRESH = 30
POSITIONS_TTL = 10

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantBox Terminal",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Bloomberg CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Terminal base */
html, body, [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #E0E0E0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 16px;
}
[data-testid="stApp"] > div { background-color: #000000 !important; }
[data-testid="stSidebar"]   { display: none; }
.block-container {
    padding: 4px 16px !important;
    max-width: 1280px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header   { visibility: hidden; }
[data-testid="stToolbar"]   { display: none; }
[data-testid="stDecoration"]{ display: none; }

hr { border-color: #2a2a2a !important; margin: 6px 0 !important; }

/* ── HEADER BAR ── */
.ticker {
    border-top: 2px solid #FF8C00;
    border-bottom: 1px solid #222;
    padding: 7px 12px;
    font-size: 0.97rem;
    letter-spacing: 0.03em;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ticker .logo { color: #FF8C00; font-weight: bold; font-size: 1.05rem; letter-spacing: 0.22em; }
.ticker .sep  { color: #2a2a2a; margin: 0 14px; }

/* ── TAB OVERRIDES ── */
[data-testid="stTabs"] button {
    background: #000 !important;
    color: #888 !important;
    border: none !important;
    border-bottom: 2px solid #1e1e1e !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    padding: 6px 18px !important;
    border-radius: 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF8C00 !important;
    border-bottom: 2px solid #FF8C00 !important;
    background: #000 !important;
}
[data-testid="stTabs"] button:hover { color: #E0E0E0 !important; }
[data-testid="stTabsContent"] { padding-top: 8px !important; }

/* ── SECTION LABELS ── */
.bb-header {
    color: #FF8C00; font-size: 0.88rem; font-weight: bold;
    letter-spacing: 0.16em; border-bottom: 1px solid #1e1e1e;
    padding-bottom: 3px; margin: 10px 0 5px 0;
}

/* ── STAT STRIP ── */
.statstrip { display: flex; gap: 0; border: 1px solid #1e1e1e; margin-bottom: 10px; }
.stat { flex: 1; padding: 8px 14px; border-right: 1px solid #1e1e1e; }
.stat:last-child { border-right: none; }
.stat .lbl { color: #888; font-size: 0.80rem; letter-spacing: 0.12em; margin-bottom: 3px; }
.stat .val { color: #E8E8E8; font-size: 1.30rem; font-weight: bold; }
.stat .sub { color: #777; font-size: 0.80rem; margin-top: 2px; }

/* ── MINI STAT GRID (Analytics) ── */
.mini-grid { display: flex; flex-wrap: wrap; gap: 0; border: 1px solid #1e1e1e; margin-bottom: 8px; }
.mini-stat { flex: 1 0 16%; padding: 7px 12px; border-right: 1px solid #1e1e1e; min-width: 120px; }
.mini-stat:last-child { border-right: none; }
.mini-stat .lbl { color: #888; font-size: 0.75rem; letter-spacing: 0.10em; }
.mini-stat .val { font-size: 1.05rem; font-weight: bold; }

/* ── TABLES ── */
.bb-table { width: 100%; border-collapse: collapse; font-size: 0.95rem; font-family: 'Courier New', monospace; }
.bb-table th { color: #999; border-bottom: 1px solid #1e1e1e; padding: 4px 10px;
               text-align: right; font-size: 0.80rem; letter-spacing: 0.10em; font-weight: normal; }
.bb-table th:first-child { text-align: left; }
.bb-table td { padding: 5px 10px; border-bottom: 1px solid #111; color: #D8D8D8; text-align: right; }
.bb-table td:first-child { text-align: left; }
.bb-table tr:last-child td { border-bottom: none; }
.bb-table tr:hover td { background: #0a0a0a; }

/* Color tokens */
.pos  { color: #00C853 !important; font-weight: bold; }
.neg  { color: #D50000 !important; font-weight: bold; }
.neu  { color: #999 !important; }
.sym  { color: #FFFFFF !important; font-weight: bold; letter-spacing: 0.04em; }
.tag-vr { color: #4A90D9; font-size: 0.88rem; }
.tag-vr-s { color: #C084FC; font-size: 0.88rem; }
.tag-vr-d { color: #4A90D9; font-size: 0.88rem; }
.tag-ma { color: #C8960C; font-size: 0.88rem; }
.long-tag  { color: #00C853; font-size: 0.88rem; letter-spacing: 0.06em; }
.short-tag { color: #D50000; font-size: 0.88rem; letter-spacing: 0.06em; }

/* ── BAR CHART ── */
.bar-row   { display: flex; align-items: center; margin: 4px 0; font-size: 0.92rem; }
.bar-label { color: #999; width: 200px; letter-spacing: 0.04em; }
.bar-fill  { color: #FF8C00; letter-spacing: -2px; }
.bar-stat  { color: #D0D0D0; margin-left: 10px; font-size: 0.88rem; }

.tbl-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }

/* ── MOBILE ── */
@media (max-width: 600px) {
    html, body, [data-testid="stApp"] { font-size: 14px; }
    .block-container { padding: 2px 8px !important; max-width: 100% !important; margin: 0 !important; }
    .ticker { flex-wrap: wrap; gap: 4px 0; padding: 6px 8px; font-size: 0.88rem; }
    .ticker .logo { width: 100%; margin-bottom: 2px; }
    .ticker .sep  { display: none; }
    .statstrip { flex-wrap: wrap; }
    .stat { flex: 0 0 50%; border-right: none; border-bottom: 1px solid #1e1e1e; box-sizing: border-box; }
    .stat:nth-child(odd)  { border-right: 1px solid #1e1e1e; }
    .stat:nth-last-child(-n+2) { border-bottom: none; }
    .stat .val { font-size: 1.10rem; }
    .bar-label { width: 110px; }
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly dark theme ────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#000000",
    plot_bgcolor="#050505",
    font=dict(family="Courier New, monospace", color="#D0D0D0", size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#111", showgrid=True, zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#111", showgrid=True, zeroline=False, tickfont=dict(size=10)),
    legend=dict(bgcolor="#000", bordercolor="#1e1e1e", borderwidth=1, font=dict(size=10)),
    hoverlabel=dict(bgcolor="#111", bordercolor="#333", font=dict(family="Courier New", size=11)),
)


# ─── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client():
    return Client(
        api_key=os.environ.get("BINANCE_API_KEY", ""),
        api_secret=os.environ.get("BINANCE_API_SECRET", ""),
    )


@st.cache_data(ttl=28, show_spinner=False)
def fetch_account() -> dict:
    try:
        acct = get_client().futures_account()
        return {
            "wallet": float(acct["totalWalletBalance"]),
            "unrealized": float(acct["totalUnrealizedProfit"]),
            "margin": float(acct["totalInitialMargin"]),
            "available": float(acct["availableBalance"]),
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=POSITIONS_TTL, show_spinner=False)
def fetch_positions() -> list[dict]:
    try:
        rows = []
        for p in get_client().futures_position_information():
            amt = float(p["positionAmt"])
            if abs(amt) == 0:
                continue
            entry = float(p["entryPrice"])
            mark  = float(p["markPrice"])
            upnl  = float(p["unRealizedProfit"])
            pct   = (mark - entry) / entry * 100 if amt > 0 else (entry - mark) / entry * 100
            rows.append({
                "symbol": p["symbol"],
                "side":   p["positionSide"],
                "entry":  entry,
                "mark":   mark,
                "pct":    pct,
                "upnl":   upnl,
                "notional": abs(float(p["notional"])),
            })
        return sorted(rows, key=lambda x: x["pct"], reverse=True)
    except Exception as e:
        st.error(f"positions error: {e}")
        return []


def load_state_bars() -> dict:
    if not STATE.exists():
        return {}
    try:
        s = json.load(open(STATE))
        return {sym: pos.get("bars_held", 0) for sym, pos in s.get("positions", {}).items()}
    except Exception:
        return {}


@st.cache_data(ttl=28, show_spinner=False)
def load_trades() -> pd.DataFrame:
    if not DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        f"SELECT * FROM trades WHERE {REAL} ORDER BY id", conn
    )
    conn.close()
    if not df.empty and "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True, errors="coerce")
    return df


def win_streak(df: pd.DataFrame) -> tuple[str, int]:
    if df.empty:
        return ("–", 0)
    seq = df["pnl_usd"].values[::-1]
    current = "W" if seq[0] > 0 else "L"
    count = 0
    for v in seq:
        if (v > 0) == (current == "W"):
            count += 1
        else:
            break
    return (current, count)


def unicode_bar(value: float, max_val: float, width: int = 16) -> str:
    if max_val == 0:
        return ""
    frac = min(abs(value) / max_val, 1.0)
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _pf(df: pd.DataFrame) -> float:
    """Profit factor = gross wins / gross losses."""
    if df.empty:
        return 0.0
    wins   = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    losses = df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum()
    return wins / abs(losses) if losses != 0 else float("inf")


def _sharpe(df: pd.DataFrame) -> float:
    """Simplified daily Sharpe (annualised, 365d)."""
    if df.empty or len(df) < 2:
        return 0.0
    pnl = df["pnl_pct"].values
    mu  = np.mean(pnl)
    sig = np.std(pnl, ddof=1)
    if sig == 0:
        return 0.0
    return float(mu / sig * np.sqrt(len(pnl)))


def _max_drawdown(df: pd.DataFrame) -> float:
    """Max drawdown on equity curve (balance_after) in %."""
    if df.empty or "balance_after" not in df.columns:
        return 0.0
    bal = df["balance_after"].dropna().values
    if len(bal) == 0:
        return 0.0
    peak = np.maximum.accumulate(bal)
    dd   = (peak - bal) / peak * 100
    return float(dd.max())


# ─── Chart builders ───────────────────────────────────────────────────────────
def chart_equity(df: pd.DataFrame) -> go.Figure | None:
    """Equity curve: balance_after over time."""
    if df.empty or "balance_after" not in df.columns:
        return None
    d = df.dropna(subset=["balance_after", "exit_time"]).sort_values("exit_time")
    if d.empty:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["exit_time"], y=d["balance_after"],
        mode="lines",
        line=dict(color="#FF8C00", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,140,0,0.06)",
        name="Balance",
        hovertemplate="<b>%{x|%m/%d %H:%M}</b><br>$%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="EQUITY CURVE", font=dict(color="#FF8C00", size=11), x=0))
    return fig


def chart_pnl_bars(df: pd.DataFrame) -> go.Figure | None:
    """Individual trade P&L bar chart."""
    if df.empty:
        return None
    d = df.sort_values("id")
    colors = ["#00C853" if v > 0 else "#D50000" for v in d["pnl_usd"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(d) + 1)),
        y=d["pnl_usd"],
        marker_color=colors,
        name="Trade PnL",
        hovertemplate="Trade #%{x}<br><b>%{y:+.2f} USD</b><extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#333", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="TRADE P&L  (chronological)", font=dict(color="#FF8C00", size=11), x=0),
                      yaxis_title=None, xaxis_title=None)
    return fig


def chart_trades_scatter(df: pd.DataFrame) -> go.Figure | None:
    """Trade scatter: time vs PnL, colored by strategy."""
    if df.empty or "exit_time" not in df.columns:
        return None
    d = df.dropna(subset=["exit_time"]).sort_values("exit_time")
    if d.empty:
        return None

    fig = go.Figure()
    for strat, grp in d.groupby("strategy_type"):
        color = "#C084FC" if "straddle" in str(strat) else ("#4A90D9" if "vr" in str(strat) else "#C8960C")
        label = "S_d8" if "straddle" in str(strat) else ("D_mom8" if "directional" in str(strat) else "MA CLUSTER")
        fig.add_trace(go.Scatter(
            x=grp["exit_time"], y=grp["pnl_usd"],
            mode="markers",
            marker=dict(color=[color if v > 0 else "#FF4444" for v in grp["pnl_usd"]],
                        size=8, line=dict(width=0)),
            name=label,
            hovertemplate=(
                "<b>%{x|%m/%d %H:%M}</b><br>"
                + "PnL: <b>%{y:+.2f} USD</b><extra></extra>"
            ),
        ))
    fig.add_hline(y=0, line_color="#333", line_width=1)
    fig.update_layout(**PLOT_LAYOUT, title=dict(text="TRADE TIMELINE", font=dict(color="#FF8C00", size=11), x=0))
    return fig


def chart_rolling_wr(df: pd.DataFrame, window: int = 20) -> go.Figure | None:
    """Rolling win rate over last N trades."""
    if df.empty or len(df) < window:
        return None
    d = df.sort_values("id")
    wr = d["pnl_usd"].gt(0).rolling(window).mean() * 100
    fig = go.Figure()
    fig.add_hline(y=50, line_color="#333", line_dash="dash", line_width=1)
    fig.add_trace(go.Scatter(
        x=list(range(1, len(d) + 1)), y=wr,
        mode="lines",
        line=dict(color="#4A90D9", width=2),
        fill="tozeroy",
        fillcolor="rgba(74,144,217,0.08)",
        name=f"WR {window}T",
        hovertemplate="Trade #%{x}<br>WR: <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=f"ROLLING WIN RATE  ({window}-trade)", font=dict(color="#FF8C00", size=11), x=0),
        yaxis_range=[0, 100],
        yaxis_ticksuffix="%",
    )
    return fig


def chart_mfe_mae(df: pd.DataFrame) -> go.Figure | None:
    """MFE vs MAE scatter."""
    if df.empty or "mfe_pct" not in df.columns:
        return None
    d = df.dropna(subset=["mfe_pct", "mae_pct"])
    if d.empty:
        return None
    colors = ["#00C853" if v > 0 else "#D50000" for v in d["pnl_usd"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["mfe_pct"], y=d["mae_pct"],
        mode="markers",
        marker=dict(color=colors, size=7, opacity=0.8, line=dict(width=0)),
        hovertemplate="MFE: %{x:+.2f}%<br>MAE: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT,
                      title=dict(text="MFE vs MAE  (green=win)", font=dict(color="#FF8C00", size=11), x=0),
                      xaxis_title="MFE %", yaxis_title="MAE %")
    return fig


# ─── Render ───────────────────────────────────────────────────────────────────
def render():
    now_utc = pd.Timestamp.now(tz="UTC")
    now_str = now_utc.strftime("%Y-%m-%d  %H:%M UTC")

    acct      = fetch_account()
    positions = fetch_positions()
    bars_map  = load_state_bars()
    df        = load_trades()

    wallet     = acct.get("wallet", 0)
    unrealized = acct.get("unrealized", 0)
    margin     = acct.get("margin", 0)
    margin_pct = margin / wallet * 100 if wallet else 0
    n_pos      = len(positions)
    total_pnl  = df["pnl_usd"].sum() if not df.empty else 0
    upnl_pct   = unrealized / wallet * 100 if wallet else 0

    # ── TICKER BAR (always visible, above tabs) ───────────────────────────────
    upnl_class   = "pos" if unrealized >= 0 else "neg"
    totpnl_class = "pos" if total_pnl >= 0 else "neg"

    st.markdown(f"""
<div class="ticker">
  <span class="logo">QUANTBOX TERMINAL</span>
  <span class="sep">|</span>
  <span>WALLET <b>${wallet:,.2f}</b></span>
  <span class="sep">|</span>
  <span>UNREALIZED <span class="{upnl_class}">{unrealized:+,.2f} ({upnl_pct:+.2f}%)</span></span>
  <span class="sep">|</span>
  <span>MARGIN <span class="neu">${margin:,.0f} / {margin_pct:.1f}%</span></span>
  <span class="sep">|</span>
  <span>POSITIONS <b>{n_pos}</b>/10</span>
  <span class="sep">|</span>
  <span>CUMULATIVE PnL <span class="{totpnl_class}"><b>${total_pnl:+,.2f}</b></span></span>
  <span class="sep">|</span>
  <span class="neu">{now_str}</span>
</div>
""", unsafe_allow_html=True)

    # ── SCORECARD (always visible) ────────────────────────────────────────────
    n_trades    = len(df)
    wr          = df["pnl_usd"].gt(0).mean() * 100 if n_trades > 0 else 0
    best        = df.loc[df["pnl_usd"].idxmax()] if n_trades > 0 else None
    worst       = df.loc[df["pnl_usd"].idxmin()] if n_trades > 0 else None
    streak_type, streak_n = win_streak(df)
    avg_bars    = df["bars_held"].mean() if n_trades > 0 else 0
    wins        = int(df["pnl_usd"].gt(0).sum()) if n_trades > 0 else 0
    losses      = int(df["pnl_usd"].le(0).sum()) if n_trades > 0 else 0

    wr_color     = "#00C853" if wr >= 50 else "#D50000"
    totpnl_color = "#00C853" if total_pnl >= 0 else "#D50000"
    streak_label = f"{'WIN' if streak_type=='W' else 'LOSS'} STREAK" if streak_type != "–" else "STREAK"
    streak_color = "#00C853" if streak_type == "W" else ("#D50000" if streak_type == "L" else "#555")
    best_str     = f"+${best['pnl_usd']:.2f}  {best['symbol']}" if best is not None else "—"
    worst_str    = f"${worst['pnl_usd']:.2f}  {worst['symbol']}" if worst is not None else "—"

    st.markdown(f"""
<div class="statstrip">
  <div class="stat">
    <div class="lbl">TRADES</div>
    <div class="val">{n_trades}</div>
    <div class="sub">avg {avg_bars:.0f} bars</div>
  </div>
  <div class="stat">
    <div class="lbl">WIN RATE</div>
    <div class="val" style="color:{wr_color}">{wr:.1f}%</div>
    <div class="sub">{wins}W  {losses}L</div>
  </div>
  <div class="stat">
    <div class="lbl">CLOSED PnL</div>
    <div class="val" style="color:{totpnl_color}">${total_pnl:+,.2f}</div>
    <div class="sub">realized only</div>
  </div>
  <div class="stat">
    <div class="lbl">BEST TRADE</div>
    <div class="val" style="color:#00C853;font-size:1.05rem">{best_str}</div>
    <div class="sub">{best['exit_reason'] if best is not None else ''}</div>
  </div>
  <div class="stat">
    <div class="lbl">WORST TRADE</div>
    <div class="val" style="color:#D50000;font-size:1.05rem">{worst_str}</div>
    <div class="sub">{worst['exit_reason'] if worst is not None else ''}</div>
  </div>
  <div class="stat">
    <div class="lbl">{streak_label}</div>
    <div class="val" style="color:{streak_color}">{streak_n}</div>
    <div class="sub">{'consecutive' if streak_n > 1 else ''}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊  OVERVIEW", "📈  EQUITY", "🕐  TRADES", "📉  ANALYTICS"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown('<div class="bb-header">OPEN POSITIONS</div>', unsafe_allow_html=True)
            if not positions:
                st.markdown('<span class="neu" style="font-size:0.85rem">— No open positions —</span>', unsafe_allow_html=True)
            else:
                rows_html = ""
                for p in positions:
                    sym_key   = f"{p['symbol']}:{p['side']}"
                    bars      = bars_map.get(sym_key, bars_map.get(p["symbol"], "?"))
                    pct_class = "pos" if p["pct"] >= 0 else "neg"
                    side_cls  = "long-tag" if p["side"] == "LONG" else "short-tag"
                    rows_html += f"""
<tr>
  <td class="sym">{p['symbol']}</td>
  <td><span class="{side_cls}">{p['side']}</span></td>
  <td>${p['entry']:,.4f}</td>
  <td>${p['mark']:,.4f}</td>
  <td class="{pct_class}">{p['pct']:+.2f}%</td>
  <td class="{pct_class}">${p['upnl']:+.2f}</td>
  <td class="neu">{bars}/192</td>
  <td class="neu">${p['notional']:,.0f}</td>
</tr>"""
                st.markdown(f"""
<div class="tbl-wrap"><table class="bb-table">
<tr><th style="text-align:left">SYMBOL</th><th>SIDE</th><th>ENTRY</th><th>MARK</th>
<th>PnL%</th><th>PnL USD</th><th>BARS</th><th>NOTIONAL</th></tr>
{rows_html}
</table></div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="bb-header">STRATEGY BREAKDOWN</div>', unsafe_allow_html=True)
            if not df.empty:
                strat_rows = ""
                for strat, grp in df.groupby("strategy_type"):
                    n_s     = len(grp)
                    wr_s    = grp["pnl_usd"].gt(0).mean() * 100
                    tot_s   = grp["pnl_usd"].sum()
                    avg_mfe = grp["mfe_pct"].mean() if "mfe_pct" in grp else 0
                    avg_mae = grp["mae_pct"].mean() if "mae_pct" in grp else 0
                    avg_b   = grp["bars_held"].mean()
                    tot_cls = "pos" if tot_s >= 0 else "neg"
                    tag_cls = "tag-vr-s" if "straddle" in strat else ("tag-vr-d" if "directional" in strat else "tag-ma")
                    label   = "S_d8" if "straddle" in strat else ("D_mom8" if "directional" in strat else "MA CLU")
                    strat_rows += f"""
<tr>
  <td><span class="{tag_cls}">{label}</span></td>
  <td>{n_s}</td>
  <td>{wr_s:.0f}%</td>
  <td class="{tot_cls}">${tot_s:+.0f}</td>
  <td class="neu">{avg_mfe:+.2f}%</td>
  <td class="neu">{avg_mae:.2f}%</td>
  <td class="neu">{avg_b:.0f}b</td>
</tr>"""
                st.markdown(f"""
<div class="tbl-wrap"><table class="bb-table">
<tr><th style="text-align:left">STRAT</th><th>N</th><th>WIN%</th>
<th>PnL USD</th><th>MFE AVG</th><th>MAE AVG</th><th>AVG BARS</th></tr>
{strat_rows}
</table></div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        col_t, col_b = st.columns([3, 2])

        with col_t:
            st.markdown('<div class="bb-header">RECENT TRADES  <span class="neu">(15)</span></div>', unsafe_allow_html=True)
            if df.empty:
                st.markdown('<span class="neu">No closed trades.</span>', unsafe_allow_html=True)
            else:
                recent     = df.sort_values("id", ascending=False).head(15)
                trade_rows = ""
                for _, r in recent.iterrows():
                    pnl      = r["pnl_usd"]
                    pct_cls  = "pos" if pnl >= 0 else "neg"
                    side_cls = "long-tag" if r["direction"] == "LONG" else "short-tag"
                    tag_cls  = "tag-vr-s" if "straddle" in str(r["strategy_type"]) else ("tag-vr-d" if "directional" in str(r["strategy_type"]) else "tag-ma")
                    t        = str(r["exit_time"])[:16]
                    mfe      = r.get("mfe_pct", 0) or 0
                    strat_label = "S_d8" if "straddle" in str(r["strategy_type"]) else ("D_mom8" if "directional" in str(r["strategy_type"]) else "MA")
                    trade_rows += f"""
<tr>
  <td class="neu" style="text-align:left">{t}</td>
  <td class="sym" style="text-align:left">{r['symbol']}</td>
  <td><span class="{side_cls}">{r['direction']}</span></td>
  <td><span class="{tag_cls}">{strat_label}</span></td>
  <td class="{pct_cls}">${pnl:+.2f}</td>
  <td class="neu">{mfe:+.1f}%</td>
  <td class="neu">{r['bars_held']}</td>
  <td class="neu" style="text-align:left">{r['exit_reason']}</td>
</tr>"""
                st.markdown(f"""
<div class="tbl-wrap"><table class="bb-table">
<tr><th style="text-align:left">EXIT TIME</th><th style="text-align:left">SYMBOL</th>
<th>DIR</th><th>STR</th><th>PnL USD</th><th>MFE</th><th>BARS</th>
<th style="text-align:left">REASON</th></tr>
{trade_rows}
</table></div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="bb-header">EXIT REASON DISTRIBUTION</div>', unsafe_allow_html=True)
            if not df.empty:
                grp = df.groupby("exit_reason").agg(
                    n=("pnl_usd", "count"),
                    avg_pnl=("pnl_usd", "mean"),
                ).sort_values("n", ascending=False)
                max_n    = grp["n"].max()
                bars_html = ""
                for reason, row in grp.iterrows():
                    bar     = unicode_bar(row["n"], max_n, 12)
                    avg_cls = "pos" if row["avg_pnl"] >= 0 else "neg"
                    bars_html += f"""
<div class="bar-row">
  <span class="bar-label">{str(reason)[:22]}</span>
  <span class="bar-fill">{bar}</span>
  <span class="bar-stat"> {int(row['n'])}x <span class="{avg_cls}">avg ${row['avg_pnl']:+.0f}</span></span>
</div>"""
                st.markdown(bars_html, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — EQUITY
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        if df.empty:
            st.markdown('<span class="neu">No trade data yet.</span>', unsafe_allow_html=True)
        else:
            # ── Equity curve ─────────────────────────────────────────────────
            fig_eq = chart_equity(df)
            if fig_eq:
                st.plotly_chart(fig_eq, use_container_width=True)
            else:
                st.markdown('<span class="neu">balance_after data unavailable.</span>', unsafe_allow_html=True)

            # ── Individual P&L bars ───────────────────────────────────────────
            fig_pnl = chart_pnl_bars(df)
            if fig_pnl:
                st.plotly_chart(fig_pnl, use_container_width=True)

            # ── Summary strip ─────────────────────────────────────────────────
            if "balance_after" in df.columns and df["balance_after"].notna().any():
                start_bal = df.dropna(subset=["balance_after"]).iloc[0]["balance_before"] \
                    if "balance_before" in df.columns else None
                end_bal   = df.dropna(subset=["balance_after"]).iloc[-1]["balance_after"]
                max_dd    = _max_drawdown(df)
                dd_color  = "#00C853" if max_dd < 10 else ("#FF8C00" if max_dd < 20 else "#D50000")
                start_str = f"${start_bal:,.2f}" if start_bal is not None else "—"
                st.markdown(f"""
<div class="mini-grid">
  <div class="mini-stat">
    <div class="lbl">START BALANCE</div>
    <div class="val">{start_str}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">CURRENT BALANCE</div>
    <div class="val">${end_bal:,.2f}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">TOTAL REALIZED</div>
    <div class="val" style="color:{'#00C853' if total_pnl >= 0 else '#D50000'}">${total_pnl:+,.2f}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">MAX DRAWDOWN</div>
    <div class="val" style="color:{dd_color}">{max_dd:.1f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — TRADES
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        if df.empty:
            st.markdown('<span class="neu">No trade data yet.</span>', unsafe_allow_html=True)
        else:
            # ── Timeline scatter ──────────────────────────────────────────────
            fig_scatter = chart_trades_scatter(df)
            if fig_scatter:
                st.plotly_chart(fig_scatter, use_container_width=True)

            # ── Full trade table (all, newest first) ─────────────────────────
            st.markdown(f'<div class="bb-header">ALL TRADES  <span class="neu">({n_trades})</span></div>',
                        unsafe_allow_html=True)
            all_sorted = df.sort_values("id", ascending=False)
            trade_rows = ""
            for _, r in all_sorted.iterrows():
                pnl      = r["pnl_usd"]
                pct_cls  = "pos" if pnl >= 0 else "neg"
                side_cls = "long-tag" if r["direction"] == "LONG" else "short-tag"
                tag_cls  = "tag-vr-s" if "straddle" in str(r["strategy_type"]) else ("tag-vr-d" if "directional" in str(r["strategy_type"]) else "tag-ma")
                t        = str(r["exit_time"])[:16]
                mfe      = r.get("mfe_pct", 0) or 0
                mae      = r.get("mae_pct", 0) or 0
                strat_label = "S_d8" if "straddle" in str(r["strategy_type"]) else ("D_mom8" if "directional" in str(r["strategy_type"]) else "MA")
                trade_rows += f"""
<tr>
  <td class="neu">{t}</td>
  <td class="sym">{r['symbol']}</td>
  <td><span class="{side_cls}">{r['direction']}</span></td>
  <td><span class="{tag_cls}">{strat_label}</span></td>
  <td>${r['entry_price']:,.4f}</td>
  <td>${r['exit_price']:,.4f}</td>
  <td class="{pct_cls}">{r['pnl_pct']:+.2f}%</td>
  <td class="{pct_cls}">${pnl:+.2f}</td>
  <td class="neu">{mfe:+.1f}%</td>
  <td class="neu">{mae:.1f}%</td>
  <td class="neu">{r['bars_held']}</td>
  <td class="neu">{r['exit_reason']}</td>
</tr>"""
            st.markdown(f"""
<div class="tbl-wrap"><table class="bb-table">
<tr><th>EXIT TIME</th><th>SYMBOL</th><th>DIR</th><th>STR</th>
<th>ENTRY</th><th>EXIT</th><th>PnL%</th><th>PnL USD</th>
<th>MFE</th><th>MAE</th><th>BARS</th><th>REASON</th></tr>
{trade_rows}
</table></div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if df.empty:
            st.markdown('<span class="neu">No trade data yet.</span>', unsafe_allow_html=True)
        else:
            pf_val   = _pf(df)
            sh_val   = _sharpe(df)
            dd_val   = _max_drawdown(df)
            avg_mfe  = df["mfe_pct"].mean() if "mfe_pct" in df.columns else 0
            avg_mae  = df["mae_pct"].mean() if "mae_pct" in df.columns else 0
            avg_hold = df["bars_held"].mean()

            pf_color = "#00C853" if pf_val >= 1.5 else ("#FF8C00" if pf_val >= 1.0 else "#D50000")
            sh_color = "#00C853" if sh_val >= 1.0 else ("#FF8C00" if sh_val >= 0 else "#D50000")
            dd_color = "#00C853" if dd_val < 10 else ("#FF8C00" if dd_val < 20 else "#D50000")
            pf_str   = f"{pf_val:.2f}" if pf_val != float("inf") else "∞"

            st.markdown(f"""
<div class="mini-grid">
  <div class="mini-stat">
    <div class="lbl">PROFIT FACTOR</div>
    <div class="val" style="color:{pf_color}">{pf_str}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">SHARPE (session)</div>
    <div class="val" style="color:{sh_color}">{sh_val:.2f}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">MAX DRAWDOWN</div>
    <div class="val" style="color:{dd_color}">{dd_val:.1f}%</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">AVG MFE</div>
    <div class="val" style="color:#00C853">{avg_mfe:+.2f}%</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">AVG MAE</div>
    <div class="val" style="color:#D50000">{avg_mae:.2f}%</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">AVG HOLD</div>
    <div class="val">{avg_hold:.0f} bars</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Rolling WR ─────────────────────────────────────────────────
            col_wr, col_mfe = st.columns(2)
            with col_wr:
                fig_rwr = chart_rolling_wr(df, 20)
                if fig_rwr:
                    st.plotly_chart(fig_rwr, use_container_width=True)
                else:
                    st.markdown('<span class="neu">Need ≥ 20 trades for rolling WR.</span>', unsafe_allow_html=True)

            with col_mfe:
                fig_mfe = chart_mfe_mae(df)
                if fig_mfe:
                    st.plotly_chart(fig_mfe, use_container_width=True)

            # ── Exit reason breakdown (Plotly bar) ─────────────────────────
            st.markdown('<div class="bb-header">EXIT REASON  —  PnL BREAKDOWN</div>', unsafe_allow_html=True)
            grp_exit = df.groupby("exit_reason").agg(
                n=("pnl_usd", "count"),
                avg_pnl=("pnl_usd", "mean"),
                total_pnl=("pnl_usd", "sum"),
                wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
            ).sort_values("total_pnl", ascending=False).reset_index()

            colors_exit = ["#00C853" if v >= 0 else "#D50000" for v in grp_exit["total_pnl"]]
            fig_exit = go.Figure()
            fig_exit.add_trace(go.Bar(
                x=grp_exit["exit_reason"], y=grp_exit["total_pnl"],
                marker_color=colors_exit,
                text=[f"{int(n)}T / WR {wr:.0f}%" for n, wr in zip(grp_exit["n"], grp_exit["wr"])],
                textposition="outside",
                textfont=dict(size=9, color="#888"),
                hovertemplate="<b>%{x}</b><br>Total: $%{y:+.2f}<extra></extra>",
            ))
            fig_exit.add_hline(y=0, line_color="#333", line_width=1)
            fig_exit.update_layout(**PLOT_LAYOUT,
                                   title=dict(text="TOTAL PnL BY EXIT REASON", font=dict(color="#FF8C00", size=11), x=0))
            st.plotly_chart(fig_exit, use_container_width=True)

            # ── Strategy comparison ────────────────────────────────────────
            if df["strategy_type"].nunique() > 1:
                st.markdown('<div class="bb-header">STRATEGY COMPARISON</div>', unsafe_allow_html=True)
                grp_strat = df.groupby("strategy_type").agg(
                    n=("pnl_usd", "count"),
                    wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
                    pf=("pnl_usd", lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if (x < 0).any() else float("inf")),
                    total=("pnl_usd", "sum"),
                ).reset_index()

                strat_comp = ""
                for _, r in grp_strat.iterrows():
                    label   = "S_d8" if "straddle" in str(r["strategy_type"]) else ("D_mom8" if "directional" in str(r["strategy_type"]) else "MA CLUSTER")
                    tag_cls = "tag-vr-s" if "straddle" in str(r["strategy_type"]) else ("tag-vr-d" if "directional" in str(r["strategy_type"]) else "tag-ma")
                    tot_cls = "pos" if r["total"] >= 0 else "neg"
                    pf_str2 = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
                    strat_comp += f"""
<tr>
  <td><span class="{tag_cls}">{label}</span></td>
  <td>{int(r['n'])}</td>
  <td>{r['wr']:.1f}%</td>
  <td>{pf_str2}</td>
  <td class="{tot_cls}">${r['total']:+,.2f}</td>
</tr>"""
                st.markdown(f"""
<div class="tbl-wrap"><table class="bb-table">
<tr><th style="text-align:left">STRATEGY</th><th>TRADES</th><th>WIN RATE</th>
<th>PROFIT FACTOR</th><th>TOTAL PnL</th></tr>
{strat_comp}
</table></div>""", unsafe_allow_html=True)

    # ── Auto refresh ──────────────────────────────────────────────────────────
    time.sleep(REFRESH)
    st.cache_data.clear()
    st.rerun()


render()
