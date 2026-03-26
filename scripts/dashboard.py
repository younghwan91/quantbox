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

_DATA = Path(__file__).parent.parent / "data"
DB_LIVE = _DATA / "journal.sqlite"
DB_PAPER = _DATA / "journal_paper.sqlite"
STATE_LIVE = _DATA / "state" / "scalper_state.json"
STATE_PAPER = _DATA / "state" / "scalper_state_paper.json"
# Active paths — overridden per-render based on mode toggle
DB = DB_LIVE
STATE = STATE_LIVE
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
.tag-ma  { color: #C8960C; font-size: 0.88rem; }
.tag-unk { color: #8B949E; font-size: 0.88rem; }
.long-tag  { color: #00C853; font-size: 0.88rem; letter-spacing: 0.06em; }
.short-tag { color: #D50000; font-size: 0.88rem; letter-spacing: 0.06em; }
.reason-sl    { color: #F85149; font-size: 0.82rem; }
.reason-recon { color: #D29922; font-size: 0.82rem; }
.reason-trail { color: #58A6FF; font-size: 0.82rem; }
.reason-norm  { color: #8B949E; font-size: 0.82rem; }

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
            entry    = float(p["entryPrice"])
            mark     = float(p["markPrice"])
            upnl     = float(p["unRealizedProfit"])
            leverage = int(p.get("leverage", 30))
            pct      = (mark - entry) / entry * 100 if amt > 0 else (entry - mark) / entry * 100
            roi_pct  = pct * leverage
            rows.append({
                "symbol":   p["symbol"],
                "side":     p["positionSide"],
                "entry":    entry,
                "mark":     mark,
                "pct":      pct,
                "roi_pct":  roi_pct,
                "leverage": leverage,
                "upnl":     upnl,
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


def load_state_strategies() -> dict:
    """Return {pos_key: strategy_type} from state file. Key format: 'SYMBOL:SIDE'."""
    if not STATE.exists():
        return {}
    try:
        s = json.load(open(STATE))
        result = {}
        for sym, pos in s.get("positions", {}).items():
            strat = pos.get("strategy_type", "vr_directional")
            is_straddle = pos.get("is_straddle", False)
            # Patch in-flight: state written before code fix may have vr_squeeze for straddles
            if is_straddle and strat == "vr_directional":
                strat = "vr_straddle"
            result[sym] = strat
        return result
    except Exception:
        return {}


def load_state_pos_details() -> dict:
    """Return {pos_key: {sl, tp, trail_active, peak_gain, max_hold}} from state."""
    if not STATE.exists():
        return {}
    try:
        s = json.load(open(STATE))
        result = {}
        for sym, pos in s.get("positions", {}).items():
            result[sym] = {
                "sl": pos.get("sl", 0),
                "tp": pos.get("tp", 0),
                "trail_active": pos.get("trail_active", False),
                "peak_gain": pos.get("peak_gain", 0),
                "max_hold": pos.get("max_hold", 192),
                "server_sl_order_id": pos.get("server_sl_order_id"),
                "server_tp_order_id": pos.get("server_tp_order_id"),
            }
        return result
    except Exception:
        return {}


@st.cache_data(ttl=28, show_spinner=False)
def load_trades(_db_path: str = "") -> pd.DataFrame:
    db = Path(_db_path) if _db_path else DB
    if not db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        f"SELECT * FROM trades WHERE {REAL} ORDER BY id", conn
    )
    conn.close()
    if not df.empty and "exit_time" in df.columns:
        # format="ISO8601" handles mixed precision (seconds vs microseconds) in pandas 2.x
        df["exit_time"] = pd.to_datetime(df["exit_time"], format="ISO8601", utc=True, errors="coerce")
    return df


def _strat_tag(strategy_type: str) -> tuple[str, str]:
    """Return (css_class, display_label) for a strategy_type value."""
    s = str(strategy_type).lower()
    if "vr_straddle" in s:
        return "tag-vr-s", "STRD"
    if "squeeze" in s or "vr" in s:
        return "tag-vr-d", "VR"
    if "ma" in s or "cluster" in s:
        return "tag-ma", "MA CLU"
    return "tag-unk", "?"


def _reason_tag(reason: str) -> tuple[str, str]:
    """Return (css_class, display_label) for an exit_reason value."""
    r = str(reason)
    if r in ("server_sl", "stop_loss"):
        return "reason-sl", "SL"
    if r == "reconciliation_close":
        return "reason-recon", "RECON"
    if r == "trailing_stop":
        return "reason-trail", "TRAIL"
    if r == "drawdown_halt":
        return "reason-sl", "DD HALT"
    if r == "max_hold":
        return "reason-norm", "MAX HOLD"
    return "reason-norm", r


def _fmt_pct(v: object) -> str:
    """Format a pct float, returning '–' for None / NaN."""
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{float(v):+.1f}%"


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
    """Equity curve: cumulative pnl_usd over time (start_balance + cumsum)."""
    if df.empty or "pnl_usd" not in df.columns or "exit_time" not in df.columns:
        return None
    d = df.dropna(subset=["pnl_usd", "exit_time"]).sort_values("exit_time").copy()
    if d.empty:
        return None

    # Derive start balance from first non-NULL balance_before; fallback to 0
    start_balance = 0.0
    if "balance_before" in d.columns:
        first_valid = d["balance_before"].dropna()
        if not first_valid.empty:
            start_balance = float(first_valid.iloc[0])

    d["equity"] = start_balance + d["pnl_usd"].cumsum()

    fig = go.Figure()

    # If balance_source column exists, split solid vs dashed segments
    if "balance_source" in d.columns:
        real_mask = d["balance_source"] != "interpolated"
        interp_mask = d["balance_source"] == "interpolated"
        if real_mask.any():
            dr = d[real_mask]
            fig.add_trace(go.Scatter(
                x=dr["exit_time"], y=dr["equity"],
                mode="lines",
                line=dict(color="#FF8C00", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,140,0,0.06)",
                name="Equity",
                hovertemplate="<b>%{x|%m/%d %H:%M}</b><br>$%{y:,.2f}<extra></extra>",
            ))
        if interp_mask.any():
            di = d[interp_mask]
            fig.add_trace(go.Scatter(
                x=di["exit_time"], y=di["equity"],
                mode="lines",
                line=dict(color="#FF8C00", width=1, dash="dot"),
                name="Equity (interpolated)",
                hovertemplate="<b>%{x|%m/%d %H:%M}</b><br>$%{y:,.2f} (interp)<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter(
            x=d["exit_time"], y=d["equity"],
            mode="lines",
            line=dict(color="#FF8C00", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,140,0,0.06)",
            name="Equity",
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
        color = "#C084FC" if "vr_straddle" in str(strat) else ("#4A90D9" if "vr" in str(strat) else "#C8960C")
        label = "S_d8" if "vr_straddle" in str(strat) else ("D_mom8" if "directional" in str(strat) else "MA CLUSTER")
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
def _fetch_account_paper() -> dict:
    """Read balance from paper state file (no Binance API call)."""
    if not STATE_PAPER.exists():
        return {"wallet": 0, "unrealized": 0, "margin": 0, "available": 0}
    try:
        s = json.load(open(STATE_PAPER))
        bal = s.get("paper_balance", 1000.0)
        return {"wallet": bal, "unrealized": 0, "margin": 0, "available": bal}
    except Exception:
        return {"wallet": 0, "unrealized": 0, "margin": 0, "available": 0}


def _fetch_positions_paper() -> list[dict]:
    """Read positions from paper state file (no Binance API call)."""
    if not STATE_PAPER.exists():
        return []
    try:
        s = json.load(open(STATE_PAPER))
        rows = []
        for sym, pos in s.get("positions", {}).items():
            entry = pos.get("entry_price", 0)
            signal = pos.get("signal", 1)
            leverage = pos.get("leverage", 30)
            rows.append({
                "symbol": sym.split(":")[0],
                "side": pos.get("position_side", "LONG" if signal == 1 else "SHORT"),
                "entry": entry,
                "mark": entry,  # no mark price in paper mode
                "pct": 0,
                "roi_pct": 0,
                "leverage": leverage,
                "upnl": 0,
                "notional": pos.get("notional", 0),
            })
        return rows
    except Exception:
        return []


def render():
    global DB, STATE

    # ── Mode toggle (persists across 30s auto-refresh via session_state) ──
    mode = st.radio(
        "Mode", ["LIVE", "PAPER"],
        horizontal=True,
        key="dashboard_mode",
        label_visibility="collapsed",
    )
    is_paper = mode == "PAPER"
    DB = DB_PAPER if is_paper else DB_LIVE
    STATE = STATE_PAPER if is_paper else STATE_LIVE

    now_utc = pd.Timestamp.now(tz="UTC")
    now_str = now_utc.strftime("%Y-%m-%d  %H:%M UTC")

    acct          = _fetch_account_paper() if is_paper else fetch_account()
    positions     = _fetch_positions_paper() if is_paper else fetch_positions()
    bars_map      = load_state_bars()
    strategies_map = load_state_strategies()
    pos_details   = load_state_pos_details()
    df            = load_trades(str(DB))

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

    _badge_bg = "#FFD600" if is_paper else "#00C853"
    _badge_text = "PAPER" if is_paper else "LIVE"
    _wallet_label = "PAPER BAL" if is_paper else "WALLET"

    st.markdown(f"""
<div class="ticker">
  <span class="logo">QUANTBOX TERMINAL</span>
  <span style="background:{_badge_bg};color:#000;padding:1px 8px;font-weight:bold;border-radius:3px;margin:0 6px;">{_badge_text}</span>
  <span class="sep">|</span>
  <span>{_wallet_label} <b>${wallet:,.2f}</b></span>
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
                    strat_type = strategies_map.get(sym_key, strategies_map.get(p["symbol"], "vr_directional"))
                    strat_cls, strat_label = _strat_tag(strat_type)
                    pct_class = "pos" if p["pct"] >= 0 else "neg"
                    side_cls  = "long-tag" if p["side"] == "LONG" else "short-tag"
                    # SL/TP/trail from state
                    det = pos_details.get(sym_key, pos_details.get(p["symbol"], {}))
                    sl_val = det.get("sl", 0)
                    tp_val = det.get("tp", 0)
                    max_hold = det.get("max_hold", 192)
                    sl_ok = "✓" if det.get("server_sl_order_id") else "✗"
                    tp_ok = "✓" if det.get("server_tp_order_id") else "✗"
                    if det.get("trail_active"):
                        peak = det.get("peak_gain", 0) * 100
                        trail_str = f"✓({peak:+.1f}%)"
                    else:
                        trail_str = "✗"
                    rows_html += f"""
<tr>
  <td class="sym">{p['symbol']}</td>
  <td><span class="{side_cls}">{p['side']}</span></td>
  <td><span class="{strat_cls}">{strat_label}</span></td>
  <td>${p['entry']:,.4f}</td>
  <td>${p['mark']:,.4f}</td>
  <td class="{pct_class}">{p['pct']:+.2f}%</td>
  <td class="{pct_class}">${p['upnl']:+.2f}</td>
  <td class="neu">{sl_val * 100:.1f}%{sl_ok}</td>
  <td class="neu">{tp_val * 100:.0f}%{tp_ok}</td>
  <td class="neu">{trail_str}</td>
  <td class="neu">{p['leverage']}x</td>
  <td class="neu">{bars}/{max_hold}</td>
  <td class="neu">${p['notional']:,.0f}</td>
</tr>"""
                st.markdown(f"""
<div class="tbl-wrap"><table class="bb-table">
<tr><th style="text-align:left">SYMBOL</th><th>SIDE</th><th>STRAT</th><th>ENTRY</th><th>MARK</th>
<th>Price%</th><th>PnL USD</th><th>SL</th><th>TP</th><th>TRAIL</th><th>LEV</th><th>BARS</th><th>NOTIONAL</th></tr>
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
                    _mfe    = grp["mfe_pct"].mean() if "mfe_pct" in grp else None
                    _mae    = grp["mae_pct"].mean() if "mae_pct" in grp else None
                    avg_mfe_disp = _fmt_pct(_mfe)
                    avg_mae_disp = _fmt_pct(_mae)
                    avg_b   = grp["bars_held"].mean()
                    tot_cls = "pos" if tot_s >= 0 else "neg"
                    tag_cls, label = _strat_tag(strat)
                    strat_rows += f"""
<tr>
  <td><span class="{tag_cls}">{label}</span></td>
  <td>{n_s}</td>
  <td>{wr_s:.0f}%</td>
  <td class="{tot_cls}">${tot_s:+.0f}</td>
  <td class="neu">{avg_mfe_disp}</td>
  <td class="neu">{avg_mae_disp}</td>
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
                recent     = df.sort_values("exit_time", ascending=False).head(15)
                trade_rows = ""
                for _, r in recent.iterrows():
                    pnl      = r["pnl_usd"]
                    pct_cls  = "pos" if pnl >= 0 else "neg"
                    side_cls = "long-tag" if r["direction"] == "LONG" else "short-tag"
                    tag_cls, strat_label = _strat_tag(r["strategy_type"])
                    rsn_cls, rsn_label   = _reason_tag(r["exit_reason"])
                    t        = str(r["exit_time"])[:16]
                    mfe_disp = _fmt_pct(r.get("mfe_pct"))
                    trade_rows += f"""
<tr>
  <td class="neu" style="text-align:left">{t}</td>
  <td class="sym" style="text-align:left">{r['symbol']}</td>
  <td><span class="{side_cls}">{r['direction']}</span></td>
  <td><span class="{tag_cls}">{strat_label}</span></td>
  <td class="{pct_cls}">${pnl:+.2f}</td>
  <td class="neu">{mfe_disp}</td>
  <td class="neu">{r['bars_held']}</td>
  <td><span class="{rsn_cls}">{rsn_label}</span></td>
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
                st.markdown('<span class="neu">No pnl_usd data for equity curve.</span>', unsafe_allow_html=True)

            # ── Individual P&L bars ───────────────────────────────────────────
            fig_pnl = chart_pnl_bars(df)
            if fig_pnl:
                st.plotly_chart(fig_pnl, use_container_width=True)

            # ── Summary strip ─────────────────────────────────────────────────
            start_bal: float | None = None
            end_bal:   float | None = None
            if "balance_before" in df.columns:
                first_bb = df["balance_before"].dropna()
                if not first_bb.empty:
                    start_bal = float(first_bb.iloc[0])
            if "balance_after" in df.columns:
                last_ba = df["balance_after"].dropna()
                if not last_ba.empty:
                    end_bal = float(last_ba.iloc[-1])
            max_dd    = _max_drawdown(df)
            dd_color  = "#00C853" if max_dd < 10 else ("#FF8C00" if max_dd < 20 else "#D50000")
            start_str = f"${start_bal:,.2f}" if start_bal is not None else "—"
            end_str   = f"${end_bal:,.2f}"   if end_bal   is not None else "—"
            st.markdown(f"""
<div class="mini-grid">
  <div class="mini-stat">
    <div class="lbl">START BALANCE</div>
    <div class="val">{start_str}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">CURRENT BALANCE</div>
    <div class="val">{end_str}</div>
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
            all_sorted = df.sort_values("exit_time", ascending=False)
            trade_rows = ""
            for _, r in all_sorted.iterrows():
                pnl      = r["pnl_usd"]
                pct_cls  = "pos" if pnl >= 0 else "neg"
                side_cls = "long-tag" if r["direction"] == "LONG" else "short-tag"
                tag_cls, strat_label = _strat_tag(r["strategy_type"])
                rsn_cls, rsn_label   = _reason_tag(r["exit_reason"])
                t        = str(r["exit_time"])[:16]
                mfe_disp = _fmt_pct(r.get("mfe_pct"))
                mae_disp = _fmt_pct(r.get("mae_pct"))
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
  <td class="neu">{mfe_disp}</td>
  <td class="neu">{mae_disp}</td>
  <td class="neu">{r['bars_held']}</td>
  <td><span class="{rsn_cls}">{rsn_label}</span></td>
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
            avg_mfe  = df["mfe_pct"].mean() if "mfe_pct" in df.columns else None
            avg_mae  = df["mae_pct"].mean() if "mae_pct" in df.columns else None
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
    <div class="val" style="color:#00C853">{_fmt_pct(avg_mfe)}</div>
  </div>
  <div class="mini-stat">
    <div class="lbl">AVG MAE</div>
    <div class="val" style="color:#D50000">{_fmt_pct(avg_mae)}</div>
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
                    tag_cls, label = _strat_tag(r["strategy_type"])
                    if "vr" not in str(r["strategy_type"]):
                        label = "MA CLUSTER"
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

            # ── STRD monthly WR monitor (cliff sensitivity warning) ────────
            strd_df = df[df["strategy_type"].str.contains("straddle", na=False, case=False)]
            if len(strd_df) >= 5:
                st.markdown('<div class="bb-header">STRD MONTHLY WIN RATE  —  CLIFF MONITOR</div>', unsafe_allow_html=True)
                st.markdown(
                    '<span class="neu" style="font-size:11px">'
                    "⚠️ r77: s_trail_activate=0.015 is a sharp cliff — ±0.003 → PF drops 60-75%. "
                    "OOS WR baseline: ~65%. Alert if monthly WR falls below 55%."
                    "</span>",
                    unsafe_allow_html=True,
                )

                strd_m = strd_df.copy()
                strd_m["month"] = strd_m["exit_time"].dt.to_period("M").astype(str)
                monthly = (
                    strd_m.groupby("month")
                    .agg(
                        n=("pnl_usd", "count"),
                        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
                        total=("pnl_usd", "sum"),
                    )
                    .reset_index()
                    .sort_values("month")
                )

                STRD_WR_OOS = 65.0   # r77 OOS WR baseline
                STRD_WR_WARN = 55.0  # alert threshold

                rows_html = ""
                any_warn = False
                for _, row in monthly.iterrows():
                    wr_color = (
                        "#00C853" if row["wr"] >= STRD_WR_OOS
                        else ("#FF8C00" if row["wr"] >= STRD_WR_WARN else "#D50000")
                    )
                    warn = " ⚠️" if row["wr"] < STRD_WR_WARN else ""
                    if warn:
                        any_warn = True
                    tot_cls = "pos" if row["total"] >= 0 else "neg"
                    rows_html += (
                        f'<tr><td>{row["month"]}</td>'
                        f'<td>{int(row["n"])}</td>'
                        f'<td style="color:{wr_color};font-weight:bold">'
                        f'{row["wr"]:.1f}%{warn}</td>'
                        f'<td class="{tot_cls}">${row["total"]:+,.2f}</td></tr>'
                    )

                st.markdown(
                    f'<div class="tbl-wrap"><table class="bb-table">'
                    f"<tr><th>MONTH</th><th>TRADES</th>"
                    f"<th>WIN RATE (OOS={STRD_WR_OOS:.0f}%)</th><th>TOTAL PnL</th></tr>"
                    f"{rows_html}</table></div>",
                    unsafe_allow_html=True,
                )

                if any_warn:
                    st.error(
                        "🚨 STRD WR < 55% detected in one or more months. "
                        "Check s_trail_activate (cliff at 0.015±0.003). "
                        "Consider pausing STRD or running r81 sweep."
                    )

    # ── Auto refresh ──────────────────────────────────────────────────────────
    time.sleep(REFRESH)
    st.cache_data.clear()
    st.rerun()


render()
