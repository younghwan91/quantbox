"""QuantBox Terminal — Bloomberg-style live dashboard.

Usage:
    uv run streamlit run scripts/dashboard.py --server.headless true
Access:
    http://localhost:8501
"""
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

DB = Path(__file__).parent.parent / "data" / "journal.sqlite"
STATE = Path(__file__).parent.parent / "data" / "state" / "scalper_state.json"
REAL = "ABS(pnl_pct) < 10 AND symbol NOT LIKE 'TEST%' AND symbol NOT LIKE '%:%'"
REFRESH = 30
POSITIONS_TTL = 10  # faster refresh for open positions

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantBox Terminal",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Bloomberg CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Bloomberg Terminal base */
html, body, [data-testid="stApp"] {
    background-color: #000000 !important;
    color: #FFFFFF;
    font-family: 'Courier New', Courier, monospace;
    font-size: 16px;
}
[data-testid="stApp"] > div { background-color: #000000 !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0.5rem 1rem !important; max-width: 100% !important; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Divider */
hr { border-color: #FF8C00 !important; margin: 4px 0 !important; }

/* Ticker bar */
.ticker {
    background: #0a0a0a;
    border: 1px solid #FF8C00;
    border-radius: 2px;
    padding: 8px 16px;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.ticker .logo { color: #FF8C00; font-weight: bold; font-size: 1.18rem; }
.ticker .sep { color: #444; margin: 0 10px; }

/* Section headers */
.bb-header {
    color: #FF8C00;
    font-size: 0.95rem;
    font-weight: bold;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #333;
    padding-bottom: 2px;
    margin: 8px 0 4px 0;
}

/* Score cards */
.scorecard {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    margin-bottom: 8px;
}
.card {
    background: #0d0d0d;
    border: 1px solid #222;
    border-left: 3px solid #FF8C00;
    padding: 8px 10px;
    border-radius: 2px;
}
.card .label { color: #888; font-size: 0.86rem; letter-spacing: 0.1em; }
.card .value { color: #FFF; font-size: 1.42rem; font-weight: bold; margin-top: 2px; }
.card .sub   { color: #555; font-size: 0.80rem; margin-top: 1px; }
.card.green  { border-left-color: #00D964; }
.card.red    { border-left-color: #FF3B3B; }
.card.orange { border-left-color: #FF8C00; }

/* Tables */
.bb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.95rem;
    font-family: 'Courier New', monospace;
}
.bb-table th {
    color: #FF8C00;
    border-bottom: 1px solid #333;
    padding: 4px 10px;
    text-align: left;
    font-size: 0.84rem;
    letter-spacing: 0.08em;
}
.bb-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #111;
    color: #DDD;
}
.bb-table tr:hover td { background: #0d0d0d; }
.pos  { color: #00D964 !important; font-weight: bold; }
.neg  { color: #FF3B3B !important; font-weight: bold; }
.neu  { color: #888 !important; }
.sym  { color: #FFF !important; font-weight: bold; }
.tag-vr { color: #58a6ff; font-size: 0.82rem; }
.tag-ma { color: #d29922; font-size: 0.82rem; }
.long-tag  { color: #00D964; font-size: 0.87rem; }
.short-tag { color: #FF3B3B; font-size: 0.87rem; }

/* Bar chart */
.bar-row { display: flex; align-items: center; margin: 3px 0; font-size: 0.93rem; }
.bar-label { color: #888; width: 220px; }
.bar-fill  { color: #FF8C00; letter-spacing: -1px; }
.bar-stat  { color: #DDD; margin-left: 8px; }
</style>
""", unsafe_allow_html=True)


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
    import json
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
    return df


def win_streak(df: pd.DataFrame) -> tuple[str, int]:
    """Returns ('W'/'L', count) for current streak."""
    if df.empty:
        return ("–", 0)
    seq = df["pnl_usd"].values[::-1]
    if len(seq) == 0:
        return ("–", 0)
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


# ─── Render ───────────────────────────────────────────────────────────────────
def render():
    now_utc = pd.Timestamp.now(tz="UTC")
    now_str = now_utc.strftime("%Y-%m-%d  %H:%M UTC")

    acct = fetch_account()
    positions = fetch_positions()
    bars_map = load_state_bars()
    df = load_trades()

    wallet    = acct.get("wallet", 0)
    unrealized= acct.get("unrealized", 0)
    margin    = acct.get("margin", 0)
    margin_pct= margin / wallet * 100 if wallet else 0
    n_pos     = len(positions)

    total_pnl = df["pnl_usd"].sum() if not df.empty else 0
    upnl_pct  = unrealized / wallet * 100 if wallet else 0

    # ── TICKER BAR ────────────────────────────────────────────────────────────
    upnl_class = "pos" if unrealized >= 0 else "neg"
    totpnl_class = "pos" if total_pnl >= 0 else "neg"

    st.markdown(f"""
<div class="ticker">
  <span class="logo">▪ QUANTBOX TERMINAL</span>
  <span class="sep">|</span>
  <span>💰 <b>${wallet:,.2f}</b></span>
  <span class="sep">|</span>
  <span>Unrealized <span class="{upnl_class}">{unrealized:+,.2f} ({upnl_pct:+.2f}%)</span></span>
  <span class="sep">|</span>
  <span>Margin <span class="neu">${margin:,.0f} / {margin_pct:.1f}%</span></span>
  <span class="sep">|</span>
  <span>Positions <b>{n_pos}</b>/10</span>
  <span class="sep">|</span>
  <span>Cumulative <span class="{totpnl_class}"><b>${total_pnl:+,.2f}</b></span></span>
  <span class="sep">|</span>
  <span class="neu">{now_str}</span>
</div>
""", unsafe_allow_html=True)

    # ── SCORECARD ─────────────────────────────────────────────────────────────
    n_trades = len(df)
    wr = df["pnl_usd"].gt(0).mean() * 100 if n_trades > 0 else 0
    best  = df.loc[df["pnl_usd"].idxmax()] if n_trades > 0 else None
    worst = df.loc[df["pnl_usd"].idxmin()] if n_trades > 0 else None
    streak_type, streak_n = win_streak(df)
    avg_bars = df["bars_held"].mean() if n_trades > 0 else 0

    streak_class = "green" if streak_type == "W" else ("red" if streak_type == "L" else "orange")
    streak_label = f"{'WIN' if streak_type=='W' else 'LOSS'} STREAK" if streak_type != "–" else "STREAK"
    wr_class = "green" if wr >= 50 else "red"
    totpnl_card_class = "green" if total_pnl >= 0 else "red"

    best_str  = f"+${best['pnl_usd']:.0f} ({best['symbol']})"  if best is not None else "–"
    worst_str = f"${worst['pnl_usd']:.0f} ({worst['symbol']})" if worst is not None else "–"

    st.markdown(f"""
<div class="scorecard">
  <div class="card orange">
    <div class="label">TOTAL TRADES</div>
    <div class="value">{n_trades}</div>
    <div class="sub">avg {avg_bars:.0f} bars held</div>
  </div>
  <div class="card {wr_class}">
    <div class="label">WIN RATE</div>
    <div class="value">{wr:.1f}%</div>
    <div class="sub">{int(df['pnl_usd'].gt(0).sum())}W / {int(df['pnl_usd'].le(0).sum())}L</div>
  </div>
  <div class="card {totpnl_card_class}">
    <div class="label">TOTAL PnL</div>
    <div class="value">${total_pnl:+,.2f}</div>
    <div class="sub">closed trades only</div>
  </div>
  <div class="card green">
    <div class="label">BEST TRADE</div>
    <div class="value" style="font-size:1.0rem">{best_str}</div>
    <div class="sub">{best['exit_reason'] if best is not None else ''}</div>
  </div>
  <div class="card {streak_class}">
    <div class="label">{streak_label.upper()}</div>
    <div class="value">{streak_n}</div>
    <div class="sub">{'in a row' if streak_n > 0 else ''}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── ROW 2: POSITIONS | STRATEGY ──────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="bb-header">OPEN POSITIONS</div>', unsafe_allow_html=True)
        if not positions:
            st.markdown('<span class="neu" style="font-size:0.85rem">— No open positions —</span>', unsafe_allow_html=True)
        else:
            rows_html = ""
            for p in positions:
                sym_key = f"{p['symbol']}:{p['side']}"
                bars = bars_map.get(sym_key, bars_map.get(p["symbol"], "?"))
                pct_class = "pos" if p["pct"] >= 0 else "neg"
                side_class = "long-tag" if p["side"] == "LONG" else "short-tag"
                rows_html += f"""
<tr>
  <td class="sym">{p['symbol']}</td>
  <td><span class="{side_class}">{p['side']}</span></td>
  <td>${p['entry']:,.4f}</td>
  <td>${p['mark']:,.4f}</td>
  <td class="{pct_class}">{p['pct']:+.2f}%</td>
  <td class="{pct_class}">${p['upnl']:+.2f}</td>
  <td class="neu">{bars}/192</td>
  <td class="neu">${p['notional']:,.0f}</td>
</tr>"""
            st.markdown(f"""
<table class="bb-table">
<tr><th>SYMBOL</th><th>SIDE</th><th>ENTRY</th><th>MARK</th><th>PnL%</th><th>PnL$</th><th>BARS</th><th>NOTIONAL</th></tr>
{rows_html}
</table>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="bb-header">STRATEGY BREAKDOWN</div>', unsafe_allow_html=True)
        if not df.empty:
            strat_rows = ""
            for strat, grp in df.groupby("strategy_type"):
                n_s = len(grp)
                wr_s = grp["pnl_usd"].gt(0).mean() * 100
                tot_s = grp["pnl_usd"].sum()
                avg_mfe = grp["mfe_pct"].mean() * 100 if "mfe_pct" in grp else 0
                avg_mae = grp["mae_pct"].mean() * 100 if "mae_pct" in grp else 0
                avg_b = grp["bars_held"].mean()
                tot_class = "pos" if tot_s >= 0 else "neg"
                tag_cls = "tag-vr" if "vr" in strat else "tag-ma"
                label = "VR SQZ" if "vr" in strat else "MA CLU"
                strat_rows += f"""
<tr>
  <td><span class="{tag_cls}">{label}</span></td>
  <td>{n_s}</td>
  <td>{wr_s:.0f}%</td>
  <td class="{tot_class}">${tot_s:+.0f}</td>
  <td class="neu">{avg_mfe:+.2f}%</td>
  <td class="neu">{avg_mae:.2f}%</td>
  <td class="neu">{avg_b:.0f}b</td>
</tr>"""
            st.markdown(f"""
<table class="bb-table">
<tr><th>STRAT</th><th>N</th><th>WR</th><th>PnL$</th><th>MFE</th><th>MAE</th><th>AVG</th></tr>
{strat_rows}
</table>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── ROW 3: RECENT TRADES | EXIT REASON DIST ───────────────────────────────
    col_t, col_b = st.columns([3, 2])

    with col_t:
        st.markdown('<div class="bb-header">RECENT TRADES  <span class="neu">(15)</span></div>', unsafe_allow_html=True)
        if df.empty:
            st.markdown('<span class="neu">No closed trades.</span>', unsafe_allow_html=True)
        else:
            recent = df.sort_values("id", ascending=False).head(15)
            trade_rows = ""
            for _, r in recent.iterrows():
                pnl = r["pnl_usd"]
                pct_class = "pos" if pnl >= 0 else "neg"
                win_icon = "▲" if pnl >= 0 else "▼"
                side_cls = "long-tag" if r["direction"] == "LONG" else "short-tag"
                tag_cls = "tag-vr" if "vr" in str(r["strategy_type"]) else "tag-ma"
                t = str(r["exit_time"])[:16]
                mfe = r.get("mfe_pct", 0) or 0
                mae = r.get("mae_pct", 0) or 0
                trade_rows += f"""
<tr>
  <td class="neu">{t}</td>
  <td class="sym">{r['symbol']}</td>
  <td><span class="{side_cls}">{r['direction']}</span></td>
  <td><span class="{tag_cls}">{'VR' if 'vr' in str(r['strategy_type']) else 'MA'}</span></td>
  <td class="{pct_class}">{win_icon} ${pnl:+.2f}</td>
  <td class="neu">MFE {mfe*100:+.1f}%</td>
  <td class="neu">{r['bars_held']}b</td>
  <td class="neu" style="font-size:0.65rem">{r['exit_reason']}</td>
</tr>"""
            st.markdown(f"""
<table class="bb-table">
<tr><th>TIME</th><th>SYMBOL</th><th>DIR</th><th>STR</th><th>PnL$</th><th>MFE</th><th>BARS</th><th>REASON</th></tr>
{trade_rows}
</table>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="bb-header">EXIT REASON DISTRIBUTION</div>', unsafe_allow_html=True)
        if not df.empty:
            grp = df.groupby("exit_reason").agg(
                n=("pnl_usd", "count"),
                avg_pnl=("pnl_usd", "mean"),
                total=("pnl_usd", "sum"),
            ).sort_values("n", ascending=False)

            max_n = grp["n"].max()
            bars_html = ""
            for reason, row in grp.iterrows():
                bar = unicode_bar(row["n"], max_n, 12)
                avg_cls = "pos" if row["avg_pnl"] >= 0 else "neg"
                bars_html += f"""
<div class="bar-row">
  <span class="bar-label">{reason[:22]}</span>
  <span class="bar-fill">{bar}</span>
  <span class="bar-stat"> {int(row['n'])}x <span class="{avg_cls}">avg ${row['avg_pnl']:+.0f}</span></span>
</div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

    # ── Auto refresh ─────────────────────────────────────────────────────────
    time.sleep(REFRESH)
    st.cache_data.clear()
    st.rerun()


render()
