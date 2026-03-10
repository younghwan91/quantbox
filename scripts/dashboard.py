"""QuantBox Live Dashboard — scripts/dashboard.py

실행:
    uv run streamlit run scripts/dashboard.py --server.headless true
접속:
    http://localhost:8501   (같은 컴퓨터)
    http://[PC_IP]:8501     (같은 네트워크)
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

DB_LIVE = Path(__file__).parent.parent / "data" / "journal.sqlite"
REAL_FILTER = "ABS(pnl_pct) < 10 AND symbol NOT LIKE 'TEST%' AND symbol NOT LIKE '%:%'"
REFRESH_SEC = 30

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantBox Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main { background-color: #0d1117; }
    .stMetric { background: #161b22; border-radius: 8px; padding: 10px; }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── 데이터 로더 ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    return Client(
        api_key=os.environ.get("BINANCE_API_KEY", ""),
        api_secret=os.environ.get("BINANCE_API_SECRET", ""),
    )


def fetch_account(client: Client) -> dict:
    try:
        acct = client.futures_account()
        return {
            "wallet": float(acct["totalWalletBalance"]),
            "unrealized": float(acct["totalUnrealizedProfit"]),
            "margin": float(acct["totalInitialMargin"]),
            "available": float(acct["availableBalance"]),
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_positions(client: Client) -> list[dict]:
    try:
        raw = client.futures_position_information()
        result = []
        for p in raw:
            amt = float(p["positionAmt"])
            if abs(amt) == 0:
                continue
            entry = float(p["entryPrice"])
            mark = float(p["markPrice"])
            pnl_pct = (mark - entry) / entry * 100 if amt > 0 else (entry - mark) / entry * 100
            result.append(
                {
                    "심볼": p["symbol"],
                    "방향": p["positionSide"],
                    "진입가": f"${entry:,.4f}",
                    "현재가": f"${mark:,.4f}",
                    "PnL%": f"{pnl_pct:+.2f}%",
                    "_pnl": pnl_pct,
                    "미실현$": f"${float(p['unrealizedProfit']):+.2f}",
                    "Notional": f"${abs(float(p['notional'])):,.0f}",
                }
            )
        return sorted(result, key=lambda x: x["_pnl"], reverse=True)
    except Exception:
        return []


def fetch_state_bars() -> dict:
    """state.json에서 bars_held 읽기."""
    import json

    state_path = Path(__file__).parent.parent / "data" / "state" / "scalper_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.load(open(state_path))
        return {sym: pos.get("bars_held", 0) for sym, pos in state.get("positions", {}).items()}
    except Exception:
        return {}


def fetch_recent_trades(n: int = 20) -> pd.DataFrame:
    if not DB_LIVE.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_LIVE)
        df = pd.read_sql_query(
            f"""
            SELECT symbol, strategy_type, direction, pnl_usd, exit_reason,
                   bars_held, exit_time
            FROM trades
            WHERE {REAL_FILTER}
            ORDER BY id DESC LIMIT {n}
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def fetch_strategy_stats() -> pd.DataFrame:
    if not DB_LIVE.exists():
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_LIVE)
        df = pd.read_sql_query(
            f"""
            SELECT
                strategy_type as 전략,
                COUNT(*) as 거래수,
                ROUND(100.0 * SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as WR,
                ROUND(SUM(pnl_usd), 2) as 합계PnL,
                ROUND(AVG(pnl_usd), 2) as 평균PnL,
                ROUND(AVG(bars_held), 1) as 평균보유바
            FROM trades WHERE {REAL_FILTER}
            GROUP BY strategy_type
            ORDER BY 합계PnL DESC
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# ─── 렌더링 ───────────────────────────────────────────────────────────────────
def render():
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    st.title("📈 QuantBox Live Dashboard")
    st.caption(f"마지막 갱신: {now}  •  {REFRESH_SEC}초마다 자동 새로고침")

    client = get_client()

    # ── 1. 계좌 요약 ──────────────────────────────────────────────────────────
    acct = fetch_account(client)

    if "error" not in acct:
        total_equity = acct["wallet"] + acct["unrealized"]
        margin_pct = acct["margin"] / acct["wallet"] * 100 if acct["wallet"] else 0
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 잔고 (Wallet)", f"${acct['wallet']:,.2f}")
        col2.metric(
            "📊 미실현 PnL",
            f"${acct['unrealized']:+,.2f}",
            delta=f"{acct['unrealized'] / acct['wallet'] * 100:+.2f}%" if acct["wallet"] else None,
        )
        col3.metric("🔒 마진 사용", f"${acct['margin']:,.2f}", delta=f"{margin_pct:.1f}% 사용 중")
        col4.metric("✅ 여유 잔고", f"${acct['available']:,.2f}")
    else:
        st.error(f"API 연결 실패: {acct['error']}")

    st.divider()

    # ── 2. 열린 포지션 ────────────────────────────────────────────────────────
    bars_map = fetch_state_bars()
    positions = fetch_positions(client)

    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"🟡 열린 포지션 ({len(positions)}개)")
        if not positions:
            st.info("현재 열린 포지션 없음")
        else:
            for pos in positions:
                sym_key = f"{pos['심볼']}:{pos['방향']}"
                bars = bars_map.get(sym_key, bars_map.get(pos["심볼"], "?"))
                pnl_val = pos["_pnl"]
                color = "🟢" if pnl_val >= 0 else "🔴"
                st.markdown(
                    f"{color} **{pos['심볼']}** `{pos['방향']}`  "
                    f"진입 {pos['진입가']} → 현재 {pos['현재가']}  "
                    f"**{pos['PnL%']}** ({pos['미실현$']})  "
                    f"bars={bars}/192  notional={pos['Notional']}"
                )

    # ── 3. 전략 성과 ──────────────────────────────────────────────────────────
    with right:
        st.subheader("📊 전략별 성과")
        stats = fetch_strategy_stats()
        if stats.empty:
            st.info("트레이드 기록 없음")
        else:
            for _, row in stats.iterrows():
                pnl = row["합계PnL"]
                color = "🟢" if pnl >= 0 else "🔴"
                wr = row["WR"]
                st.markdown(
                    f"{color} **{row['전략']}**  "
                    f"{int(row['거래수'])}건  WR={wr:.0f}%  "
                    f"합계=**${pnl:+.2f}**  평균=${row['평균PnL']:+.2f}  "
                    f"평균보유={row['평균보유바']}바"
                )

    st.divider()

    # ── 4. 최근 트레이드 ──────────────────────────────────────────────────────
    st.subheader("📋 최근 트레이드 (최근 20건)")
    df = fetch_recent_trades(20)
    if df.empty:
        st.info("아직 청산된 트레이드 없음")
    else:
        df["결과"] = df["pnl_usd"].apply(lambda x: "✅ WIN" if x > 0 else "❌ LOSS")
        df["PnL$"] = df["pnl_usd"].apply(lambda x: f"${x:+.2f}")
        df["청산시각"] = df["exit_time"].str[:16]
        display = df[["결과", "청산시각", "symbol", "direction", "strategy_type", "PnL$", "exit_reason", "bars_held"]].copy()
        display.columns = ["결과", "시각", "심볼", "방향", "전략", "PnL$", "청산이유", "보유바"]

        def color_pnl(val):
            if "+" in str(val):
                return "color: #3fb950; font-weight: bold"
            elif "-" in str(val):
                return "color: #f85149; font-weight: bold"
            return ""

        styled = display.style.map(color_pnl, subset=["PnL$"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── 자동 새로고침 ─────────────────────────────────────────────────────────
    time.sleep(REFRESH_SEC)
    st.rerun()


render()
