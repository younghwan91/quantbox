#!/usr/bin/env python3
"""r85: Execution Quality Analysis — Live vs Backtest Edge Comparison.

Measures how much edge is lost to execution gaps (bracket delays, ratchet
misses, ghost losses, restart gaps) vs the theoretical backtest edge.

Usage:
    python research/active/r81_r85/r85_execution_quality.py
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent.parent.parent

from quantbox.core.bot_utils import GIT_COMMIT  # noqa: E402
from quantbox.strategy.vol_straddle.config import VolStraddleConfig  # noqa: E402

cfg = VolStraddleConfig()
# 왕복 비용 = fee×2 + slippage×2
ROUND_TRIP_COST = cfg.round_trip_cost  # config에서 읽음 (하드코딩 금지)


# ── 1. 라이브 거래 데이터 로드 ──────────────────────────────────────────────────


def load_live_trades(journal_path: str = "data/journal.sqlite") -> list[dict]:
    """journal.sqlite에서 전체 스트래들 거래를 로드한다.

    strategy_type이 'straddle' 또는 'vol_straddle'인 레코드를 반환.
    DB가 없으면 빈 리스트 반환.
    """
    db = ROOT / journal_path
    if not db.exists():
        return []

    try:
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, symbol, strategy_type, direction,
                   entry_price, exit_price, pnl_pct, pnl_usd,
                   mfe_pct, mae_pct, notional, bars_held,
                   entry_time, exit_time, exit_reason,
                   balance_before, balance_after, session, commit_sha
            FROM trades
            WHERE strategy_type IN ('straddle', 'vol_straddle')
            ORDER BY entry_time ASC
            """
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[경고] journal.sqlite 로드 실패: {exc}")
        return []


# ── 2. 정산 이벤트 로드 ────────────────────────────────────────────────────────


def load_recon_events(recon_path: str = "data/journal/recon.db") -> list[dict]:
    """recon.db에서 정산 이벤트를 로드한다.

    ghost 감지, bracket 수리, 서버 SL/TP 체결 등을 반환.
    DB가 없으면 빈 리스트 반환.
    """
    db = ROOT / recon_path
    if not db.exists():
        return []

    try:
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, timestamp, event_type, symbol, action, result, details_json, session_id "
            "FROM recon_events ORDER BY timestamp ASC"
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d.pop("details_json", "{}"))
            result.append(d)
        return result
    except Exception as exc:
        print(f"[경고] recon.db 로드 실패: {exc}")
        return []


# ── 3. 스트래들 페어 지표 계산 ──────────────────────────────────────────────────


def compute_pair_metrics(trades: list[dict]) -> list[dict]:
    """거래를 스트래들 페어로 묶어 페어별 지표를 계산한다.

    스트래들 페어 = 동일 심볼 + entry_time이 같은(또는 60초 이내) LONG + SHORT 한 쌍.

    Returns:
        페어별 지표 딕셔너리 리스트. 각 항목:
            symbol, entry_time, long_pnl_usd, short_pnl_usd,
            winning_leg_pnl, losing_leg_pnl, net_pnl_usd,
            fee_usd, net_edge_usd, exit_reason_long, exit_reason_short
    """
    # symbol + entry_time(초 단위 버킷) 기준으로 그룹화
    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        entry_raw = t.get("entry_time") or ""
        # entry_time 문자열을 epoch초로 변환 (30초 버킷으로 스냅)
        try:
            dt = datetime.fromisoformat(entry_raw.replace("Z", "+00:00"))
            bucket_ts = int(dt.timestamp() // 60) * 60  # 1분 버킷
        except Exception:
            bucket_ts = 0
        key = f"{t['symbol']}_{bucket_ts}"
        buckets[key].append(t)

    pairs: list[dict] = []
    for key, group in buckets.items():
        longs = [t for t in group if t.get("direction") == "LONG"]
        shorts = [t for t in group if t.get("direction") == "SHORT"]

        if not longs or not shorts:
            # 미완성 페어는 단독 레그로 기록
            for t in group:
                fee = _calc_fee(t.get("notional") or 0.0)
                pnl = t.get("pnl_usd") or 0.0
                pairs.append(
                    {
                        "symbol": t["symbol"],
                        "entry_time": t.get("entry_time"),
                        "pair_type": "orphan",
                        "long_pnl_usd": pnl if t.get("direction") == "LONG" else 0.0,
                        "short_pnl_usd": pnl if t.get("direction") == "SHORT" else 0.0,
                        "winning_leg_pnl": max(pnl, 0.0),
                        "losing_leg_pnl": min(pnl, 0.0),
                        "gross_pnl_usd": pnl,
                        "fee_usd": fee,
                        "net_pnl_usd": pnl - fee,
                        "net_edge_usd": pnl - fee,
                        "exit_reason_long": t.get("exit_reason") if t.get("direction") == "LONG" else "-",
                        "exit_reason_short": t.get("exit_reason") if t.get("direction") == "SHORT" else "-",
                        "bars_held_long": t.get("bars_held") if t.get("direction") == "LONG" else None,
                        "bars_held_short": t.get("bars_held") if t.get("direction") == "SHORT" else None,
                    }
                )
            continue

        # 가장 가까운 LONG-SHORT 쌍으로 매칭
        used_shorts: set[int] = set()
        for long_t in longs:
            best_short = None
            best_idx = -1
            for idx, short_t in enumerate(shorts):
                if idx in used_shorts:
                    continue
                best_short = short_t
                best_idx = idx
                break

            if best_short is None:
                continue
            used_shorts.add(best_idx)

            long_pnl = long_t.get("pnl_usd") or 0.0
            short_pnl = best_short.get("pnl_usd") or 0.0
            gross_pnl = long_pnl + short_pnl

            long_notional = long_t.get("notional") or 0.0
            short_notional = best_short.get("notional") or 0.0
            fee = _calc_fee(long_notional) + _calc_fee(short_notional)

            pairs.append(
                {
                    "symbol": long_t["symbol"],
                    "entry_time": long_t.get("entry_time"),
                    "pair_type": "full",
                    "long_pnl_usd": long_pnl,
                    "short_pnl_usd": short_pnl,
                    "winning_leg_pnl": max(long_pnl, short_pnl),
                    "losing_leg_pnl": min(long_pnl, short_pnl),
                    "gross_pnl_usd": gross_pnl,
                    "fee_usd": fee,
                    "net_pnl_usd": gross_pnl - fee,
                    "net_edge_usd": gross_pnl - fee,
                    "exit_reason_long": long_t.get("exit_reason"),
                    "exit_reason_short": best_short.get("exit_reason"),
                    "bars_held_long": long_t.get("bars_held"),
                    "bars_held_short": best_short.get("bars_held"),
                }
            )

    return pairs


def _calc_fee(notional: float) -> float:
    """왕복 수수료 계산 (진입+청산 × fee_rate + slippage_rate)."""
    return notional * ROUND_TRIP_COST


# ── 4. 실행 갭 계산 ────────────────────────────────────────────────────────────


def compute_execution_gaps(trades: list[dict], recon_events: list[dict]) -> dict:
    """실행 갭을 정량화한다: bracket 지연, ratchet 미스, ghost 손실, 재시작 갭.

    Returns:
        갭 유형별 집계 딕셔너리:
            bracket_repairs: int
            bracket_repair_fails: int
            ghost_detections: int
            ghost_closed_count: int
            server_sl_fills: int
            server_tp_fills: int
            total_ghost_pnl_usd: float  (ghost 청산 손실 추정)
            ratchet_miss_count: int     (SL가 아닌 max_hold로 청산된 trail_activate 후 거래)
    """
    gaps: dict[str, int | float] = {
        "bracket_repairs": 0,
        "bracket_repair_fails": 0,
        "ghost_detections": 0,
        "ghost_closed_count": 0,
        "server_sl_fills": 0,
        "server_tp_fills": 0,
        "total_ghost_pnl_usd": 0.0,
        "ratchet_miss_count": 0,
    }

    for ev in recon_events:
        etype = ev.get("event_type", "")
        if etype == "bracket_repair":
            gaps["bracket_repairs"] = int(gaps["bracket_repairs"]) + 1
        elif etype == "bracket_repair_fail":
            gaps["bracket_repair_fails"] = int(gaps["bracket_repair_fails"]) + 1
        elif etype == "ghost_detected":
            gaps["ghost_detections"] = int(gaps["ghost_detections"]) + 1
        elif etype == "ghost_closed":
            gaps["ghost_closed_count"] = int(gaps["ghost_closed_count"]) + 1
            # details에 pnl이 있으면 합산
            pnl = ev.get("details", {}).get("pnl_usd", 0.0) or 0.0
            gaps["total_ghost_pnl_usd"] = float(gaps["total_ghost_pnl_usd"]) + pnl
        elif etype == "server_sl_fill":
            gaps["server_sl_fills"] = int(gaps["server_sl_fills"]) + 1
        elif etype == "server_tp_fill":
            gaps["server_tp_fills"] = int(gaps["server_tp_fills"]) + 1

    # ratchet 미스: trailing이 활성화됐을 텐데 max_hold로 청산된 거래
    trail_activate = cfg.d_trail_activate
    for t in trades:
        mfe = t.get("mfe_pct") or 0.0
        reason = t.get("exit_reason") or ""
        # MFE가 trail_activate를 넘었는데 max_hold/time으로 청산 → ratchet 미스 의심
        if mfe > trail_activate and "hold" in reason.lower():
            gaps["ratchet_miss_count"] = int(gaps["ratchet_miss_count"]) + 1

    return gaps


# ── 5. 리포트 생성 ─────────────────────────────────────────────────────────────


def generate_report(pair_metrics: list[dict], execution_gaps: dict, output_path: Path) -> None:
    """Markdown 리포트를 생성한다.

    포함 항목:
    - 요약 테이블: 총 거래 수, 승률, 총 수익, 수수료, 실행 갭
    - 페어별 상세 테이블
    - 실행 갭 분류: bracket 지연, ratchet 미스, ghost 손실, 재시작 갭
    - 판정: 실행 갭 < 0.1% → 시그널 튜닝 정당화
    """
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit_short = GIT_COMMIT.get("short", "unknown")

    full_pairs = [p for p in pair_metrics if p["pair_type"] == "full"]
    orphans = [p for p in pair_metrics if p["pair_type"] == "orphan"]

    n_pairs = len(full_pairs)
    n_orphans = len(orphans)
    total_gross = sum(p["gross_pnl_usd"] for p in full_pairs)
    total_fees = sum(p["fee_usd"] for p in full_pairs)
    total_net = sum(p["net_pnl_usd"] for p in full_pairs)
    wins = sum(1 for p in full_pairs if p["net_pnl_usd"] > 0)
    wr = wins / n_pairs if n_pairs > 0 else 0.0

    # 실행 갭 USD 합산
    ghost_loss = float(execution_gaps.get("total_ghost_pnl_usd", 0.0))
    total_exec_gap = ghost_loss  # 현재는 ghost 손실만 직접 USD로 측정 가능

    # 실행 갭 비율 (총 gross 대비)
    exec_gap_pct = abs(total_exec_gap) / abs(total_gross) if total_gross != 0 else 0.0

    lines: list[str] = [
        "# r85: Execution Quality Analysis",
        "",
        f"생성일시: {now_str}  ",
        f"커밋: `{commit_short}`  ",
        f"분석 설정: fee={cfg.fee_rate * 100:.3f}%, slippage={cfg.slippage_rate * 100:.3f}%, "
        f"round_trip={ROUND_TRIP_COST * 100:.3f}%",
        "",
        "---",
        "",
        "## 요약",
        "",
        "| 항목 | 값 |",
        "|------|----|",
        f"| 스트래들 페어 수 | {n_pairs} |",
        f"| 미완성 레그 (orphan) | {n_orphans} |",
        f"| 승률 | {wr * 100:.1f}% |",
        f"| 총 수익 (gross) | ${total_gross:+.2f} |",
        f"| 총 수수료 | ${total_fees:.2f} |",
        f"| 순 수익 (net) | ${total_net:+.2f} |",
        f"| Ghost 손실 추정 | ${ghost_loss:+.2f} |",
        f"| 실행 갭 비율 | {exec_gap_pct * 100:.3f}% |",
        "",
    ]

    # 판정
    if exec_gap_pct < 0.001:  # 0.1% 미만
        decision = "실행 갭 < 0.1% → **시그널 튜닝 정당화**. 실행보다 시그널 품질 개선 우선."
    else:
        decision = (
            f"실행 갭 = {exec_gap_pct * 100:.3f}% → **실행 개선 우선**. 시그널 튜닝 전 bracket/ratchet 수정 필요."
        )

    lines += [
        "### 판정",
        "",
        f"> {decision}",
        "",
        "---",
        "",
        "## 실행 갭 분류",
        "",
        "| 갭 유형 | 건수 | 비고 |",
        "|---------|------|------|",
        f"| Bracket 수리 성공 | {execution_gaps.get('bracket_repairs', 0)} | SL/TP 재배치 |",
        f"| Bracket 수리 실패 | {execution_gaps.get('bracket_repair_fails', 0)} | 수리 불가 케이스 |",
        f"| Ghost 포지션 감지 | {execution_gaps.get('ghost_detections', 0)} | 상태 불일치 |",
        f"| Ghost 포지션 청산 | {execution_gaps.get('ghost_closed_count', 0)} | 강제 청산 |",
        f"| 서버 SL 체결 | {execution_gaps.get('server_sl_fills', 0)} | Binance 서버 SL |",
        f"| 서버 TP 체결 | {execution_gaps.get('server_tp_fills', 0)} | Binance 서버 TP |",
        f"| Ratchet 미스 의심 | {execution_gaps.get('ratchet_miss_count', 0)} | MFE > trail_activate + max_hold 청산 |",
        f"| Ghost 손실 (USD) | ${ghost_loss:+.2f} | 직접 측정 가능한 갭 |",
        "",
        "---",
        "",
    ]

    if full_pairs:
        lines += [
            f"## 페어별 상세 ({min(n_pairs, 50)}건 표시, 최대 50건)",
            "",
            "| # | 심볼 | 진입시각 | Long PnL | Short PnL | Gross | 수수료 | Net | LONG 청산 | SHORT 청산 |",
            "|---|------|----------|----------|-----------|-------|--------|-----|-----------|------------|",
        ]
        for i, p in enumerate(full_pairs[:50], 1):
            entry_str = (p.get("entry_time") or "")[:19]
            lines.append(
                f"| {i} | {p['symbol']} | {entry_str} "
                f"| ${p['long_pnl_usd']:+.2f} | ${p['short_pnl_usd']:+.2f} "
                f"| ${p['gross_pnl_usd']:+.2f} | ${p['fee_usd']:.2f} | ${p['net_pnl_usd']:+.2f} "
                f"| {p.get('exit_reason_long', '?')} | {p.get('exit_reason_short', '?')} |"
            )
        lines.append("")
    else:
        lines += [
            "## 페어별 상세",
            "",
            "_데이터 없음 — 라이브 거래 누적 후 재실행_",
            "",
        ]

    if orphans:
        lines += [
            f"## 미완성 레그 (Orphan, {len(orphans)}건)",
            "",
            "짝이 맞지 않는 LONG 또는 SHORT 단독 레그. 재시작/강제청산 관련 갭.",
            "",
            "| # | 심볼 | 진입시각 | 방향 | PnL | 청산 이유 |",
            "|---|------|----------|------|-----|-----------|",
        ]
        for i, p in enumerate(orphans[:20], 1):
            entry_str = (p.get("entry_time") or "")[:19]
            direction = "LONG" if p["long_pnl_usd"] != 0 else "SHORT"
            pnl_v = p["long_pnl_usd"] if direction == "LONG" else p["short_pnl_usd"]
            reason = p.get("exit_reason_long") if direction == "LONG" else p.get("exit_reason_short")
            lines.append(f"| {i} | {p['symbol']} | {entry_str} | {direction} | ${pnl_v:+.2f} | {reason or '?'} |")
        lines.append("")

    lines += [
        "---",
        "",
        f"_자동 생성: r85_execution_quality.py / commit {commit_short}_",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ── 6. 진입점 ──────────────────────────────────────────────────────────────────


def main() -> None:
    trades = load_live_trades()
    recon_events = load_recon_events()

    if not trades:
        print("스트래들 거래 데이터가 없습니다. 라이브 데이터 누적 후 재실행하세요 (data/journal.sqlite 필요).")
        return

    pair_metrics = compute_pair_metrics(trades)
    execution_gaps = compute_execution_gaps(trades, recon_events)

    commit_short = GIT_COMMIT.get("short", "unknown")
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = ROOT / f"docs/backtest_results/202603/{date_str}_{commit_short}_r85_execution_quality.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_report(pair_metrics, execution_gaps, output_path)
    print(f"리포트 작성 완료: {output_path}")

    # 요약 출력
    full_pairs = [p for p in pair_metrics if p["pair_type"] == "full"]
    n = len(full_pairs)
    if n > 0:
        total_net = sum(p["net_pnl_usd"] for p in full_pairs)
        wins = sum(1 for p in full_pairs if p["net_pnl_usd"] > 0)
        print(f"페어 수: {n}, 승률: {wins / n * 100:.1f}%, 순 수익: ${total_net:+.2f}")
        print(f"Ghost 손실: ${execution_gaps.get('total_ghost_pnl_usd', 0.0):+.2f}")
        print(f"Ratchet 미스: {execution_gaps.get('ratchet_miss_count', 0)}건")


if __name__ == "__main__":
    main()
