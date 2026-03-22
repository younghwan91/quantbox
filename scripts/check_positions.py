#!/usr/bin/env python3
"""Live position diagnostic — reads state + Binance to show accurate SL/TP/PnL.

Usage:
    python3 scripts/check_positions.py

Always reads strategy_type from state to pick the correct sl_pct.
Never calculates from memory.
"""
import json
import os
import sys
import time
import hmac
import hashlib
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
STATE_PATH = ROOT / "data" / "state" / "scalper_state.json"

# strategy_type → (sl_pct, trail_activate, trail_distance)
_SL_PARAMS: dict[str, tuple[float, float, float]] = {
    "vr_directional": (0.01, 0.01, 0.002),
    "vr_straddle":    (0.015, 0.015, 0.001),
    "ma_cluster":     (0.02, 0.02, 0.003),
}

BASE = "https://fapi.binance.com"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")


def _signed(path: str, params: dict | None = None) -> dict | list:
    params = params or {}
    params["timestamp"] = int(time.time() * 1000)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(
        f"{BASE}{path}?{qs}&signature={sig}",
        headers={"X-MBX-APIKEY": API_KEY},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    if not API_KEY or not API_SECRET:
        sys.exit("ERROR: BINANCE_API_KEY/SECRET not set")

    state = json.loads(STATE_PATH.read_text())
    positions = state.get("positions", {})

    if not positions:
        print("No open positions in state.")
        return

    # Fetch all open algo orders once
    algo_orders_raw = _signed("/fapi/v1/openAlgoOrders")
    algo_by_id: dict[str, dict] = {}
    if isinstance(algo_orders_raw, list):
        for o in algo_orders_raw:
            algo_by_id[str(o["algoId"])] = o

    print(f"{'─'*72}")
    print(f"{'POSITION DIAGNOSTIC':^72}")
    print(f"{'─'*72}")

    for pos_key, pos in positions.items():
        sym = pos.get("symbol", pos_key.split(":")[0])
        side = pos.get("position_side", "N/A")
        entry = float(pos.get("entry_price", 0))
        strategy_type = pos.get("strategy_type", "vr_directional")
        bars = pos.get("bars_held", 0)
        max_hold = pos.get("max_hold", 192)
        peak_pct = (pos.get("peak_gain") or 0.0) * 100
        trail_active = pos.get("trail_active", False)

        sl_pct, trail_act, trail_dist = _SL_PARAMS.get(
            strategy_type, _SL_PARAMS["vr_directional"]
        )

        # Current mark price
        mark_data = _signed("/fapi/v1/premiumIndex", {"symbol": sym})
        mark = float(mark_data["markPrice"])

        # PnL
        if side == "LONG":
            pnl_pct = (mark - entry) / entry * 100
            sl_calc = entry * (1 - sl_pct)
            dist_to_sl_pct = (mark - sl_calc) / entry * 100
        else:
            pnl_pct = (entry - mark) / entry * 100
            sl_calc = entry * (1 + sl_pct)
            dist_to_sl_pct = (sl_calc - mark) / entry * 100

        # Server SL — read actual trigger from Binance (not calculated)
        sl_order_id = str(pos.get("server_sl_order_id", ""))
        tp_order_id = str(pos.get("server_tp_order_id", ""))
        sl_algo = algo_by_id.get(sl_order_id, {})
        tp_algo = algo_by_id.get(tp_order_id, {})

        sl_trigger = sl_algo.get("triggerPrice", f"~{sl_calc:.5f}(calc)")
        tp_trigger = tp_algo.get("triggerPrice", "N/A")
        sl_status = sl_algo.get("algoStatus", "NOT IN openAlgoOrders ⚠️")
        tp_status = tp_algo.get("algoStatus", "NOT IN openAlgoOrders")

        # Notional / dollar PnL
        qty = float(sl_algo.get("quantity", 0) or tp_algo.get("quantity", 0))
        notional = qty * entry if qty else 0
        pnl_usd = notional * pnl_pct / 100 if notional else 0
        sl_usd = notional * (-sl_pct if side == "LONG" else -sl_pct) if notional else 0

        # Trail stop level
        trail_stop_pct = peak_pct - trail_dist * 100
        trail_triggered = trail_active and (pnl_pct < trail_stop_pct)

        print(f"\n{pos_key}  [{strategy_type}]")
        print(f"  Entry:       {entry:.5f}  |  Mark: {mark:.5f}  |  Side: {side}")
        print(f"  PnL:         {pnl_pct:+.3f}%  (${pnl_usd:+.2f})")
        print(f"  Bars:        {bars}/{max_hold}")
        print()
        print(f"  SL (server): trigger={sl_trigger}  status={sl_status}")
        print(f"  TP (server): trigger={tp_trigger}  status={tp_status}")
        print(f"  SL calc:     entry×{1-sl_pct if side=='LONG' else 1+sl_pct:.3f} = {sl_calc:.5f}  [{sl_pct*100:.1f}%  {strategy_type}]")
        print(f"  Dist to SL:  {dist_to_sl_pct:+.3f}%  |  Max loss if hit: ${abs(sl_usd):.2f}")
        print()
        print(f"  Trail:       active={trail_active}  peak={peak_pct:+.3f}%  trail_stop={trail_stop_pct:+.3f}%  exit_now={trail_triggered}")
        print(f"{'─'*72}")

    # Orphaned algo orders (position closed but algo order remains)
    state_sl_ids = {
        str(pos.get("server_sl_order_id", ""))
        for pos in positions.values()
    }
    state_tp_ids = {
        str(pos.get("server_tp_order_id", ""))
        for pos in positions.values()
    }
    state_algo_ids = state_sl_ids | state_tp_ids

    orphans = [o for oid, o in algo_by_id.items() if oid not in state_algo_ids]
    if orphans:
        print(f"\n⚠️  ORPHANED ALGO ORDERS ({len(orphans)}):")
        for o in orphans:
            print(f"  algoId={o['algoId']} sym={o['symbol']} type={o['orderType']} trigger={o['triggerPrice']} status={o['algoStatus']}")
    else:
        print("\n✅ No orphaned algo orders.")


if __name__ == "__main__":
    main()
