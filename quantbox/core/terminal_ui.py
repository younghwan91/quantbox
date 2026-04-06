"""
QuantBox Terminal UI — Startup banner, warnings, and formatted output.

Provides:
- ASCII art startup banner with system info
- Formatted cycle/trade output helpers
- Color constants for terminal readability
"""

import sys
from datetime import datetime, timezone

# ─────────────────── ANSI Color Codes ────────────────────────────────


class Color:
    """ANSI escape codes for terminal colors."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_GREEN = "\033[42m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY or CI)."""
        for attr in dir(cls):
            if attr.isupper() and not attr.startswith("_"):
                setattr(cls, attr, "")


# Disable colors if not a real terminal
if not sys.stdout.isatty():
    Color.disable()


# ─────────────────── Constants ───────────────────────────────────────

BANNER = r"""
{cyan}{bold}
   ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗██████╗  ██████╗ ██╗  ██╗
  ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔══██╗██╔═══██╗╚██╗██╔╝
  ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██████╔╝██║   ██║ ╚███╔╝
  ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██╔══██╗██║   ██║ ██╔██╗
  ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ██████╔╝╚██████╔╝██╔╝ ██╗
   ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
{reset}"""

DIVIDER_HEAVY = "━" * 72
DIVIDER_LIGHT = "─" * 72
DIVIDER_DOT = "┄" * 72

VERSION = "2.0.0"


# ─────────────────── Startup Display ─────────────────────────────────


def print_banner() -> None:
    """Print the QuantBox ASCII art banner."""
    print(
        BANNER.format(
            cyan=Color.CYAN,
            bold=Color.BOLD,
            reset=Color.RESET,
        )
    )


def print_system_info(
    mode: str,
    strategy: str,
    balance: float,
    n_symbols: int,
    max_slots: int,
    leverage_base: int,
    leverage_max: int,
    lookback: int,
    threshold: float,
    target: float,
    stop_mult: float,
    max_hold: int,
    d_trail_activate: float = 0.04,
    d_trail_distance: float = 0.005,
    min_confidence: int = 0,
    testnet: bool = False,
    ma_cluster_enabled: bool = False,
    ma_cluster_pct: float = 0.03,
    ma_cluster_min_duration: int = 8,
    ma_cluster_max_duration: int = 24,
    ma_cluster_min_momentum: float = 0.01,
    ma_cluster_sl: float = 0.02,
    ma_cluster_tp: float = 0.08,
    ma_cluster_trail_activate: float = 0.02,
    ma_cluster_trail_distance: float = 0.005,
    ma_cluster_leverage: float = 12.0,
    donchian_enabled: bool = False,
    donchian_lookback: int = 50,
    donchian_vol_mult: float = 2.5,
    donchian_sl: float = 0.015,
    donchian_tp: float = 0.20,
    donchian_trail_activate: float = 0.02,
    donchian_trail_distance: float = 0.008,
    donchian_leverage: float = 15.0,
    risk_per_trade: float = 0.01,
) -> None:
    """Print formatted system configuration table.

    Args:
        mode: 'LIVE' or 'PAPER'.
        strategy: Strategy name.
        balance: Starting USDT balance.
        n_symbols: Number of symbols in universe.
        max_slots: Max concurrent positions.
        leverage_base: Base leverage.
        leverage_max: Max leverage.
        lookback: Signal lookback bars.
        threshold: Min move threshold.
        target: Target retrace.
        stop_mult: Stop loss multiplier.
        max_hold: Max hold bars.
        min_confidence: Minimum confidence score.
        testnet: Whether using testnet.
        ma_cluster_enabled: Whether MA cluster strategy is active.
        ma_cluster_pct: MA cluster spread threshold.
        ma_cluster_min_duration: Min bars in cluster.
        ma_cluster_max_duration: Max bars — zombie filter.
        ma_cluster_min_momentum: Min |momentum| for entry.
        ma_cluster_sl: MA cluster stop loss.
        ma_cluster_tp: MA cluster take profit.
        ma_cluster_leverage: MA cluster leverage.
    """
    C = Color
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Mode badge
    if mode == "LIVE":
        mode_badge = f"{C.BG_RED}{C.WHITE}{C.BOLD}  ● LIVE TRADING  {C.RESET}"
    else:
        mode_badge = f"{C.BG_GREEN}{C.WHITE}{C.BOLD}  ● PAPER MODE  {C.RESET}"

    net_badge = f"{C.YELLOW}[TESTNET]{C.RESET}" if testnet else ""

    print(f"{C.DIM}{DIVIDER_HEAVY}{C.RESET}")
    print(f"  {mode_badge} {net_badge}")
    print(f"  {C.DIM}v{VERSION}  |  {now}{C.RESET}")
    print(f"{C.DIM}{DIVIDER_HEAVY}{C.RESET}")
    print()

    # System table
    _print_section("SYSTEM")
    _print_row("Strategy", strategy.upper())
    _print_row("Balance", f"${balance:,.2f}")
    _print_row("Universe", f"{n_symbols} symbols")
    _print_row("Max Slots", str(max_slots))
    print()

    _print_section("STRATEGIES")
    # Show risk-based leverage: L = risk_per_trade × max_concurrent / SL
    _vr_lev = risk_per_trade * max_slots / stop_mult if stop_mult > 0 else leverage_base
    _print_row("VR Directional", f"{Color.GREEN}ENABLED{Color.RESET} (D_mom8, {_vr_lev:.0f}x)")
    if ma_cluster_enabled:
        _ma_lev = risk_per_trade * max_slots / ma_cluster_sl if ma_cluster_sl > 0 else ma_cluster_leverage
        _print_row("MA Cluster", f"{Color.GREEN}ENABLED{Color.RESET} ({_ma_lev:.0f}x)")
    else:
        _print_row("MA Cluster", f"{Color.RED}DISABLED{Color.RESET}")
    if donchian_enabled:
        _dc_lev = risk_per_trade * max_slots / donchian_sl if donchian_sl > 0 else donchian_leverage
        _print_row("Donchian", f"{Color.GREEN}ENABLED{Color.RESET} ({_dc_lev:.0f}x)")
    else:
        _print_row("Donchian", f"{Color.RED}DISABLED{Color.RESET}")
    print()

    _print_section("VR DIRECTIONAL")
    _print_row("VR Threshold", f"{threshold * 100:.1f}%")
    _print_row("SL / TP", f"{stop_mult * 100:.0f}% / {target * 100:.0f}%")
    _print_row("Trail", f"{d_trail_activate * 100:.1f}%→{d_trail_distance * 100:.1f}% (activate→distance)")
    _print_row("Max Hold", f"{max_hold} bars")
    _print_row("Leverage", f"{_vr_lev:.0f}x (risk={risk_per_trade * 100:.0f}%×N={max_slots}/SL={stop_mult * 100:.0f}%)")
    print()

    if ma_cluster_enabled:
        _print_section("MA CLUSTER")
        _print_row("Spread", f"{ma_cluster_pct * 100:.0f}%")
        _print_row("Duration", f"{ma_cluster_min_duration}-{ma_cluster_max_duration} bars (4h)")
        _print_row("Min Momentum", f"{ma_cluster_min_momentum * 100:.1f}%")
        _print_row("SL / TP", f"{ma_cluster_sl * 100:.0f}% / {ma_cluster_tp * 100:.0f}%")
        _print_row(
            "Trail",
            f"{ma_cluster_trail_activate * 100:.1f}%→{ma_cluster_trail_distance * 100:.1f}% (activate→distance)",
        )
        _print_row("Leverage", f"{ma_cluster_leverage:.0f}x")
        print()

    if donchian_enabled:
        _print_section("DONCHIAN")
        _print_row("Lookback", f"{donchian_lookback} bars (4h)")
        _print_row("Vol Spike", f"{donchian_vol_mult}x SMA(20)")
        _print_row("SL / TP", f"{donchian_sl * 100:.1f}% / {donchian_tp * 100:.0f}%")
        _print_row(
            "Trail",
            f"{donchian_trail_activate * 100:.1f}%→{donchian_trail_distance * 100:.1f}% (activate→distance)",
        )
        _print_row("Leverage", f"{donchian_leverage:.0f}x")
        print()


def format_section_header(title: str) -> str:
    """Format a colored section header for cycle output.

    Args:
        title: Section name (e.g. 'POSITIONS', 'SIGNALS', 'SUMMARY').

    Returns:
        Formatted string with CYAN+BOLD brackets.
    """
    C = Color
    return f"  {C.CYAN}{C.BOLD}▸ {title}{C.RESET}"


def format_section_divider() -> str:
    """Return a lightweight dot divider for visual separation between cycle sections."""
    return f"  {Color.DIM}{DIVIDER_DOT}{Color.RESET}"


def format_position_line(
    pos_key: str,
    strategy_type: str,
    direction: str,
    entry_price: float,
    current_price: float,
    upnl_pct: float,
    bars: int,
    max_hold: int,
    sl_pct: float,
    sl_active: bool,
    tp_pct: float,
    tp_active: bool,
    trail_active: bool,
    peak_gain: float | None = None,
) -> str:
    """Format a colored, structured position status line.

    All arguments are pre-computed display values — this function is a pure
    display formatter with no knowledge of position internals.

    Args:
        pos_key: Position compound key (e.g. 'SUIUSDT:LONG').
        strategy_type: Strategy name (e.g. 'vr_directional').
        direction: 'L' or 'S'.
        entry_price: Entry price.
        current_price: Current mark price.
        upnl_pct: Unrealized PnL percentage.
        bars: Bars held.
        max_hold: Max hold bars.
        sl_pct: Stop loss percentage (0-1 scale).
        sl_active: Whether server SL order is active.
        tp_pct: Take profit percentage (0-1 scale).
        tp_active: Whether server TP order is active.
        trail_active: Whether trailing stop is active.
        peak_gain: Peak gain percentage (0-1 scale) if trail active.

    Returns:
        Multi-line formatted string.
    """
    C = Color
    dir_color = C.GREEN if direction == "L" else C.RED
    pnl_color = C.GREEN if upnl_pct >= 0 else C.RED

    sl_check = f"{C.GREEN}✓{C.RESET}" if sl_active else f"{C.RED}✗{C.RESET}"
    tp_check = f"{C.GREEN}✓{C.RESET}" if tp_active else f"{C.RED}✗{C.RESET}"

    if trail_active:
        peak_str = f"({peak_gain * 100:+.1f}%)" if peak_gain is not None else ""
        trail_str = f"{C.GREEN}✓{C.RESET}{C.DIM}{peak_str}{C.RESET}"
    else:
        trail_str = f"{C.DIM}✗{C.RESET}"

    return (
        f"    {C.BOLD}{pos_key}{C.RESET}  {C.DIM}{strategy_type}{C.RESET}\n"
        f"      {dir_color}{direction}{C.RESET} "
        f"${entry_price:.4f} → ${current_price:.4f}  "
        f"{pnl_color}{C.BOLD}uPnL={upnl_pct:+.2f}%{C.RESET}  "
        f"{C.DIM}{bars}/{max_hold}bars{C.RESET}\n"
        f"      SL={sl_pct * 100:.1f}%{sl_check}  "
        f"TP={tp_pct * 100:.0f}%{tp_check}  "
        f"trail={trail_str}"
    )


def _print_section(title: str) -> None:
    """Print a section header (startup banner use)."""
    print(format_section_header(title))


def _print_row(label: str, value: str) -> None:
    """Print a key-value row, aligned."""
    print(f"  {Color.DIM}{label:<16}{Color.RESET}{value}")


# ─────────────────── Cycle Formatting ────────────────────────────────


def format_cycle_header(
    cycle: int,
    balance: float,
    n_active: int,
    max_slots: int,
    timestamp: datetime | None = None,
) -> str:
    """Format cycle header line.

    Args:
        cycle: Cycle number.
        balance: Current balance.
        n_active: Active position count.
        max_slots: Max slot count.
        timestamp: Optional UTC timestamp.

    Returns:
        Formatted string.
    """
    C = Color
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%H:%M")

    # Balance color
    bal_str = f"${balance:,.2f}"

    # Slots color
    if n_active == max_slots:
        slot_str = f"{C.YELLOW}{n_active}/{max_slots}{C.RESET}"
    elif n_active == 0:
        slot_str = f"{C.DIM}{n_active}/{max_slots}{C.RESET}"
    else:
        slot_str = f"{C.GREEN}{n_active}/{max_slots}{C.RESET}"

    return (
        f"{C.DIM}{DIVIDER_LIGHT}{C.RESET}\n"
        f"  {C.BOLD}#{cycle:<5}{C.RESET}"
        f"{C.DIM}{ts}{C.RESET}  "
        f"{bal_str}  "
        f"Slots {slot_str}"
    )


def format_entry(
    symbol: str,
    side: str,
    price: float,
    leverage: int,
    size_usd: float,
    confidence: int | None = None,
) -> str:
    """Format position entry log line.

    Args:
        symbol: Trading pair.
        side: 'LONG' or 'SHORT'.
        price: Entry price.
        leverage: Applied leverage.
        size_usd: Position size in USD.
        confidence: Optional confidence score.

    Returns:
        Formatted string.
    """
    C = Color
    side_color = C.GREEN if side == "LONG" else C.RED
    conf_str = f"  C={confidence}" if confidence is not None else ""

    return (
        f"  {side_color}{C.BOLD}▶ {side}{C.RESET} "
        f"{C.BOLD}{symbol}{C.RESET}  "
        f"@ {price:.4f}  "
        f"{leverage}x  "
        f"${size_usd:,.0f}"
        f"{C.DIM}{conf_str}{C.RESET}"
    )


def format_exit(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_usd: float,
    pnl_pct: float,
    reason: str,
    bars_held: int,
) -> str:
    """Format position exit log line.

    Args:
        symbol: Trading pair.
        side: 'LONG' or 'SHORT'.
        entry_price: Entry price.
        exit_price: Exit price.
        pnl_usd: PnL in USD.
        pnl_pct: PnL as percentage.
        reason: Exit reason.
        bars_held: Bars held.

    Returns:
        Formatted string.
    """
    C = Color
    pnl_color = C.GREEN if pnl_usd >= 0 else C.RED
    icon = "✓" if pnl_usd >= 0 else "✗"

    return (
        f"  {pnl_color}{C.BOLD}{icon}{C.RESET} "
        f"{C.BOLD}{symbol}{C.RESET}  "
        f"{entry_price:.4f} → {exit_price:.4f}  "
        f"{pnl_color}{C.BOLD}${pnl_usd:+,.2f}{C.RESET} "
        f"{pnl_color}({pnl_pct:+.2f}%){C.RESET}  "
        f"{C.DIM}{reason} · {bars_held}bars{C.RESET}"
    )


def format_scan_summary(
    n_candidates: int,
    top_movers: list[tuple[str, float]],
    threshold: float,
) -> str:
    """Format scan results summary.

    Args:
        n_candidates: Number of qualifying candidates.
        top_movers: List of (symbol, abs_move) tuples.
        threshold: Move threshold.

    Returns:
        Formatted string.
    """
    C = Color
    if n_candidates > 0:
        return f"  {C.CYAN}⎯ {n_candidates} candidates passed threshold{C.RESET}"
    else:
        if top_movers:
            top3 = [f"{s} {m * 100:.2f}%" for s, m in top_movers[:3]]
            return f"  {C.DIM}⎯ No signals  Top: {', '.join(top3)}  (need >{threshold * 100:.1f}%){C.RESET}"
        return f"  {C.DIM}⎯ No signals{C.RESET}"


def format_session_summary(
    cycles: int,
    n_trades: int,
    win_rate: float,
    total_pnl: float,
    balance: float,
    session_roi: float,
    best_trade: dict | None = None,
    worst_trade: dict | None = None,
) -> str:
    """Format session end summary block.

    Args:
        cycles: Total cycles run.
        n_trades: Total trades.
        win_rate: Win rate percentage.
        total_pnl: Total PnL in USD.
        balance: Final balance.
        session_roi: Session return percentage.
        best_trade: Optional best trade dict.
        worst_trade: Optional worst trade dict.

    Returns:
        Formatted multi-line string.
    """
    C = Color
    pnl_color = C.GREEN if total_pnl >= 0 else C.RED

    lines = [
        "",
        f"{C.DIM}{DIVIDER_HEAVY}{C.RESET}",
        f"  {C.BOLD}SESSION SUMMARY{C.RESET}",
        f"{C.DIM}{DIVIDER_HEAVY}{C.RESET}",
    ]

    _row = lambda k, v: f"  {C.DIM}{k:<16}{C.RESET}{v}"

    lines.append(_row("Cycles", str(cycles)))
    lines.append(_row("Trades", f"{n_trades}  (WR {win_rate:.0f}%)"))
    lines.append(_row("PnL", f"{pnl_color}{C.BOLD}${total_pnl:+,.2f}{C.RESET}"))
    lines.append(_row("Balance", f"${balance:,.2f}"))
    lines.append(_row("Session ROI", f"{pnl_color}{session_roi:+.2f}%{C.RESET}"))

    if best_trade:
        lines.append(_row("Best", f"{C.GREEN}{best_trade['symbol']} ${best_trade['pnl_usd']:+,.2f}{C.RESET}"))
    if worst_trade:
        lines.append(_row("Worst", f"{C.RED}{worst_trade['symbol']} ${worst_trade['pnl_usd']:+,.2f}{C.RESET}"))

    lines.append(f"{C.DIM}{DIVIDER_HEAVY}{C.RESET}")
    lines.append("")

    return "\n".join(lines)


def print_startup_complete(mode: str) -> None:
    """Print the final 'ready' message after startup.

    Args:
        mode: 'LIVE' or 'PAPER'.
    """
    C = Color
    print(f"  {C.GREEN}{C.BOLD}● READY{C.RESET}  Waiting for next 15m bar close...")
    print(f"{C.DIM}{DIVIDER_HEAVY}{C.RESET}")
    print()
