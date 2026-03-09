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
    entry_mode: str,
    min_confidence: int,
    testnet: bool = False,
    ma_cluster_enabled: bool = False,
    ma_cluster_pct: float = 0.03,
    ma_cluster_min_duration: int = 8,
    ma_cluster_max_duration: int = 24,
    ma_cluster_min_momentum: float = 0.01,
    ma_cluster_sl: float = 0.02,
    ma_cluster_tp: float = 0.08,
    ma_cluster_leverage: float = 12.0,
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
        entry_mode: Entry mode name.
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
    _print_row("Entry Mode", entry_mode)
    _print_row("Balance", f"${balance:,.2f}")
    _print_row("Universe", f"{n_symbols} symbols")
    _print_row("Max Slots", str(max_slots))
    print()

    _print_section("PARAMETERS")
    _print_row("Lookback", f"{lookback} bars")
    _print_row("Threshold", f"{threshold * 100:.1f}%")
    _print_row("Target", f"{target * 100:.1f}% retrace")
    _print_row("Stop", f"{stop_mult}x move")
    _print_row("Max Hold", f"{max_hold} bars")
    _print_row("Confidence", f">= {min_confidence}")
    print()

    _print_section("LEVERAGE")
    _print_row("Base", f"{leverage_base}x")
    _print_row("Max", f"{leverage_max}x")
    print()

    if ma_cluster_enabled:
        _print_section("MA CLUSTER")
        _print_row("Status", f"{Color.GREEN}ENABLED{Color.RESET}")
        _print_row("Spread", f"{ma_cluster_pct * 100:.0f}%")
        _print_row("Min Duration", f"{ma_cluster_min_duration} bars (4h)")
        _print_row("Max Duration", f"{ma_cluster_max_duration} bars (4h)")
        _print_row("Min Momentum", f"{ma_cluster_min_momentum * 100:.1f}%")
        _print_row("SL / TP", f"{ma_cluster_sl * 100:.0f}% / {ma_cluster_tp * 100:.0f}%")
        _print_row("Leverage", f"{ma_cluster_leverage:.0f}x")
        print()


def _print_section(title: str) -> None:
    """Print a section header."""
    print(f"  {Color.CYAN}{Color.BOLD}[ {title} ]{Color.RESET}")


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
