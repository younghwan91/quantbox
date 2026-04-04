"""Canonical strategy naming — single source of truth.

All strategy_type string literals MUST be imported from here.
Never hardcode strategy type strings elsewhere in production code.
"""

# Canonical strategy_type values (used in code, SQLite, state, logs)
VR_DIRECTIONAL = "vr_directional"
VR_STRADDLE = "vr_straddle"
MA_CLUSTER = "ma_cluster"
DONCHIAN = "donchian"

STRATEGY_TYPES = frozenset({VR_DIRECTIONAL, VR_STRADDLE, MA_CLUSTER, DONCHIAN})

# Human-readable display names (for banner, Discord, docs)
DISPLAY_NAMES: dict[str, str] = {
    VR_DIRECTIONAL: "VR Directional",
    VR_STRADDLE: "VR Straddle",
    MA_CLUSTER: "MA Cluster",
    DONCHIAN: "Donchian",
}

# Signal/research aliases (for docs cross-reference)
SIGNAL_NAMES: dict[str, str] = {
    VR_DIRECTIONAL: "D_mom8",
    VR_STRADDLE: "S_d8",
    MA_CLUSTER: "MA Cluster",
    DONCHIAN: "DC_vol50",
}

# Old → new mapping for backward compat (log parsing, old SQLite records)
STRATEGY_ALIASES: dict[str, str] = {
    "vr_squeeze": VR_DIRECTIONAL,
    "straddle": VR_STRADDLE,
    "ma_cluster": MA_CLUSTER,
    "donchian": DONCHIAN,
}


def normalize_strategy_type(raw: str) -> str:
    """Normalize old strategy_type values to canonical names.

    Accepts both old and new names, always returns canonical.
    Unknown values pass through unchanged.
    """
    return STRATEGY_ALIASES.get(raw, raw)
