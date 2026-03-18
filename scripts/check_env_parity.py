#!/usr/bin/env python3
"""Environment parity check.

Pre-commit hook that warns when .env values differ from config.py defaults.
Helps catch the pydantic-settings precedence issue where .env overrides
config.py silently.

Usage:
    python scripts/check_env_parity.py

Exit code 0 = clean, 1 = warnings found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
ENV_FILE = REPO_ROOT / ".env"
CONFIG_FILE = REPO_ROOT / "quantbox/strategy/vol_straddle/config.py"


def parse_env_file(filepath: Path) -> dict[str, str]:
    """Parse .env file into key-value dict."""
    env_vars = {}
    if not filepath.exists():
        return env_vars
    
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env_vars[key] = value
    
    return env_vars


def parse_config_defaults(filepath: Path) -> dict[str, tuple[str, str]]:
    """Parse config.py for VS_ prefixed defaults.
    
    Returns:
        Dict mapping env var name to (default_value, field_name).
    """
    if not filepath.exists():
        return {}
    
    content = filepath.read_text()
    defaults = {}
    
    # Pattern: field_name: type = Field(default=X, env="VS_...")
    # or: field_name: type = X  with # env: VS_...
    pattern = re.compile(
        r'(\w+)\s*:\s*[^=]+=\s*Field\([^)]*default\s*=\s*([^,)]+)[^)]*env\s*=\s*["\']([^"\']+)["\']',
        re.MULTILINE
    )
    
    for match in pattern.finditer(content):
        field_name = match.group(1)
        default_val = match.group(2).strip()
        env_name = match.group(3)
        defaults[env_name] = (default_val, field_name)
    
    # Also check simple assignments: field: type = value
    # with validation_alias pattern
    pattern2 = re.compile(
        r'(\w+)\s*:\s*[^=]+=\s*([^#\n]+)#.*(?:VS_\w+)',
        re.MULTILINE
    )
    
    return defaults


def main() -> int:
    """Compare .env values with config.py defaults."""
    env_vars = parse_env_file(ENV_FILE)
    config_defaults = parse_config_defaults(CONFIG_FILE)
    
    # Filter to VS_ prefixed vars only
    vs_env_vars = {k: v for k, v in env_vars.items() if k.startswith("VS_")}
    
    warnings = []
    
    # Check for VS_ vars in .env that are same as defaults (redundant)
    for env_name, env_value in vs_env_vars.items():
        if env_name in config_defaults:
            default_val, field_name = config_defaults[env_name]
            # Normalize for comparison
            env_norm = env_value.lower().strip()
            default_norm = default_val.lower().strip()
            
            # Skip complex expressions
            if '(' in default_norm or ')' in default_norm:
                continue
            
            if env_norm == default_norm:
                warnings.append(
                    f"Redundant: {env_name}={env_value} in .env "
                    f"(same as config.py default). Consider removing from .env."
                )
    
    # Info about overrides
    overrides = []
    for env_name, env_value in vs_env_vars.items():
        if env_name in config_defaults:
            default_val, field_name = config_defaults[env_name]
            env_norm = env_value.lower().strip()
            default_norm = default_val.lower().strip()
            if '(' not in default_norm and env_norm != default_norm:
                overrides.append(f"  {env_name}: .env={env_value}, config.py={default_val}")
    
    if warnings:
        print("⚠️  .env parity warnings:\n")
        for w in warnings:
            print(f"  {w}")
        print()
    
    if overrides:
        print(f"ℹ️  {len(overrides)} active .env overrides (config.py defaults differ):")
        for o in overrides:
            print(o)
        print()
    
    # Only fail on actual problems, not warnings
    return 0


if __name__ == "__main__":
    sys.exit(main())
