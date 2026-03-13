"""Stability tests for deleted strategies (Elliott + Alpha Engine).

These tests guarantee that:
1. Deleted modules are permanently gone (import fails as expected)
2. strategy_name allowlist blocks removed names at construction
3. No lingering code references to removed modules exist at runtime
4. All remaining 44 quantbox modules import cleanly

Regression guard: if anyone re-adds elliott.py or alpha_engine.py,
or accidentally re-imports them, these tests will immediately fail.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest


# ──────────────────────────────────────────────
# 1. Deleted modules must NOT be importable
# ──────────────────────────────────────────────


def test_elliott_module_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quantbox.strategy.elliott")


def test_alpha_engine_module_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quantbox.strategy.alpha_engine")


# ──────────────────────────────────────────────
# 2. strategy_name allowlist (ValueError on unknown)
# ──────────────────────────────────────────────


def test_strategy_name_elliott_raises():
    from quantbox.scalper import RubberBandScalper

    with pytest.raises(ValueError, match="elliott"):
        RubberBandScalper(paper_mode=True, strategy_name="elliott")


def test_strategy_name_alpha_raises():
    from quantbox.scalper import RubberBandScalper

    with pytest.raises(ValueError, match="alpha"):
        RubberBandScalper(paper_mode=True, strategy_name="alpha")


def test_strategy_name_bad_raises():
    from quantbox.scalper import RubberBandScalper

    with pytest.raises(ValueError, match="bad_strategy"):
        RubberBandScalper(paper_mode=True, strategy_name="bad_strategy")


def test_strategy_name_vol_straddle_ok():
    from quantbox.scalper import RubberBandScalper
    from quantbox.strategy.vol_straddle import VolStraddleStrategy

    s = RubberBandScalper(paper_mode=True, strategy_name="vol_straddle")
    assert isinstance(s.strategy, VolStraddleStrategy)
    assert s.strategy_name == "vol_straddle"


def test_strategy_name_rubberband_ok():
    from quantbox.scalper import RubberBandScalper
    from quantbox.strategy.rubberband import RubberBandStrategy

    s = RubberBandScalper(paper_mode=True, strategy_name="rubberband")
    assert isinstance(s.strategy, RubberBandStrategy)
    assert s.strategy_name == "rubberband"


# ──────────────────────────────────────────────
# 3. Runtime code inspection — no deleted imports
# ──────────────────────────────────────────────


def test_positions_no_elliott_import():
    """positions.py must not reference ElliottConfig at runtime."""
    from quantbox.execution import positions

    src = inspect.getsource(positions)
    assert "ElliottConfig" not in src, "ElliottConfig still referenced in positions.py"
    assert "elliott" not in src.lower(), "elliott still referenced in positions.py"


def test_positions_no_alpha_import():
    """positions.py must not reference AlphaConfig at runtime."""
    from quantbox.execution import positions

    src = inspect.getsource(positions)
    assert "AlphaConfig" not in src, "AlphaConfig still referenced in positions.py"
    assert "alpha_engine" not in src.lower(), "alpha_engine still referenced in positions.py"


def test_scalper_no_elliott_import():
    """scalper.py must not reference ElliottConfig/ElliottStrategy at runtime."""
    import quantbox.scalper as scalper_mod

    src = inspect.getsource(scalper_mod)
    assert "ElliottConfig" not in src
    assert "ElliottStrategy" not in src


def test_scalper_no_alpha_import():
    """scalper.py must not reference AlphaConfig/AlphaStrategy at runtime."""
    import quantbox.scalper as scalper_mod

    src = inspect.getsource(scalper_mod)
    assert "AlphaConfig" not in src
    assert "AlphaStrategy" not in src


# ──────────────────────────────────────────────
# 4. strategy/__init__.py exports must NOT include deleted names
# ──────────────────────────────────────────────


def test_strategy_init_no_elliott_exports():
    import quantbox.strategy as strategy_mod

    all_exports = getattr(strategy_mod, "__all__", [])
    deleted = {"ElliottConfig", "ElliottStrategy", "compute_elliott_signals", "scan_multi_scale_patterns"}
    found = deleted & set(all_exports)
    assert not found, f"Deleted Elliott exports still in __all__: {found}"


def test_strategy_init_no_alpha_exports():
    import quantbox.strategy as strategy_mod

    all_exports = getattr(strategy_mod, "__all__", [])
    deleted = {"AlphaConfig", "AlphaStrategy", "compute_alpha_signals", "compute_hybrid_signals"}
    found = deleted & set(all_exports)
    assert not found, f"Deleted Alpha exports still in __all__: {found}"


# ──────────────────────────────────────────────
# 5. Full package import walk — nothing broken
# ──────────────────────────────────────────────


def test_quantbox_package_fully_importable():
    """Every module in quantbox/ must be importable without error."""
    errors: list[str] = []
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        path=["quantbox"], prefix="quantbox.", onerror=lambda x: None
    ):
        try:
            importlib.import_module(modname)
        except Exception as e:
            errors.append(f"{modname}: {e}")

    assert not errors, "Import errors found:\n" + "\n".join(errors)
