"""Verify both strategies satisfy the TradingStrategy protocol."""

from __future__ import annotations

import inspect

import pytest

from quantbox.strategy.protocol import TradingStrategy

# Every method/property name that TradingStrategy declares.
_PROTOCOL_MEMBERS: list[str] = [
    name
    for name, _ in inspect.getmembers(
        TradingStrategy,
        predicate=lambda m: callable(m) or isinstance(m, property),
    )
    if not name.startswith("_")
]


# ── structural checks (no instantiation needed) ────────────────────


class TestVolStraddleSatisfiesProtocol:
    """VolStraddleStrategy must expose every protocol method/property."""

    @pytest.fixture(autouse=True)
    def _import_class(self):
        from quantbox.strategy.vol_straddle.strategy import VolStraddleStrategy

        self.cls = VolStraddleStrategy

    @pytest.mark.parametrize("name", _PROTOCOL_MEMBERS)
    def test_has_member(self, name: str):
        assert hasattr(self.cls, name), f"VolStraddleStrategy missing: {name}"

    def test_isinstance_check(self):
        """runtime_checkable isinstance works on an instance."""
        instance = self.cls()
        assert isinstance(instance, TradingStrategy)


class TestRubberBandSatisfiesProtocol:
    """RubberBandStrategy must expose every protocol method/property."""

    @pytest.fixture(autouse=True)
    def _import_class(self):
        from quantbox.strategy.rubberband.strategy import RubberBandStrategy

        self.cls = RubberBandStrategy

    @pytest.mark.parametrize("name", _PROTOCOL_MEMBERS)
    def test_has_member(self, name: str):
        assert hasattr(self.cls, name), f"RubberBandStrategy missing: {name}"

    def test_isinstance_check(self):
        """runtime_checkable isinstance works on an instance."""
        instance = self.cls()
        assert isinstance(instance, TradingStrategy)


# ── signature compatibility ─────────────────────────────────────────


def _public_methods(proto) -> dict[str, inspect.Signature]:
    """Extract non-dunder callable signatures from a Protocol."""
    result = {}
    for name in dir(proto):
        if name.startswith("_"):
            continue
        obj = getattr(proto, name)
        if callable(obj) and not isinstance(obj, property):
            result[name] = inspect.signature(obj)
    return result


@pytest.mark.parametrize(
    "strategy_path",
    [
        "quantbox.strategy.vol_straddle.strategy.VolStraddleStrategy",
        "quantbox.strategy.rubberband.strategy.RubberBandStrategy",
    ],
)
def test_method_param_count_compatible(strategy_path: str):
    """Strategy methods accept at least as many params as the protocol."""
    import importlib

    module_path, cls_name = strategy_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)

    proto_methods = _public_methods(TradingStrategy)
    for method_name, proto_sig in proto_methods.items():
        impl = getattr(cls, method_name, None)
        assert impl is not None, f"{cls_name} missing {method_name}"

        impl_sig = inspect.signature(impl)
        proto_required = [
            p
            for p in proto_sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        impl_params = list(impl_sig.parameters.values())

        # Implementation must accept all required protocol params.
        assert len(impl_params) >= len(proto_required), (
            f"{cls_name}.{method_name}: expected ≥{len(proto_required)} params, got {len(impl_params)}"
        )
