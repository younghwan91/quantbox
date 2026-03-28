"""Unit tests for to_api_symbol() and round_to_tick() in core/types.py."""

from quantbox.core.types import round_to_tick, to_api_symbol


class TestToApiSymbol:
    def test_plain_symbol_passthrough(self):
        assert to_api_symbol("BTCUSDT") == "BTCUSDT"

    def test_compound_key_strips_side(self):
        assert to_api_symbol("BTCUSDT:LONG") == "BTCUSDT"
        assert to_api_symbol("XRPUSDT:SHORT") == "XRPUSDT"

    def test_pos_symbol_takes_priority(self):
        pos = {"symbol": "ETHUSDT", "entry_price": 100}
        assert to_api_symbol("ETHUSDT:LONG", pos) == "ETHUSDT"

    def test_pos_symbol_with_colon_is_stripped(self):
        # Corrupt state: symbol field has compound key
        pos = {"symbol": "ETHUSDT:LONG"}
        assert to_api_symbol("ETHUSDT:LONG", pos) == "ETHUSDT"

    def test_pos_without_symbol_falls_back_to_key(self):
        pos = {"entry_price": 100}  # no "symbol" field
        assert to_api_symbol("DOGEUSDT:SHORT", pos) == "DOGEUSDT"

    def test_pos_none_uses_key(self):
        assert to_api_symbol("SUIUSDT:LONG", None) == "SUIUSDT"

    def test_empty_pos_uses_key(self):
        assert to_api_symbol("SUIUSDT:LONG", {}) == "SUIUSDT"


class TestRoundToTick:
    def test_exact_tick_unchanged(self):
        assert round_to_tick(100.0, 0.1, 1) == 100.0

    def test_rounds_to_nearest_tick(self):
        assert round_to_tick(100.03, 0.1, 1) == 100.0

    def test_small_tick_size(self):
        # DOGEUSDT: tick=0.00001, precision=5
        assert round_to_tick(0.123456, 0.00001, 5) == 0.12346

    def test_large_tick_size(self):
        # BTCUSDT: tick=0.10, precision=2
        assert round_to_tick(67890.37, 0.10, 2) == 67890.40

    def test_precision_limits_decimals(self):
        result = round_to_tick(1.23456789, 0.01, 2)
        assert result == 1.23

    def test_matches_legacy_inline_pattern(self):
        # Verify equivalence with the old inline pattern
        price = 0.0554
        tick_size = 0.0001
        precision = 4
        legacy = round(round(price / tick_size) * tick_size, precision)
        assert round_to_tick(price, tick_size, precision) == legacy
