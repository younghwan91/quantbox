"""ChaseConfig가 하드코딩이 아니라 config에서 온다는 것을 고정한다 (Rule #6).

배경: `scalper.py`가 4개 값을 하드코딩하고 `initial_offset_ticks`는 아예 전달하지
않았다. dataclass 기본값 0으로 돌았으므로 config가 의도한 "첫 주문을 1틱 유리하게"는
프로덕션에서 한 번도 실행되지 않았다. config를 고쳐도 아무 일이 안 일어나는 상태였다.
"""

from __future__ import annotations

import inspect

import pytest

from quantbox.core.limit_chaser import ChaseConfig
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

_FIELDS = (
    "max_chase_ticks",
    "chase_interval_seconds",
    "timeout_seconds",
    "fallback_to_market",
    "initial_offset_ticks",
)


def test_config_declares_every_chase_field() -> None:
    """ChaseConfig의 모든 필드가 VolStraddleConfig에 대응 항목을 갖는다."""
    for name in ChaseConfig.__dataclass_fields__:
        assert name in VolStraddleConfig.model_fields, f"config에 없는 chaser 파라미터: {name}"


def test_scalper_wires_chase_config_from_config() -> None:
    """`scalper.py`가 리터럴이 아니라 vs_config를 넘긴다."""
    from quantbox import scalper as scalper_mod

    src = inspect.getsource(scalper_mod)
    start = src.index("config=ChaseConfig(")
    block = src[start : src.index(")", src.index("initial_offset_ticks", start))]

    for name in _FIELDS:
        assert f"{name}=self.vs_config.{name}" in block, f"{name}이 config에서 오지 않는다"


@pytest.mark.parametrize("name", _FIELDS)
def test_chase_defaults_agree_between_config_and_dataclass(name: str) -> None:
    """두 기본값이 어긋나면 배선 누락이 조용히 다른 동작을 만든다.

    `initial_offset_ticks`가 정확히 그 사례였다 — config 1, dataclass 0.
    어긋난 채로 두려면 왜 그런지 여기서 먼저 설명해야 한다.
    """
    cfg_default = VolStraddleConfig.model_fields[name].default
    dc_default = ChaseConfig.__dataclass_fields__[name].default
    assert cfg_default == dc_default, (
        f"{name}: config={cfg_default} vs ChaseConfig={dc_default} — 배선이 빠지면 후자가 쓰인다"
    )
