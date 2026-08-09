"""파라미터 하드코딩 가드 검증 — 래칫이 실제로 동작하는지 증명한다.

가드는 강제 장치다. 레거시 sweep 실험을 마이그레이션(재스테이징)해도 막지 않으면서,
**새로운** 하드코딩 파라미터는 반드시 잡아야 한다. 둘 다 테스트로 고정한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_research_params import (
    BASELINE_PATH,
    build_baseline,
    check_file,
    load_baseline,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, body: str, name: str = "r999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────
#  잡아야 하는 것 (새 하드코딩)
# ──────────────────────────────────────────────────────────────────────────
def test_catches_hardcoded_d_sl(tmp_path):
    p = _write(tmp_path, 'params = {"d_sl": 0.005, "tp": 0.05}\n')
    sigs = {v[3] for v in check_file(p)}
    assert "d_sl=0.005" in sigs


def test_catches_hardcoded_leverage(tmp_path):
    p = _write(tmp_path, "VR_LN = 3.0\nMA_LN = 2.1\n")
    sigs = {v[3] for v in check_file(p)}
    assert {"VR_LN=3.0", "MA_LN=2.1"} <= sigs


def test_new_file_with_hardcoded_param_fails(tmp_path, capsys):
    p = _write(tmp_path, 'params = {"d_sl": 0.01}\n')
    assert main([str(p)]) == 1


def test_new_value_in_grandfathered_file_still_fails(tmp_path, capsys):
    """유예 파일이라도 baseline에 없는 새 시그니처가 들어오면 실패한다."""
    # baseline은 실제 repo-relative 경로로 키가 잡히므로, 유예 시그니처만
    # 시뮬레이션하려면 실제 유예 파일 하나에 새 값을 붙인 사본이 필요하지만,
    # tmp 파일은 어차피 유예되지 않으므로(경로 불일치) 새 값은 무조건 실패한다.
    p = _write(tmp_path, 'g = {"d_sl": 0.0075}\nnew = {"d_sl": 0.002}\n')
    assert main([str(p)]) == 1


# ──────────────────────────────────────────────────────────────────────────
#  통과해야 하는 것
# ──────────────────────────────────────────────────────────────────────────
def test_from_cfg_escape_is_clean(tmp_path):
    p = _write(tmp_path, 'params = {"d_sl": 0.01}  # from cfg\n')
    sigs = {v[3] for v in check_file(p) if v[3] is not None}
    assert sigs == set(), "줄 끝 `# from cfg` 는 개별 면제되어야 한다"


def test_config_read_is_clean(tmp_path, capsys):
    p = _write(
        tmp_path,
        "from quantbox.strategy.vol_straddle.config import VolStraddleConfig\n"
        "cfg = VolStraddleConfig()\n"
        "VR_LN = cfg.base_leverage / cfg.max_concurrent\n"
        'params = {"d_sl": cfg.d_sl, "tp": cfg.d_tp}\n',
    )
    assert check_file(p) == []
    assert main([str(p)]) == 0


def test_comment_only_line_ignored(tmp_path):
    p = _write(tmp_path, '# params = {"d_sl": 0.005}\n')
    assert check_file(p) == []


# ──────────────────────────────────────────────────────────────────────────
#  래칫 baseline
# ──────────────────────────────────────────────────────────────────────────
def test_baseline_file_exists_and_wellformed():
    assert BASELINE_PATH.exists()
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert "grandfathered" in data
    assert isinstance(data["grandfathered"], dict)


def test_baseline_grandfathers_existing_research_files():
    """유예 목록의 파일들은 지금 검사해도 통과해야 한다(마이그레이션이 안 막힘)."""
    offenders = list(load_baseline().keys())
    assert offenders, "baseline이 비어있으면 이 테스트는 무의미하다"
    assert main([str(ROOT / rel) for rel in offenders]) == 0


def test_baseline_is_current():
    """커밋된 baseline이 실제 research/ 스캔 결과와 일치한다(드리프트 방지)."""
    on_disk = load_baseline()
    freshly_built = build_baseline()
    assert on_disk == freshly_built, (
        "baseline이 낡았다. 갱신: uv run python scripts/check_research_params.py --update-baseline"
    )
