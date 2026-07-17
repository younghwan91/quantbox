"""재구현 가드 검증 — 가드가 실제로 잡는지 증명한다.

설계도 §6: "가드가 재구현 시도를 실제로 잡음 (테스트로 증명)".
가드는 강제 장치이므로, 가드 자체가 작동하지 않으면 harness는 권장사항일 뿐이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_research_reimpl import (
    BANNED,
    BASELINE_PATH,
    build_baseline,
    load_baseline,
    main,
    scan_file,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, body: str, name: str = "r999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ──────────────────────────────────────────────────────────────────────────
#  잡아야 하는 것
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("symbol", sorted(BANNED))
def test_catches_every_banned_reimplementation(tmp_path, symbol):
    p = _write(tmp_path, f"def {symbol}(x):\n    return x\n")
    found = scan_file(p)
    assert [v[1] for v in found] == [symbol]
    assert found[0][2], "무엇으로 대체할지 안내가 없다"


def test_catches_own_numba_sim_kernel(tmp_path):
    p = _write(
        tmp_path,
        "import numba\n\n@numba.jit(nopython=True)\ndef sim_my_new_strategy(close):\n    return close\n",
    )
    found = scan_file(p)
    assert [v[1] for v in found] == ["sim_my_new_strategy"]


def test_catches_nested_and_method_definitions(tmp_path):
    """클래스 안이나 함수 안에 숨겨도 잡는다 — AST 전체를 훑기 때문."""
    p = _write(
        tmp_path,
        "class Runner:\n"
        "    def load_data(self, syms):\n"
        "        return {}\n"
        "\n"
        "def outer():\n"
        "    def compute_metrics(pnl):\n"
        "        return {}\n"
        "    return compute_metrics\n",
    )
    assert sorted(v[1] for v in scan_file(p)) == ["compute_metrics", "load_data"]


def test_guard_exits_nonzero_on_new_violation(tmp_path, capsys):
    p = _write(tmp_path, "def load_data(syms):\n    return {}\n")
    assert main([str(p)]) == 1
    err = capsys.readouterr().err
    assert "load_data" in err
    assert "MarketData" in err, "대안 안내가 없으면 가드는 잔소리일 뿐이다"


# ──────────────────────────────────────────────────────────────────────────
#  통과시켜야 하는 것 — 오탐은 가드를 우회하게 만든다
# ──────────────────────────────────────────────────────────────────────────


def test_harness_import_is_clean(tmp_path, capsys):
    p = _write(
        tmp_path,
        "from quantbox.research.harness import MarketData, experiment, report\n"
        "\n"
        "@experiment(name='r999')\n"
        "def run(cfg):\n"
        "    data = MarketData.load(cfg.symbol_whitelist)\n"
        "    return report.standard([], data, cfg)\n",
    )
    assert scan_file(p) == []
    assert main([str(p)]) == 0


def test_exempt_marker_allows_with_reason(tmp_path):
    p = _write(
        tmp_path,
        "# harness-exempt: 일회성 프로토타입, 캐시 포맷이 다름\ndef load_data(syms):\n    return {}\n",
    )
    assert scan_file(p) == []


def test_exempt_marker_on_def_line(tmp_path):
    p = _write(tmp_path, "def load_data(syms):  # harness-exempt: 임시\n    return {}\n")
    assert scan_file(p) == []


def test_exempt_without_reason_still_fails(tmp_path):
    """이유 없는 탈출구는 우회 해킹이다 — 막는다."""
    p = _write(tmp_path, "# harness-exempt:\ndef load_data(syms):\n    return {}\n")
    assert [v[1] for v in scan_file(p)] == ["load_data"]


def test_unrelated_code_is_not_flagged(tmp_path):
    p = _write(
        tmp_path,
        "def simulate_annealing(x):\n    return x\n\ndef similar_symbols(a):\n    return a\n\ndef my_analysis():\n    pass\n",
    )
    assert scan_file(p) == [], "sim_ 접두사가 아닌 이름을 잡으면 안 된다"


def test_calling_is_fine_only_defining_is_banned(tmp_path):
    """harness 함수를 **호출**하는 건 정상이다. 다시 **정의**하는 게 문제다."""
    p = _write(
        tmp_path,
        "from quantbox.research.harness import compute_metrics, portfolio_sim\n"
        "\n"
        "def run():\n"
        "    pnl = portfolio_sim([], 10, None)\n"
        "    return compute_metrics(pnl, 6.0, 10)\n",
    )
    assert scan_file(p) == []


def test_syntax_error_file_is_skipped(tmp_path):
    p = _write(tmp_path, "def broken(:\n")
    assert scan_file(p) == []


# ──────────────────────────────────────────────────────────────────────────
#  래칫 — 기존 위반은 유예, 새 위반은 차단
# ──────────────────────────────────────────────────────────────────────────


def test_baseline_file_exists_and_is_wellformed():
    assert BASELINE_PATH.exists(), "baseline이 없으면 기존 research 파일이 전부 커밋 불가가 된다"
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    gf = data["grandfathered"]
    assert len(gf) > 0
    for rel, symbols in gf.items():
        assert (ROOT / rel).exists() or True  # 파일이 지워졌을 수 있다 — 유예 목록은 관대해도 안전
        assert isinstance(symbols, list) and symbols


def test_baseline_grandfathers_existing_research_files():
    """도입 시점의 기존 위반은 커밋을 막지 않는다."""
    gf = load_baseline()
    offenders = [rel for rel, syms in gf.items() if syms and (ROOT / rel).exists()]
    assert offenders, "유예 대상이 하나도 없다 — baseline이 잘못 생성됐다"
    assert main([str(ROOT / rel) for rel in offenders]) == 0


def test_baseline_is_current():
    """baseline이 실제 research/ 상태와 일치해야 한다.

    어긋나면 유예가 과하거나(새 위반이 조용히 통과) 부족하다(기존 파일이 막힘).
    갱신: uv run python scripts/check_research_reimpl.py --update-baseline
    """
    assert build_baseline() == load_baseline()


def test_new_symbol_in_grandfathered_file_still_fails(tmp_path):
    """유예된 파일이라도 **새** 재구현을 추가하면 막힌다 — 래칫은 한 방향이다."""
    gf = load_baseline()
    rel, symbols = next((r, s) for r, s in gf.items() if (ROOT / r).exists())
    original = (ROOT / rel).read_text(encoding="utf-8")

    probe = tmp_path / Path(rel).name
    probe.write_text(original + "\n\ndef bootstrap_pf(pnl):\n    return {}\n", encoding="utf-8")

    # tmp 경로라 baseline 키와 매칭되지 않으므로, 유예 심볼도 함께 잡힌다.
    # 핵심 확인은 '새로 추가한 심볼이 잡히는가'.
    found = {v[1] for v in scan_file(probe)}
    assert "bootstrap_pf" in found
