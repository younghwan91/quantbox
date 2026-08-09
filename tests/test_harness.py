"""실험 harness 검증.

핵심 관심사는 두 가지다:
1. **parity** — harness의 MarketData.load가 기존 research load_data와 완전히 같은
   데이터를 내놓는가. 어긋나면 harness로 옮긴 실험의 숫자가 바뀐다(북극성 위반).
2. **기본값이 안전한가** — 갈라진 구현들이 틀렸던 지점(_offset 누락, portfolio_size=1
   폭발, data_years 산출식)이 harness에서는 기본으로 올바른가.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from quantbox.research.harness import MarketData, experiment, get_commit, report
from quantbox.research.harness.data import MARKET_REF_SYMBOL
from quantbox.research.harness.report import BARS_PER_YEAR_15M
from quantbox.strategy.vol_straddle.config import VolStraddleConfig

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_extended"
R120 = ROOT / "research" / "active" / "r116_willy_nfi" / "r120_ma_cluster_per_tf.py"


def _make_data(lengths: dict[str, int]) -> MarketData:
    """정렬 검증용 최소 MarketData. 값이 아니라 정렬 규칙만 본다."""
    data: dict[str, dict] = {}
    for sym, n in lengths.items():
        arr = np.linspace(1.0, 2.0, n)
        data[sym] = {"close": arr, "high": arr * 1.01, "low": arr * 0.99}
    MarketData._align(data)
    return MarketData(data=data, source=Path("test"))


# ──────────────────────────────────────────────────────────────────────────
#  정렬 불변식 — 갈라진 구현 16벌 중 4벌이 놓쳤던 지점
# ──────────────────────────────────────────────────────────────────────────


def test_offset_aligns_to_longest_symbol():
    md = _make_data({"AAA": 100, "BBB": 60, "CCC": 100})
    assert md["AAA"]["_offset"] == 0
    assert md["BBB"]["_offset"] == 40  # 짧은 심볼은 앞쪽 여백만큼 밀린다
    assert md["CCC"]["_offset"] == 0


def test_offset_missing_is_rejected():
    """_offset 없는 데이터는 생성 자체가 막힌다 — 조용히 틀리느니 시끄럽게 죽는다."""
    bad = {"AAA": {"close": np.ones(10), "high": np.ones(10), "low": np.ones(10)}}
    with pytest.raises(ValueError, match="_offset"):
        MarketData(data=bad, source=Path("test"))


def test_align_empty_is_noop():
    md = MarketData(data={}, source=Path("test"))
    assert len(md) == 0
    assert report.data_years(md) == 0.0


# ──────────────────────────────────────────────────────────────────────────
#  data_years — 6종으로 갈라져 연수익을 서로 다르게 스케일하던 지점
# ──────────────────────────────────────────────────────────────────────────


def test_data_years_uses_aligned_timeline_not_average():
    """짧은 심볼이 섞여도 기간은 공통 타임라인 전체다.

    평균 봉수를 쓰면 기간이 짧게 잡혀 연수익이 부풀려진다 — 갈라진 구현 다수가 그랬다.
    """
    n_long = int(BARS_PER_YEAR_15M * 2)  # 2년
    md = _make_data({"AAA": n_long, "BBB": n_long // 4})
    assert report.data_years(md) == pytest.approx(2.0, rel=1e-6)


def test_data_years_counts_offset_bars():
    """짧은 심볼의 len+_offset이 공통 타임라인 길이가 된다."""
    md = _make_data({"AAA": 35064, "BBB": 100})
    assert len(md["BBB"]["close"]) + md["BBB"]["_offset"] == 35064
    assert report.data_years(md) == pytest.approx(1.0, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────────
#  report.standard — portfolio_size=1 폭발 함정
# ──────────────────────────────────────────────────────────────────────────


def test_standard_defaults_to_config_portfolio_size():
    """기본값이 max_concurrent여야 한다. 1이면 연수익이 폭발한다(실측 r116 아티팩트)."""
    cfg = VolStraddleConfig()
    md = _make_data({"AAA": int(BARS_PER_YEAR_15M)})
    trades = [("AAA", i * 100, i * 100 + 10, 0.02) for i in range(200)]

    m = report.standard(trades, md, cfg, apply_portfolio_sim=False, n_boot=0)

    # 200 트레이드 × 2% 를 1/max_concurrent 로 나눠 복리
    expected = (np.cumprod(np.full(200, 1.0 + 0.02 / cfg.max_concurrent))[-1] - 1.0) * 100
    assert m["ret_pct"] == pytest.approx(expected, rel=1e-3)
    assert m["N"] == 200
    assert m["years"] == pytest.approx(1.0, rel=1e-3)


def test_standard_portfolio_sim_enforces_slot_contention():
    """슬롯 경합이 기본 적용된다 — 라이브가 슬롯을 공유하므로."""
    cfg = VolStraddleConfig()
    md = _make_data({"AAA": 5000, "BBB": 5000})
    # 전 구간 겹치는 트레이드를 슬롯 수보다 많이 만든다
    overlapping = [("AAA", 100, 4000, 0.01) for _ in range(cfg.max_concurrent + 5)]

    filled = report.standard(overlapping, md, cfg, n_boot=0)
    raw = report.standard(overlapping, md, cfg, apply_portfolio_sim=False, n_boot=0)

    assert filled["n_filled"] < raw["n_filled"], "슬롯 경합이 적용되지 않았다"
    assert filled["n_raw"] == len(overlapping)


def test_render_contains_key_metrics():
    cfg = VolStraddleConfig()
    md = _make_data({"AAA": int(BARS_PER_YEAR_15M)})
    trades = [("AAA", i * 100, i * 100 + 10, 0.01 if i % 2 else -0.005) for i in range(40)]
    text = report.render(report.standard(trades, md, cfg, apply_portfolio_sim=False), title="t")
    for token in ("PF=", "maxDD=", "calmar=", "bootstrap PF:"):
        assert token in text


# ──────────────────────────────────────────────────────────────────────────
#  experiment 데코레이터
# ──────────────────────────────────────────────────────────────────────────


def test_experiment_injects_config_and_records_commit():
    seen: dict = {}

    @experiment(name="t_exp", quiet=True)
    def run(cfg):
        seen["cfg"] = cfg
        return 42

    res = run()
    assert res.value == 42
    assert res.name == "t_exp"
    assert res.commit == get_commit()
    assert res.elapsed_s >= 0.0
    assert isinstance(seen["cfg"], VolStraddleConfig), "실험은 config를 주입받아야 한다"


def test_get_commit_is_stable_and_short():
    c = get_commit()
    assert c == get_commit()
    if c != "unknown":
        assert len(c) <= 12


# ──────────────────────────────────────────────────────────────────────────
#  실데이터 parity — 골든 load_data와 완전 일치해야 한다
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.skipif(not CACHE.is_dir() or not R120.exists(), reason="확장 캐시 또는 r120 없음")
def test_marketdata_matches_golden_load_data():
    """harness 로더 == r120의 골든 load_data (배열 단위 완전 일치).

    이게 깨지면 harness로 옮긴 실험의 숫자가 바뀐다 = 기존 백테스트 결과 무효화.
    """
    spec = importlib.util.spec_from_file_location("r120_golden", R120)
    r120 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r120)

    wl = list(VolStraddleConfig().symbol_whitelist)
    golden = r120.load_data(wl)
    md = MarketData.load(wl)

    assert len(golden) > 0, "골든 로더가 아무것도 못 읽었다 — 테스트가 공허하다"
    assert set(golden) == set(md.data)

    for sym in golden:
        g, h = golden[sym], md[sym]
        assert set(g) == set(h), f"{sym}: 키 불일치"
        for key in g:
            if key == "_offset":
                assert g[key] == h[key], f"{sym}: _offset 불일치"
            else:
                np.testing.assert_array_equal(g[key], h[key], err_msg=f"{sym}.{key}")


@pytest.mark.slow
@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_marketdata_load_real_invariants():
    md = MarketData.load(["BTCUSDT", "ETHUSDT"])
    assert len(md) >= 1
    assert MARKET_REF_SYMBOL in md
    assert all(len(md[s]["close"]) >= 500 for s in md.symbols)
    # 캐시는 6년(2190일)치 — 정렬된 기간이 그 근방이어야 한다
    assert 5.5 < report.data_years(md) < 6.5


@pytest.mark.slow
@pytest.mark.skipif(not CACHE.is_dir(), reason="확장 캐시 없음")
def test_resample_delegates_to_production():
    from quantbox.backtest.trade_sim import build_resampled

    md = MarketData.load(["BTCUSDT"])
    got = md.resample(16)
    expected = build_resampled(md.data, 16)
    assert set(got) == set(expected)
    np.testing.assert_array_equal(got["BTCUSDT"]["close"], expected["BTCUSDT"]["close"])


def test_load_missing_cache_raises():
    with pytest.raises(FileNotFoundError):
        MarketData.load(["BTCUSDT"], cache_dir="data/__nope__")
