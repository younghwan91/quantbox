"""r121 — MA 클러스터 per-TF 스윕, harness 판.

r120과 **같은 질문**에 답한다: MA 클러스터 브레이크아웃을 4h가 아닌 다른 TF에서
돌리면 어떻게 되나?

차이는 답이 아니라 **비용**이다.
- r120: 이 질문에 답하려고 numba sim을 통째로 fork하고, load_data / resample /
  compute_metrics / get_commit 을 새로 짰다 (~300줄).
- r121: sim·로딩·지표 재구현 **0줄**. TF는 `ratio` 인자일 뿐이다.

harness가 재수출하는 함수는 라이브가 쓰는 바로 그 함수라, parity가 구조적으로 보장된다.
`run_ma_cluster(ratio=16)`이 프로덕션 `collect_ma_cluster_trades`와 bit-parity라는 것은
tests/test_sim_core_parity.py 가 실데이터로 검증한다.

⚠️ **r120과 숫자를 직접 비교하지 말 것.** 같은 질문이지만 조건이 다르다:
  1. r120은 BTC MA50(4h) 게이트로 롱을 거른다. r121은 게이트 없는 소지(素地) 버전이다.
  2. r120은 tp=0.20(사실상 TP 해제)을 쓴다. r121은 cfg 기본값을 쓴다.
  3. r120은 `compute_metrics(pnl, DATA_YEARS)`를 portfolio_size 없이 호출한다 → 기본값 1
     = 매 트레이드 전액 복리. r121은 `report.standard`가 max_concurrent(=10)를 넘긴다
     = 라이브 슬롯 사이징과 같은 의미. r120 쪽 수익률이 구조적으로 부풀려져 있다.
문제 (3)은 harness가 존재하는 이유 그 자체다 — 실험이 지표 호출을 각자 짜면 이렇게 어긋난다.
"""

from __future__ import annotations

from quantbox.research.harness import MarketData, experiment, report, run_ma_cluster

#: (라벨, 15m 대비 ratio, max_hold 봉수) — 각 TF에서 동일한 보유 시간이 되도록 맞춘다.
TF_CONFIGS: list[tuple[str, int, int]] = [
    ("15m", 1, 192),  # 192 × 15m = 48h
    ("30m", 2, 96),  # 96 × 30m = 48h
    ("1h", 4, 48),  # 48 × 1h = 48h
    ("4h", 16, 6),  # 6 × 4h = 24h — 현재 프로덕션 설정
]


@experiment(name="r121_ma_cluster_per_tf_harness")
def run(cfg):
    data = MarketData.load(cfg.symbol_whitelist)

    results: dict[str, dict] = {}
    for label, ratio, max_hold in TF_CONFIGS:
        trades = run_ma_cluster(data.data, ratio=ratio, max_hold=max_hold)
        results[label] = report.standard(trades, data, cfg)
    return results


if __name__ == "__main__":
    res = run()
    for label, metrics in res.value.items():
        print()
        print(report.render(metrics, title=f"MA cluster @ {label}"))
