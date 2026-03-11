# r54: Entry Precision Research

**Date:** 2026-03-11  
**Commit:** `6b14f34`  
**Branch:** `research/r54-entry-precision`  
**Data:** 2020-02-22 ~ 2026-03-11 (6.00yr)  
**Slots:** 10 | **Cost:** 0.08% round-trip | **L/N:** 3.0  
**Model:** 1x compound (compute_metrics baseline)  

---

## Backtest-Live Parity Check

- [x] config.py에서 파라미터 읽음 (하드코딩 없음)
- [x] D_TRAIL_ACTIVATE=0.005, D_TRAIL_DISTANCE=0.002
- [x] 룩어헤드 없음 (완성된 바만, `i` 인덱스 사용)
- [x] 슬롯 수=10, cost=0.0008

## Results

**Best config:** close_pos≥0.50, max_delay=0

| Metric | Baseline | r54 Best | Delta |
|--------|----------|----------|-------|
| N | 4684 | 4341 | — |
| PF | 2.686 | 2.483 | — |
| WR% | 49.9 | 44.6 | — |
| EV(bp) | 38.1 | 31.2 | — |
| Calmar | 179.8 | 70.1 | — |
| MaxDD% | 9.4 | 11.1 | — |

Bootstrap CI: [2.240, 2.754]  
OOS/IS ratio: 0.55  
