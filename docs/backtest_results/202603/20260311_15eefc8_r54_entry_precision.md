# r54: Entry Precision Research

**Date:** 2026-03-11  
**Commit:** `15eefc8`  
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
| N | 4829 | 4520 | — |
| PF | 2.512 | 2.288 | — |
| WR% | 4630.0 | 4020.0 | — |
| EV(bp) | 34.7 | 27.6 | — |
| Calmar | 148.7 | 27.4 | — |
| MaxDD% | 940.0 | 2310.0 | — |

Bootstrap CI: [2.054, 2.518]  
OOS/IS ratio: 0.52  
