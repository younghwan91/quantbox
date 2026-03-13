# r76: S_d8 s_trail_distance 최적화 (Trail Server Ratchet 이후 재검증)

**날짜:** 2026-03-14
**커밋:** d037ab9
**스크립트:** `research/active/r76_r80/r76_trail_distance.py`
**데이터:** 60 syms (symbol_whitelist) · 5.67yr · Binance USDT-M
**비용:** 0.16% round-trip (straddle) · 10슬롯 (max_concurrent)
**Parity:** config.py 직접 읽음 ✅ · symbol_whitelist ✅ · 룩어헤드 없음 ✅

## 배경

r67 sensitivity (2026-03-12) — S_d8 straddle trail_distance 스윕 (당시 s_trail_activate=0.005 기준):
- 0.001: PF 3.772, OOS 2.193 (r67 ❌ — PROD_IDX 버그로 0.003 기준 판정, 실제 PROD=0.002)
- 0.0015: PF 3.223, OOS 1.854 (r67 ✅)
- 0.002 (현재): PF 2.772, OOS 1.576 (✅)

r74 (2026-03-13) — s_trail_activate=0.015 채택 후 S_d8 baseline: N=7730, PF=4.948
Trail Server Ratchet (e473e4d) 구현 완료 → live gap 문제 해소.
r76 OOS 기준: OOS PF ≥ 1.5 AND Bootstrap P(PF≥1.0) ≥ 95%
(IS/OOS ≤ 1.5 제외 — 고PF 전략에서 IS > OOS는 정상; OOS 절대값이 핵심)

---

## Phase 0: Baseline 검증 (s_trail_distance=0.002)

| 항목 | 값 | r74 reference |
|------|---|--------------|
| N | 7748 | 7730 (0.2% drift) |
| PF | 4.876 | 4.948 (1.5% drift) |
| WR | 49.3% | — |
| EV | 41.3bp | — |
| Calmar | 1180.7 | — |

## Phase 1: S_d8 s_trail_distance Sweep

| s_trail_distance | N | PF | WR | EV(bp) | MaxDD | OOS PF | IS/OOS | Boot≥95% | Robust |
|---|---|---|---|---|---|---|---|---|---|
| 0.0008 | 7748 | 10.957 | 60.2% | 53.3 | 3.7% | 5.265 | 2.08 | ✅ | ✅ |
| 0.0010 | 7748 | 9.349 | 58.2% | 51.3 | 4.3% | 4.444 | 2.10 | ✅ | ✅ |
| 0.0012 | 7748 | 8.069 | 56.1% | 49.3 | 5.3% | 3.796 | 2.13 | ✅ | ✅ |
| 0.0015 | 7748 | 6.583 | 53.3% | 46.3 | 9.1% | 3.052 | 2.16 | ✅ | ✅ |
| 0.0018 | 7748 | 5.470 | 50.8% | 43.3 | 14.1% | 2.502 | 2.19 | ✅ | ✅ |
| 0.0020 ★ | 7748 | 4.876 | 49.3% | 41.3 | 21.2% | 2.210 | 2.21 | ✅ | ✅ |

★ = current production (s_trail_distance=0.0020)

**추천:** ✅ s_trail_distance=0.0008 채택 권장

---

## Phase 2: Combined Portfolio (D+S+MA)

| | N | PF | Calmar | MaxDD | vs r73 |
|---|---|---|---|---|---|
| s_dist=0.0008 (proposed) | 9775 | 2.484 | 9882.8 | 37.1% | 1.24x |
| s_dist=0.002 (current) | 9775 | 2.089 | 1880.5 | 41.8% | 1.04x |
| r73 canonical (reference) | 9,773 | 2.002 | 1,695.6 | 40.5% | 1.00x |

---

## 결론

✅ s_trail_distance=0.0008 채택 권장
변경 위치: `config.py: s_trail_distance=0.0008` + `.env: VS_S_TRAIL_DISTANCE=0.0008`

**구현:** 미구현
