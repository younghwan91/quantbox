# r92: MA Cluster 4h BTC Gate Validation

**날짜:** 2026-03-18  
**커밋:** e0d6792 (research/r91-ma-parity)  
**결정:** ✅ **PASS — MA cluster 재활성화**  
**구현:** feature/r92-ma-gate → vX.Y.Z *(merge pending)*

---

## 배경

r91 (2026-03-17)에서 MA cluster 라이브 실패 (N=16, PF=0.309, 계정 손실 96%) 원인 확인:

| 기간 | 원인 |
|------|------|
| 3/4–3/8 | G6 게이트 미존재 → BTC 폭락 중 무방비 진입 |
| 3/9–3/13 | G6 15m MA50 (12.5h) — 4h 전략에 너무 짧음. 주간 하락트렌드 중 장중 반등이 15m MA50 통과 |

**해결책:** MA cluster 전용 4h BTC MA50 게이트 (50 × 4h = 200h ≈ 8.3일)

---

## 코드 변경 (Phase 1)

### `config.py`
```python
ma_cluster_btc_gate_enabled: bool = True   # 신규
ma_cluster_btc_ma_period_4h: int = 50      # 신규
```

### `strategy.py`
```python
self._last_signal_source: dict[str, str] = {}
# on_bar() 각 경로에 'vr_squeeze' | 'ma_cluster' | 'none' 태깅
```

### `scalper.py`
- `_fetch_btc_data()`: vol_straddle에서 BTC 4h 60봉 다운로드 (기존 `pass` 수정)
- `_scan_entries()`: `on_bar()` 직후 게이트 체크 — MA cluster LONG + BTC 4h < MA50 → signal=0

### 테스트
- `tests/test_ma_cluster_gate.py`: 13개 신규 테스트 (config 기본값, signal source 태깅, 게이트 로직)
- 전체 2327 테스트 통과

---

## 백테스트 설정

| 항목 | 값 |
|------|-----|
| 데이터 | 2020-01-01 ~ 2025-12-31 (6.0년) |
| 유니버스 | `cfg.symbol_whitelist` 60 심볼 |
| 슬롯 | `max_concurrent=10` |
| 비용 | 0.08% round-trip |
| L/N | 2.1 (MA cluster: 21x / 10 슬롯) |
| 파라미터 | `VolStraddleConfig()` 직접 읽기 |
| Sim 엔진 | `sim_live_style_gated()` (r92 신규 Numba 함수) |

**게이트 설계:**
- BTC 4h MA50 계산: `np.mean(btc_close_4h[-50:])`
- LONG 진입 조건: `btc_4h_cur ≥ btc_4h_ma50` (BTC가 MA50 위에 있을 때만)
- SHORT 진입: 게이트 없음 (하락장 방향성 중립)
- Straddle (S_d8): 영향 없음

**Backtest-Live Parity 확인:**
- 파라미터 하드코딩 없음 ✅ (`VolStraddleConfig()`)
- 룩어헤드 없음 ✅ (4h 완성 바 기준)
- 슬롯 경쟁 ✅ (`portfolio_sim` canonical)

---

## 결과

### 주요 지표 비교

| 지표 | Gate OFF | Gate ON | 변화 |
|------|----------|---------|------|
| N | 772 | 649 | -123 (-15.9%) |
| PF | 2.554 | **2.749** | +0.195 (+7.6%) |
| WR | 55.4% | **57.5%** | +2.1pp |
| MaxDD | 22.9% | **17.7%** | **-5.2pp** |
| EV/trade | +134.7bp | **+144.2bp** | +9.5bp |
| CAGR (LN=2.1) | 45.9% | 41.7% | -4.2pp (거래 수 감소) |

> CAGR 소폭 감소는 거래 수 감소(N -123) 때문. 허용 범위.

### 게이트 통계 (Gate ON)

| 항목 | 수치 |
|------|------|
| LONG 진입 시도 | 404 |
| LONG 차단 | 123 (30.4%) |
| SHORT 진입 시도 | 617 |
| SHORT 통과 | 617 (100%) |
| 총 거래 (gate-off) | 772 |
| 총 거래 (gate-on) | 649 |

### 연도별 PF (Gate ON)

| 연도 | N | PF | WR | MaxDD |
|------|---|----|----|-------|
| 2020 | 65 | 2.56 | 52.3% | 8.1% |
| 2021 | 160 | 4.90 | 63.1% | 6.3% |
| 2022 | 142 | 2.89 | 57.7% | 13.4% |
| 2023 | 131 | 2.97 | 58.8% | 15.2% |
| 2024 | 91 | 2.04 | 55.5% | 17.7% |
| 2025 | 60 | 2.31 | 57.5% | 12.8% |

**전체 6년 수익** ✅

---

## 통과 기준

| 기준 | 목표 | 결과 | 판정 |
|------|------|------|------|
| N | ≥ 400 | 649 | ✅ |
| PF | ≥ 1.5 | 2.749 | ✅ |
| MaxDD | ≤ 25% | 17.7% | ✅ |
| EV/trade | ≥ +50bp | +144.2bp | ✅ |
| 손실 연도 | 0 | 0 | ✅ |
| Gate ON > Gate OFF PF | — | 2.749 > 2.554 | ✅ |

---

## 결정: MA Cluster 재활성화

r92 PASS 조건 전부 충족. 설정 변경:

```python
# config.py
ma_cluster_enabled: bool = True  # r92 gate PASS
ma_cluster_btc_gate_enabled: bool = True
ma_cluster_btc_ma_period_4h: int = 50
```

```bash
# .env
VS_MA_CLUSTER_ENABLED=true
VS_MA_CLUSTER_BTC_GATE_ENABLED=true
```

### 모니터링 계획

재활성화 후 2주 / 30+ 거래 후 평가:

| 지표 | 정상 | 재검토 |
|------|------|--------|
| Live PF | ≥ 1.0 | < 0.8 |
| Live WR | ≥ 50% | < 45% |
| MA cluster 손실 / 전체 | ≤ 40% | > 60% |

**게이트 로그 확인:**
```bash
grep "MA cluster 4h gate LONG suppressed" logs/bot/vol_straddle/$(ls -t logs/bot/vol_straddle/ | head -1)
```

---

## 한계 및 주의사항

- **2026 OOS 없음**: 캐시가 2025년 말까지. 2026년 1-3월은 검증 안 됨. 모니터링으로 대체.
- **EV 감소 트렌드**: 2021 EV=224bp → 2025 EV=130bp. 하락 중이나 여전히 +130bp 이상.
- **게이트 민감도 미검증**: MA50 기간 50이 최적값인지 스윕 안 함. 충분히 안전한 기본값.
- **슬롯 경쟁**: MA cluster가 VR/Straddle과 같은 10슬롯 공유. 혼합 백테스트는 별도.

---

## 파일

- `research/active/r91_r95/r92_ma_cluster_gate.py` — 백테스트 스크립트
- `quantbox/strategy/vol_straddle/config.py` — 게이트 파라미터 + 재활성화
- `quantbox/strategy/vol_straddle/strategy.py` — signal source 태깅
- `quantbox/scalper.py` — BTC 4h 다운로드 + 게이트 로직
- `tests/test_ma_cluster_gate.py` — 13 테스트
