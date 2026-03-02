# R41 — Johansen Multi-Cointegration Spread Research

**Date:** 2026-03-03  
**Baseline:** v13.0.1 (tag)  
**Branch:** `feature/multi-cointegration`  
**Script:** `research/active/johansen_vecm_r41.py`  
**Objective:** 수수료보다 큰 수익을 만들어라 — VR squeeze 보완 전략으로 저변동 구간 수익 확보

## 실험 설계

### 데이터
- 54 바이낸스 USDT-M 선물 심볼, 6년 (2190일) 15m 캔들
- 4h 리샘플링 (resample factor=16) → 바당 ~13,000개
- 1h/15m은 스프레드 진폭이 비용을 넘지 못해 제외

### Walk-Forward 방법론
1. **Formation Window** (125 bars = 21일 @4h): Johansen 공적분 테스트
2. **OOS Window** (125 bars = 21일 @4h): 동결된 가중치로 매매 시뮬레이션
3. 125 bars씩 슬라이딩, 바스켓당 최대 30 WF steps
4. **룩어헤드 방지**: OOS 구간에서 formation 통계(평균, 표준편차)만 사용

### 비용 모델
- 수수료: 0.05%/side (maker, LimitChaser 가정)
- 슬리피지: 0.05%/side
- 펀딩비: 0.01%/8h × 총 |weight| × 양방향 × 추정 보유 기간
- Round-trip = sum(|weight|) × 2 × 0.10%
- 3-asset 평균: ~46 bps, 4-asset: ~51 bps, 5-asset: ~57 bps

### 매매 규칙
- 진입: |z-score| ≥ 1.5 (formation 통계 기반)
- 청산: |z-score| ≤ 0.3 (평균 회귀)
- 손절: |z-score| ≥ 3.0
- 최대 보유: 12 bars (48h @4h)

## 결과 요약

| 지표 | 3-asset | 4-asset | 5-asset |
|------|---------|---------|---------|
| 테스트 바스켓 수 | 398 | 399 | 400 |
| 수익 바스켓 수 | 17 (4.3%) | 13 (3.3%) | 5 (1.3%) |
| 평균 비용/trade (bps) | 45.6 | 50.8 | 56.8 |
| 평균 순 PnL/trade (bps) | -52.3 | -53.7 | -61.9 |
| 최고 순 PnL/trade (bps) | +146.4 | +45.2 | +44.9 |

**전체: 36 / 1,199 (3.0%) 수익성 확보**

## TOP 5 바스켓

### #1: ADA + ICP + TON (4h)
- 가중치: [1.0, -0.52, -0.16], 총 |w|=1.68
- 순 PnL: +146.4 bps/trade, 비용: 40.4 bps
- 17 trades, WR=64.7%, PF=2.89
- 반감기: 2.7 bars (~11h), Hurst: 0.249
- 공적분 안정성: 10% (30 WF 중 3 window만)
- ⚠ **17 trades — 통계적 유의성 부족**

### #2: LTC + OP + UNI (4h)
- 순 PnL: +72.8 bps/trade, 비용: 58.3 bps
- 23 trades, WR=47.8%, PF=1.69
- 공적분 안정성: 3% → **극도로 불안정**

### #3: APT + COMP + RUNE (4h)
- 순 PnL: +51.2 bps/trade, 14 trades
- 공적분 안정성: 7% → **불안정**

### #4: AXS + CRV + DYDX + SNX (4h) ⭐
- 가중치: [1.0, -0.20, -0.28, -0.69], 총 |w|=2.17
- 순 PnL: +45.2 bps/trade, 비용: 49.9 bps
- **114 trades**, WR=44.7%, PF=1.32
- 반감기: 2.0 bars (~8h), Hurst: 0.095
- 공적분 안정성: **27% (8/30 windows)**
- ADF t-stat: -3.00
- 총 누적: +5,152 bps (51.5%)
- ⭐ **가장 유망 — 높은 트레이드 수 + 양호한 안정성**

### #5: 1000SHIB + BCH + ICP (4h)
- 순 PnL: +46.6 bps/trade, 비용: 49.5 bps
- 52 trades, WR=48.1%, PF=1.52
- 공적분 안정성: 20%

## 핵심 발견

### 1. 비용이 대부분의 스프레드를 소멸시킨다
- 1,199개 바스켓 중 **97%가 비용 후 손실**
- 평균 비용 ~50 bps vs 스프레드 평균 회귀 진폭이 빈약
- 이전 pairs 레포와 동일한 결론: "round-trip cost > mean-reversion profit"

### 2. 3-asset이 4/5-asset보다 유리
- 다리(legs) 수가 적을수록 비용이 낮아 수익 확률 증가
- 5-asset: 1.3% 수익률 → 비용이 진폭을 완전히 흡수

### 3. 공적분 안정성이 낮다
- 최고 안정성: 43% (DOT+FIL+NEAR+XLM, 하지만 순 +21 bps로 미미)
- 대부분 10-30% → 30개 window 중 3-9개만 공적분 관계 유지
- **암호화폐 공적분은 일시적이며 불안정**

### 4. 4h가 유일하게 작동하는 타임프레임
- 1h: 0% 수익 (스프레드 진폭 < 비용)
- 15m: 테스트 생략 (더 심할 것으로 판단)
- 4h: 일부 수익 가능 (스프레드 움직임이 충분히 큼)

### 5. 선택 편향(Selection Bias) 위험
- 1,199개 바스켓 다중 검정 → 일부는 순전히 우연으로 수익
- Bonferroni 보정 후 유의한 바스켓: 가능성 낮음
- #4 (114 trades, PF=1.32)만이 통계적으로 어느 정도 신뢰 가능

## PnL 단위 버그 수정 이력

초기 구현에서 z-score 변화량(무차원)과 비용(가격 비율)을 직접 비교하는 심각한 버그 발견.
- **수정 전**: `trade_pnl = z_change - cost_rt` → 비용이 무시됨 → 42% 수익률 (허위)
- **수정 후**: `trade_pnl = z_change × form_std - total_cost` → 올바른 단위 → 3% 수익률 (현실)

## 결론

**Multi-cointegration 전략은 현재 상태로는 VR squeeze의 보완 전략으로 부적합.**

근거:
1. 수익 바스켓 비율 3% — 선택 편향 위험
2. 공적분 안정성 <30% — 실시간에서 "지금 공적분인가?" 판단 어려움
3. 최고 단일 바스켓(#4) 연 19 trades → 연 수익 ~$860 (@$10K) → 실전 ROI 미미
4. 저변동 구간 보완이라는 목적 달성 불확실 — 공적분은 변동성과 무관하게 불안정

### 다음 단계 (만약 계속한다면)
1. **적응형 진입**: 실시간 Johansen 테스트 → 공적분 확인된 기간만 매매
2. **더 많은 심볼**: 54개 → 200+ 바이낸스 심볼로 확장
3. **Formation 길이 최적화**: 125 bars 외에 60-300 bars 범위 탐색
4. **다른 접근법 고려**: 
   - Funding rate arbitrage (안정적 수익, 낮은 리스크)
   - Cross-exchange arbitrage
   - Volatility-regime adaptive strategy (현행 VR squeeze의 파라미터를 변동성에 따라 조절)
