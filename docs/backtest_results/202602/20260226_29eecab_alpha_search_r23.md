# R23: New Alpha Search — 4 Dimensions

| Field | Value |
|-------|-------|
| **Commit** | `29eecab` |
| **Date** | 2026-02-26 14:58 UTC |
| **Script** | `research/active/r21_r30/r23_alpha_search.py` |
| **Data** | 2190d (6yr), 50 symbols, 210,335×15m bars |
| **Conditions** | ZERO cost, 1x leverage, zero lookahead |
| **Runtime** | 57 seconds |

## Executive Summary

**❌ 27개 변형 전부 PF < 1.0. 15m 크립토 선물에서 OHLCV 데이터만으로는 알파가 없다.**

- Best: **C2-Range-MR-0.2** (PF=0.905) — 극도로 횡보인 구간에서만 MR
- 모멘텀은 MR보다 **더 나쁘다** (Best momentum: B4 PF=0.817)
- 볼륨 시그널은 **방향성 예측에 무능하다** (WR ~50% but PF 0.66-0.70)
- 타임프레임을 높일수록 **성능이 하락한다** (15m → 1h → 4h)
- 4가지 방향 모두 실패 → OHLCV 기반 접근의 근본적 한계

## Phase A: Timeframe (1h/4h Mean-Reversion)

| Variant | N | WR% | R:R | PF | EV(bp) | SL% |
|---------|------|------|------|-------|--------|-----|
| **A1-MR-15m** (baseline) | 100,581 | 35.9 | 1.60 | **0.896** | -0.085 | 62% |
| A2-MR-1h | 61,007 | 33.3 | 1.51 | 0.752 | -0.415 | 66% |
| A3-MR-1h-conf≥3 | 21,343 | 33.5 | 1.55 | 0.780 | -0.384 | 65% |
| A4-MR-4h | 16,715 | 35.6 | 1.25 | 0.694 | -0.904 | 61% |

**결론: 타임프레임을 높이면 MR이 더 나빠진다.** 15m가 최선이며 그조차도 PF < 1.0.
- 1h: PF 0.896 → 0.752 (−16%)
- 4h: PF 0.896 → 0.694 (−23%)
- SL%가 62→66%로 증가 = 높은 TF에서 mean-reversion이 더 약하다

## Phase B: Momentum (Follow the Trend)

| Variant | N | WR% | R:R | PF | EV(bp) | SL% |
|---------|------|------|------|-------|--------|-----|
| B1-Mom-15m (baseline) | 127,267 | 34.3 | 1.41 | 0.737 | -0.210 | 64% |
| B2-Mom-15m-asym (2.5x/0.15) | 127,273 | 11.0 | 5.87 | 0.723 | -0.122 | 89% |
| B3-Mom-15m-noBTC | 448,621 | 34.3 | 1.43 | 0.750 | -0.197 | 64% |
| B4-Mom-15m-trail (3x/0.2) | 309,400 | 13.1 | 5.42 | **0.817** | -0.103 | 87% |
| B5-Mom-1h | 85,793 | 35.8 | 1.45 | 0.805 | -0.291 | 62% |

**결론: 모멘텀은 MR보다 더 나쁘다.** 
- Best momentum (B4-trail): PF=0.817 < MR baseline PF=0.896
- 15m 크립토에서 "급격한 움직임 이후" 추세도 약하고, 반전도 약하다
- 움직임 이후에는 그냥 **노이즈**가 지배적
- BTC 필터 유무 차이 미미 (B1 vs B3: 0.737 vs 0.750)

## Phase C: Regime-Conditional

| Variant | N | WR% | R:R | PF | EV(bp) | SL% |
|---------|------|------|------|-------|--------|-----|
| **C2-Range-MR-0.2** | 43,143 | 35.0 | 1.68 | **0.905** | -0.069 | 64% |
| C1-Range-MR-0.3 | 72,897 | 35.3 | 1.63 | 0.889 | -0.084 | 63% |
| C5-Adaptive (MR+Mom) | 76,380 | 29.9 | 2.04 | 0.870 | -0.090 | 69% |
| C6-Adaptive-wide | 67,031 | 29.9 | 2.02 | 0.862 | -0.098 | 69% |
| C4-Trend-Mom-0.4 | 18,166 | 19.2 | 2.88 | 0.686 | -0.256 | 80% |
| C3-Trend-Mom-0.5 | 3,481 | 19.6 | 2.31 | 0.564 | -0.487 | 79% |

**결론: Range-only MR이 전체 27개 중 최고 (PF=0.905), 하지만 여전히 < 1.0.**
- ER < 0.2 (극도로 횡보) 필터가 R:R을 1.60→1.68로 개선
- 추세장 모멘텀(C3, C4)은 최악: PF=0.56-0.69, SL%=79-80%
- Adaptive(MR+Mom 혼합)은 각각보다 나쁨 → 시너지 없음

## Phase D: Volume-Based Alpha

| Variant | N | WR% | R:R | PF | EV(bp) | SL% |
|---------|------|------|------|-------|--------|-----|
| D5-VolEnhanced-2x (MR+vol) | 83,243 | 35.6 | 1.60 | 0.885 | -0.099 | 63% |
| D5b-VolEnhanced-3x | 33,099 | 35.5 | 1.59 | 0.876 | -0.113 | 63% |
| D6-VolBreakout-20 | 168,723 | 47.6 | 0.81 | 0.737 | -0.133 | 52% |
| D6b-VolBreak-48 | 81,634 | 42.7 | 0.96 | 0.718 | -0.158 | 57% |
| D2-TakerExhaust (0.35/0.65) | 203,507 | 50.2 | 0.70 | 0.706 | -0.121 | 50% |
| D3b-CVD-Div-24 | 376,195 | 51.8 | 0.65 | 0.703 | -0.139 | 48% |
| D2b-TakerMod (0.38/0.62) | 321,339 | 50.1 | 0.70 | 0.700 | -0.126 | 50% |
| D3-CVD-Div-48 | 370,423 | 51.8 | 0.65 | 0.693 | -0.144 | 48% |
| D1-VolClimax-2.5x | 24,536 | 50.3 | 0.68 | 0.684 | -0.149 | 50% |
| D1b-VolClimax-3.5x | 9,747 | 49.9 | 0.68 | 0.679 | -0.155 | 50% |
| D4b-VWAP-672-2.5 | 2,608,795 | 56.4 | 0.52 | 0.678 | -0.139 | 44% |
| D4-VWAP-96-2.0 | 1,596,477 | 56.7 | 0.51 | 0.661 | -0.150 | 43% |

**결론: 볼륨 시그널은 방향 예측에 실패한다.**
- D5 (MR+vol 필터)는 기본 MR과 거의 동일 (PF 0.896→0.885) → 볼륨이 알파를 추가하지 않음
- 순수 볼륨 시그널 (D1-D4): WR ~50-57%이지만 R:R 0.51-0.70 → 이길 때 적게 벌고 질 때 많이 잃음
- CVD divergence: WR=51.8% (coin flip)
- Taker exhaustion: WR=50.2% (coin flip)
- VWAP: 가장 많은 시그널 (2.6M) 중 WR=56.4%이지만 R:R=0.52 → PF=0.678
- **볼륨 데이터는 15m 크립토 선물에서 방향성 edge를 제공하지 않는다**

## Full Ranking (27 variants, PF descending)

| Rank | Label | N | WR% | R:R | PF | Phase |
|------|-------|------|------|------|-------|-------|
| 1 | **C2-Range-MR-0.2** | 43,143 | 35.0 | 1.68 | **0.905** | C |
| 2 | A1-MR-15m | 100,581 | 35.9 | 1.60 | 0.896 | A |
| 3 | C1-Range-MR-0.3 | 72,897 | 35.3 | 1.63 | 0.889 | C |
| 4 | D5-VolEnhanced-2x | 83,243 | 35.6 | 1.60 | 0.885 | D |
| 5 | D5b-VolEnhanced-3x | 33,099 | 35.5 | 1.59 | 0.876 | D |
| 6 | C5-Adaptive | 76,380 | 29.9 | 2.04 | 0.870 | C |
| ... | ... | ... | ... | ... | ... | ... |
| 27 | C3-Trend-Mom-0.5 | 3,481 | 19.6 | 2.31 | 0.564 | C |

## Diagnosis: 왜 4가지 방향 모두 실패하는가

### 1. 15m 크립토는 "Near-Random Walk"
모든 예측 시그널 (MR, 모멘텀, 볼륨, 레짐)이 PF < 1.0이라는 것은 15m 크립토 선물 가격이 **예측 불가능에 매우 가깝다**는 의미.

### 2. MR은 "거의" 작동하지만 부족하다
- Best PF = 0.905 (gap: 0.095). WR이 35% → 38.6%만 되면 PF > 1.0
- 그러나 이 3.6%p 갭은 6년/10만 트레이드/27개 변형에서 일관적 → 노이즈가 아닌 구조적 한계

### 3. 모멘텀은 MR보다 확실히 나쁘다
- 급격한 움직임 이후 가격은 **반전도, 지속도 하지 않는다** → 그냥 noise
- MR이 0.896, Momentum이 0.737 → 둘 다 coin flip보다 나쁘지만, MR이 "덜 나쁘다"
- 이는 약한 mean-reversion 경향이 있지만 수익화할 정도는 아님을 의미

### 4. 볼륨은 방향 예측력이 없다
- OHLCV 기반 볼륨 지표 (스파이크, CVD, taker ratio, VWAP)는 **시장 활동도**를 알려주지만 **방향**을 예측하지 못함
- 실시간 호가창(orderbook depth, L2/L3 data)이 필요할 수 있지만, 이는 15m 바 데이터로는 불가능

## Verdict

```
❌ 27/27 variants: PF < 1.0 at ZERO cost.

  Best: C2-Range-MR-0.2 (PF=0.905, gap=0.095)
  The 15m crypto futures market on OHLCV candle data is 
  effectively unpredictable with these approaches.

  OHLCV-based technical analysis on 15m crypto futures 
  does not contain exploitable alpha.
```

## What Could Work (Outside This Framework)

1. **Real-time orderbook data** — L2 depth, order imbalance, spoofing detection
2. **Cross-exchange arbitrage** — price discrepancies between venues
3. **Funding rate arbitrage** — long/short perp vs spot when funding is extreme
4. **On-chain data** — whale wallet movements, exchange deposit/withdrawal flows
5. **Sentiment/news** — social media sentiment spikes, news event prediction
6. **Inter-asset relationships** — lead-lag between correlated assets (not just BTC as regime)
7. **Higher-frequency** — tick data, sub-second execution where market microstructure creates edge
8. **Different markets** — Crypto is extremely efficient at 15m. Illiquid altcoins or new listings might have edge.
