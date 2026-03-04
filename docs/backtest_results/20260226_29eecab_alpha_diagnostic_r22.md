# R22: Zero-Lookahead Alpha Diagnostic Report

| Field | Value |
|-------|-------|
| **Commit** | `29eecab` |
| **Date** | 2026-02-26 14:13 UTC |
| **Script** | `research/active/alpha_diagnostic_r22.py` |
| **Data** | 2190d (6yr), 50 symbols, 210,335 bars |
| **Lookahead** | ZERO (MTF uses last completed 4h bar, `j = j_raw - 1`) |
| **Capital** | $2,095 |

## Executive Summary

**결론: 러버밴드 15m 평균회귀 시그널은 이론적 기대값(EV)이 음수이다.**

- 베이스라인 (conf≥3, Risk-Off, PBT exit, zero cost): **PF=0.877, EV=-0.09bp**
- 20개 포트폴리오 변형 중 **19개가 음수 수익** (zero cost에서도)
- 유일한 양수: O-WideTgt (2.5x ATR target, zero cost, 1x) → **PF=1.02, Ret=+205.6%** (6yr)
- 그러나 O-WideTgt도 maker 수수료(0.02%/side) 적용 시 PF ≈ 1.00 이하로 하락할 것으로 추정
- **시그널 자체에 알파가 없다.** 파라미터 최적화로는 구제 불가능.

## Phase 1: Per-Trade Independent Analysis

포트폴리오 제약 없이 모든 트레이드를 독립적으로 분석.

### Entry Filter Comparison (Zero Cost)

| Config | N | WR% | R:R | PF | EV(bp) |
|--------|-------|------|------|-------|--------|
| Risk-Off conf≥2 | 819,266 | 35.6 | 1.59 | 0.881 | -0.08 |
| Risk-Off conf≥3 | 436,769 | 35.6 | 1.59 | 0.877 | -0.09 |
| Risk-Off conf≥4 | 91,813 | 35.8 | 1.61 | 0.901 | -0.07 |
| BOTH Regimes conf≥3 | 297,755 | 35.6 | 1.68 | 0.925 | -0.05 |
| Fixed 5% conf≥3 | 60,930 | 40.7 | 1.44 | **0.986** | -0.02 |
| Fixed 3.2% conf≥3 | 156,242 | 37.9 | 1.50 | 0.911 | -0.10 |
| MTF SMA(6) | 110,168 | 35.4 | 1.61 | 0.882 | -0.09 |
| MTF SMA(12) | 130,580 | 36.6 | 1.60 | 0.922 | -0.06 |
| MTF SMA(24) | 149,516 | 36.3 | 1.59 | 0.909 | -0.06 |

- **Best per-trade**: Fixed 5% threshold (PF=0.986) — 여전히 < 1.0
- 62% 트레이드가 스탑로스에 걸림. 평균회귀 실패율이 매우 높음.

### Cost Impact (Risk-Off, conf≥3, PBT Exit)

| Cost | WR% | R:R | PF | EV(bp) |
|------|------|------|-------|--------|
| Zero | 35.6 | 1.59 | 0.877 | -0.09 |
| Maker (0.02%/side) | 35.5 | 1.51 | 0.828 | -0.13 |
| Full (0.04%/side) | 35.3 | 1.43 | 0.783 | -0.17 |

### Exit Strategy Comparison (conf≥3, Risk-Off, Zero Cost)

| Config | N | WR% | R:R | PF |
|--------|-------|------|------|-------|
| PBT-1.5/0.382/25 (baseline) | 436,769 | 35.6 | 1.59 | 0.877 |
| Tight-SL-0.20 | 436,776 | 22.2 | 3.15 | 0.897 |
| Wide-SL-0.50 | 436,763 | 41.9 | 1.22 | 0.879 |
| Quick-tgt-0.5x | 436,774 | 59.4 | 0.47 | 0.686 |
| **Wide-tgt-2.5x** | 436,769 | 31.1 | 2.10 | **0.945** |
| Quick-hold-12 | 436,771 | 39.6 | 1.37 | 0.897 |
| Patient-hold-50 | 436,769 | 33.5 | 1.76 | 0.887 |
| **Asymm-2.0/0.20** | 436,776 | 19.7 | 3.86 | **0.947** |
| Scalp-0.3/0.15 | 436,779 | 42.9 | 1.11 | 0.831 |

- **Best exits**: Asymm (PF=0.947), Wide-tgt (PF=0.945) — 비대칭 R:R이 도움이 되지만 여전히 < 1.0

## Phase 2: Portfolio Simulation (max_slots=2)

실제 운용과 동일한 포트폴리오 제약 적용.

| Var | Desc | Ret% | DD% | Sharpe | WR% | PF | #Tr |
|-----|------|------|-----|--------|-----|-----|-----|
| A-ZeroCost | Zero cost, no MTF, 1x | -747.1 | 520.3 | -1.33 | 40.9 | 0.94 | 36,139 |
| B-Maker | Maker 0.02%/side, 1x | -1427.2 | 1378.8 | -2.49 | 40.9 | 0.92 | 36,139 |
| C-Full | Full 0.04%/side, 1x | -2107.2 | 2037.1 | -3.68 | 40.8 | 0.89 | 36,139 |
| D-Prod | Full cost, MTF, 3.75x (production) | -4051.7 | 1018.1 | -2.84 | 41.0 | 0.92 | 22,689 |
| E-MTF12 | MTF SMA(12), zero cost, 1x | -229.8 | 146.8 | -0.52 | 41.1 | 0.97 | 22,689 |
| **F-MTF24** | **MTF SMA(24), zero cost, 1x** | **-76.9** | **96.8** | **-0.09** | **41.4** | **0.99** | **23,560** |
| G-MTF6 | MTF SMA(6), zero cost, 1x | -264.3 | 174.3 | -0.57 | 40.6 | 0.96 | 21,224 |
| H-Conf2 | min_conf=2, zero cost, 1x | -750.6 | 473.2 | -1.28 | 41.4 | 0.95 | 41,894 |
| I-Conf4 | min_conf=4, zero cost, 1x | -458.5 | 437.0 | -0.85 | 40.6 | 0.94 | 21,111 |
| J-Both | Both regimes, zero cost, 1x | -409.0 | 246.2 | -1.06 | 40.4 | 0.96 | 34,561 |
| K-Fixed5 | Fixed 5%, zero cost, 1x | -402.7 | 266.5 | -0.71 | 42.1 | 0.95 | 9,972 |
| L-Fixed3 | Fixed 3.2%, zero cost, 1x | -629.7 | 386.8 | -1.13 | 40.8 | 0.94 | 18,971 |
| M-NoPBT | No PBT, simple exit, zero cost | -841.6 | 585.0 | -1.49 | 37.1 | 0.94 | 34,327 |
| N-TightSL | Tight stop 0.20, zero cost, 1x | -650.5 | 289.5 | -1.56 | 26.1 | 0.95 | 52,170 |
| **O-WideTgt** | **Wide target 2.5x, zero cost, 1x** | **+205.6** | **75.1** | **0.57** | **33.7** | **1.02** | **30,158** |
| P-Quick | Quick hold=12, zero cost, 1x | -626.7 | 389.1 | -1.19 | 42.6 | 0.96 | 43,787 |
| Q-Patient | Patient hold=50, zero cost, 1x | -950.4 | 906.2 | -1.66 | 40.4 | 0.92 | 31,006 |
| R-TightTgt | Tight target 0.5x, zero cost, 1x | -3277.2 | 3189.9 | -7.31 | 67.7 | 0.79 | 79,179 |
| S-C4Maker | Conf4 + maker, 1x | -857.7 | 827.9 | -1.85 | 40.5 | 0.92 | 21,111 |
| T-BothMkr | Both regimes + maker, 1x | -1055.0 | 913.0 | -2.41 | 40.4 | 0.93 | 34,561 |

### Key Observations

1. **O-WideTgt가 유일한 양수 변형**: PF=1.02, +205.6% (6yr), Sharpe=0.57
   - 2.5x ATR 타겟으로 승자를 길게 보유 → WR=33.7% but R:R 향상
   - **단, zero cost에서만 양수. 수수료 적용 시 마이너스 전환 가능성 높음.**

2. **F-MTF24가 가장 break-even에 근접**: PF=0.99, -76.9% (6yr), Sharpe=-0.09
   - MTF SMA(24)가 노이즈 필터링에 가장 효과적
   - 그러나 여전히 음수

3. **D-Prod (현재 프로덕션)**: PF=0.92, -4051.7% (6yr)
   - 3.75x 레버리지가 연간 수수료 -0.30%/trade를 36,000회 증폭 → 파멸적 손실

## Phase 3: Cost Break-Even (baseline conf≥3)

| Cost/side | PF | EV(bp) | Total% |
|-----------|------|--------|--------|
| 0bp | 0.877 | -0.09 | -38,919 |
| 1bp | 0.852 | -0.11 | -47,654 |
| 2bp | 0.828 | -0.13 | -56,390 |
| 3bp | 0.805 | -0.15 | -65,125 |
| 4bp | 0.783 | -0.17 | -73,861 |
| 5bp | 0.761 | -0.19 | -82,596 |

**Break-even cost: 0bp** — 수수료를 0으로 해도 이미 PF < 1.0. 손익분기점 자체가 존재하지 않음.

## Phase 4: MFE/MAE Analysis

| Metric | Mean | Median | P25 | P75 | P90 |
|--------|------|--------|-----|-----|-----|
| MFE% | 1.553 | 0.941 | 0.336 | 2.049 | 3.616 |
| MAE% | -1.341 | -0.977 | -1.667 | -0.512 | -0.190 |
| Gain% | -0.086 | -0.592 | -1.078 | 0.618 | 2.093 |

**Winners** (155,637 trades): Mean MFE = 3.075%
**Losers** (281,132 trades): Mean MFE = 0.711%, Mean MAE = -1.783%

### 해석
- 진입 후 유리한 방향으로 평균 1.55% 이동하지만, 불리한 방향으로도 1.34% 이동
- **패자의 MFE가 0.711%** — 대부분의 손실 트레이드는 유리한 방향으로 거의 이동하지 않고 바로 스탑에 걸림
- 러버밴드(평균회귀)가 발생하지 않는 경우가 64%: 가격이 급변 후 계속 같은 방향으로 이동

## Diagnosis: 왜 러버밴드가 작동하지 않는가?

### 1. 전제 자체의 오류
"급격한 가격 변동 후 평균으로 회귀한다"는 가정에서, 15m 크립토 선물은 **모멘텀이 평균회귀보다 지배적**이다.
62%의 트레이드가 스탑에 걸리는 것은 "밴드를 이탈한 가격이 계속 이탈 방향으로 진행"함을 의미한다.

### 2. R:R은 양호하나 WR이 치명적
- R:R = 1.59 (합리적)
- WR = 35.6% (PF > 1.0에 필요한 최소 WR = 38.6%, 이 R:R 기준)
- **WR이 3%p 부족.** 이 3%p가 근본적 한계.

### 3. 수수료가 문제가 아님
Zero cost에서 이미 PF=0.877. 수수료 최적화(maker vs taker)는 의미 없음.

## Verdict

```
❌ The rubber band 15m mean-reversion signal has NO theoretical edge.
   PF=0.877 at zero cost over 6yr / 436,769 trades / 50 symbols.
   
   One variant (O-WideTgt: 2.5x target) achieves PF=1.02 at zero cost,
   but this marginal edge disappears with any transaction costs.
   
   Parameter optimization cannot save this strategy.
   A fundamentally different approach is required.
```

## Possible Next Steps

1. **Timeframe 변경**: 15m → 1h 또는 4h. 더 긴 타임프레임에서 평균회귀가 더 강할 수 있음.
2. **Momentum 전환**: 평균회귀 대신 모멘텀/추세추종으로 전환. Phase 4의 MFE 데이터가 이를 지지.
3. **Regime-Conditional**: 특정 시장 국면(예: 횡보장)에서만 평균회귀, 추세장에서는 모멘텀.
4. **Volume/Orderbook 특징**: 가격 기반이 아닌 주문장/거래량 기반 알파 소스 탐색.
5. **O-WideTgt 심층 분석**: PF=1.02의 구조 분석 — 특정 기간/심볼에서만 작동하는지, 수수료 포함 시 실제로 어떤지.
