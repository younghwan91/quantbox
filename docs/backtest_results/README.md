# Backtest Results Archive

각 파일은 특정 git 커밋에 연결된 백테스트/A-B 테스트 결과를 기록합니다.

## 폴더 구조

```
docs/backtest_results/
├── 202602/   # 2026-02 (R2~R23, 방향비대칭, 클라이맥스 실험 등 — 31개)
└── 202603/   # 2026-03 (R31~R45, v13/v14 배포 — 19개)
```

## 네이밍 규칙

```
{YYYYMMDD}_{commit_short}_{description}.md
```

예시: `20260302_e4b5831_v13_deployment.md`

파일 이름의 날짜 순으로 `ls | sort`만 해도 시간 순 정렬 완성.

## 불변 베이스라인 (변경 금지)

| 파트 | 레퍼런스 파일 | 핵심 수치 |
|---|---|---|
| **VR Squeeze** | `20260302_d70cd2f_yearly_r39.md` (202603/) | N=2,265 (D+S), PF=1.280, WR=43.4% |
| **MA Cluster** | `20260304_d4e70d8_r45_param_sweep.md` (202603/) | N=1,631, PF=1.365 (R45 필터 적용) |
| **Combined** | `20260304_d4e70d8_r45_param_sweep.md` (202603/) | N=3,748, PF=1.304, Calmar=18.8 |
| **v13 배포** | `20260302_e4b5831_v13_deployment.md` (202603/) | AnnRet+687%, DD=68%, Calmar=10.12 |
| **v14 배포** | `20260304_ef55e2e_quantbox_v14_strategy_report.md` (202603/) | AnnRet+1,086%(1x), Calmar=19.4 |

> `scaling_r38.py`의 `collect_trades()` + `portfolio_sim()`이 모든 수치의 canonical sim. 재구현 금지.

## 인덱스 — 202602/ (Rubber Band era + VR Alpha 탐색)

| 날짜 | 커밋 | 설명 | 핵심 결과 |
|---|---|---|---|
| 2026-02-18 | `8eb95b2` | v11.4 Adaptive Threshold 365d | Sharpe 7.78 |
| 2026-02-19 | `27d6339` | R3+R4: Smart Exit A/B | **Y-Kitchen** WR 67%, Sharpe 8.42, PF 2.07 |
| 2026-02-19 | `6618f55` | 3-Year A/B Test (7 dimensions) | Sharpe 5.61 |
| 2026-02-19 | `76d71af` | Performance Assessment A/B/C | **B wins**: Sharpe 6.07 |
| 2026-02-19 | `f26299b` | Bidirectional Flip Assessment | MTF Block wins |
| 2026-02-19 | `6eb0212` | Universe 20→30 expansion | **B-30**: Sharpe 7.09 (+19%) |
| 2026-02-19 | `dab1f80` | Universe B-30/40/50 sweep | **B-50**: Sharpe 7.24, +6812% |
| 2026-02-19 | `be25665` | Filter Relaxation A/B | **E-3Slot**: Sharpe 6.41, 26.1 trades/day |
| 2026-02-19 | `5900fa5` | Partial Exit + Trailing A/B | **E-BE15**: R:R 0.98→1.51 |
| 2026-02-20 | `0690b0f` | R5 Y-Kitchen Combos + 6yr crash | **J-YK+2Slt** survives crash era |
| 2026-02-20 | `3fbf1d6` | R6 Combo Variants | **M-2SSTSX**: DD 8.0%, Sharpe 5.50, PF 1.32 |
| 2026-02-20 | `bdd85f0` | R7 3D Exit Grid (101 variants) | **T62_S79_PBT**: R:R +103% |
| 2026-02-20 | `3f63b4c` | R8 Stress Test | All PBT Grade A+, T100S38 most stable |
| 2026-02-21 | `a09a0da` | R10 VWMA Multi-Factor | VWMA(100) #1 predictor |
| 2026-02-21 | `25b983d` | R11 VWMA+POC R:R Max | **U-RR+VWS** Calmar 1004 |
| 2026-02-21 | `0b7936d` | R12 Trailing Stop Innovation | Runner near-optimal, ATR×0.7 best |
| 2026-02-21 | `614a449` | Direction Asymmetry R1+R2 (6yr) | Longs+25% EV, baseline wins Calmar |
| 2026-02-21 | `7f6bd77` | Adaptive Direction A/B (6yr) | DualStopTgt first to beat Return |
| 2026-02-23 | `ca2ceba` | Trail Factor R3 (6yr) | C-TF10 monotonic: lower=better |
| 2026-02-23 | `cb1d9cb` | Adaptive Exit R13 | K-ConfHold Calmar 1039 |
| 2026-02-24 | `d80623f` | **R21: Climax+ATR Stop → v12.6** | **Calmar 728→1171 (+60.8%), DD 5.8%** |
| 2026-02-26 | `29eecab` | R22 Zero-Lookahead Diagnostic | ❌ No edge — 4h MTF lookahead fix |
| 2026-02-26 | `29eecab` | R23 New Alpha Search | ❌ 27/27 PF < 1.0 — RB 알파 없음 확인 |

## 인덱스 — 202603/ (VR Squeeze + MA Cluster, 배포)

| 날짜 | 커밋 | 설명 | 핵심 결과 |
|---|---|---|---|
| 2026-03-01 | `43fc452` | R36 Frequency Boost | N+46%, PF 1.382 |
| 2026-03-02 | `500478f` | R37 Frequency Boost v2 | |
| 2026-03-02 | `9d01aa4` | **R38 Scaling (canonical sim)** | **`scaling_r38.py` 정식 베이스라인** |
| 2026-03-02 | `0b01f9a` | R38b Grid | |
| 2026-03-02 | `d70cd2f` | **R39 Yearly (VR 6/6yr)** | **VR PF=1.280, WR=43.4% — 불변 기준** |
| 2026-03-02 | `149c0ec` | R40 Tournament | |
| 2026-03-02 | `748630b` | R40 Deep Stats | Bootstrap CI [1.158, 1.479], P=100% |
| 2026-03-02 | `e4b5831` | **v13.0 배포** | **25x/10s, AnnRet+687%, Calmar 10.12** |
| 2026-03-03 | `bb7cb38` | R41 Johansen multi-cointegration | ❌ 3% profitability — 미채택 |
| 2026-03-03 | `03b2b6b` | R42 Regime-Adaptive VR | MARGINAL — static params robust |
| 2026-03-03 | `4bb90af` | **R43 Exit Improvement** | **Trailing stop → v13.1: d_tp=12%, trail +2%/0.3%** |
| 2026-03-03 | `42034c4` | R44 High-Vol Complement | ❌ 161+ configs, PF<1.0 |
| 2026-03-04 | `d4e70d8` | **R45 MA Cluster Param Sweep** | **Rank#6: PF=1.304, Calmar=18.8 — 배포** |
| 2026-03-04 | `ef55e2e` | **v14.0.0 Strategy Report** | **Dual AnnRet+1,086%(1x), Calmar=19.4** |
| 2026-03-04 | `6fdd86c` | **v14.0.0 Current Status** | **868 tests, Paper $1,668** |
