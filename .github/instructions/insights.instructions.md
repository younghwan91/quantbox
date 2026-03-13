# LinkedIn Quant Insights Skill

> **목적:** 전략 노출 없이, 연구에서 나온 데이터를 전문적인 LinkedIn 포스트로 변환하는 표준 절차.
> 목표 독자: Two Sigma / Citadel 리서처가 RT할 만한 수준. "좋은 알고 블로그"가 아니라 "퀀트 커뮤니티 레퍼런스".

---

## 0. Wall Street Quant Quality Bar (필수, 미충족 시 발행 금지)

**필수 3요소 — 모두 있어야 발행:**

### 0-1. 메커니즘 (Mechanism)
"왜"를 반드시 설명. 단순 상관 관찰 금지.

| Bad | Good |
|-----|------|
| "Max drawdown nearly doubled when including crash years" | "During 4 crash periods, cross-asset crypto ρ spiked 0.12 → 0.61. A 4-position portfolio that behaves as 4 independent bets normally effectively becomes ~1.6 positions during drawdowns — that's where the 6.7% → 13.2% gap comes from." |
| "Win rate is more stable than R:R" | "Win rate is regime-sensitive: 68% in trending markets, potentially 58% in mean-reverting conditions. R:R is determined by exit logic — it doesn't shift with regime changes. That asymmetry is the structural argument." |

### 0-2. 통계 근거 (Statistical Grounding)
OOS 또는 bootstrap을 한 줄이라도 언급. 없으면 한계를 명시.

- "Bootstrap resampling (1,000 paths): edge held in 94% of cases"
- "Consistent across OOS period 2024-2025 (PF 1.21 vs IS 1.27)"
- "In-sample only — OOS validation pending" (최소한 이 정도 명시)

### 0-3. 지적 정직성 (Intellectual Honesty)
데이터 한계를 한 줄로 명시. 이게 있어야 퀀트가 신뢰한다.

- "This data spans 2020-2026, including the 2020-2022 bull run. EV estimates from that window likely overstate what's available in calmer regimes."
- "Measured on USDT-M perpetuals only — transferability to spot or TradFi assets is untested."

### 0-4. Precision Test (발행 직전 체크)
- [ ] 독자가 데이터 설명만으로 결과를 재현할 수 있는가?
- [ ] 모든 숫자가 한 가지 측정 방법으로만 나올 수 있는가?
- [ ] "the data suggests" / "in this dataset" 없이 단정하는 문장이 있는가? → 수정
- [ ] 메커니즘이 "숫자가 다르다"가 아니라 "왜 다른지"를 설명하는가?

---

## 1. 포스트 구조 (6-Block Formula)

```
[Block 1] HOOK — 반직관적 주장 or 놀라운 숫자 (1-2줄)
           → 첫 2줄이 "더 보기" 클릭 여부 결정
           → 퀀트용: 구체적 수치로 믿음을 깨는 문장

[Block 2] SETUP — 실험 조건 간결 설명 (2-3줄)
           → 기간, 심볼 수, 무엇을 비교했는지

[Block 3] DATA — 핵심 수치 제시 (3-5줄)
           → 퀀트용: 숫자 + 단위 + 비교 기준이 모두 명시

[Block 4] MECHANISM — "왜"를 설명 (2-4줄, 핵심)
           → 상관 관계가 아닌 인과 메커니즘
           → 간단한 수식이나 비율이 있으면 더 신뢰도 높음

[Block 5] HONEST LIMITS — 데이터 한계 + 조건부 결론 (1-2줄)
           → 이게 있어야 퀀트가 RT한다

[Block 6] FOOTER — 데이터 출처 credibility
           → 심볼 수 · 기간 · 거래 건수 · 거래소

[Tags] 3-5개. 하단.
```

---

## 2. 스타일 규칙

| 규칙 | 퀀트 기준 | 일반 블로그 수준 |
|------|-----------|----------------|
| 숫자 | "ρ 0.12 → 0.61 during drawdowns" | "correlation spikes" |
| 메커니즘 | 인과 + 수식/비율 | 상관 관계 언급 |
| 결론 | 조건부 ("in this regime...") | 단정 ("always better") |
| 한계 | 명시 ("2020-22 bull included") | 생략 |
| 실용성 | 구체적 체크 방법 | "worth thinking about" |
| 길이 | 350-450 단어 (밀도 있으면 더 읽힘) | 300 단어 |

**톤 원칙 — 겸손한 연구자 관점:**
- 발견을 공유하되, 독자에게 결론을 강요하지 않는다
- 단정적 선언 대신 관찰: "the data suggests" > "the data proves"
- 독자를 깎아내리는 표현 금지
- 전략 자랑 금지

**절대 쓰지 말 것:**
- "my strategy", "my system", "I use this in live trading"
- 구체적 진입 조건, SL/TP 파라미터
- 특정 전략 이름 (VR Squeeze, MA Cluster 등)
- 과도한 자신감: "this is not a fluke", "structural flaw"

**항상 쓸 것:**
- 데이터 출처: "X symbols · Y years · Binance USDT-M perpetuals"
- 메커니즘 설명 (왜 그런 결과가 나왔는지)
- 지적 정직성 한 줄
- **포스트 첫 줄에 타이틀**: 짧고 궁금증 유발 (LinkedIn 미리보기에 노출됨)

---

## 3. 소재 발굴 절차 (매일 1포스트)

### Step 1: 리서치 문서 스캔
```bash
ls docs/backtest_results/202602/ docs/backtest_results/202603/
```
→ 날짜 오래된 것부터. 아직 안 쓴 것 선택.

### Step 2: 인사이트 적합성 체크
아래 3가지 중 하나를 만족하면 소재로 사용:
- [ ] **반직관적**: 상식과 반대되는 결과인가?
- [ ] **보편적 교훈**: 퀀트도 "맞아!" 할 내용인가?
- [ ] **놀라운 숫자**: 숫자 자체가 임팩트가 있는가?

### Step 3: 안전성 체크 (전략 노출 여부)
- [ ] 신호 로직 없음 (VR ratio, momentum, squeeze 등 언급 금지)
- [ ] 구체적 파라미터 없음 (SL=1%, trail_activate 등 금지)
- [ ] "내 전략" 언급 없음
- [ ] 시장 구조 관찰로만 프레이밍 가능한가?

**전략 노출 위험 단어 (문맥 무관 금지):**

| 단어/표현 | 위험 이유 | 대체 표현 |
|-----------|-----------|-----------|
| breakout / 브레이크아웃 | 전략 유형 직접 노출 | "directional momentum", "trending move" |
| squeeze | VR Squeeze 직접 노출 | "low-volatility period" |
| ATR / volatility ratio | 신호 계산 방식 노출 | "volatility measure" |
| 15-minute / 15m bars | 구체적 타임프레임 노출 | "intraday data" |
| trailing stop activate | 파라미터 명 노출 | "profit protection threshold" |
| concurrent positions | 슬롯 구조 암시 | "position limit", "simultaneous trades" |

**검토 방법 (포스트 작성 후 반드시):**
```
grep -iE "breakout|squeeze|ATR|15m|15-minute|trail.*activ|concurrent" insights/post_NNN*.txt
```
→ 출력이 있으면 수정 후 커밋.

### Step 4: 파일 생성

LinkedIn은 마크다운 미지원. 포스트별 폴더 구조 사용:

```
insights/
├── post_001/
│   ├── post_001_short_desc.py    ← 차트 생성 스크립트
│   ├── post_001_short_desc.png   ← 업로드할 이미지 (dpi=150, dark theme)
│   ├── post_001_short_desc.txt   ← LinkedIn 복붙용 plain text (EN+KR)
│   └── post_001_short_desc.md    ← 연구 노트/데이터 테이블 보관용
```

**txt 파일 규칙:**
- `**bold**` 사용 금지 → 그냥 텍스트로
- EN 포스트 → `---` 구분선 → KR 포스트 순서
- 해시태그는 EN 버전에만 (KR 버전은 데이터 푸터로 끝)
- 마지막에 hashtag 5개 이하

### Step 5: Quality Gate (발행 전)
```
Section 0의 필수 3요소 모두 충족?
[ ] 메커니즘 (인과, 수식/비율 포함)
[ ] 통계 근거 (OOS/bootstrap/한계 명시)
[ ] 지적 정직성 (데이터 기간/조건 명시)
```

### Step 6: 커밋
```bash
git add insights/post_NNN*
git commit -m "feat(insights): post #NNN — {제목}"
```

---

## 4. 차트 스타일 표준

```python
# 색상 팔레트 (항상 동일)
BG_DARK    = '#0d1117'   # 배경
BG_CARD    = '#161b22'   # 패널
BG_BORDER  = '#30363d'   # 테두리
TEXT_MAIN  = '#f0f6fc'   # 주요 텍스트
TEXT_SUB   = '#8b949e'   # 보조 텍스트
GREEN      = '#3fb950'   # 좋음
RED        = '#f85149'   # 나쁨
ACCENT     = '#58a6ff'   # 강조 (파란)
ORANGE     = '#d29922'   # 경고

# 이미지 설정
figsize = (16, 9)   # 와이드 (LinkedIn 최적)
dpi = 150
bbox_inches = 'tight'

# 규칙
# - 서브타이틀에 항상: 심볼 수 · 기간 · 비용 포함
# - 이모지 쓰지 말 것 (한국 폰트 없으면 □ 박스로 깨짐)
# - 범례는 facecolor='#21262d', edgecolor='#30363d'
```

### 텍스트 겹침 방지 규칙 (필수)

**1. 바 위에 숫자 레이블**
```python
ax.set_ylim(0, max_val * 1.20)
ypos = val + 0.05 if val >= 0 else val - 0.4
```

**2. 라인 차트 포인트 레이블**
```python
offset = 0.15 if i % 2 == 0 else -0.30
ax.text(i, val + offset, f'{val:.2f}', ...)
```

**3. Annotation**
```python
bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=color, alpha=0.85)
```

**4. 서브플롯 간격**
```python
gs = fig.add_gridspec(..., hspace=0.52, wspace=0.35,
                      top=0.87, bottom=0.09, left=0.07, right=0.97)
```

**5. 차트 검증 체크리스트**
- [ ] 각 패널 타이틀이 데이터와 겹치지 않는가?
- [ ] 바/포인트 레이블이 인접 원소와 겹치지 않는가?
- [ ] footer 텍스트가 x축 레이블과 겹치지 않는가?
- [ ] 범례가 데이터 영역과 겹치지 않는가?

---

## 5. 발행된 포스트 목록

| # | 제목 | 소재 | 발행일 |
|---|------|------|--------|
| 001 | Your 3-Year Backtest Might Be Missing the Hardest Part | r5 crash validation | 2026-03-08 |
| 002 | Win Rate Tells Half the Story | r8 stress test | 2026-03-08 |
| 003 | The Long Bias That Took Six Years to Show Up | r1 direction asymmetry | 2026-03-08 |
| 004 | More Symbols, More Alpha? | universe expansion | 2026-03-08 |
| 005 | The Slot Count Sweet Spot | r39 slot sweep | 2026-03-08 |
| 006 | Your Trailing Stop Is Probably Already Good Enough | r12 trailing | 2026-03-08 |
| 007 | I Tried Dynamic Parameters. Static Won. | r42 regime | 2026-03-08 |
| 008 | More Leverage or More Positions? | r52 RCB analysis | 2026-03-08 |
| 009 | Monte Carlo Said I'd Have $390K. It Also Said 21% Chance of Ruin. | r46 MC | 2026-03-08 |
| 010 | The Exit You Never Optimize (But Should) | r7 exit grid | 2026-03-08 |
| 011 | Your IS PF Was 2.16. Your Live PF Will Be 1.38. | r59 OOS validation | 2026-03-13 |
| 012 | Every Config Lost Money in One Year. Except This One. | r39 yearly consistency | 2026-03-13 |
| 013 | Kelly Said 154x. We Run 30x. Here's Why That's Right. | r63 Kelly audit | 2026-03-13 |
| 014 | The Backtest Said +$125. Live Delivered -$33. Same Signal. | r65 MFE exit gap | 2026-03-13 |

---

## 6. 미래 포스트 타이틀 큐

| 우선순위 | # | 타이틀 아이디어 | 핵심 숫자 | 리서치 출처 |
|---------|---|----------------|-----------|------------|
| 🔥 | 015 | Confidence Filtering: When Fewer Trades Is Better | conf≥3 vs conf≥2: fewer trades, higher EV, lower DD | r5/r6 (20260220) |
| 🔥 | 016 | One Parameter Changed. Every Year Got Better. | d_trail_activate 0.005→0.010, 2023+2024+2025 모두 PF 개선 | r69 yearly OOS (20260313) |
| ⭐ | 017 | The Calmar Ratio Is the Only Metric That Matters for Sizing | AnnRet/MaxDD vs Sharpe — why Calmar aligns with position sizing reality | r39 ranking (20260302) |
| ⭐ | 018 | Trail Factor AB Test: When "Better" Looks Identical | 18 trail variants, max diff = 2% return | r12/trail_factor (20260223) |

