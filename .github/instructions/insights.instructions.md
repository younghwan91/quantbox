# LinkedIn Quant Insights Skill

> **목적:** 전략 노출 없이, 연구에서 나온 데이터를 전문적인 LinkedIn 포스트로 변환하는 표준 절차.
> 매일 하나씩 옛날 리서치 문서에서 인사이트를 추출해 시리즈로 발행.

---

## 1. 포스트 구조 (5-Block Formula)

```
[Block 1] HOOK — 반직관적 주장 or 놀라운 숫자 (1-2줄)
           → 첫 2줄이 "더 보기" 클릭 여부 결정

[Block 2] STORY — 구체적 숫자로 상황 설명 (3-5줄)
           → "I ran X through Y" 아님. 데이터가 주어가 되게.

[Block 3] REVERSAL — 예상 외 결과 or 반전 (1-2줄, bold)
           → 이게 라이크를 만드는 핵심. 독자가 "어?" 하게.

[Block 4] INSIGHT — 한 줄 원칙으로 압축 (bold)
           → 기억에 남는 문장. 저장/공유 유발.

[Block 5] FOOTER — 데이터 출처 credibility (1줄, italic)
           → 숫자 · 심볼 수 · 기간 · 거래소

[Tags] 3-5개만. 하단 배치.
```

---

## 2. 스타일 규칙

| 규칙 | 좋은 예 | 나쁜 예 |
|------|---------|---------|
| 주어 | "The 3-year winner hit 13.2% DD" | "I found that my strategy..." |
| 숫자 | 구체적 ("6.7% → 13.2%") | 모호함 ("almost doubled") |
| 톤 | 관찰을 공유하는 연구자 | 독자를 가르치는 사람, 자기 자랑 |
| 길이 | 300-380단어 | 400+ (너무 김), 150- (너무 짧음) |
| 전략 노출 | 시장 구조 관찰만 | 신호 로직, 파라미터, 진입 조건 |

**톤 원칙 — 겸손한 연구자 관점:**
- 발견을 공유하되, 독자에게 결론을 강요하지 않는다
- "worth looking at", "turned out to be", "one thing that keeps coming up" → 부드럽게
- 독자가 틀렸다고 암시하는 표현 금지: "you don't have an edge", "you're wrong", "most traders make this mistake"
- 단정적 선언 대신 관찰: "the data suggests" > "the data proves"
- 전략 자랑 금지: 수익률 숫자는 비교 맥락에서만 사용

**절대 쓰지 말 것:**
- "my strategy", "my system", "I use this in live trading"
- 구체적 진입 조건, SL/TP 파라미터
- 특정 전략 이름 (VR Squeeze, MA Cluster 등)
- 독자를 깎아내리는 표현: "you're just validating a bull market assumption", "you don't have an edge — you have a trend dependency"
- 과도한 자신감: "this is not a fluke", "the data says the opposite", "structural flaw"

**항상 쓸 것:**
- 데이터 출처: "X symbols · Y years · Binance USDT-M perpetuals"
- 메커니즘 설명 (왜 그런 결과가 나왔는지): 숫자만 나열하면 부실함
- 열린 결말 or 부드러운 제안: "worth checking", "one approach worth testing"
- **포스트 첫 줄에 타이틀** (LinkedIn 미리보기에 노출됨): 짧고 궁금증을 유발하는 문장
  - 예: "Win Rate Tells Half the Story", "The Long Bias That Took Six Years to Show Up"
  - 타이틀 다음 빈 줄 한 칸 → 본문 시작

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
- [ ] **보편적 교훈**: 전략 모르는 사람도 "맞아!" 할 내용인가?
- [ ] **놀라운 숫자**: 숫자 자체가 임팩트가 있는가?

### Step 3: 안전성 체크 (전략 노출 여부)
- [ ] 신호 로직 없음 (VR ratio, momentum, squeeze, compression 등 언급 금지)
- [ ] 구체적 파라미터 없음 (SL=1%, trail_activate 등 금지)
- [ ] "내 전략" 언급 없음
- [ ] 시장 구조 관찰로만 프레이밍 가능한가?
- [ ] **전략 유형 힌트 단어 없음** → 아래 금지어 리스트 참조

**전략 노출 위험 단어 (문맥 무관 금지):**

| 단어/표현 | 위험 이유 | 대체 표현 |
|-----------|-----------|-----------|
| breakout / 브레이크아웃 | 전략 유형 직접 노출 | "directional momentum", "trending move" |
| squeeze | VR Squeeze 직접 노출 | "low-volatility period", "compression" 금지 |
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
│   ├── post_001_short_desc.txt   ← LinkedIn 복붙용 plain text
│   └── post_001_short_desc.md    ← 연구 노트/데이터 테이블 보관용
├── post_002/
│   └── ...
└── post_003/
    └── ...
```

**txt 파일 규칙:**
- `**bold**` 사용 금지 → 그냥 텍스트로
- `---` 구분선 사용 금지
- 줄바꿈으로만 단락 구분
- 마지막에 hashtag 5개 이하

### Step 5: 커밋
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

차트 생성 후 PNG를 반드시 시각적으로 검증. 겹침이 자주 발생하는 상황과 대응책:

**1. 바 위에 숫자 레이블**
```python
# 바 높이가 작을 때 레이블이 다음 바와 겹침 → 충분한 y 여유 확보
ax.set_ylim(0, max_val * 1.20)   # 상단 20% 여유 (레이블 공간)

# 음수 바는 아래로 레이블
ypos = val + 0.05 if val >= 0 else val - 0.4   # 부호 따라 방향 분기
```

**2. 라인 차트 포인트 레이블**
```python
# 가까운 포인트끼리 겹침 → 교대로 위/아래 배치
offset = 0.15 if i % 2 == 0 else -0.30
ax.text(i, val + offset, f'{val:.2f}', ...)
```

**3. Annotation (화살표 + 텍스트 박스)**
```python
# xytext 좌표를 데이터 범위의 여유 영역으로 지정
# 데이터가 1.2~1.5 범위면 annotation은 1.6+ 또는 1.1- 배치
# bbox 반드시 사용 (배경 박스가 겹침 가독성 확보)
bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor=color, alpha=0.85)
```

**4. 서브플롯 간격**
```python
# hspace/wspace 부족이 타이틀-축 겹침 원인
gs = fig.add_gridspec(..., hspace=0.52, wspace=0.35,
                      top=0.87, bottom=0.09, left=0.07, right=0.97)
# top=0.87: 메인 타이틀 + 서브타이틀 공간
# bottom=0.09: footer 텍스트 공간
```

**5. x축 레이블 겹침**
```python
# 항목이 5개 이상이면 rotation 적용
ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha='right')
# 또는 줄바꿈 사용: "Config A\n(Balanced)"
```

**6. 차트 생성 후 검증 체크리스트**
- [ ] 각 패널 타이틀이 데이터와 겹치지 않는가?
- [ ] 바/포인트 레이블이 인접 원소와 겹치지 않는가?
- [ ] annotation bbox가 데이터 라인을 가리지 않는가?
- [ ] footer 텍스트가 x축 레이블과 겹치지 않는가? (`bottom` 충분히 확보)
- [ ] 범례가 데이터 영역과 겹치지 않는가? (`loc='best'` 대신 명시적 위치)

---

## 5. 발행된 포스트 목록

| # | 제목 | 소재 | 발행일 |
|---|------|------|--------|
| 001 | Why Your 3-Year Backtest is Lying | r5 crash validation | 2026-03-08 |
| 002 | WR 70% vs R:R 2:1 — Which Survives? | r8 stress test | 2026-03-08 |
| 003 | The Long Bias That Stopped Working | r1 direction asymmetry | 2026-03-08 |
| 004 | More Symbols, More Alpha? | universe expansion | 2026-03-08 |

---

## 6. 미래 포스트 타이틀 큐

옛날 리서치 문서에서 추출. 전략 노출 없이 시장 구조 관찰만 사용 가능한 소재.

| 우선순위 | # | 타이틀 아이디어 | 핵심 숫자 | 리서치 출처 |
|---------|---|----------------|-----------|------------|
| 🔥 | 005 | The Slot Count Sweet Spot — More Parallelism Isn't Always Better | 10슬롯 Calmar 최대, 12+는 역효과 | r39 (20260302) |
| 🔥 | 006 | Your Trailing Stop Is Probably Already Good Enough | 99.9% WR after TP1, 18 variants ±2% from baseline | r12 trailing (20260221) |
| 🔥 | 007 | I Tried Dynamic Parameters. Static Won. | regime-adaptive swept 224 variants → +0.3% PF 개선, 복잡성 대비 가치 없음 | r42 regime (20260303) |
| ⭐ | 008 | The Leverage Level That Maximizes Risk-Adjusted Return | LN=2.0 Calmar 최고, LN=5.0 DD=99.9% | r52 RCB analysis (20260308) |
| ⭐ | 009 | Why Monte Carlo Will Break Your Overconfidence | P50=$390K but P50 DD=73.6%, 1yr compound 10,000 paths | r46 MC (20260307) |
| ⭐ | 010 | The Exit You Never Optimize (But Should) | exit grid 64 variants, winner +8% Calmar vs baseline | r7 exit grid (20260220) |
| ☆ | 011 | Confidence Filtering: When Fewer Trades Is Better | conf≥3 vs conf≥2: fewer trades, higher EV, lower DD | r5/r6 (20260220) |
| ☆ | 012 | How 3 Years of Bear Market Changed Everything | 2022-2023 regime shift, long EV halved, short WR plateau | r39 yearly breakdown (20260302) |
| ☆ | 013 | The Calmar Ratio Is the Only Metric That Matters for Sizing | AnnRet/MaxDD vs Sharpe — why Calmar aligns with position sizing reality | r39 ranking (20260302) |
| ☆ | 014 | Trail Factor AB Test: When "Better" Looks Identical | 18 trail variants, max diff = 2% return, lesson about diminishing returns | r12/trail_factor (20260223) |

---

## 6. 소재 우선순위 큐

| 우선순위 | 제목 아이디어 | 리서치 출처 | 핵심 숫자 |
|------|------|------|------|
| 🔥 | WR 70% vs R:R 2:1 | r8 (20260220) | WR 6pp↓ = 파산, R:R 21pp↓도 생존 |
| 🔥 | Long/Short 역전 | r1/r2 (20260221) | 2020-22 롱 +25% EV → 2024 역전 |
| ⭐ | 심볼 수와 수익 | universe_expansion (20260219) | 20→30심볼 수익 1.5배, PF도 상승 |
| ⭐ | 레버리지 스윗스팟 | r52 today analysis | LN 2.0 Calmar 최고, 5.0은 파산 |
| ☆ | 슬롯 최적화 | r39 (20260302) | 10슬롯 Calmar 최대, 12+는 역효과 |
| ☆ | 백테스트 비용 민감도 | r8 stress test B | 0.06% 슬리피지도 PF 5% 감소 |
