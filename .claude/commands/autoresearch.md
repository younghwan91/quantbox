---
description: "Autonomous research agent. Give it a goal + a single file to hack + a metric to optimize. It experiments forever until you stop it."
allowed-tools: Read, Grep, Glob, Bash, Agent, Write, Edit
model: opus
---

# autoresearch

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The idea: give an AI agent a small but real experiment setup and let it experiment autonomously. It modifies the code, runs the experiment, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and (hopefully) better results.

## How it works

The repo is deliberately kept to three things that matter:

- **prepare** — fixed infrastructure. Data loading, evaluation function, constants. Not modified.
- **train** — the single file the agent edits. Everything is fair game: parameters, logic, architecture, filters, new indicators, radical rewrites. **This file is edited and iterated on by the agent**.
- **program** — this file. Baseline instructions. **This file is edited and iterated on by the human**.

The metric and the evaluation function are fixed. Everything else is fair game.

**The user specifies the research goal: $ARGUMENTS**

## Interview (Socratic, one question at a time)

If `$ARGUMENTS` is specific enough to define goal + metric + constraints (e.g. "D_mom8 Calmar 최적화, PF>1.0 유지"), skip the interview and go straight to Setup.

Otherwise, run a short Socratic interview. Rules:
- **한 번에 한 질문.** 절대 여러 질문을 한꺼번에 하지 않는다.
- **코드베이스를 먼저 탐색한 뒤 질문한다.** 코드가 이미 알려주는 것을 사용자에게 묻지 않는다.
- **가장 모호한 차원을 타겟한다.** 매 질문이 가장 약한 부분을 공략한다.
- **3~5 라운드 안에 끝낸다.** deep-interview가 아니라 research kickoff다.

### Clarity dimensions (scored 0-1 after each answer)

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Goal | 40% | 뭘 개선/발견하려는지 한 문장으로 말할 수 있는가? |
| Metric | 30% | 성공/실패를 판단할 구체적 숫자가 있는가? |
| Constraints | 30% | 건드리면 안 되는 것, 유지할 최소 기준이 있는가? |

**Readiness gate**: `clarity = goal×0.4 + metric×0.3 + constraints×0.3`. Proceed when clarity ≥ 0.8 (ambiguity ≤ 20%).

### Interview flow

1. **Explore first.** Before asking anything, read the relevant parts of the codebase (config.py, signals.py, recent research scripts). Use this context to ask informed questions.

2. **Ask round 1** — target the weakest dimension. Show the score after the answer:
   ```
   Round 1 | Target: Goal | Clarity: 30%
   {question informed by codebase context}
   ```

3. **Score and repeat.** After each answer, rescore all dimensions. If clarity ≥ 80%, stop and proceed to Setup. Otherwise ask the next question targeting the new weakest dimension.

4. **Round 4+: Simplifier challenge.** If still unclear, ask: "가장 단순한 버전으로 줄이면 뭐가 남아?" — 불필요한 복잡성을 제거한다.

5. **Hard cap: 5 rounds.** 5라운드 후에도 clarity < 80%이면, 현재 이해도를 보여주고 부족한 부분을 명시한 뒤 진행할지 물어본다.

### Example

```
[explores: config.py has d_vr_threshold=0.50, signals.py has VR squeeze logic]

Round 1 | Target: Goal | Clarity: 0%
현재 D_mom8이 d_vr=0.50으로 돌고 있는데, 진입 타이밍을 바꾸고 싶은 건지
아니면 exit 로직(trail/SL)을 개선하고 싶은 건지, 아니면 둘 다?

> "진입이 너무 늦어서 타이밍을 앞당기고 싶어"
→ Goal: 0.9, Metric: 0.2, Constraints: 0.1 | Clarity: 42%

Round 2 | Target: Constraints | Clarity: 42%
타이밍을 앞당기면 트레이드 수가 줄어들 수 있는데,
최소 몇 개는 유지해야 돼? 그리고 MaxDD는 어디까지 허용?

> "N>500, MaxDD<40%"
→ Goal: 0.9, Metric: 0.2, Constraints: 0.9 | Clarity: 63%

Round 3 | Target: Metric | Clarity: 63%
성공 기준이 Calmar? PF? 아니면 연수익률?

> "Calmar"
→ Goal: 0.9, Metric: 0.9, Constraints: 0.9 | Clarity: 90% ✓

Ready. Proceeding to Setup.
```

## Setup

Work with the context to:

1. **Understand the goal.** What is being optimized? Read the relevant code. Examples:
   - "D_mom8 Calmar 최적화" → read config.py, signals.py, trade_sim.py
   - "새 RSI reversal 전략 프로토타입" → read signals.py, protocol.py
   - "포트폴리오 슬롯 배분 최적화" → read trade_sim.py portfolio_sim()
   - "MA Cluster 진입 필터 개선" → read config.py ma_cluster section, signals.py

2. **Create the branch**: `git checkout -b autoresearch/<tag> develop`

3. **Write the train file.** One single Python file that:
   - Contains everything the agent will modify (params, signal logic, filters, etc.)
   - Imports fixed infrastructure (data loading, evaluation) from the codebase
   - Prints results in a grep-friendly `key: value` format
   - Can be run with `uv run <file> > run.log 2>&1`

   The file must be self-contained enough that modifying it is the ONLY thing needed for each experiment. Put it outside `research/` to avoid the param guard hook (e.g. `autoresearch/train.py`).

4. **Initialize results.tsv** with a header row matching the metrics. Do not commit this file.

5. **Run baseline.** The very first run establishes the baseline.

6. **Confirm and go.**

## Experimentation

Each experiment runs a backtest or analysis. You launch it simply as: `uv run autoresearch/train.py > run.log 2>&1`

**What you CAN do:**
- Modify the train file — this is the only file you edit. Everything is fair game: parameters, signal logic, new indicators, new filters, radical architectural changes, import new libraries from the existing deps.

**What you CANNOT do:**
- Modify `quantbox/backtest/trade_sim.py`. It is read-only. It contains the fixed evaluation and simulation.
- Modify production code in `quantbox/`. Your experiments stay in the train file.
- Install new packages or add dependencies.
- Use synthetic/random data. Real Binance data only.
- **Reduce SL below production value.** Backtest executes SL at exact trigger price, but live server-side STOP orders fill via market order with tick rounding + slippage (observed: 1% SL → 1.23% actual on SKLUSDT). Tighter SL always looks better in backtest but the improvement is illusory. SL must stay >= production config value. Same applies to any parameter where backtest assumes perfect execution but live has slippage (e.g. trail_distance near the 0.5% Binance floor).

**The goal is simple: optimize the target metric.** Everything in the train file is fair game.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. An improvement of ~0 but much simpler code? Keep.

## Output format

The train file must print a summary block:

```
---
primary_metric: <value>
secondary_1:    <value>
secondary_2:    <value>
...
```

Extract the key metric: `grep "^primary_metric:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns depend on the research goal. Always include:
- commit hash (short, 7 chars)
- primary metric value
- status: `keep`, `discard`, or `crash`
- short description of what this experiment tried

## The experiment loop

LOOP FOREVER:

1. Look at the current state: results.tsv history, what's been tried, what worked.
2. Think of ONE experimental idea. Modify the train file.
3. `git commit -m "exp: <description>"`
4. `uv run autoresearch/train.py > run.log 2>&1`
5. `grep "^primary_metric:" run.log` — if empty, run crashed. `tail -n 50 run.log` for the traceback.
6. Record results in results.tsv.
7. If improved → keep the commit (new baseline).
8. If equal or worse → `git reset --hard HEAD~1`.
9. Go to 1.

**One change at a time.** Each experiment tests exactly ONE hypothesis.

**NEVER STOP.** Once the loop has begun, do NOT pause to ask the human. Do NOT ask "should I keep going?". The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the codebase for new angles, try combining previous near-misses, try more radical changes. Read papers. The loop runs until the human interrupts you, period.

**SILENT MODE.** Do not output text to the user between experiments. No play-by-play commentary. No "Exp 3: Calmar 4.5, discard." The user is not watching. All results go to `results.tsv` — that is the log. The only text output during the loop is the progress report every 10 experiments.

**Timeout**: If a run exceeds 15 minutes, kill it and treat as failure.

**Crashes**: If it's a typo or easy fix, fix and re-run. If the idea is fundamentally broken, skip it, log "crash", move on.

## Progress reports

Every 10 experiments, print a brief summary (this is the ONLY output during the loop):

```
=== Progress Report (after N experiments) ===
Baseline: X.XX → Current Best: Y.YY (+Z.Z%)
Kept: K / Discarded: D / Crashed: C
Key insight: <what pattern is emerging>
```

## When the human returns

Provide:
1. Final `results.tsv`
2. Summary: baseline vs best, what worked, what didn't
3. The final train file with best configuration
4. `git log --oneline` of kept experiments
5. Recommended changes to production code (if applicable)
