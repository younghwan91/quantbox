# R44 — High-Volatility Complement Strategy Research

**Commit**: `42034c4` (base), `TBD` (this report)  
**Date**: 2026-03-03  
**Status**: ❌ **NEGATIVE RESULT** — No viable complement found

---

## Objective

The VR Squeeze strategy (v13.1) only trades when VR < 0.50 (volatility compression).
When all 50 symbols have VR > 0.6 (as of 2026-03-03), the bot is completely idle.

**Goal**: Find a complementary strategy that trades specifically during high-volatility
periods, so the combined system generates alpha in more market conditions.

## Methodology

- **Data**: 50 symbols, 6yr, 15m/1h/4h candles (same dataset as v13.1)
- **Infrastructure**: R38 canonical sim (collect_trades, portfolio_sim, compute_metrics)
- **Baseline**: v13.1 VR Squeeze (N=2265, PF=1.280, EV=+32.1bp, Calmar=5.8)
- **10 slots**: Same as production
- **Costs**: 0.05% per trade (COST_DIRECTIONAL)

## R44: Mean Reversion Strategies (26 configs)

| Vector | Hypothesis | Configs | Best PF | Result |
|--------|-----------|---------|---------|--------|
| V1: VR Expansion MR | High VR → fade momentum | 7 | 0.820 | ❌ |
| V2: RSI Extreme MR | RSI < 20 or > 80 → fade | 5 | 0.874 | ❌ |
| V3: Bollinger Band MR | Price beyond bands → fade | 5 | 0.876 | ❌ |
| V4: Momentum Exhaustion | Large move + volume → reversal | 5 | 0.804 | ❌ |
| V5: VR Regime Switch | VR drops from high→low → follow | 4 | 0 trades | ❌ |

**All 26 mean reversion configs lost money.** Best PF was 0.876 (V3: BB(20,2.0)).
Mean reversion does not work in crypto at 15m-4h timeframes.

## R44b: Momentum/Trend-Following Strategies (135+ configs)

| Vector | Hypothesis | Configs | Best PF | Result |
|--------|-----------|---------|---------|--------|
| V6: VR + Momentum Following | High VR + follow momentum | 60 | 0.959 | ❌ |
| V7: Momentum Breakout | Strong move → ride continuation | 20 | 0.928 | ❌ |
| V8: Multi-TF Alignment | 15m + 4h same direction | 27 | 0.945 | ❌ |
| V9: VR Expansion Onset | VR just crossed threshold | 27 | 0.959 | ❌ |
| V10: Momentum + RSI | Momentum + RSI confirmation | 27+ | 0.922 | ❌ |

**All trend-following configs also lost money.** Best PF was 0.959 (V9: VR>0.7 lb16).

## Key Finding

**Neither mean reversion NOR trend-following produces an edge at high-vol without
the specific VR squeeze microstructure.** Across 161+ parameter configs:

- Mean reversion PF range: **0.568 – 0.876**
- Trend-following PF range: **0.581 – 0.959**

The VR squeeze edge comes specifically from the compression→expansion regime
transition, not from momentum direction or VR level alone.

## Why VR Squeeze Works and These Don't

The VR squeeze strategy isn't "trading when VR is low" — it trades the
**regime transition** from compression to expansion. This captures:

1. **Coiled spring effect**: Energy stored during compression releases
2. **Breakout directionality**: momentum(8) gives reliable direction after squeeze
3. **Low false positive rate**: squeeze duration filter (≥4 bars) eliminates noise

Simple threshold-based entries (VR > X, momentum > Y, RSI extremes, BB bands)
lack this structural edge. They fire too frequently (~100K-450K raw trades)
with no selective filter, resulting in PF < 1.0 everywhere.

## Practical Implication

The current VR squeeze idle periods are a feature, not a bug:

- The bot is designed to be patient and selective
- Multi-day gaps between trades are normal (backtest avg ~1 trade/day)
- Current market (all symbols VR > 0.6) = the bot correctly stays out
- **No simple complementary strategy can fill these gaps profitably**

## Possible Future Directions (Not Tested)

1. **Order book / funding rate signals** — Microstructure data not available in OHLCV
2. **Cross-asset correlation** — BTC dominance, sector rotation
3. **Adaptive VR threshold** — Dynamic threshold based on market regime
4. **Different asset class** — Spot, options, or DeFi strategies
5. **Longer timeframe** — Daily/weekly trend-following may have edge

## Runtime

- R44 (26 MR configs): 25.1s
- R44b (135+ TF configs): ~3 min (interrupted after V10 partial, all V6-V9 complete)
