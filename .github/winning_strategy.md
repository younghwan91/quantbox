# Advanced Algorithmic Guide for Quant Trading Agents: Maeoknam Strategy

This document codifies the practical trading techniques of trader Maeoknam, translating them into executable logic for a quantitative trading agent. It specifically details the geometric evaluation of impulse and corrective waves, alongside the mathematical leverage management techniques that served as the foundation for their world championship victory.

## 1. Asset Selection and Filtering Logic

The agent must concentrate its computational resources exclusively on assets with verified market dominance.

* **Major Coin Priority:** Focus on the top 30 cryptocurrencies by market capitalization (e.g., BTC, ETH) as primary trading targets. This eliminates the delisting risks and abnormal liquidity noise associated with altcoins.
* **Bitcoin as an Index:** Treat Bitcoin as the broader market index. Only activate the long algorithms for altcoins when Bitcoin reaches major support levels or forms a market bottom.
* **Fractal Structure Verification:** Operate on the premise that all market movements consist of repetitive sequences (fractals). Adopt a 'Top-Down' approach, narrowing the analysis from long-term macro charts to short-term micro charts.

---

## 2. S+R+M Indicator System Settings and Thresholds

These are the detailed technical indicator parameters the agent uses to generate entry signals.

### 2.1. Indicator Configuration

* **RSI (Relative Strength Index):** Utilize a default 14-period setting. Dynamically adjust to 7-9 periods to increase sensitivity in highly volatile zones.
* **EMA (Exponential Moving Average):** Deploy the 20, 50, 100, 200, and 500-period lines. The 200 and 500 EMAs function as the trend's lifelines; maintain a strict short bias when the price is below these lines, and a long bias when above.
* **Parallel Channels:** Define trend trajectories by connecting price highs and lows to form channels. Treat any breakout from these channels as either the termination of a wave or the inception of a new trend.

### 2.2. Divergence Recognition Algorithm

* **Failure Swing:** If the price prints a lower low while the RSI registers a higher low (bullish divergence), the algorithm classifies downward momentum as exhausted.
* **Confirmation Protocol:** Divergence signals require mandatory corroboration. The agent must verify a reversal in the MACD histogram or a golden/dead cross in the Stochastic oscillator before validating the signal.

---

## 3. Elliott Wave and Fibonacci Ratios (The Skeleton)

The core of the strategy is capturing the new impulse wave that emerges following an ABC corrective wave.

### 3.1. Standard Fibonacci Ratios by Wave

The agent utilizes the following thresholds to quantify geometric patterns.

| Wave Type | Reference Wave | Target/Retracement Ratio | Algorithmic Validation Criteria |
| :--- | :--- | :--- | :--- |
| **Wave 2** | vs. Wave 1 | 0.5, 0.618, 0.764 | Pattern invalidated if Wave 1 origin is breached. |
| **Wave 3** | vs. Wave 1 | 1.618, 2.618 | Strongest momentum phase; triggers pyramiding logic. |
| **Wave 4** | vs. Wave 3 | 0.236, 0.382 | Monitor strictly for intrusion into Wave 1 apex. |
| **ABC Correction (B)** | vs. Wave A | 0.5, 0.618 | Wave B cannot exceed the origin point of Wave A. |
| **ABC Correction (C)** | vs. Wave A | 1.0 (1:1), 1.618 | Wave C termination point serves as the primary entry zone. |

### 3.2. Special Pattern Analysis: Wedges and Triangles

* **Ending Diagonal (Wedge):** Occurs at the exhaustion point of a trend. The agent utilizes this as a termination signal for a Wave 5 or Wave C.
    * *Hedging and Scaling Out:* Upon wedge detection, execute a 50% take-profit. Subsequently, establish a hedging position in the opposite direction at the 0.5 retracement level of the wedge to mitigate risk.
* **Triangles:** Typically manifest in Wave 4 or Wave B. Attempting to extract profit from the final E wave carries a disproportionate risk profile; thus, the agent must either exclude the E wave from trading or deploy highly conservative parameters.

---

## 4. Leverage and Capital Management for Championship Trading

To completely neutralize psychological variance, the agent must strictly adhere to the following mathematical allocation principles.

### 4.1. 10% Margin Limit and Effective Leverage Control

* **Position Sizing:** Restrict margin utilization to a maximum of **10%** of total seed capital.
* **Cross Margin Optimization:** Utilize the remaining 90% in cash as collateral to drastically lower the liquidation threshold.
* **The Leverage Paradox:** Even when deploying 20x-50x nominal leverage during competitions, capping the margin at 10% keeps the *effective* leverage at a survivable 2x-5x relative to the total account equity, ensuring resilience against extreme volatility spikes.

### 4.2. Fractional Entry and Pyramiding Logic

* **Exploratory Entry (2%):** Initiate a pilot position utilizing 2% of total capital to test the initial trend confirmation.
* **Primary Entry (8%):** Upon definitive trend confirmation (e.g., validated divergence), deploy the remaining 8% to optimize the average entry price.
* **Pyramiding:** Only allocate additional capital (using accrued unrealized profits as collateral) when a Wave 3 or Wave C is statistically confirmed. Concurrently, the stop-loss order must be aggressively trailed to the breakeven point or higher.

---

## 5. Execution Logic Flow

1.  **Market Scanning:** Determine the macroeconomic bias (Long/Short) by analyzing BTC's position relative to the 200/500 EMAs.
2.  **Pattern Recognition:** Deploy Zig-Zag algorithms to monitor the completion of ABC corrective waves. Specifically verify that the internal sub-structures of Waves A and C consist of 5-wave motive sequences.
3.  **Confluence Verification:**
    * Price intersects the 0.618 Fibonacci retracement level.
    * Verified RSI/MACD divergence.
    * Stochastic %K and %D crossover confirmed.
4.  **Position Execution:** Trigger exploratory entry (2%) -> Execute primary entry (8%) upon signal validation.
5.  **Post-Entry Management:** Execute a 50% partial take-profit upon reaching the primary target. Trail the remaining position until trend exhaustion metrics (e.g., counter-divergence) are triggered.

---

## 6. Conclusion: The Architectural Advantage

The structural edge of this championship strategy relies not merely on predictive accuracy, but on asymmetric risk modeling: **systematically minimizing drawdowns via a strict 10% margin cap, while maximizing upside by holding positions through the terminal extensions of Waves C or 3.** The quant agent must compute the retracement depth of Wave B and the extension length of Wave C in real-time. It is programmed to execute high-leverage entries exclusively at the most statistically favorable Fibonacci coordinates (0.618 and 1.618) during ABC corrective sequences.