# Edge Decay Research — Complete Documentation Index

## 📚 Documentation Files

### 1. **EDGE_DECAY_QUICK_START.py** ⭐ START HERE
- **Purpose**: Minimal working example (~180 lines)
- **What it does**: Data loading → Trade collection → Portfolio sim → Yearly decomposition → Edge decay analysis
- **How to use**: `python EDGE_DECAY_QUICK_START.py`
- **Output**: Yearly PF/WR/Calmar breakdown + decay metrics

### 2. **EDGE_DECAY_RESEARCH_GUIDE.md** 📖 DETAILED REFERENCE
- **Purpose**: Complete API reference with code patterns
- **Sections**:
  - R86 Production Parity pattern
  - trade_sim.py functions (collect_trades, sim_directional_trailing, etc.)
  - signals.py VR computation and rolling percentiles
  - config.py conviction parameters
  - 5 reusable code patterns
  - Bootstrap CI methodology
- **Length**: 355 lines

### 3. **EDGE_DECAY_API_REFERENCE.txt** 🔧 QUICK LOOKUP
- **Purpose**: Structured API reference for all 4 files
- **Format**: ASCII art diagrams + bullet points
- **Includes**:
  - Function signatures and return types
  - Data flow diagram
  - Critical gotchas
  - Key parameters table
- **Length**: 187 lines

---

## 🎯 Quick Navigation by Task

### **I need to run an edge decay study NOW**
→ Use `EDGE_DECAY_QUICK_START.py`
```bash
python EDGE_DECAY_QUICK_START.py
```
Then modify to analyze specific aspects (by exit type, by conviction score, etc.)

### **I need to understand how to load data**
→ Section 1 of EDGE_DECAY_RESEARCH_GUIDE.md + load_all_data() in QUICK_START.py

### **I need to know what collect_trades() returns**
→ EDGE_DECAY_API_REFERENCE.txt "FUNCTION A: collect_trades_trailing()"
- Returns: `[(sym, entry_bar_15m, exit_bar_15m, pnl_fraction), ...]`

### **I need to understand trade simulation**
→ EDGE_DECAY_API_REFERENCE.txt "FUNCTION B/C: sim_directional_trailing / sim_straddle_trailing()"
- Shows exit priority: SL → TRAIL → TP → MAXHOLD

### **I need to compute metrics**
→ EDGE_DECAY_API_REFERENCE.txt "FUNCTION E: compute_metrics()"
- Returns 14 keys: N, PF, WR, RR, Calmar, MaxDD, AnnRet, etc.

### **I need to decompose by year**
→ yearly_breakdown() in QUICK_START.py + yearly_breakdown pattern in GUIDE

### **I need conviction parameters**
→ Section 4 of EDGE_DECAY_RESEARCH_GUIDE.md + config.py reference

### **I need to understand VR computation**
→ Section 3 of EDGE_DECAY_RESEARCH_GUIDE.md (compute_vr formula)

---

## 🔑 Key Concepts

### **Data Loading Pattern** (lines 1-30 of QUICK_START.py)
```
cache_extended/*.npz 
  → load by symbol 
  → resample via build_resampled() 
  → precompute VR/momentum IN-PLACE 
  → estimate data_years
```

### **Trade Collection Pattern** (lines 50-70 of QUICK_START.py)
```
collect_trades_trailing(cfg, data_15m, data_1h, data_4h, "D_mom8"|"S_d8")
  ↓
[(sym, entry_15m, exit_15m, pnl_frac), ...]
```

### **Portfolio Simulation Pattern** (lines 80-90 of QUICK_START.py)
```
combined = vr_trades + strd_trades
portfolio_sim(combined, max_concurrent=10, straddle_syms=frozenset)
  ↓
pnl array (N trades accepted after slot filtering)
```

### **Yearly Decomposition Pattern** (lines 100-150 of QUICK_START.py)
```
For each year:
  • bars_per_year = (max_bar - min_bar) / data_years
  • Filter: yr_start ≤ entry_bar < yr_end
  • Run: portfolio_sim() + compute_metrics() on subset
  • Store: results[year] = metrics dict
```

### **Edge Decay Analysis Pattern** (lines 155-180 of QUICK_START.py)
```
Compare yearly PF/WR/Calmar:
  • 2020: PF=3.25, WR=48.2%
  • 2025: PF=1.52, WR=39.8%
  • Decay: -53% PF, -17% WR over 5 years
  • Annual rate: ~8-9% exponential decay
```

---

## 📊 Understanding Edge Decay Metrics

| Metric | Good Range | Interpretation |
|--------|-----------|---|
| **PF** | ≥ 1.5 | Profit factor (sum wins / abs sum losses) |
| **WR** | 40-50% | Win rate % |
| **RR** | 2-4 | Reward:Risk (avg_win / avg_loss) |
| **Calmar** | ≥ 0.5 | Return / MaxDD (risk-adjusted) |
| **MaxDD** | < 50% | Maximum drawdown |
| **AnnRet** | 20-40% | Annualized return % |

**Decay = year-over-year PF/WR decline**

---

## 🛠️ How to Extend QUICK_START.py

### **Analyze by Exit Type** (requires sim function modification)
```python
def analyze_by_exit_type(trades):
    """Group trades by exit reason: SL vs TRAIL vs TP vs MAXHOLD"""
    # Requires extending sim_directional_trailing to return exit codes
    # Then filter trades and yearly breakdown per exit type
```

### **Analyze by Conviction Score**
```python
def analyze_by_conviction(trades, cfg):
    """Group trades by CW-Tier conviction score (0-4)"""
    # Requires computing conviction score per entry
    # Then filter trades and analyze PF decay per score
```

### **Analyze by Timeframe** (easy!)
```python
def analyze_by_tf(cfg):
    """Separate 15m/30m/1h/4h edge decay"""
    vr_15m = collect_trades_trailing(..., tf_group=[TF_15m])
    vr_1h = collect_trades_trailing(..., tf_group=[TF_1h])
    # Then yearly breakdown per TF
```

### **Bootstrap Confidence Intervals** (already in QUICK_START.py)
```python
boot = bootstrap_pf(pnl, n_boot=5000)
print(f"PF [{boot['pf_ci_lower']:.3f}, {boot['pf_ci_upper']:.3f}]")
print(f"P(profitable) = {boot['P_PF_ge_1']:.1%}")
```

---

## 🚀 Typical Edge Decay Study Workflow

```
1. Run QUICK_START.py baseline
   ↓ (Full 2020-2025: PF=X, WR=Y%, Calmar=Z)

2. Modify yearly_breakdown() to also report by exit type
   ↓ (Which exit type is decaying fastest? SL, TRAIL, TP?)

3. Decompose by conviction score
   ↓ (Do high-conviction entries decay slower?)

4. Analyze by timeframe
   ↓ (Is 15m decaying faster than 1h?)

5. Bootstrap analysis
   ↓ (Are yearly declines statistically significant?)

6. Conclusion report
   → "Edge decays ~8% annually; mostly from SL shortening"
   → "High-conviction entries decay -2% vs -10% low-conviction"
   → "30m TF most resilient; 4h most decay-prone"
```

---

## 🔐 Critical Things to Remember

### **Don't Hardcode Config**
```python
# ❌ WRONG
cfg_dict = {"sp": 8, "lp": 56, "d_vr": 0.50, ...}

# ✅ RIGHT
cfg = VolStraddleConfig()
# (reads from .env or defaults)
```

### **Trailing Stops Have Priority**
```
Exit order: SL (intra-bar) → TRAIL (if active) → TP → MAXHOLD
SL always fires first (server-side)
```

### **Straddle = 2 Slots**
```python
# Each straddle entry opens LONG + SHORT simultaneously
# Must pass straddle_syms to portfolio_sim for correct slot accounting
portfolio_sim(combined, 10, straddle_syms=frozenset)
```

### **Yearly Decomposition ≠ Calendar Years**
```python
# bars_per_year estimated from data
# Years keyed 2020, 2021, etc. (not actual calendar)
# Assumes data starts ~2020
```

### **One-Time Load Only**
```python
# Load data ONCE at startup
# All analyses use same precomputed VR/momentum arrays
# This is fast (seconds) and memory-efficient
```

---

## 📋 File Locations in Quantbox

```
quantbox/
├── quantbox/backtest/trade_sim.py (1,184 lines)
│   ├─ collect_trades_trailing()
│   ├─ sim_directional_trailing()
│   ├─ sim_straddle_trailing()
│   ├─ portfolio_sim()
│   ├─ compute_metrics()
│   └─ bootstrap_pf()
│
├── quantbox/strategy/vol_straddle/
│   ├─ signals.py (412 lines)
│   │   ├─ compute_vr()
│   │   ├─ compute_momentum()
│   │   ├─ compute_squeeze_dur()
│   │   └─ make_direction_momentum()
│   │
│   └─ config.py (446 lines)
│       ├─ VolStraddleConfig dataclass
│       ├─ VR parameters
│       ├─ Conviction (CW-Tier) parameters
│       └─ Position sizing & risk management
│
├── research/active/r86_r90/
│   └─ r86_production_parity.py (canonical pattern for backtests)
│
└── data/cache_extended/
    └─ *_15m_2190d_*.npz (base data files)
```

---

## ✅ Checklist: Before Running Your Script

- [ ] VolStraddleConfig loads from .env (check VS_* env vars)
- [ ] cache_extended/*.npz files exist and readable
- [ ] Symbol whitelist not empty
- [ ] data_years calculated correctly (should be 5-6 years)
- [ ] collected trades > 0 (at least 100+)
- [ ] pnl array length matches accepted trades (portfolio_sim filtering)
- [ ] yearly results have entries for all years in data
- [ ] bootstrap CI lower > 0 (profitable backtest)

---

## 🤝 Questions? Gotchas?

### **collect_trades() returns ONLY (sym, e, x, pnl)**
To add exit type metadata (SL/TRAIL/TP/MAXHOLD), you need to modify sim functions to return an additional array.

### **Why does VR have NaNs?**
VR warmup period: first (long_period - 1) bars are NaN (need enough data for ATR).

### **Why is yearly breakdown important?**
Edge decay manifests as year-over-year PF/WR decline. This is the core metric for your research.

### **Can I use different symbols?**
Yes! Any symbol in cache_extended can be included. Whitelist controls which symbols to use.

### **Can I backtest without straddles?**
Yes! Just collect VR trades only, skip straddle collection.

---

## 📖 File Sizes & Complexity

| File | Lines | Purpose | Complexity |
|------|-------|---------|-----------|
| QUICK_START.py | 180 | Minimal working example | Low |
| RESEARCH_GUIDE.md | 355 | Detailed reference | Medium |
| API_REFERENCE.txt | 187 | Quick lookup | Low |
| trade_sim.py | 1,184 | Core engine | High |
| signals.py | 412 | VR/momentum | Medium |
| config.py | 446 | Configuration | Low |

**Recommended reading order**:
1. QUICK_START.py (5 min)
2. QUICK_START.py output (2 min)
3. yearly_breakdown() in QUICK_START.py (5 min)
4. EDGE_DECAY_API_REFERENCE.txt (10 min)
5. EDGE_DECAY_RESEARCH_GUIDE.md (20 min)
6. trade_sim.py source (as needed)

---

## 🎓 Next Steps for Your Edge Decay Study

After running QUICK_START.py:

1. **Understand the decay**: Is it uniform or accelerating?
2. **Find the root cause**: Is it SL shortening? WR declining? RR compressing?
3. **Propose solutions**: Conviction filtering? Regime detection? Parameter adaptation?
4. **Test countermeasures**: Do higher conviction entries hold up better?
5. **Estimate actionable edge**: What's profitable after decay adjustment?

Good luck! 🚀

