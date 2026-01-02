# Realistic Training Guide: Avoiding Paper Trading Overfitting

**Problem**: Paper trading assumes instant fills at mid-price. This is unrealistic and causes the agent to learn strategies that fail in live trading.

**Solution**: Train with slippage simulation to learn strategies that work with realistic execution constraints.

---

## The Overfitting Problem

### Paper Trading Assumptions (WRONG)
```python
# Current paper trading logic
if action == Action.BUY:
    entry_price = current_prob  # Instant fill at mid-price
    shares = size / entry_price
    # Position opened immediately
```

**What's wrong**:
- ✗ Zero latency (decision → fill is instant)
- ✗ No spread cost (buy at mid, not ask)
- ✗ No market impact (large orders don't move price)
- ✗ Perfect fills (100% fill rate, no partials)
- ✗ No adverse selection (fills always at expected price)

### What Actually Happens (Live Trading)
```
1. Agent decides BUY at prob=0.450 (mid-price)
2. Latency: 150ms for order to reach exchange
3. Spread: Must pay ask = 0.450 + 0.005 = 0.455
4. Market impact: $500 order moves price by +0.002
5. Price movement: During 150ms, price moves to 0.458
6. Adverse selection: Fast move → fill at 0.461
7. Actual entry: 0.461 (not 0.450)
8. Slippage: +0.011 (+2.4%)
```

**Result**: Strategy profitable on paper loses money live.

---

## Solution 1: Slippage Simulation During Training

### Enable Realistic Execution

Modify `run.py` to use `SlippageSimulator`:

```python
# Add to run.py imports
from helpers.slippage_simulator import SlippageSimulator, RealisticPaperTrading

# In TradingEngine.__init__
def __init__(self, strategy, trade_size=10.0, live_mode=False,
             realistic_execution=True):  # NEW FLAG

    # ... existing code ...

    # Slippage simulator (if realistic mode enabled)
    self.realistic_execution = realistic_execution
    self.slippage_sim = None

    if realistic_execution and not live_mode:
        print("✓ Realistic execution simulation enabled")
        print("  Training will include slippage, latency, and market impact")

        self.slippage_sim = SlippageSimulator(
            avg_spread_bps=15.0,     # 0.15% spread (calibrate from live data)
            avg_latency_ms=150.0,    # 150ms average latency
            impact_coefficient=0.02, # Price impact
            orderbook_depth=5000.0,  # Typical liquidity
        )
```

### Modify _execute_paper() to Use Simulator

```python
def _execute_paper(self, cid, action, state):
    """Execute paper trade with optional slippage simulation."""
    if action == Action.HOLD:
        return

    pos = self.positions.get(cid)
    if not pos:
        return

    price = state.prob
    trade_amount = self.trade_size * action.size_multiplier

    # REALISTIC EXECUTION MODE
    if self.slippage_sim:
        # Determine side
        side = "BUY" if action.is_buy else "SELL"

        # Build market state for simulator
        market_state = {
            "spread": state.spread,
            "volatility": state.realized_vol_5m,
            "momentum": state.returns_1m,  # Recent momentum
            "time_to_expiry": state.time_remaining,
        }

        # Simulate realistic fill
        result = self.slippage_sim.simulate_fill(
            side=side,
            expected_price=price,
            size=trade_amount,
            spread=market_state["spread"],
            volatility=market_state["volatility"],
            momentum=market_state["momentum"],
            time_to_expiry=market_state["time_to_expiry"],
        )

        # Handle fill result
        if not result.filled:
            # Order didn't fill (adverse selection)
            print(f"    ✗ Order rejected: {pos.asset} {side}")
            return

        # Use actual fill price (with slippage)
        actual_price = result.fill_price
        actual_size = result.fill_size

        # Log slippage
        if abs(result.slippage) > 0.005:  # >0.5%
            slippage_bps = result.slippage * 10000
            print(f"    💸 Slippage: {slippage_bps:+.1f} bps | "
                  f"Latency: {result.latency_ms:.0f}ms")

        if result.partial_fill:
            print(f"    ⚠ Partial fill: ${actual_size:.0f} / ${trade_amount:.0f}")

    else:
        # ORIGINAL PAPER TRADING (instant fills)
        actual_price = price
        actual_size = trade_amount

    # Rest of execution logic uses actual_price, actual_size
    # ... (existing position management code) ...
```

---

## Solution 2: Training with Realistic Constraints

### Phase 1: Train on Paper (Baseline)

```bash
# Standard paper trading - get baseline performance
python run.py rl --train --size 50
```

**Expected**: High ROI, unrealistic

### Phase 2: Train with Slippage Simulation

```bash
# Enable realistic execution
python run.py rl --train --size 50 --realistic
```

**Expected**: Lower ROI (20-50% degradation), but realistic

### Phase 3: Compare Performance

```python
# After training both models, compare:

# Paper trading model:
# - ROI: 170%
# - Win rate: 23%
# - Trades frequently (low cost assumed)

# Realistic model:
# - ROI: 85% (50% degradation)
# - Win rate: 22%
# - Trades less frequently (accounts for costs)
# - Better suited for live trading
```

---

## Solution 3: Calibrate Simulator from Live Data

After initial live trading, calibrate simulator:

```python
# Collect live fills
live_fills = [
    {
        "expected_price": 0.450,
        "fill_price": 0.461,
        "latency_ms": 187,
        "size": 50,
        "filled": True,
    },
    # ... more fills from live trading
]

# Calibrate simulator
slippage_sim.calibrate_from_live_data(live_fills)

# Now simulator uses ACTUAL measured slippage/latency
# Retrain model with calibrated simulator
```

---

## Realistic Training Parameters

### Conservative (Recommended)

```python
SlippageSimulator(
    avg_spread_bps=20.0,      # 0.20% spread (pessimistic)
    avg_latency_ms=200.0,     # 200ms latency (pessimistic)
    impact_coefficient=0.03,  # High impact (small market)
    orderbook_depth=3000.0,   # Low liquidity (conservative)
    adverse_selection_factor=0.2,  # 20% adverse selection risk
)
```

**Why conservative?** Better to train on worse conditions, succeed in better ones.

### After Live Calibration

```python
SlippageSimulator(
    avg_spread_bps=12.0,      # Measured from live data
    avg_latency_ms=145.0,     # Measured from live data
    impact_coefficient=0.015, # Calibrated from fills
    orderbook_depth=6000.0,   # Observed liquidity
    adverse_selection_factor=0.1,  # Measured rejection rate
)
```

---

## Expected Performance Degradation

Based on realistic execution constraints:

### Slippage Impact

```python
# Example trade:
# Paper:     Entry 0.450, Exit 0.500 → PnL = +0.050 (+11.1%)
# Realistic: Entry 0.461, Exit 0.494 → PnL = +0.033 (+7.2%)
#
# Degradation: 34% worse due to slippage on both entry and exit
```

### Latency Impact

```
Fast market move:
- Paper:     Sees move, enters instantly
- Realistic: Sees move, 150ms delay → price already moved
- Result:    Missed opportunity or worse entry
```

### Size Impact

```
Small orders ($10-50):   10-15% degradation
Medium orders ($100-250): 20-30% degradation
Large orders ($500+):     30-50% degradation
```

### Overall Degradation

**Conservative estimate**: 30-50% worse PnL vs paper trading

```
Paper trading results:  $50K PnL (2,500% ROI)
Realistic simulation:   $25K-35K PnL (1,250-1,750% ROI)
Expected live results:  $15K-30K PnL (750-1,500% ROI)
```

---

## Training Workflow: Paper → Realistic → Live

### Step 1: Initial Training (Paper)

```bash
python run.py rl --train --size 50
# Train for 10+ hours, save model
```

**Metrics**:
- PnL: Track total profit
- Win rate: Percentage profitable trades
- Sharpe ratio: Risk-adjusted returns

### Step 2: Retrain with Slippage

```bash
python run.py rl --train --size 50 --realistic --slippage-factor 1.5
```

**Metrics**:
- PnL degradation: Compare to paper
- Slippage cost: Average per trade
- Fill rate: Percentage orders filled

**Goal**: Still profitable after realistic costs

### Step 3: Analyze Differences

```python
# Load both models, compare on same test data:

paper_actions = []
realistic_actions = []

for state in test_states:
    paper_actions.append(paper_model.act(state))
    realistic_actions.append(realistic_model.act(state))

# Differences:
# - Paper model trades more (assumes zero cost)
# - Realistic model is selective (accounts for costs)
# - Realistic model avoids volatile periods (high slippage risk)
```

### Step 4: Live Testing (Tiny Size)

```bash
python run.py rl --live --private-key $KEY --size 5 --load realistic_model
```

**Monitor**:
- Actual slippage vs simulated
- Fill rate vs simulated
- PnL vs realistic simulation

### Step 5: Calibrate and Iterate

```python
# After 100+ live trades, calibrate simulator
live_data = load_live_fills("live_fills.csv")
slippage_sim.calibrate_from_live_data(live_data)

# Retrain with calibrated simulator
python run.py rl --train --size 50 --realistic --calibrated
```

---

## Key Metrics to Track

### During Training

1. **Ideal PnL** (paper trading, no slippage)
2. **Realistic PnL** (with slippage simulation)
3. **Degradation %** = (Ideal - Realistic) / Ideal

### During Live Testing

1. **Expected PnL** (from realistic simulation)
2. **Actual PnL** (from live fills)
3. **Calibration error** = Actual - Expected

### Calibration Quality

If simulation is accurate:
- Actual PnL ≈ Simulated PnL (within ±20%)
- Actual slippage ≈ Simulated slippage (within ±5 bps)
- Fill rate ≈ Simulated fill rate (within ±10%)

---

## Red Flags (Overfitting to Paper Trading)

🚩 **High paper PnL, negative live PnL**
- Likely: Agent learned to exploit zero-latency assumption
- Fix: Increase latency simulation, add momentum-based adverse selection

🚩 **High trade frequency**
- Likely: Not accounting for transaction costs
- Fix: Add explicit cost per trade, increase spread simulation

🚩 **Large position sizes**
- Likely: Not accounting for market impact
- Fix: Increase impact coefficient, reduce orderbook depth

🚩 **Trading during high volatility**
- Likely: Assuming fills at expected price during fast moves
- Fix: Increase adverse selection factor during volatile periods

🚩 **Simulation PnL >> Live PnL**
- Likely: Simulator too optimistic
- Fix: Use conservative parameters, calibrate from live data

---

## Hardware Considerations for Realistic Training

### Training Time Impact

**Without slippage sim**:
- 256 experiences → 1 PPO update
- ~30 seconds per update
- 10 hours = 1200 updates

**With slippage sim**:
- Extra compute per fill: ~0.1ms (negligible)
- Training time: Same (slippage is simple math)
- Memory: Same (no extra storage)

**Verdict**: Slippage simulation adds <1% overhead

### Cloud Training Not Needed

For this model size:
- M1 Mac: Perfect for training
- Cloud GPU: Overkill
- Cloud CPU: Slower than M1

**Recommendation**:
- Train on Mac with slippage simulation
- Deploy to cloud for live trading (latency benefit)

---

## Command Reference

```bash
# Paper trading (unrealistic, baseline)
python run.py rl --train --size 50

# Realistic training (recommended)
python run.py rl --train --size 50 --realistic

# Realistic training with custom parameters
python run.py rl --train --size 50 --realistic \
    --spread-bps 20 \
    --latency-ms 200 \
    --impact 0.03

# Train with live-calibrated simulator
python run.py rl --train --size 50 --realistic --calibrated calibration.json

# Compare paper vs realistic models
python analyze_models.py --paper paper_model --realistic realistic_model

# Live test with realistic model
python run.py rl --live --private-key $KEY --size 5 --load realistic_model
```

---

## Summary

**Problem**: Paper trading overfits to unrealistic assumptions

**Solutions**:
1. ✅ **Slippage simulation** - Model realistic execution during training
2. ✅ **Conservative parameters** - Train on worse conditions than expected
3. ✅ **Live calibration** - Update simulator from actual data
4. ✅ **Iterative refinement** - Paper → Realistic → Live → Calibrate → Repeat

**Expected outcome**:
- 30-50% PnL degradation from paper to live
- But: Model actually works in live trading
- Better: Profitable at small sizes, scalable strategy

**Hardware**: M1 Mac is perfect for this - no cloud needed for training.
