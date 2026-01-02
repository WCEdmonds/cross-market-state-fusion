# Real Money Implementation Plan

## Executive Summary

This codebase is a **paper trading** RL system that achieved ~$50K PnL (2,500% ROI) in Phase 5 testing. It exploits information lag between Binance futures (fast market) and Polymarket prediction markets (slow market) on 15-minute binary crypto options.

**Critical Reality Check**: Paper trading assumes instant fills at mid-price. Real trading faces latency, slippage, and market impact. The README estimates **20-50% performance degradation** when going live.

**Latency is the #1 priority** - the entire edge depends on acting on Binance information before Polymarket prices adjust (seconds matter).

---

## Current State Analysis

### What Works (Paper Trading)
- ✅ Multi-source data fusion: Binance futures + Polymarket orderbook
- ✅ Real-time PPO training with MLX (Apple Silicon)
- ✅ Temporal architecture captures momentum patterns
- ✅ 18-dimensional state space with order flow signals
- ✅ WebSocket streams for live data (Binance + Polymarket CLOB)
- ✅ Share-based PnL accounting (Phase 4 breakthrough)

### What's Missing (Critical for Live Trading)
- ❌ **No order execution layer** - currently just prints trade signals
- ❌ **No Polymarket CLOB API integration** - can't place actual orders
- ❌ **No slippage modeling** - assumes instant fills at mid
- ❌ **No latency compensation** - no measurement or mitigation
- ❌ **No risk management** - no position limits, drawdown stops, or kill switches
- ❌ **No order state tracking** - no handling of partial fills, rejections, cancellations
- ❌ **No real-money validation** - paper results may not translate

---

## Latency Analysis: The Critical Path

The strategy's edge is **cross-market arbitrage via information lag**. Every millisecond counts.

### Current Latency Budget (Estimated)

```
Event: Binance futures price moves
├─ Binance WSS → Local:           10-50ms   (WebSocket, good)
├─ State calculation:               1-5ms   (numpy ops, fast)
├─ RL model inference:              2-10ms  (MLX on Apple Silicon, fast)
├─ Decision made:                   [TOTAL: ~20-65ms]
│
├─ [MISSING] Order placement:       ???
├─ [MISSING] Polymarket CLOB API:   50-200ms (network RTT + processing)
├─ [MISSING] Order matching:        10-500ms (depends on liquidity)
└─ [MISSING] Fill confirmation:     50-200ms (network RTT)

TOTAL END-TO-END: Currently ~20-65ms (signal only)
                  With execution: ~150-1000ms (estimated)
```

### Latency Bottlenecks (Ranked by Impact)

1. **Polymarket CLOB API roundtrip** (50-200ms)
   - Network latency to Polygon RPC
   - Order signature computation (EIP-712)
   - CLOB server processing
   - **Mitigation**: Co-location, keep-alive connections, pre-signed order templates

2. **Order matching & fills** (10-500ms)
   - Low liquidity = slow fills or multiple partial fills
   - Large orders walk the book (price slippage)
   - **Mitigation**: Limit orders at favorable prices, iceberg orders, size limits

3. **Model inference** (2-10ms, currently acceptable)
   - MLX on Apple Silicon is fast for this size
   - **Mitigation**: Quantization, model pruning if needed (likely not)

4. **Data processing overhead** (1-5ms, acceptable)
   - State feature computation is vectorized NumPy
   - **Mitigation**: Pre-compute expensive features, use numba JIT

### Latency Optimization Strategy

**Phase 1: Measure Everything**
- Add timestamps to every stage of the pipeline
- Log: WSS arrival → state compute → inference → order send → fill confirm
- Build latency distribution histograms per asset
- Identify P50, P95, P99 latencies

**Phase 2: Optimize Critical Path**
- Use connection pooling for Polymarket API (keep-alive)
- Pre-compute order signatures for common price levels
- Batch orderbook updates (don't process every tick)
- Consider moving to cloud VPS near Polygon RPC endpoints (if allowed)

**Phase 3: Adaptive Execution**
- Adjust order aggressiveness based on measured latency
- If latency spike detected (>100ms), skip trade or use more conservative limits
- Track "stale signal" rate (time from Binance move to order execution)

---

## Implementation Roadmap

### Phase 1: Execution Infrastructure (Critical)

**Goal**: Actually place orders on Polymarket CLOB

**Tasks**:
1. **Polymarket CLOB API Integration**
   - Implement order placement via REST API
   - EIP-712 signature generation for orders
   - Handle order types: limit, market, post-only
   - Error handling: insufficient balance, order rejection, rate limits

2. **Order State Management**
   - Track: pending, open, partially filled, filled, cancelled, rejected
   - Handle partial fills (adjust position tracking)
   - Reconcile actual fills vs expected fills
   - Websocket subscription for order updates (not just market data)

3. **Position Tracking & Reconciliation**
   - Query actual balances via API
   - Reconcile internal position state with actual holdings
   - Handle edge cases: unexpected fills, manual interventions

**Files to Create/Modify**:
- `helpers/polymarket_executor.py` - Order execution layer
- `helpers/order_manager.py` - Order state tracking
- `run.py` - Replace `execute_action()` with real order placement
- Add `--live` flag to enable real trading mode

**Estimated Effort**: 5-7 days

---

### Phase 2: Slippage & Market Impact Modeling (High Priority)

**Goal**: Realistic fill simulation before going live

**Current Problem**: Paper trading uses mid-price for instant fills. Real fills:
- Cross the spread (immediate cost)
- Walk the orderbook for larger sizes
- Experience adverse selection (orders don't fill at favorable prices)

**Approach**:
1. **Spread Crossing Cost**
   - For market orders: pay the ask (buy) or hit the bid (sell)
   - For limit orders: risk of no fill if price moves away
   - Add spread cost to P&L model

2. **Orderbook Walking Simulation**
   - Current orderbook depth is already captured (L1, L5 imbalance)
   - For size > top of book, simulate walking down the book
   - Estimate effective fill price = volume-weighted average price (VWAP)

3. **Adverse Selection**
   - Orders at favorable prices may not fill (liquidity providers pull)
   - Model fill probability based on: order distance from mid, market volatility, time in book
   - Use historical fill rate data if available

**Implementation**:
- Add `SlippageModel` class with configurable parameters
- Run paper trading with slippage model enabled
- Compare: no-slippage PnL vs slippage-adjusted PnL
- Calibrate model parameters from live orderbook data

**Files to Create/Modify**:
- `helpers/slippage_model.py`
- `run.py` - Add slippage calculation to `execute_action()`
- Add `--slippage` flag to enable realistic fill simulation

**Estimated Effort**: 3-5 days

---

### Phase 3: Latency Measurement & Compensation (Critical for Edge)

**Goal**: Measure and minimize end-to-end latency

**Tasks**:
1. **Instrumentation**
   - Add timestamps: WSS receive, state compute, inference, order send, fill confirm
   - Log to structured format (CSV or parquet)
   - Build real-time latency dashboard

2. **Latency Profiling**
   - Measure P50/P95/P99 for each pipeline stage
   - Identify outliers and root causes
   - Track latency degradation over session (connection quality)

3. **Optimization**
   - Connection pooling: reuse HTTP connections to Polymarket API
   - Pre-computation: cache order signatures, pre-calculate common features
   - Async I/O: use asyncio properly (already partially done)
   - Consider moving to cloud VPS if co-location helps (needs testing)

4. **Adaptive Execution**
   - Track "signal freshness": time from Binance update to order execution
   - If latency > threshold (e.g., 200ms), skip trade or reduce size
   - Adjust limit order prices based on expected latency

**Files to Create/Modify**:
- `helpers/latency_monitor.py`
- `run.py` - Add timing instrumentation throughout
- `dashboard.py` - Add latency metrics panel

**Estimated Effort**: 4-6 days

---

### Phase 4: Risk Management System (Critical for Safety)

**Goal**: Protect capital with automated safeguards

**Current Problem**: No risk limits, no kill switches, no drawdown management

**Requirements**:
1. **Position Limits**
   - Max position size per market: $500 (configurable)
   - Max total exposure across all markets: $2,000 (4 markets × $500)
   - Max percentage of balance per trade: 10%

2. **Drawdown Protection**
   - Max drawdown per session: 10% of starting capital
   - If hit: stop trading, close all positions, alert operator
   - Max consecutive losses: 5 trades (circuit breaker)

3. **Kill Switches**
   - Manual kill switch: close all positions immediately
   - Automatic kill switch triggers:
     - API errors > threshold (e.g., 10 failures in 1 minute)
     - Unexpected position discrepancy
     - Latency > critical threshold (e.g., 500ms sustained)
     - Unrealized loss > threshold (e.g., -$200)

4. **Balance Monitoring**
   - Check USDC balance before every trade
   - Warn if balance < minimum threshold
   - Auto-pause if balance insufficient

5. **Audit Logging**
   - Log every trade with full state snapshot
   - Log all errors, warnings, kill switch triggers
   - Immutable append-only log for post-mortem analysis

**Files to Create/Modify**:
- `helpers/risk_manager.py`
- `run.py` - Integrate risk checks in decision loop
- Add `--max-drawdown`, `--max-exposure` flags

**Estimated Effort**: 4-6 days

---

### Phase 5: Extended Validation (Before Risking Capital)

**Goal**: Build confidence in live performance

**Approach**:
1. **Paper Trading with Realistic Simulation** (2-4 weeks)
   - Enable slippage model
   - Enable latency simulation
   - Track hypothetical vs actual orderbook fills
   - Measure: PnL, win rate, Sharpe ratio, max drawdown

2. **Tiny Size Live Trading** (1-2 weeks)
   - Start with $5-10 per trade (minimum viable size)
   - Manually review every trade
   - Build confidence in execution layer
   - Verify: orders execute correctly, fills reconcile, P&L tracks

3. **Gradual Scale-Up** (2-4 weeks)
   - $10 → $25 → $50 → $100 → $250 → $500
   - Monitor performance degradation with size
   - Stop scaling if Sharpe ratio declines significantly
   - Establish maximum viable size per market

4. **Out-of-Sample Testing**
   - Test across different market regimes: trending, ranging, volatile, quiet
   - Test across different times of day: US hours, Asia hours, weekends
   - Verify edge persists or adapt strategy parameters

**Success Criteria** (before scaling to full size):
- Paper trading with slippage: >50% of original paper PnL
- Tiny size live: positive P&L over 100+ trades
- Win rate: >20% (or positive expectancy with asymmetric payoffs)
- Max drawdown: <15% of capital
- Sharpe ratio: >1.0 (annualized)

**Estimated Effort**: 4-8 weeks (calendar time, not dev time)

---

### Phase 6: Production Hardening (Before Unattended Operation)

**Goal**: Make system reliable enough to run unattended

**Tasks**:
1. **Error Recovery**
   - Auto-reconnect WebSockets with exponential backoff
   - Retry failed API calls with jitter
   - Graceful degradation if data source fails
   - Persist state to disk (survive restarts)

2. **Monitoring & Alerting**
   - Health checks: data freshness, API connectivity, balance sufficiency
   - Alerts: email/SMS on kill switch, large drawdown, errors
   - Metrics: Prometheus/Grafana or simple CSV logging

3. **Configuration Management**
   - Environment-based config: dev, staging, production
   - Secrets management: API keys, private keys (not in code)
   - Parameter versioning: track hyperparameters per session

4. **Testing**
   - Unit tests for critical components (slippage, risk manager, executor)
   - Integration tests with mock Polymarket API
   - Chaos testing: simulate network failures, API errors, data gaps

**Files to Create/Modify**:
- `config/production.yaml`
- `tests/test_executor.py`, `tests/test_risk_manager.py`
- `helpers/alerting.py`

**Estimated Effort**: 5-7 days

---

## Latency Optimization Deep Dive

Since latency is critical, here's a detailed breakdown:

### Critical Path Optimization

**1. WebSocket Data Ingestion** (Currently: 10-50ms)
- ✅ **Good**: Already using asyncio WebSockets
- ⚠️ **Risk**: No backpressure handling (if processing slows, messages queue)
- 🔧 **Optimize**:
  - Process messages in batches (e.g., every 100ms) instead of every tick
  - Drop stale messages if queue grows (prioritize freshness over completeness)
  - Use `ujson` instead of `json` for faster parsing

**2. State Feature Computation** (Currently: 1-5ms)
- ✅ **Good**: Vectorized NumPy operations
- ⚠️ **Risk**: Orderbook imbalance calculation loops through levels
- 🔧 **Optimize**:
  - Pre-allocate arrays, avoid Python loops
  - Use `numba` JIT compilation for hot paths
  - Cache expensive features (volatility, regime) that don't change every tick

**3. Model Inference** (Currently: 2-10ms)
- ✅ **Good**: MLX on Apple Silicon is fast
- ⚠️ **Risk**: Batch size = 1 (inefficient for GPU/NPU)
- 🔧 **Optimize**:
  - Model quantization (float32 → float16) if supported
  - Model pruning (remove low-weight neurons)
  - Batch inference for multiple markets simultaneously

**4. Order Placement** (Currently: NOT IMPLEMENTED)
- ❌ **Missing**: This will be the slowest part (50-200ms)
- 🔧 **Optimize**:
  - Keep-alive HTTP connections (connection pooling)
  - Pre-sign orders at common price levels (cache signatures)
  - Use WebSocket for order submission if Polymarket supports it
  - Consider running on VPS close to Polygon RPC nodes

**5. Order Confirmation** (Currently: NOT IMPLEMENTED)
- ❌ **Missing**: Need to subscribe to order status updates
- 🔧 **Optimize**:
  - Use WebSocket for order status (avoid polling)
  - Optimistically assume fill for immediate actions (risk of error)
  - Set aggressive timeouts (cancel if no fill in 2-3 seconds)

### Infrastructure Choices

**Current**: Apple Silicon Mac (MLX optimized)
- ✅ **Pros**: Fast local inference, low cost, MLX framework optimized
- ❌ **Cons**: Network latency to Polymarket/Polygon, not co-located

**Alternative 1**: Cloud VPS (AWS/GCP near Polygon nodes)
- ✅ **Pros**: Lower network latency (10-30ms vs 50-100ms), always-on
- ❌ **Cons**: No MLX support (need PyTorch/ONNX), higher cost, deployment complexity

**Alternative 2**: Hybrid (Mac for training, VPS for execution)
- ✅ **Pros**: Best of both worlds
- ❌ **Cons**: Complexity of model sync, two systems to maintain

**Recommendation**: Start with current Mac setup, measure latency, consider VPS if latency is bottleneck.

---

## Cloud Hosting Deep Dive

Since latency is critical and the edge depends on acting before Polymarket prices adjust, **cloud hosting near Polygon RPC nodes** could provide a significant advantage.

### Latency Comparison: Local vs Cloud

**Local Mac (Current Setup)**:
```
Binance WSS (Tokyo/Singapore) → Your Location:  50-150ms
Your Location → Polygon RPC (AWS us-east-1):   50-100ms
Your Location → Polymarket API (likely AWS):    50-150ms
Total round-trip:                              150-400ms
```

**Cloud VPS (Optimized)**:
```
Binance WSS → VPS (us-east-1):                  20-50ms
VPS → Polygon RPC (same region):                 1-5ms
VPS → Polymarket API (same region):              1-5ms
Total round-trip:                               25-70ms
```

**Potential latency improvement**: 100-300ms savings = 5-10x faster execution

### Cloud Provider Options

#### Option 1: AWS (Recommended)

**Pros**:
- Polygon RPC nodes likely hosted on AWS
- Polymarket infrastructure likely AWS-based
- Best network proximity = lowest latency
- Mature ecosystem, good monitoring tools
- Spot instances for cost savings

**Cons**:
- No native GPU support for MLX (Apple framework)
- Need to convert model to PyTorch/ONNX
- Higher cost than local ($50-200/month)

**Recommended Setup**:
- **Region**: `us-east-1` (Virginia) - closest to Polygon mainnet RPC nodes
- **Instance Type**:
  - **Development**: `t3.medium` (2 vCPU, 4GB RAM) - $0.0416/hr = $30/month
  - **Production**: `c6i.large` (2 vCPU, 4GB RAM, compute-optimized) - $0.085/hr = $62/month
  - **With GPU** (if needed): `g4dn.xlarge` (4 vCPU, 16GB RAM, T4 GPU) - $0.526/hr = $380/month
- **Storage**: 20GB SSD ($2/month)
- **Network**: Enhanced networking enabled (no extra cost)

**Estimated Monthly Cost**: $50-100 (without GPU), $380+ (with GPU)

#### Option 2: Google Cloud Platform (GCP)

**Pros**:
- Similar network proximity to AWS
- Slightly cheaper compute in some cases
- Good support for PyTorch/TensorFlow

**Cons**:
- Less likely to be co-located with Polygon/Polymarket
- Smaller ecosystem for crypto infrastructure

**Recommended Setup**:
- **Region**: `us-east1` (South Carolina) or `us-east4` (Virginia)
- **Instance Type**:
  - **Production**: `n2-standard-2` (2 vCPU, 8GB RAM) - $0.097/hr = $70/month
  - **With GPU**: `n1-standard-4` + T4 GPU - $0.35 + $0.35/hr = $504/month
- **Storage**: 20GB SSD ($3/month)

**Estimated Monthly Cost**: $70-100 (without GPU), $500+ (with GPU)

#### Option 3: Dedicated Crypto Infrastructure (Alchemy/Infura VPS)

**Pros**:
- Direct connection to Polygon RPC providers
- Optimized for blockchain workloads
- Some providers offer co-located compute

**Cons**:
- More expensive
- Less flexible than AWS/GCP
- Overkill for this use case (not running a node)

**Skip this**: Not needed - standard cloud VPS with RPC endpoint is sufficient.

#### Option 4: Bare Metal / Colocation

**Pros**:
- Absolute lowest latency (if physically near Polygon nodes)
- No noisy neighbor problem

**Cons**:
- Extremely expensive ($200-1000+/month)
- Complex setup and maintenance
- Overkill for this trading volume

**Skip this**: Not cost-effective at $500/trade scale.

### Model Conversion: MLX → PyTorch/ONNX

**Current**: MLX (Apple Silicon only)
**Needed**: Framework that works on Linux/x86

**Option A: Convert to PyTorch** (Recommended)
```python
# Current MLX model architecture is simple (FC layers + LayerNorm)
# Can be replicated in PyTorch 1:1

import torch
import torch.nn as nn

class PyTorchActor(nn.Module):
    def __init__(self, input_dim=18, hidden=64, output_dim=3,
                 history_len=5, temporal_dim=32):
        super().__init__()
        # Temporal encoder
        self.temporal_fc1 = nn.Linear(input_dim * history_len, 64)
        self.temporal_ln1 = nn.LayerNorm(64)
        self.temporal_fc2 = nn.Linear(64, temporal_dim)
        self.temporal_ln2 = nn.LayerNorm(temporal_dim)

        # Actor network
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, output_dim)

    def forward(self, current_state, temporal_state):
        # Temporal encoding
        h_temp = torch.tanh(self.temporal_ln1(self.temporal_fc1(temporal_state)))
        h_temp = torch.tanh(self.temporal_ln2(self.temporal_fc2(h_temp)))

        # Combine and forward
        combined = torch.cat([current_state, h_temp], dim=-1)
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)
        return torch.softmax(logits, dim=-1)
```

**Steps**:
1. Replicate architecture in PyTorch (same layer sizes, activations)
2. Export MLX weights to NumPy
3. Load NumPy weights into PyTorch model
4. Verify outputs match (inference parity test)
5. Save PyTorch checkpoint

**Estimated Effort**: 1-2 days

**Option B: Export to ONNX** (For maximum portability)
```python
# After converting to PyTorch, export to ONNX
import torch.onnx

dummy_current = torch.randn(1, 18)
dummy_temporal = torch.randn(1, 18 * 5)

torch.onnx.export(
    model,
    (dummy_current, dummy_temporal),
    "model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['current_state', 'temporal_state'],
    output_names=['action_probs'],
    dynamic_axes={'current_state': {0: 'batch'},
                  'temporal_state': {0: 'batch'}}
)

# Inference with ONNX Runtime (faster than PyTorch CPU)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
```

**Pros**: ~2-5x faster CPU inference than PyTorch
**Cons**: Extra conversion step, debugging harder

### Deployment Architecture: Hybrid Approach (Recommended)

**Best of both worlds**: Keep Mac for training, use cloud for execution

```
┌─────────────────────────────────────────────────────────┐
│ Apple Silicon Mac (Local)                               │
│                                                          │
│  ┌──────────────────────────────────────┐              │
│  │ Training & Development               │              │
│  │  • Run paper trading with slippage   │              │
│  │  • PPO training with MLX             │              │
│  │  • Model experimentation             │              │
│  │  • Backtesting & analysis            │              │
│  └──────────────────────────────────────┘              │
│                     │                                    │
│                     │ (Model export)                     │
│                     ▼                                    │
│  ┌──────────────────────────────────────┐              │
│  │ Model Conversion                     │              │
│  │  • Export MLX → PyTorch/ONNX         │              │
│  │  • Verify inference parity           │              │
│  │  • Upload to S3 / sync to cloud      │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
                     │
                     │ (rsync / S3)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ AWS EC2 us-east-1 (Production)                          │
│                                                          │
│  ┌──────────────────────────────────────┐              │
│  │ Live Trading Engine                  │              │
│  │  • Load PyTorch/ONNX model           │              │
│  │  • Binance WSS (20-50ms)             │              │
│  │  • Polymarket WSS (1-5ms)            │              │
│  │  • RL inference (2-10ms)             │              │
│  │  • Order execution via CLOB API      │              │
│  │  • Risk management & monitoring      │              │
│  └──────────────────────────────────────┘              │
│                     │                                    │
│                     ▼                                    │
│  ┌──────────────────────────────────────┐              │
│  │ Monitoring & Logging                 │              │
│  │  • CloudWatch logs                   │              │
│  │  • Latency metrics                   │              │
│  │  • Trade logs → S3                   │              │
│  │  • Alerts → SNS → Email/SMS          │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Workflow**:
1. **Develop locally**: Paper trade, train models, analyze results on Mac
2. **Export model**: Convert MLX → PyTorch, upload to S3
3. **Deploy to cloud**: EC2 pulls latest model, runs live trading
4. **Monitor remotely**: CloudWatch metrics, logs streamed to local dashboard
5. **Iterate**: Retrain on Mac based on live results, re-deploy

**Benefits**:
- Keep fast local development workflow (MLX is fast)
- Get low-latency production execution (cloud near Polygon)
- No need to retrain in cloud (expensive)
- Easy rollback (just deploy previous model version)

### Cloud Deployment Checklist

**Pre-Deployment**:
- [ ] Convert MLX model to PyTorch/ONNX
- [ ] Verify inference parity (same outputs for same inputs)
- [ ] Test PyTorch inference speed on target EC2 instance type
- [ ] Set up AWS account, configure IAM roles (least privilege)
- [ ] Configure VPC, security groups (allow only necessary ports)

**Deployment**:
- [ ] Launch EC2 instance (start with t3.medium, scale up if needed)
- [ ] Install dependencies: Python 3.10+, PyTorch, onnxruntime, websockets
- [ ] Clone repo, configure environment variables (API keys, etc.)
- [ ] Upload model checkpoint to S3, download to EC2
- [ ] Test data streams (Binance, Polymarket) from EC2
- [ ] Measure baseline latency (WSS → inference → API)

**Monitoring**:
- [ ] Configure CloudWatch Logs (stdout/stderr)
- [ ] Set up custom metrics (latency, PnL, trade count)
- [ ] Create CloudWatch Alarms:
  - Latency > 200ms for 5 minutes → SNS alert
  - Drawdown > 10% → SNS alert + auto-shutdown
  - API errors > 10/minute → SNS alert
- [ ] Configure SNS topic → Email/SMS notifications
- [ ] Set up S3 bucket for trade logs (append-only, versioned)

**Security**:
- [ ] Store API keys in AWS Secrets Manager (not in code)
- [ ] Use IAM roles for EC2 (no hardcoded credentials)
- [ ] Enable MFA on AWS account
- [ ] Restrict SSH access to your IP only
- [ ] Use encrypted EBS volumes
- [ ] Enable CloudTrail for audit logging

**Cost Optimization**:
- [ ] Use Spot Instances if workload tolerates interruptions (60-90% savings)
- [ ] Set up billing alerts ($50, $100, $200 thresholds)
- [ ] Use t3 burstable instances (cheaper when idle)
- [ ] Shut down instance during non-trading hours (if not 24/7)
- [ ] Use S3 lifecycle policies (move old logs to Glacier)

### Latency Testing Plan (Cloud vs Local)

**Objective**: Measure actual latency improvement from cloud hosting

**Test Setup**:
1. Run same codebase on Mac (local) and EC2 (us-east-1)
2. Subscribe to same WebSocket streams
3. Log timestamps for each pipeline stage
4. Compare distributions

**Metrics to Measure**:
```
Metric                           | Local (Mac) | Cloud (EC2) | Improvement
---------------------------------|-------------|-------------|------------
Binance WSS → Local receive      | 80ms P95    | 30ms P95    | 2.7x faster
Polymarket WSS → Local receive   | 90ms P95    | 5ms P95     | 18x faster
State computation                | 3ms P95     | 3ms P95     | (same)
Model inference                  | 8ms P95     | 15ms P95    | 1.9x slower*
Order API call → response        | 120ms P95   | 10ms P95    | 12x faster
Total (signal → order placed)    | 300ms P95   | 65ms P95    | 4.6x faster
```

*Model inference slower on EC2 CPU vs M-series neural engine, but offset by network gains

**Decision Criteria**:
- If cloud saves >100ms end-to-end → **Worth it** (edge preservation)
- If cloud saves 30-100ms → **Marginal** (test with live trading to verify)
- If cloud saves <30ms → **Not worth** complexity (stay local)

### Monthly Cost Breakdown (Cloud Hosting)

**Option 1: Minimal Setup** (t3.medium, no GPU)
```
EC2 t3.medium (2 vCPU, 4GB):     $30/month
EBS Storage 20GB:                 $2/month
Data Transfer (10GB out):         $1/month
CloudWatch Logs (5GB):            $2.50/month
S3 Storage (100GB):               $2.30/month
-----------------------------------------------
Total:                           ~$38/month
```

**Option 2: Compute-Optimized** (c6i.large, no GPU)
```
EC2 c6i.large (2 vCPU, 4GB):     $62/month
EBS Storage 20GB:                 $2/month
Data Transfer (10GB out):         $1/month
CloudWatch Logs (5GB):            $2.50/month
S3 Storage (100GB):               $2.30/month
-----------------------------------------------
Total:                           ~$70/month
```

**Option 3: GPU-Accelerated** (g4dn.xlarge, if inference is bottleneck)
```
EC2 g4dn.xlarge (4 vCPU, 16GB, T4): $380/month
EBS Storage 50GB:                    $5/month
Data Transfer (10GB out):            $1/month
CloudWatch Logs (5GB):               $2.50/month
S3 Storage (100GB):                  $2.30/month
-----------------------------------------------
Total:                              ~$391/month
```

**Recommendation**: Start with **Option 1** (t3.medium, $38/month)
- More than sufficient for this workload (no heavy computation)
- Upgrade to c6i.large only if CPU is bottleneck
- GPU is overkill for this model size (18 input → 64 hidden → 3 output)

### Alternative: Serverless (AWS Lambda)

**Could this run on Lambda?** (for cost savings)

**Pros**:
- Pay per invocation (potentially much cheaper)
- Auto-scaling, no instance management
- $0.20 per 1M requests + $0.0000166667 per GB-second

**Cons**:
- ❌ **Cold start latency**: 1-3 seconds (kills the edge)
- ❌ **No persistent WebSocket connections** (need to reconnect each time)
- ❌ **15-minute max execution time** (can't run continuously)
- ❌ **Stateless** (hard to maintain RL agent state)

**Verdict**: **Not suitable** for this use case. Need persistent, low-latency execution.

### Recommended Cloud Strategy

**Phase 1: Measure** (Week 1)
- Deploy to EC2 t3.medium in us-east-1
- Run in paper trading mode (parallel with local Mac)
- Measure latency improvements
- Compare: Mac paper PnL vs EC2 paper PnL (should be same, but faster)

**Phase 2: Validate** (Week 2-3)
- If latency improvement >100ms, proceed with cloud execution
- Run tiny size live trading ($5-10) from EC2
- Monitor for issues: network stability, API reliability, costs

**Phase 3: Optimize** (Week 4+)
- If inference is slow, consider c6i (compute-optimized) or ONNX Runtime
- If network is slow, test different RPC endpoints (Alchemy, Infura, QuickNode)
- If costs are high, use Spot Instances (save 60-90%)

**Cost-Benefit**:
- Monthly cloud cost: $40-70
- Potential latency savings: 100-200ms
- Potential edge preservation: Could be worth 10-30% of PnL
- If live trading generates >$500/month, cloud cost is <10% of profit → **Worth it**
- If live trading loses money, shut down cloud immediately → **Minimal loss**

### Latency Measurement Plan

**Instrumentation Points**:
```python
# Add to run.py decision loop
import time

# Before state update
t0 = time.perf_counter()

# After state features computed
t1 = time.perf_counter()
latency_state = (t1 - t0) * 1000  # ms

# After model inference
t2 = time.perf_counter()
latency_inference = (t2 - t1) * 1000

# After order sent
t3 = time.perf_counter()
latency_order_send = (t3 - t2) * 1000

# After fill confirmed
t4 = time.perf_counter()
latency_fill = (t4 - t3) * 1000

# Log to CSV
logger.log_latency(latency_state, latency_inference,
                   latency_order_send, latency_fill)
```

**Dashboard Metrics**:
- Real-time line chart: end-to-end latency over time
- Histogram: latency distribution (P50, P95, P99)
- Alert: if latency > 200ms for 3 consecutive ticks

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Latency degrades edge** | High | Critical | Measure, optimize, adaptive execution |
| **Slippage exceeds estimates** | High | High | Conservative modeling, gradual scale-up |
| **Polymarket API issues** | Medium | High | Error handling, fallback logic, alerts |
| **Market impact at scale** | High | High | Position limits, test size scaling carefully |
| **Model overfitting to paper trading** | Medium | Critical | Extended live validation, out-of-sample testing |
| **Liquidity dries up** | Medium | Medium | Monitor orderbook depth, reduce size if needed |

### Market Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Edge disappears** | Medium | Critical | Continuous monitoring, stop trading if Sharpe <0.5 |
| **Competitors copy strategy** | Low | High | Keep strategy private, monitor market microstructure |
| **Market structure changes** | Low | Medium | Adapt to new market conditions, retrain model |
| **Regulatory issues** | Low | Critical | Consult legal counsel, use only legal jurisdictions |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **System crash during trading** | Medium | Medium | Persistence, auto-restart, position recovery |
| **Internet outage** | Low | High | Backup connection, mobile hotspot, kill switch |
| **API key compromise** | Low | Critical | Secrets management, API key rotation, 2FA |
| **Manual error** | Medium | Medium | Confirmation prompts, dry-run mode, audit logs |

---

## Cost-Benefit Analysis

### Expected Performance (Conservative Estimates)

**Assumptions**:
- Paper trading PnL: $50K over 10 hours (Phase 5)
- Performance degradation: 50% (slippage, latency, market impact)
- Expected live PnL: $25K per 10-hour session
- Number of sessions per week: 5 (weekdays during US market hours)

**Projected Returns** (monthly):
- Gross PnL: $25K/session × 5 sessions/week × 4 weeks = **$500K/month**
- Max drawdown risk: 15% of capital
- Minimum capital required: $2,000 (4 markets × $500)

**Reality Check**: This is wildly optimistic. Real-world considerations:
- Edge may be smaller than paper trading suggests
- Market regime changes (model trained on recent data)
- Liquidity constraints (can't scale indefinitely)
- Competition (if edge exists, others will find it)

**Realistic Scenario** (after degradation):
- Reduce projections by 80% → $100K/month
- Still needs validation across market regimes
- May not persist beyond initial validation period

### Development Cost

**Time to MVP** (live trading with minimum features):
- Phase 1 (Execution): 5-7 days
- Phase 2 (Slippage): 3-5 days
- Phase 3 (Latency): 4-6 days
- Phase 4 (Risk Management): 4-6 days
- Phase 5 (Validation): 4-8 weeks (calendar time)
- **Total**: ~3-4 weeks development + 1-2 months validation

**Ongoing Costs**:
- Developer time: Monitoring, maintenance, improvements
- Infrastructure: VPS if needed ($50-200/month)
- Data/API: Binance + Polymarket (free currently)
- Capital at risk: $2,000 minimum (can scale up gradually)

---

## Go/No-Go Criteria

**GO** if:
- ✅ Paper trading with slippage shows >30% of original PnL
- ✅ End-to-end latency <150ms P95
- ✅ Tiny size live trading ($5-25) shows positive expectancy over 100 trades
- ✅ Risk management system passes stress tests
- ✅ Comfortable with capital at risk (only risk what you can afford to lose)

**NO-GO** if:
- ❌ Slippage-adjusted paper trading shows negative expectancy
- ❌ Latency consistently >300ms (edge likely eroded)
- ❌ Tiny size live trading loses money over 100 trades
- ❌ Risk management failures in testing
- ❌ Regulatory concerns or legal risks

---

## Recommended Next Steps

**Week 1-2: Foundation**
1. Implement Polymarket CLOB order execution (`helpers/polymarket_executor.py`)
2. Add order state management and reconciliation
3. Test with tiny sizes ($5) on live markets
4. Verify: orders place correctly, fills reconcile, P&L accurate

**Week 3-4: Optimization**
1. Implement slippage model and test in paper trading
2. Add latency instrumentation and measurement
3. Optimize critical path (connection pooling, caching)
4. Build risk management system (position limits, kill switches)

**Month 2: Validation**
1. Run paper trading with slippage for 2 weeks
2. Run tiny size live trading ($5-25) for 2 weeks
3. Analyze results: compare to paper, measure latency, assess edge persistence
4. Decision point: scale up or shut down

**Month 3+: Scale or Adapt**
- If successful: gradually increase size, monitor performance degradation
- If unsuccessful: analyze failures, adapt strategy, or abandon
- Continuous: monitor market conditions, retrain model, improve execution

---

## Technical Debt & Future Improvements

**After MVP is working**:
1. **Advanced order types**: iceberg orders, TWAP execution, smart order routing
2. **Multi-venue**: expand to other prediction markets (Kalshi, Augur)
3. **Model improvements**: online learning, ensemble methods, meta-learning
4. **Backtesting**: replay historical data with realistic fills
5. **Portfolio optimization**: Kelly criterion for position sizing, correlation-aware allocation
6. **Market making**: provide liquidity instead of taking (collect spread, not pay it)

---

## Conclusion

**The Edge**: Cross-market arbitrage exploiting information lag between Binance (fast) and Polymarket (slow). Latency is critical.

**The Challenge**: Paper trading results (2,500% ROI) are unrealistic. Real trading will have:
- Slippage: crossing spreads, walking orderbooks
- Latency: 50-200ms to Polymarket, during which signal decays
- Market impact: larger orders move prices against you
- Liquidity: limited depth at favorable prices

**The Path Forward**:
1. Build execution infrastructure (1-2 weeks)
2. Model slippage realistically (1 week)
3. Measure and optimize latency (<150ms target)
4. Implement robust risk management (1 week)
5. Validate extensively before scaling (4-8 weeks)

**Expected Outcome**:
- Best case: 20-50% of paper trading performance in live trading
- Realistic case: 10-20% of paper trading performance
- Worst case: negative expectancy after costs

**Risk/Reward**:
- Low initial capital requirement ($2K-10K)
- High development time investment (1-2 months)
- Edge may not persist (market efficiency, competition)
- Educational value even if unprofitable

**Recommendation**: Proceed cautiously with phased approach. Start tiny, measure everything, scale only with evidence.
