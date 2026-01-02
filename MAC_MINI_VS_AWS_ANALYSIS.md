# Mac Mini vs AWS Deployment Analysis

**Question**: Should we deploy a Mac Mini for 24/7 trading instead of AWS, to keep MLX performance?

**TL;DR**: Mac Mini in colocation OR AWS Mac instances could work, but the numbers are tight. Local Mac Mini at home doesn't provide enough latency benefit.

---

## The Core Trade-Off

**AWS EC2 (Linux/x86)**:
- ✅ Low network latency (1-5ms to Polygon RPC)
- ✅ Low cost ($38-70/month)
- ❌ Slower inference (15-20ms vs 5-8ms on M-series)
- ❌ Requires MLX → PyTorch conversion

**Mac Mini (M2/M3)**:
- ✅ Fast inference (5-8ms with MLX)
- ✅ No model conversion needed
- ❌ Deployment location determines network latency
- ❌ Higher upfront cost ($600-800)

**Net difference**: ~10ms inference savings, but need to consider total latency.

---

## Option 1: Mac Mini at Home

### Setup
- Mac Mini M2 (base model): $599
- Your home internet
- Run 24/7 in a closet/desk

### Latency Analysis
```
Component                        | Mac at Home | AWS EC2 | Difference
---------------------------------|-------------|---------|------------
Binance WSS → receive            | 80ms P95    | 30ms    | -50ms (worse)
Polymarket WSS → receive         | 90ms P95    | 5ms     | -85ms (worse)
Model inference (MLX vs PyTorch) | 8ms P95     | 15ms    | +7ms (better)
Order API → response             | 120ms P95   | 10ms    | -110ms (worse)
---------------------------------|-------------|---------|------------
Total (signal → order)           | ~300ms      | 65ms    | -235ms (worse!)
```

**Verdict**: ❌ **Not worth it**
- You save 7ms on inference
- You lose 235ms on network latency
- Net result: 3.6x slower than AWS
- The M-series advantage is completely negated by network latency

### Cost Analysis
```
Initial: $599 (Mac Mini M2)
Monthly: ~$2 (electricity, 20W idle × 24hr × $0.12/kWh × 30 days)
Break-even: Never (it's slower than AWS)
```

---

## Option 2: Mac Mini in Colocation Datacenter

### Setup
- Mac Mini M2: $599
- Colocation near AWS us-east-1 (e.g., Equinix NY5, Telx NY2)
- 1U rack space, power, bandwidth

### Latency Analysis (Assumes colocation in NYC/Virginia area)
```
Component                        | Mac Colo | AWS EC2 | Difference
---------------------------------|----------|---------|------------
Binance WSS → receive            | 30ms     | 30ms    | 0ms (same)
Polymarket WSS → receive         | 5ms      | 5ms     | 0ms (same)
Model inference (MLX vs PyTorch) | 8ms      | 15ms    | +7ms (better)
Order API → response             | 10ms     | 10ms    | 0ms (same)
---------------------------------|----------|---------|------------
Total (signal → order)           | ~55ms    | 65ms    | +10ms (better)
```

**Verdict**: ✅ **Slightly better performance** but is it worth the cost?

### Cost Analysis
```
Colocation pricing (Equinix/Telx for 1U):
- Setup fee: $100-500
- Monthly: $50-150/month (1U, 1A power, 1Gbps)
- Cross-connect (if needed): $100-300/month
- Remote hands (if needed): $100-200/hour

Total first year:
- Hardware: $600
- Setup: $200 (avg)
- Monthly: $100/month × 12 = $1,200
- Total: $2,000 first year
- Ongoing: $1,200/year

AWS EC2 for comparison:
- Total first year: $38/month × 12 = $456
- Ongoing: $456/year
```

**Cost-benefit**:
- Extra cost: $1,544/year ($2,000 - $456)
- Performance gain: 10ms faster (15% improvement)
- Worth it if: 10ms latency = +$1,544/year in profits

**Is 10ms worth $1,544?**
- If paper trading PnL is $50K/year
- And 10ms improves fills by 3-5%
- Extra profit: $1,500-2,500/year
- **Break-even or slightly profitable**

### Risks
- Hardware failure (Mac Mini not enterprise-grade)
- Remote troubleshooting (can't easily access)
- Datacenter costs can increase
- Complexity vs AWS (no auto-restart, monitoring, etc.)

---

## Option 3: AWS Mac Instances (mac2.metal)

**AWS offers actual M1/M2 Mac instances!**

### Specs
- **mac2.metal**: M2 chip, 8 cores, 24GB RAM
- **Network**: Up to 10 Gbps in same region
- **Location**: us-east-1, us-west-2, eu-west-1
- **MLX support**: ✅ Native (it's a real Mac)

### Latency Analysis
```
Component                        | AWS Mac | AWS x86 | Difference
---------------------------------|---------|---------|------------
Binance WSS → receive            | 30ms    | 30ms    | 0ms
Polymarket WSS → receive         | 5ms     | 5ms     | 0ms
Model inference (MLX on M2)      | 8ms     | 15ms    | +7ms (better)
Order API → response             | 10ms    | 10ms    | 0ms
---------------------------------|---------|---------|------------
Total (signal → order)           | ~55ms   | 65ms    | +10ms (better)
```

**Verdict**: ✅ **Best of both worlds** - but expensive

### Cost Analysis
```
AWS mac2.metal pricing (us-east-1):
- On-Demand: $1.083/hour = $781/month
- Dedicated Host (24hr minimum): $0.653/hour = $471/month
- Storage (100GB): $10/month
- Data transfer: $1-5/month

Total: $472-792/month (vs $38/month for t3.medium)

Cost multiplier: 12-20x more expensive than Linux EC2
```

**Is it worth 12x the cost for 10ms?**
- Extra cost: $434-754/month = $5,208-9,048/year
- Need to make an extra $5-9K/year to justify
- If trading at $500/position, need 10-18 extra profitable trades/year
- At 20% win rate, need 50-90 extra trades
- **Likely not worth it unless scaling to $5K+ per trade**

### Notes
- AWS Mac instances have 24hr minimum allocation
- Can't use spot instances for Macs
- Limited availability (waitlist possible)
- Overkill for this workload

---

## Option 4: Hybrid (Train on Mac, Deploy to AWS x86)

**This is what I originally recommended** - let's revisit the model portability question.

### RL Model Portability: Does it transfer?

**Short answer**: ✅ **YES** - the policy is hardware-agnostic

**Why it works**:
1. RL policy = neural network weights (just numbers)
2. Architecture is simple: fully connected layers + LayerNorm + tanh
3. No hardware-specific tricks (unlike some vision models)
4. Deterministic inference (no dropout, no randomness)

**Conversion process**:
```python
# 1. Export MLX weights from Mac
import mlx.core as mx
import numpy as np

# Load trained MLX model
actor_weights = mx.load("rl_model/actor.npz")
critic_weights = mx.load("rl_model/critic.npz")

# Convert to NumPy
actor_np = {k: np.array(v) for k, v in actor_weights.items()}
critic_np = {k: np.array(v) for k, v in critic_weights.items()}

# Save as .npy
np.save("actor_weights.npy", actor_np)
np.save("critic_weights.npy", critic_np)

# 2. Load into PyTorch on AWS
import torch

actor_np = np.load("actor_weights.npy", allow_pickle=True).item()

# Map to PyTorch model
pytorch_actor = PyTorchActor(...)  # Same architecture
state_dict = {}
for k, v in actor_np.items():
    state_dict[k] = torch.from_numpy(v)

pytorch_actor.load_state_dict(state_dict)

# 3. Verify inference parity
test_state = np.random.randn(18).astype(np.float32)
test_temporal = np.random.randn(90).astype(np.float32)

# MLX inference
mlx_output = mlx_actor(mx.array(test_state), mx.array(test_temporal))

# PyTorch inference
torch_output = pytorch_actor(
    torch.from_numpy(test_state),
    torch.from_numpy(test_temporal)
)

# Should be identical (within floating point error)
diff = np.abs(np.array(mlx_output) - torch_output.detach().numpy())
assert diff.max() < 1e-6, f"Inference mismatch: {diff.max()}"
```

**Potential issues**:
- ⚠️ **LayerNorm differences**: MLX and PyTorch might have slightly different implementations
  - Solution: Use same epsilon value (1e-5)
  - Verify outputs match on test cases
- ⚠️ **Float32 vs Float16**: Ensure same precision
  - MLX defaults to float32
  - PyTorch can use float16 for speed (but loses precision)
  - Stick to float32 for consistency
- ⚠️ **Batch vs single inference**: Make sure to handle batch dimension
  - MLX model might expect (18,) input
  - PyTorch expects (1, 18) input
  - Add unsqueeze(0) if needed

**Performance transfer**:
- The RL policy learned on Mac will work identically on AWS
- Same states → same actions
- No "retraining" needed
- Only difference: inference speed (8ms → 15ms)

---

## Recommendation Matrix

| Scenario | Recommended Option | Why |
|----------|-------------------|-----|
| **Starting out** ($5-50/trade) | AWS t3.medium ($38/mo) | Lowest cost, proven reliability, good latency |
| **Validated** ($100-500/trade) | AWS c6i.large ($70/mo) | Slightly better compute, still cost-effective |
| **Scaling** ($500-2K/trade) | Mac Mini colocation ($100/mo) | 10ms edge might matter, costs justified |
| **High-frequency** ($5K+/trade) | AWS mac2.metal ($500/mo) | Maximum performance, costs justified by volume |
| **Budget-constrained** | Mac at home (free after $600) | Only if you already have Mac Mini, understand latency penalty |

---

## My Specific Recommendation for Your Case

**Start with AWS t3.medium ($38/month)**:

### Reasoning:
1. **Latency is king**: 235ms faster than Mac at home
2. **10ms inference difference is negligible**: Network latency (200ms saved) >> inference latency (7ms lost)
3. **Model transfers perfectly**: Train on Mac, deploy to AWS with PyTorch
4. **Low risk**: $38/month is minimal vs $600 upfront + colocation
5. **Proven**: AWS is reliable, well-monitored, easy to restart

### Workflow:
```
┌─────────────────────────────────────┐
│ Local Mac (Development)              │
│  • Paper trading + slippage sim     │
│  • RL training with MLX (fast!)     │
│  • Model analysis & tuning          │
│  • Export to PyTorch when ready     │
└──────────────┬──────────────────────┘
               │
               │ (rsync or S3 upload)
               ▼
┌─────────────────────────────────────┐
│ AWS EC2 t3.medium (Production)      │
│  • Load PyTorch model               │
│  • Live trading 24/7                │
│  • Low latency (65ms total)         │
│  • $38/month                        │
└─────────────────────────────────────┘
```

### If you later prove >$10K/month profits:
- Upgrade to Mac Mini colocation ($100/mo) for 10ms edge
- Or AWS mac2.metal ($500/mo) if you're trading $5K+ per position
- But not before you've proven profitability at small scale

---

## Model Portability Testing Plan

Before deploying to AWS, verify model transfers correctly:

### 1. Convert Model
```bash
# On Mac (after training)
python scripts/export_mlx_to_pytorch.py --input rl_model --output rl_model_pytorch
```

### 2. Create Test Suite
```python
# test_model_parity.py
import numpy as np
import mlx.core as mx
import torch

# Load both models
mlx_actor = load_mlx_model("rl_model/actor.npz")
pytorch_actor = load_pytorch_model("rl_model_pytorch/actor.pt")

# Generate 1000 random test states
np.random.seed(42)
test_states = [np.random.randn(18).astype(np.float32) for _ in range(1000)]
test_temporal = [np.random.randn(90).astype(np.float32) for _ in range(1000)]

# Compare outputs
max_diff = 0
for state, temporal in zip(test_states, test_temporal):
    mlx_out = np.array(mlx_actor(mx.array(state), mx.array(temporal)))
    torch_out = pytorch_actor(
        torch.from_numpy(state),
        torch.from_numpy(temporal)
    ).detach().numpy()

    diff = np.abs(mlx_out - torch_out).max()
    max_diff = max(max_diff, diff)

print(f"Max inference difference: {max_diff}")
assert max_diff < 1e-5, "Models don't match!"
print("✓ Models are equivalent")
```

### 3. Upload to AWS and Test
```bash
# Upload converted model
aws s3 cp rl_model_pytorch/ s3://my-bucket/models/ --recursive

# SSH to EC2
ssh ec2-user@<instance-ip>

# Download and test
aws s3 sync s3://my-bucket/models/ ./rl_model_pytorch/
python test_inference_speed.py  # Should be 10-20ms
```

### 4. Run Parallel Paper Trading
```bash
# On Mac
python run.py rl --load rl_model --size 50

# On AWS (same time)
python run.py rl --load rl_model_pytorch --size 50

# Compare results after 24 hours - should be identical trades
```

---

## Conclusion

**Q1: Mac Mini vs AWS?**
- ❌ Mac at home: Slower overall (network latency kills M-series advantage)
- ⚠️ Mac in colocation: 10ms faster, 3x cost, break-even at scale
- ⚠️ AWS Mac instances: 10ms faster, 12x cost, only worth at $5K+ scale
- ✅ **AWS x86 (recommended)**: Best latency/cost trade-off for starting

**Q2: Do RL gains transfer across machines?**
- ✅ **YES** - Policy is hardware-agnostic
- Weights transfer perfectly via MLX → NumPy → PyTorch
- Verify with inference parity tests
- Performance identical (just 7ms slower inference, but 235ms faster overall)

**Action plan**:
1. Train on Mac with MLX (fast, convenient)
2. Convert to PyTorch when ready for live trading
3. Deploy to AWS t3.medium ($38/mo)
4. If profitable at scale, consider Mac colocation or AWS Mac instances

The 7ms inference penalty is a non-issue compared to 235ms network latency savings.
