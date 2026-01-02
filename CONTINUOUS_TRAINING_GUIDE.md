# Continuous Training Architecture: Data Collection + Offline Training

**Goal**: Run multiple market observers 24/7, collect data cheaply, train models efficiently on Mac.

**TL;DR**: Use tiny AWS instances ($7/month each) to collect market data → Store in S3 → Train offline on Mac Mini → Deploy best models.

---

## The Problem with Online Training

**Current approach** (from codebase):
```
Live market data → Agent takes action → Trade executes → Reward received → Model updates
```

**Issues**:
- ❌ Requires actual trading (paper or live) to collect experiences
- ❌ Training is coupled to trading execution
- ❌ Can't easily run multiple experiments
- ❌ Expensive to run on cloud (need compute for training)

**What you want**:
```
Live market data → Record states/outcomes → Batch training on Mac → Deploy models
```

**Benefits**:
- ✅ No trading required during data collection
- ✅ Ultra-cheap data collectors ($7/month each)
- ✅ Train on Mac Mini (free, MLX-optimized)
- ✅ Run dozens of markets/experiments simultaneously
- ✅ Training can be faster than real-time (batch processing)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ AWS (Data Collection Only)                                      │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Observer 1   │  │ Observer 2   │  │ Observer N   │          │
│  │ TRUMP2024    │  │ ETH10K       │  │ BTC100K      │          │
│  │              │  │              │  │              │          │
│  │ t3.micro     │  │ t3.micro     │  │ t3.micro     │          │
│  │ $7/month     │  │ $7/month     │  │ $7/month     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                     │
│                            ▼                                     │
│                   ┌─────────────────┐                            │
│                   │   S3 Bucket     │                            │
│                   │                 │                            │
│                   │ market-data/    │                            │
│                   │  TRUMP/         │                            │
│                   │  ETH/           │                            │
│                   │  BTC/           │                            │
│                   └────────┬────────┘                            │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              │ (sync every hour)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Mac Mini (Training + Analysis)                                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Sync data from S3                                     │   │
│  │ 2. Process into training batches                         │   │
│  │ 3. Offline RL training (PPO on historical trajectories)  │   │
│  │ 4. Evaluate models on hold-out data                      │   │
│  │ 5. Upload best models to S3                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Cost: $0/month (electricity only)                              │
│  Speed: Fast (MLX on Apple Silicon)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (when ready)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ AWS EC2 (Live Trading Execution)                                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Download best model from S3                              │   │
│  │ Run live trading with PyTorch                            │   │
│  │ Collect live performance data                            │   │
│  │ Send metrics back to S3                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  t3.medium: $38/month (only when trading)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Data Collection Observer

**Purpose**: Lightweight process that watches market data and records observations.

### observer.py (New file)

```python
#!/usr/bin/env python3
"""
Market data observer - collects state observations without trading.

Runs 24/7 on cheap AWS instance, uploads data to S3 periodically.

Usage:
    python observer.py --market TRUMP2024 --s3-bucket my-market-data
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import gzip

import numpy as np
from helpers.binance_client import BinanceClient
from helpers.polymarket_client import PolymarketClient


class MarketObserver:
    """Observes market data and records states for offline training."""

    def __init__(self, cid: str, s3_bucket: str = None, local_dir: str = "data"):
        self.cid = cid
        self.s3_bucket = s3_bucket
        self.local_dir = Path(local_dir) / cid
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients (read-only)
        self.binance = BinanceClient()
        self.polymarket = PolymarketClient()

        # State history (for temporal features)
        self.state_history = []
        self.max_history = 5

        # Recording buffer
        self.observations = []
        self.buffer_size = 1000  # Write every 1000 observations

        print(f"📊 Observer initialized for {cid}")
        print(f"   Local dir: {self.local_dir}")
        print(f"   S3 bucket: {s3_bucket or 'disabled'}")

    def get_market_state(self):
        """
        Fetch current market state (same as trading logic).

        Returns:
            dict: {
                'timestamp': ...,
                'binance': {...},
                'polymarket': {...},
                'state_vector': [18 values]
            }
        """
        # Fetch Binance data
        try:
            binance_data = self.binance.get_ticker(self.cid)
        except Exception as e:
            print(f"⚠️  Binance fetch failed: {e}")
            return None

        # Fetch Polymarket data
        try:
            polymarket_data = self.polymarket.get_market(self.cid)
        except Exception as e:
            print(f"⚠️  Polymarket fetch failed: {e}")
            return None

        # Build state vector (18 dimensions)
        state = self._build_state_vector(binance_data, polymarket_data)

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'binance': binance_data,
            'polymarket': polymarket_data,
            'state_vector': state.tolist(),
        }

    def _build_state_vector(self, binance, polymarket):
        """
        Build 18-dimensional state vector.

        Same as RLStrategy.get_state() from run.py
        """
        # Binance features (10 values)
        binance_price = binance.get('price', 0)
        binance_volume = binance.get('volume', 0)
        binance_spread = binance.get('spread', 0)
        binance_momentum = binance.get('momentum', 0)
        # ... (extract all 10 features)

        # Polymarket features (8 values)
        poly_yes_price = polymarket.get('yes_price', 0)
        poly_no_price = polymarket.get('no_price', 0)
        poly_spread = polymarket.get('spread', 0)
        # ... (extract all 8 features)

        state = np.array([
            binance_price,
            binance_volume,
            binance_spread,
            binance_momentum,
            # ... all 18 features
        ], dtype=np.float32)

        return state

    def record_observation(self, state_data):
        """
        Record observation with hindsight labeling.

        For each observation, we'll later label it with:
        - What happened to price in next 30s, 1m, 5m
        - Whether a trade would have been profitable
        - Optimal action (buy/sell/hold)
        """
        self.observations.append(state_data)

        # Write to disk periodically
        if len(self.observations) >= self.buffer_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        """Write observations to local disk (compressed JSON)."""
        if not self.observations:
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.local_dir / f"obs_{timestamp}.json.gz"

        data = {
            'market': self.cid,
            'start_time': self.observations[0]['timestamp'],
            'end_time': self.observations[-1]['timestamp'],
            'count': len(self.observations),
            'observations': self.observations,
        }

        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f)

        print(f"💾 Wrote {len(self.observations)} observations to {filename.name}")

        # Upload to S3 if configured
        if self.s3_bucket:
            self.upload_to_s3(filename)

        # Clear buffer
        self.observations = []

    def upload_to_s3(self, filepath):
        """Upload file to S3."""
        try:
            import boto3
            s3 = boto3.client('s3')

            # Key format: market-data/{CID}/{date}/{filename}
            date_prefix = datetime.utcnow().strftime("%Y-%m-%d")
            s3_key = f"market-data/{self.cid}/{date_prefix}/{filepath.name}"

            s3.upload_file(str(filepath), self.s3_bucket, s3_key)
            print(f"☁️  Uploaded to s3://{self.s3_bucket}/{s3_key}")

        except Exception as e:
            print(f"⚠️  S3 upload failed: {e}")

    def run(self, tick_interval=0.5):
        """
        Main observation loop.

        Args:
            tick_interval: Seconds between observations (default 0.5s = 500ms)
        """
        print(f"\n🚀 Starting observation loop (tick={tick_interval}s)")
        print("   Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while True:
                start_time = time.time()

                # Fetch market state
                state_data = self.get_market_state()

                if state_data:
                    self.record_observation(state_data)
                    iteration += 1

                    if iteration % 100 == 0:
                        print(f"✓ {iteration} observations collected "
                              f"({len(self.observations)} in buffer)")

                # Sleep to maintain tick rate
                elapsed = time.time() - start_time
                sleep_time = max(0, tick_interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n⏸️  Stopping observer...")
            self.flush_to_disk()
            print("✓ Data saved")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.flush_to_disk()
            raise


def main():
    parser = argparse.ArgumentParser(description="Market data observer")
    parser.add_argument("--market", required=True, help="Market CID to observe")
    parser.add_argument("--s3-bucket", help="S3 bucket for data upload (optional)")
    parser.add_argument("--local-dir", default="data", help="Local data directory")
    parser.add_argument("--tick", type=float, default=0.5, help="Tick interval (seconds)")
    args = parser.parse_args()

    observer = MarketObserver(args.market, args.s3_bucket, args.local_dir)
    observer.run(tick_interval=args.tick)


if __name__ == "__main__":
    main()
```

### Deployment on AWS t3.micro

**Specs**:
- 1 vCPU, 1GB RAM
- $7.30/month ($0.0104/hour)
- Sufficient for market data collection (no training)

**Setup**:
```bash
# Launch instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --key-name my-key \
    --security-group-ids sg-xxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=Observer-TRUMP}]'

# SSH and setup
ssh ubuntu@instance-ip

# Install minimal dependencies
sudo apt update
sudo apt install -y python3-pip
pip3 install boto3 numpy requests websockets

# Clone repo (or just copy observer.py)
git clone https://github.com/your-repo/cross-market-state-fusion
cd cross-market-state-fusion

# Configure AWS credentials (for S3 upload)
aws configure

# Run observer in background
nohup python3 observer.py \
    --market "TRUMP2024" \
    --s3-bucket "my-market-data" \
    --tick 0.5 \
    > observer.log 2>&1 &

# Verify it's running
tail -f observer.log
```

### Systemd service (auto-restart)

```ini
# /etc/systemd/system/observer-trump.service
[Unit]
Description=Market Observer - TRUMP2024
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cross-market-state-fusion
ExecStart=/usr/bin/python3 observer.py --market TRUMP2024 --s3-bucket my-market-data --tick 0.5
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable observer-trump
sudo systemctl start observer-trump
sudo systemctl status observer-trump
```

---

## Component 2: Offline Training on Mac

**Purpose**: Download collected data, process into RL trajectories, train models.

### offline_trainer.py (New file)

```python
#!/usr/bin/env python3
"""
Offline RL trainer - learns from collected observation data.

Runs on Mac Mini with MLX for fast training.

Usage:
    python offline_trainer.py --data-dir ./data --output-model ./models/offline_v1
"""
import argparse
import json
import gzip
from pathlib import Path
from typing import List, Tuple

import numpy as np
import mlx.core as mx
import mlx.nn as nn


class OfflineTrainer:
    """
    Trains RL policy from offline data using hindsight labeling.

    Instead of learning from actual trades, we:
    1. Label each state with "what would have happened" if we took each action
    2. Train policy to maximize expected return
    """

    def __init__(self, data_dir: str, model_output: str):
        self.data_dir = Path(data_dir)
        self.model_output = Path(model_output)
        self.model_output.mkdir(parents=True, exist_ok=True)

        # Will load actual strategy architecture from run.py
        # This is just a placeholder
        self.actor = None
        self.critic = None

    def load_observations(self, market: str = None) -> List[dict]:
        """Load all observation files from disk."""
        print(f"📂 Loading observations from {self.data_dir}...")

        observations = []
        pattern = f"*/{market}/*.json.gz" if market else "*/*/*.json.gz"

        for filepath in self.data_dir.glob(pattern):
            try:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                    observations.extend(data['observations'])
            except Exception as e:
                print(f"⚠️  Failed to load {filepath}: {e}")

        print(f"✓ Loaded {len(observations)} observations")
        return observations

    def compute_hindsight_rewards(self, observations: List[dict]) -> List[Tuple]:
        """
        Label each observation with hindsight rewards.

        For each observation at time t:
        - Simulate what would have happened with BUY action
        - Simulate what would have happened with SELL action
        - Simulate what would have happened with HOLD action
        - Assign reward based on actual price movement

        Returns:
            List of (state, action, reward) tuples
        """
        print(f"🏷️  Computing hindsight labels...")

        labeled_data = []

        # Process in windows (need future data to compute rewards)
        window_size = 120  # 1 minute at 0.5s ticks
        lookahead = 60  # Check profit after 30 seconds

        for i in range(len(observations) - window_size):
            current_obs = observations[i]
            future_obs = observations[i + lookahead]

            state = np.array(current_obs['state_vector'])

            # Extract prices
            current_poly_yes = current_obs['polymarket']['yes_price']
            future_poly_yes = future_obs['polymarket']['yes_price']

            # Compute rewards for each action
            # BUY action: profit if price goes up
            buy_reward = (future_poly_yes - current_poly_yes) * 100  # in shares

            # SELL action: profit if price goes down
            sell_reward = (current_poly_yes - future_poly_yes) * 100

            # HOLD action: small negative (opportunity cost)
            hold_reward = -0.01

            # Find best action (hindsight optimal)
            actions_rewards = [
                (0, hold_reward),
                (1, buy_reward),
                (2, sell_reward),
            ]
            best_action, best_reward = max(actions_rewards, key=lambda x: x[1])

            # Only include if signal is strong (avoid noise)
            if abs(best_reward) > 0.5:  # Threshold
                labeled_data.append((state, best_action, best_reward))

        print(f"✓ Generated {len(labeled_data)} labeled examples")
        return labeled_data

    def train(self, labeled_data: List[Tuple], epochs=100):
        """
        Train policy using behavioral cloning + reward weighting.

        This is simpler than full offline RL:
        - Treat as supervised learning (state → action)
        - Weight examples by their reward
        - Use cross-entropy loss
        """
        print(f"\n🏋️  Training on {len(labeled_data)} examples...")

        # TODO: Load actual actor/critic from run.py
        # For now, placeholder

        # Convert to arrays
        states = np.array([x[0] for x in labeled_data])
        actions = np.array([x[1] for x in labeled_data])
        rewards = np.array([x[2] for x in labeled_data])

        # Normalize rewards to [0, 1] for weighting
        reward_weights = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)

        print(f"   States: {states.shape}")
        print(f"   Actions: {actions.shape}")
        print(f"   Rewards: min={rewards.min():.3f}, max={rewards.max():.3f}")

        # Training loop (placeholder)
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            states_shuffled = states[indices]
            actions_shuffled = actions[indices]
            weights_shuffled = reward_weights[indices]

            # Batch training
            # ... (implement actual PPO or BC updates)

            if epoch % 10 == 0:
                print(f"   Epoch {epoch}/{epochs} - Loss: ...")

        print("✓ Training complete")

        # Save model
        self.save_model()

    def save_model(self):
        """Save trained model."""
        print(f"\n💾 Saving model to {self.model_output}...")
        # TODO: Save actual actor/critic weights
        print("✓ Model saved")

    def run(self, market: str = None):
        """Full training pipeline."""
        # Load data
        observations = self.load_observations(market)

        if len(observations) < 1000:
            print("❌ Need at least 1000 observations to train")
            return

        # Label with hindsight
        labeled_data = self.compute_hindsight_rewards(observations)

        if len(labeled_data) < 100:
            print("❌ Not enough strong signals found")
            return

        # Train
        self.train(labeled_data)


def main():
    parser = argparse.ArgumentParser(description="Offline RL trainer")
    parser.add_argument("--data-dir", required=True, help="Directory with observation data")
    parser.add_argument("--output-model", required=True, help="Output model directory")
    parser.add_argument("--market", help="Filter to specific market (optional)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    args = parser.parse_args()

    trainer = OfflineTrainer(args.data_dir, args.output_model)
    trainer.run(market=args.market)


if __name__ == "__main__":
    main()
```

### Mac Mini Training Workflow

```bash
# 1. Sync data from S3 (run hourly via cron)
aws s3 sync s3://my-market-data/market-data/ ./data/

# 2. Train offline model
python offline_trainer.py \
    --data-dir ./data \
    --output-model ./models/offline_v1 \
    --market TRUMP2024 \
    --epochs 100

# 3. Evaluate on hold-out data
python evaluate_model.py \
    --model ./models/offline_v1 \
    --test-data ./data/test

# 4. Convert to PyTorch for deployment
python scripts/export_mlx_to_pytorch.py \
    --input ./models/offline_v1 \
    --output ./models/offline_v1_pytorch

# 5. Upload to S3 for production
aws s3 cp ./models/offline_v1_pytorch/ \
    s3://my-models/prod/ --recursive
```

### Automated Training (cron)

```bash
# crontab -e
# Sync data every hour
0 * * * * cd /Users/you/trading && aws s3 sync s3://my-market-data/market-data/ ./data/

# Train new model every 6 hours
0 */6 * * * cd /Users/you/trading && python offline_trainer.py --data-dir ./data --output-model ./models/latest

# Upload best model daily at 2am
0 2 * * * cd /Users/you/trading && ./scripts/deploy_best_model.sh
```

---

## Cost Analysis: Multiple Observers

### Scenario: Watching 10 Markets 24/7

```
Component                        | Cost/month | Total
─────────────────────────────────────────────────────
AWS t3.micro × 10 observers      | $7 each    | $70
S3 storage (100GB)               | $2.30      | $2.30
S3 data transfer (10GB out)      | $0.90      | $0.90
Mac Mini training (local)        | $0         | $0
─────────────────────────────────────────────────────
Total                            |            | $73/month
```

**Per market**: $7.30/month for 24/7 data collection

**Alternative: Single instance, multiple markets**
```python
# Run multiple observers on one instance
python observer.py --market TRUMP2024 &
python observer.py --market ETH10K &
python observer.py --market BTC100K &
# ... etc
```

Single t3.small (2 vCPU, 2GB): $15/month for 10-20 markets

**Even cheaper**: $1.50/month per market

---

## Advantages of This Approach

### 1. Cost Efficiency
```
Online training (AWS):   $38/mo × 10 markets = $380/mo
Offline training (Mac):  $15/mo (single collector) + $0 (Mac) = $15/mo
Savings: $365/month (96% reduction)
```

### 2. Training Speed
- Online: Limited to real-time (1 experience per 0.5s tick)
- Offline: Can train on 1 week of data in 10 minutes
- Can replay data multiple times with different hyperparameters

### 3. Experimentation
- Try different reward functions without redeploying
- A/B test multiple model architectures
- Backtest on historical data

### 4. Safety
- Train without risking capital
- Validate models thoroughly before live deployment
- Detect overfitting on hold-out data

---

## Alternative: No Training, Just Supervised Learning

**Even simpler approach**: Skip RL entirely, use supervised learning.

```python
# Label observations with simple rules
def label_action(current_price, future_price, spread):
    profit = future_price - current_price

    if profit > spread * 2:
        return "BUY"  # Strong signal
    elif profit < -spread * 2:
        return "SELL"
    else:
        return "HOLD"  # Noise

# Train classifier
from sklearn.ensemble import RandomForestClassifier

X = states  # (N, 18) features
y = actions  # (N,) labels

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Deploy
predictions = model.predict(live_states)
```

**Advantages**:
- No RL complexity
- Faster training
- Easier to debug
- Can run on t3.micro ($7/mo)

**Disadvantages**:
- Less adaptive
- Doesn't learn from long-term outcomes
- No exploration/exploitation balance

---

## Recommendation

**For cost-effective continuous training**:

1. **Phase 1: Data Collection** (Week 1-2)
   - Launch 1 t3.micro per market ($7/mo each)
   - Run observers 24/7
   - Collect at least 1 week of data per market
   - Cost: $15-70/mo depending on # of markets

2. **Phase 2: Offline Training** (Ongoing)
   - Mac Mini pulls data from S3 every hour
   - Train models every 6 hours on latest data
   - Evaluate on hold-out data
   - Cost: $0 (Mac runs 24/7 anyway)

3. **Phase 3: Deployment** (When ready)
   - Convert best model to PyTorch
   - Deploy to single t3.medium ($38/mo)
   - Run live trading on validated model
   - Continue collecting data for retraining

**Total cost for 5 markets**:
- Data collection: $35/mo (5 × t3.micro)
- Training: $0 (Mac)
- Live trading: $38/mo (1 × t3.medium)
- **Total: $73/month**

vs. online training approach: $190/month (5 × $38)
**Savings: $117/month (62% cheaper)**

---

## Next Steps

Want me to build the observer and offline trainer implementations?

I can create:
1. `observer.py` - Lightweight data collector
2. `offline_trainer.py` - Batch training on Mac
3. `deploy_observers.sh` - Script to launch multiple AWS instances
4. `sync_and_train.sh` - Automated workflow for Mac

Just let me know which markets you want to start observing!
