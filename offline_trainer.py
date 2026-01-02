#!/usr/bin/env python3
"""
Offline RL trainer - learns from collected observation data.

Runs on Mac Mini with MLX for fast training. Uses hindsight labeling
to train on collected data without needing actual trades.

Usage:
    # Train on all collected data
    python offline_trainer.py --data-dir ./data --output ./models/offline_v1

    # Train on specific market
    python offline_trainer.py --data-dir ./data --market TRUMP2024 --output ./models/trump_v1

    # Evaluate existing model
    python offline_trainer.py --evaluate ./models/offline_v1 --test-data ./data/test
"""
import argparse
import json
import gzip
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np

# Check for MLX (Mac only)
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available - this script must run on Mac with Apple Silicon")
    print("   Install: pip install mlx")
    sys.exit(1)

# Import RL strategy for architecture
from strategies.rl_mlx import Actor, Critic, RLStrategy


class OfflineTrainer:
    """
    Trains RL policy from offline data using hindsight labeling.

    Two training modes:
    1. Behavioral cloning: Learn to imitate optimal actions (simpler)
    2. Offline PPO: Full RL training on collected data (more complex)
    """

    def __init__(
        self,
        data_dir: str,
        model_output: str,
        method: str = "bc",  # "bc" or "ppo"
        input_dim: int = 18,
        hidden_size: int = 64,
        history_len: int = 5,
    ):
        self.data_dir = Path(data_dir)
        self.model_output = Path(model_output)
        self.model_output.mkdir(parents=True, exist_ok=True)
        self.method = method

        # Model architecture
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.history_len = history_len
        self.temporal_dim = 32

        # Create networks (use same architecture as online training)
        self.actor = Actor(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=3,
            history_len=history_len,
            temporal_dim=self.temporal_dim,
        )

        self.critic = Critic(
            input_dim=input_dim,
            hidden_size=96,  # Larger critic
            history_len=history_len,
            temporal_dim=self.temporal_dim,
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(learning_rate=3e-4)
        self.critic_optimizer = optim.Adam(learning_rate=1e-3)

        # Eval on init
        mx.eval(self.actor.parameters(), self.critic.parameters())

    def load_observations(self, market_filter: Optional[str] = None) -> Dict[str, List[dict]]:
        """
        Load all observation files from disk, grouped by market.

        Returns:
            dict: {cid: [observations]}
        """
        print(f"📂 Loading observations from {self.data_dir}...")

        observations_by_market = defaultdict(list)

        # Find all observation files
        pattern = "**/*.json.gz"
        files = list(self.data_dir.glob(pattern))

        if not files:
            print(f"❌ No observation files found in {self.data_dir}")
            return observations_by_market

        print(f"   Found {len(files)} observation files")

        for filepath in files:
            try:
                with gzip.open(filepath, "rt", encoding="utf-8") as f:
                    data = json.load(f)

                cid = data.get("cid")
                if not cid:
                    continue

                # Apply market filter if specified
                if market_filter and market_filter.lower() not in cid.lower():
                    continue

                observations = data.get("observations", [])
                observations_by_market[cid].extend(observations)

            except Exception as e:
                print(f"⚠️  Failed to load {filepath}: {e}")

        # Summary
        total_obs = sum(len(obs) for obs in observations_by_market.values())
        print(f"✓ Loaded {total_obs} observations across {len(observations_by_market)} markets")

        for cid, obs in observations_by_market.items():
            print(f"   • {cid[:8]}: {len(obs)} observations")

        return observations_by_market

    def extract_features(self, obs: dict) -> np.ndarray:
        """
        Extract 18-dimensional feature vector from observation.

        Simplified version - doesn't have all features from live trading,
        but includes the key ones for prediction.
        """
        state = obs["state"]

        # Helper to clamp values
        def clamp(x, min_val=-1.0, max_val=1.0):
            return max(min_val, min(max_val, float(x)))

        prob = state.get("prob", 0.5)
        spread = state.get("spread", 0.0)
        spread_pct = spread / max(0.01, prob) if prob > 0 else 0.0

        features = np.array([
            # Ultra-short momentum (3) - we don't have this from observations
            0.0,  # returns_1m - placeholder
            0.0,  # returns_5m - placeholder
            0.0,  # returns_10m - placeholder

            # Order flow (4)
            clamp(state.get("imbalance_l1", 0.0)),
            clamp(state.get("imbalance_l5", 0.0)),
            0.0,  # trade_flow_imbalance - not in observations
            0.0,  # cvd_acceleration - not in observations

            # Microstructure (3)
            clamp(spread_pct * 20),
            0.0,  # trade_intensity - not in observations
            0.0,  # large_trade_flag - not in observations

            # Volatility (2)
            0.0,  # realized_vol_5m - placeholder
            0.0,  # vol_expansion - placeholder

            # Position (4) - no position in observation mode
            0.0,  # has_position
            0.0,  # position_side
            0.0,  # position_pnl
            clamp(state.get("time_remaining", 0.5)),

            # Regime (2)
            clamp(state.get("binance_change", 0.0) * 50),
            prob,  # Current probability
        ], dtype=np.float32)

        return features

    def compute_hindsight_labels(
        self,
        observations: List[dict],
        lookahead_ticks: int = 60,  # 30s at 0.5s ticks
    ) -> List[Tuple[np.ndarray, int, float]]:
        """
        Label each observation with hindsight optimal action.

        For each observation at time t:
        - Look at price at t + lookahead
        - Determine which action would have been profitable
        - Assign reward based on price movement

        Returns:
            List of (features, action, reward) tuples
        """
        print(f"🏷️  Computing hindsight labels (lookahead={lookahead_ticks} ticks)...")

        labeled_data = []

        for i in range(len(observations) - lookahead_ticks):
            current_obs = observations[i]
            future_obs = observations[i + lookahead_ticks]

            # Extract features
            features = self.extract_features(current_obs)

            # Get price change
            current_prob = current_obs["state"].get("prob", 0.5)
            future_prob = future_obs["state"].get("prob", 0.5)

            price_change = future_prob - current_prob

            # Determine optimal action and reward
            # BUY (1): profit if price goes up
            buy_reward = price_change * 100  # Scale to reasonable range

            # SELL (2): profit if price goes down
            sell_reward = -price_change * 100

            # HOLD (0): small negative (opportunity cost)
            hold_reward = -0.1

            # Choose best action
            actions_rewards = [
                (0, hold_reward),
                (1, buy_reward),
                (2, sell_reward),
            ]
            best_action, best_reward = max(actions_rewards, key=lambda x: x[1])

            # Only include if signal is strong enough
            threshold = 0.5  # Minimum reward to include
            if abs(best_reward) > threshold:
                labeled_data.append((features, best_action, best_reward))

        print(f"✓ Generated {len(labeled_data)} labeled examples")

        # Show distribution of actions
        action_counts = defaultdict(int)
        for _, action, _ in labeled_data:
            action_counts[action] += 1

        print(f"   Action distribution:")
        print(f"      HOLD: {action_counts[0]} ({action_counts[0]/len(labeled_data)*100:.1f}%)")
        print(f"      BUY:  {action_counts[1]} ({action_counts[1]/len(labeled_data)*100:.1f}%)")
        print(f"      SELL: {action_counts[2]} ({action_counts[2]/len(labeled_data)*100:.1f}%)")

        return labeled_data

    def train_behavioral_cloning(
        self,
        labeled_data: List[Tuple[np.ndarray, int, float]],
        epochs: int = 50,
        batch_size: int = 64,
    ):
        """
        Train using behavioral cloning (supervised learning).

        Simpler than full RL - just learns to predict optimal actions.
        Weight examples by their reward (higher reward = more important).
        """
        print(f"\n🏋️  Training with Behavioral Cloning")
        print(f"   Method: Weighted cross-entropy loss")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Examples: {len(labeled_data)}")

        # Convert to arrays
        states = np.array([x[0] for x in labeled_data])
        actions = np.array([x[1] for x in labeled_data])
        rewards = np.array([x[2] for x in labeled_data])

        # Normalize rewards to [0, 1] for weighting
        reward_weights = np.abs(rewards)
        reward_weights = (reward_weights - reward_weights.min()) / (reward_weights.max() - reward_weights.min() + 1e-8)

        # Create temporal states (padded history since we don't have full history)
        # In real deployment, we'd maintain proper history
        temporal_states = np.zeros((len(states), self.history_len * self.input_dim), dtype=np.float32)

        print(f"\n   Starting training...")

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(states))
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0

            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_temporal = temporal_states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_weights = reward_weights[batch_indices]

                # Convert to MLX arrays
                states_mx = mx.array(batch_states)
                temporal_mx = mx.array(batch_temporal)
                actions_mx = mx.array(batch_actions)
                weights_mx = mx.array(batch_weights)

                # Forward pass and loss
                def loss_fn(actor):
                    probs = actor(states_mx, temporal_mx)

                    # Cross-entropy loss
                    log_probs = mx.log(probs + 1e-8)
                    # Select log probs for chosen actions
                    action_log_probs = mx.take_along_axis(
                        log_probs,
                        actions_mx.reshape(-1, 1),
                        axis=1
                    ).squeeze()

                    # Weighted negative log likelihood
                    loss = -mx.mean(weights_mx * action_log_probs)

                    return loss

                # Compute gradients and update
                loss_and_grad = nn.value_and_grad(self.actor, loss_fn)
                loss, grads = loss_and_grad(self.actor)

                self.actor_optimizer.update(self.actor, grads)
                mx.eval(self.actor.parameters())

                # Track metrics
                epoch_loss += float(loss)
                num_batches += 1

                # Compute accuracy
                probs = self.actor(states_mx, temporal_mx)
                mx.eval(probs)
                predicted_actions = np.array(mx.argmax(probs, axis=1))
                accuracy = np.mean(predicted_actions == batch_actions)
                epoch_accuracy += accuracy

            # Epoch summary
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches

            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"   Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.3f}")

        print(f"\n✓ Training complete")

    def save_model(self):
        """Save trained model weights."""
        print(f"\n💾 Saving model to {self.model_output}...")

        # Save actor
        actor_weights = self.actor.parameters()
        mx.savez(str(self.model_output / "actor.npz"), **dict(actor_weights))

        # Save critic
        critic_weights = self.critic.parameters()
        mx.savez(str(self.model_output / "critic.npz"), **dict(critic_weights))

        # Save metadata
        metadata = {
            "method": self.method,
            "input_dim": self.input_dim,
            "hidden_size": self.hidden_size,
            "history_len": self.history_len,
            "temporal_dim": self.temporal_dim,
            "created_at": datetime.utcnow().isoformat(),
        }

        with open(self.model_output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print("✓ Model saved:")
        print(f"   • actor.npz")
        print(f"   • critic.npz")
        print(f"   • metadata.json")

    def run(self, market_filter: Optional[str] = None, epochs: int = 50):
        """Full training pipeline."""
        print("\n" + "=" * 60)
        print("Offline RL Training")
        print("=" * 60)

        # Load observations
        observations_by_market = self.load_observations(market_filter)

        if not observations_by_market:
            print("❌ No observations found")
            return

        # Combine all observations
        all_observations = []
        for cid, obs in observations_by_market.items():
            all_observations.extend(obs)

        print(f"\nTotal observations: {len(all_observations)}")

        if len(all_observations) < 1000:
            print("⚠️  Warning: Less than 1000 observations - results may be poor")
            print("   Collect more data for better training")

        # Label with hindsight
        labeled_data = self.compute_hindsight_labels(all_observations)

        if len(labeled_data) < 100:
            print("❌ Not enough strong signals found")
            print("   Try:")
            print("   - Collecting more data")
            print("   - Lowering the reward threshold")
            return

        # Train
        if self.method == "bc":
            self.train_behavioral_cloning(labeled_data, epochs=epochs)
        else:
            print(f"❌ Training method '{self.method}' not implemented yet")
            print("   Use --method bc for now")
            return

        # Save
        self.save_model()

        print("\n" + "=" * 60)
        print("✓ Training Complete")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"1. Convert to PyTorch for AWS deployment:")
        print(f"   python scripts/export_mlx_to_pytorch.py \\")
        print(f"     --input {self.model_output} \\")
        print(f"     --output {self.model_output}_pytorch")
        print(f"2. Test the model:")
        print(f"   python run.py rl --load {self.model_output} --size 10")


def main():
    parser = argparse.ArgumentParser(
        description="Offline RL trainer for collected market data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on all data
  python offline_trainer.py --data-dir ./data --output ./models/offline_v1

  # Train on specific market
  python offline_trainer.py --data-dir ./data --market TRUMP --output ./models/trump_v1

  # More epochs for better convergence
  python offline_trainer.py --data-dir ./data --output ./models/offline_v2 --epochs 100
        """
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing observation data",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--market",
        help="Filter to specific market (optional)",
    )
    parser.add_argument(
        "--method",
        choices=["bc", "ppo"],
        default="bc",
        help="Training method: bc=behavioral cloning (default), ppo=offline PPO",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)",
    )

    args = parser.parse_args()

    # Create trainer
    trainer = OfflineTrainer(
        data_dir=args.data_dir,
        model_output=args.output,
        method=args.method,
    )

    # Run training
    trainer.run(market_filter=args.market, epochs=args.epochs)


if __name__ == "__main__":
    main()
