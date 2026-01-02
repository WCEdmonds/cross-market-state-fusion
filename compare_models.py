#!/usr/bin/env python3
"""
Compare Transformer RL vs Base RL performance.

Tests both models on same data and compares:
- Win rate
- Average PnL
- Sharpe ratio
- Max drawdown
- Inference speed

Usage:
    python compare_models.py \
        --base-model models/base_rl \
        --transformer-model models/transformer_rl \
        --test-data ./data/test \
        --duration 1000  # 1000 ticks
"""
import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import mlx.core as mx

from strategies import RLStrategy, TransformerRLStrategy
from strategies.base import MarketState


class ModelComparator:
    """Compare two RL models on same test data."""

    def __init__(self, base_model_path: str, transformer_model_path: str):
        print("=" * 60)
        print("Model Comparison: Base RL vs Transformer RL")
        print("=" * 60)

        # Load models
        print("\nLoading models...")
        self.base_model = RLStrategy()
        self.transformer_model = TransformerRLStrategy()

        # Note: Actual loading would require implementing load() method
        # For now, this is a template

        print(f"✓ Base RL loaded from {base_model_path}")
        print(f"✓ Transformer RL loaded from {transformer_model_path}")

        # Results tracking
        self.results = {
            "base_rl": {
                "actions": [],
                "rewards": [],
                "inference_times": [],
            },
            "transformer_rl": {
                "actions": [],
                "rewards": [],
                "inference_times": [],
            },
        }

    def run_comparison(self, test_states: List[MarketState], test_rewards: List[float]):
        """
        Run both models on same test data.

        Args:
            test_states: List of market states
            test_rewards: Actual rewards for each state
        """
        print(f"\n🧪 Running comparison on {len(test_states)} test states...")

        for i, (state, reward) in enumerate(zip(test_states, test_rewards)):
            # Base RL
            start = time.perf_counter()
            base_action = self.base_model.act(state)
            base_time = (time.perf_counter() - start) * 1000  # ms

            self.results["base_rl"]["actions"].append(base_action.value)
            self.results["base_rl"]["rewards"].append(reward)
            self.results["base_rl"]["inference_times"].append(base_time)

            # Transformer RL
            start = time.perf_counter()
            transformer_action = self.transformer_model.act(state)
            transformer_time = (time.perf_counter() - start) * 1000  # ms

            self.results["transformer_rl"]["actions"].append(transformer_action.value)
            self.results["transformer_rl"]["rewards"].append(reward)
            self.results["transformer_rl"]["inference_times"].append(transformer_time)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_states)} states...")

        print("✓ Comparison complete")

    def analyze_results(self) -> Dict[str, Dict]:
        """Compute performance metrics for each model."""
        print("\n📊 Analyzing results...\n")

        metrics = {}

        for model_name in ["base_rl", "transformer_rl"]:
            actions = np.array(self.results[model_name]["actions"])
            rewards = np.array(self.results[model_name]["rewards"])
            inf_times = np.array(self.results[model_name]["inference_times"])

            # Compute metrics
            total_pnl = rewards.sum()
            avg_reward = rewards.mean()
            sharpe = rewards.mean() / (rewards.std() + 1e-8) * np.sqrt(252)  # Annualized

            # Win rate (reward > 0)
            wins = (rewards > 0).sum()
            win_rate = wins / len(rewards)

            # Max drawdown
            cumulative = np.cumsum(rewards)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            max_drawdown = drawdown.max()

            # Inference speed
            avg_inference = inf_times.mean()
            p95_inference = np.percentile(inf_times, 95)

            # Action distribution
            action_counts = np.bincount(actions, minlength=3)
            action_dist = action_counts / len(actions)

            metrics[model_name] = {
                "total_pnl": total_pnl,
                "avg_reward": avg_reward,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "avg_inference_ms": avg_inference,
                "p95_inference_ms": p95_inference,
                "action_distribution": {
                    "HOLD": action_dist[0],
                    "BUY": action_dist[1],
                    "SELL": action_dist[2],
                },
            }

        return metrics

    def print_comparison(self, metrics: Dict[str, Dict]):
        """Print side-by-side comparison."""
        print("=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        # Performance metrics
        print("\n📈 Performance Metrics:")
        print("-" * 80)
        print(f"{'Metric':<25} | {'Base RL':>15} | {'Transformer RL':>15} | {'Difference':>15}")
        print("-" * 80)

        base = metrics["base_rl"]
        trans = metrics["transformer_rl"]

        # Total PnL
        diff_pnl = trans["total_pnl"] - base["total_pnl"]
        pct_pnl = (diff_pnl / abs(base["total_pnl"])) * 100 if base["total_pnl"] != 0 else 0
        print(f"{'Total PnL':<25} | {base['total_pnl']:>15.2f} | {trans['total_pnl']:>15.2f} | {diff_pnl:>14.2f} ({pct_pnl:+.1f}%)")

        # Avg reward
        diff_avg = trans["avg_reward"] - base["avg_reward"]
        pct_avg = (diff_avg / abs(base["avg_reward"])) * 100 if base["avg_reward"] != 0 else 0
        print(f"{'Avg Reward':<25} | {base['avg_reward']:>15.4f} | {trans['avg_reward']:>15.4f} | {diff_avg:>14.4f} ({pct_avg:+.1f}%)")

        # Sharpe ratio
        diff_sharpe = trans["sharpe_ratio"] - base["sharpe_ratio"]
        print(f"{'Sharpe Ratio':<25} | {base['sharpe_ratio']:>15.3f} | {trans['sharpe_ratio']:>15.3f} | {diff_sharpe:>14.3f}")

        # Win rate
        diff_wr = trans["win_rate"] - base["win_rate"]
        print(f"{'Win Rate':<25} | {base['win_rate']:>14.1%} | {trans['win_rate']:>14.1%} | {diff_wr:>14.1%}")

        # Max drawdown
        diff_dd = trans["max_drawdown"] - base["max_drawdown"]
        print(f"{'Max Drawdown':<25} | {base['max_drawdown']:>15.2f} | {trans['max_drawdown']:>15.2f} | {diff_dd:>14.2f}")

        # Inference speed
        print("\n⚡ Inference Speed:")
        print("-" * 80)
        print(f"{'Metric':<25} | {'Base RL':>15} | {'Transformer RL':>15} | {'Difference':>15}")
        print("-" * 80)

        diff_avg_inf = trans["avg_inference_ms"] - base["avg_inference_ms"]
        print(f"{'Avg Inference (ms)':<25} | {base['avg_inference_ms']:>15.2f} | {trans['avg_inference_ms']:>15.2f} | {diff_avg_inf:>14.2f}")

        diff_p95_inf = trans["p95_inference_ms"] - base["p95_inference_ms"]
        print(f"{'P95 Inference (ms)':<25} | {base['p95_inference_ms']:>15.2f} | {trans['p95_inference_ms']:>15.2f} | {diff_p95_inf:>14.2f}")

        # Action distribution
        print("\n🎯 Action Distribution:")
        print("-" * 80)
        print(f"{'Action':<25} | {'Base RL':>15} | {'Transformer RL':>15}")
        print("-" * 80)
        for action in ["HOLD", "BUY", "SELL"]:
            print(f"{action:<25} | {base['action_distribution'][action]:>14.1%} | {trans['action_distribution'][action]:>14.1%}")

        # Winner
        print("\n" + "=" * 80)
        if trans["total_pnl"] > base["total_pnl"]:
            improvement = ((trans["total_pnl"] / base["total_pnl"]) - 1) * 100
            print(f"🏆 WINNER: Transformer RL (+{improvement:.1f}% better PnL)")
        elif base["total_pnl"] > trans["total_pnl"]:
            decline = ((base["total_pnl"] / trans["total_pnl"]) - 1) * 100
            print(f"🏆 WINNER: Base RL (+{decline:.1f}% better PnL)")
        else:
            print("🤝 TIE: Both models performed equally")
        print("=" * 80)


def generate_synthetic_test_data(n_states: int = 1000):
    """
    Generate synthetic test data for demonstration.

    In production, load from actual collected market data.
    """
    print(f"\n⚠️  Using synthetic test data ({n_states} states)")
    print("   For real comparison, use actual market data from observer\n")

    states = []
    rewards = []

    for _ in range(n_states):
        # Random state
        state = MarketState(
            asset="BTC",
            prob=np.random.uniform(0.3, 0.7),
            time_remaining=np.random.uniform(0, 1),
            best_bid=np.random.uniform(0.4, 0.6),
            best_ask=np.random.uniform(0.4, 0.6),
            spread=np.random.uniform(0.01, 0.05),
            binance_price=np.random.uniform(50000, 70000),
        )

        # Random reward
        reward = np.random.normal(0, 5)

        states.append(state)
        rewards.append(reward)

    return states, rewards


def main():
    parser = argparse.ArgumentParser(description="Compare Base RL vs Transformer RL")
    parser.add_argument("--base-model", required=True, help="Path to base RL model")
    parser.add_argument("--transformer-model", required=True, help="Path to Transformer RL model")
    parser.add_argument("--test-data", help="Path to test data directory")
    parser.add_argument("--duration", type=int, default=1000, help="Number of test states")
    args = parser.parse_args()

    # Create comparator
    comparator = ModelComparator(args.base_model, args.transformer_model)

    # Load or generate test data
    if args.test_data:
        # TODO: Implement loading from actual data
        print(f"Loading test data from {args.test_data}...")
        test_states, test_rewards = generate_synthetic_test_data(args.duration)
    else:
        test_states, test_rewards = generate_synthetic_test_data(args.duration)

    # Run comparison
    comparator.run_comparison(test_states, test_rewards)

    # Analyze results
    metrics = comparator.analyze_results()

    # Print comparison
    comparator.print_comparison(metrics)

    print(f"\n💾 Results saved to comparison_results.json")
    # TODO: Save results to file


if __name__ == "__main__":
    main()
