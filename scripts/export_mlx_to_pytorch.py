#!/usr/bin/env python3
"""
Export MLX RL model to PyTorch format for cloud deployment.

Usage:
    python scripts/export_mlx_to_pytorch.py --input rl_model --output rl_model_pytorch
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Try to import MLX (only available on Apple Silicon)
try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("⚠️  MLX not available - this script must run on Mac with Apple Silicon")
    sys.exit(1)

# Try to import PyTorch
try:
    import torch
    import torch.nn as torch_nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available")
    print("Install: pip install torch")
    sys.exit(1)


class PyTorchActor(torch_nn.Module):
    """PyTorch version of MLX Actor network."""

    def __init__(self, input_dim=18, hidden=64, output_dim=3,
                 history_len=5, temporal_dim=32):
        super().__init__()

        # Temporal encoder (matches MLX TemporalEncoder)
        self.temporal_fc1 = torch_nn.Linear(input_dim * history_len, 64)
        self.temporal_ln1 = torch_nn.LayerNorm(64, eps=1e-5)
        self.temporal_fc2 = torch_nn.Linear(64, temporal_dim)
        self.temporal_ln2 = torch_nn.LayerNorm(temporal_dim, eps=1e-5)

        # Actor network
        combined_dim = input_dim + temporal_dim
        self.fc1 = torch_nn.Linear(combined_dim, hidden)
        self.ln1 = torch_nn.LayerNorm(hidden, eps=1e-5)
        self.fc2 = torch_nn.Linear(hidden, hidden)
        self.ln2 = torch_nn.LayerNorm(hidden, eps=1e-5)
        self.fc3 = torch_nn.Linear(hidden, output_dim)

    def forward(self, current_state, temporal_state):
        """
        Forward pass.

        Args:
            current_state: (batch, 18) tensor
            temporal_state: (batch, 90) tensor (18 * 5)

        Returns:
            action_probs: (batch, 3) tensor
        """
        # Temporal encoding
        h_temp = torch.tanh(self.temporal_ln1(self.temporal_fc1(temporal_state)))
        h_temp = torch.tanh(self.temporal_ln2(self.temporal_fc2(h_temp)))

        # Combine current + temporal
        combined = torch.cat([current_state, h_temp], dim=-1)

        # Actor forward
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)

        # Softmax to get probabilities
        return torch.softmax(logits, dim=-1)


class PyTorchCritic(torch_nn.Module):
    """PyTorch version of MLX Critic network."""

    def __init__(self, input_dim=18, hidden=64, history_len=5, temporal_dim=32):
        super().__init__()

        # Temporal encoder (shared architecture with Actor)
        self.temporal_fc1 = torch_nn.Linear(input_dim * history_len, 64)
        self.temporal_ln1 = torch_nn.LayerNorm(64, eps=1e-5)
        self.temporal_fc2 = torch_nn.Linear(64, temporal_dim)
        self.temporal_ln2 = torch_nn.LayerNorm(temporal_dim, eps=1e-5)

        # Critic network
        combined_dim = input_dim + temporal_dim
        self.fc1 = torch_nn.Linear(combined_dim, hidden)
        self.ln1 = torch_nn.LayerNorm(hidden, eps=1e-5)
        self.fc2 = torch_nn.Linear(hidden, hidden)
        self.ln2 = torch_nn.LayerNorm(hidden, eps=1e-5)
        self.fc3 = torch_nn.Linear(hidden, 1)  # Value output

    def forward(self, current_state, temporal_state):
        """
        Forward pass.

        Args:
            current_state: (batch, 18) tensor
            temporal_state: (batch, 90) tensor

        Returns:
            value: (batch, 1) tensor
        """
        # Temporal encoding
        h_temp = torch.tanh(self.temporal_ln1(self.temporal_fc1(temporal_state)))
        h_temp = torch.tanh(self.temporal_ln2(self.temporal_fc2(h_temp)))

        # Combine
        combined = torch.cat([current_state, h_temp], dim=-1)

        # Critic forward
        h = torch.tanh(self.ln1(self.fc1(combined)))
        h = torch.tanh(self.ln2(self.fc2(h)))
        value = self.fc3(h)

        return value


def load_mlx_model(model_dir):
    """Load MLX model weights from directory."""
    actor_path = Path(model_dir) / "actor.npz"
    critic_path = Path(model_dir) / "critic.npz"

    if not actor_path.exists():
        raise FileNotFoundError(f"Actor model not found: {actor_path}")
    if not critic_path.exists():
        raise FileNotFoundError(f"Critic model not found: {critic_path}")

    # Load weights
    actor_weights = mx.load(str(actor_path))
    critic_weights = mx.load(str(critic_path))

    # Convert to NumPy
    actor_np = {k: np.array(v) for k, v in actor_weights.items()}
    critic_np = {k: np.array(v) for k, v in critic_weights.items()}

    return actor_np, critic_np


def map_mlx_to_pytorch(mlx_weights, network_type="actor"):
    """
    Map MLX weight keys to PyTorch state_dict keys.

    MLX uses: "layers.0.weight", "layers.0.bias"
    PyTorch uses: "fc1.weight", "fc1.bias", "ln1.weight", "ln1.bias"
    """
    pytorch_state = {}

    # Layer mapping (depends on exact MLX architecture)
    # This is a template - adjust based on actual saved keys

    print(f"\nMLX {network_type} keys:")
    for k in mlx_weights.keys():
        print(f"  {k}: {mlx_weights[k].shape}")

    # You'll need to adjust this mapping based on actual saved keys
    # Example mapping (may need adjustment):
    layer_mapping = {
        "layers.temporal_encoder.fc1.weight": "temporal_fc1.weight",
        "layers.temporal_encoder.fc1.bias": "temporal_fc1.bias",
        "layers.temporal_encoder.ln1.weight": "temporal_ln1.weight",
        "layers.temporal_encoder.ln1.bias": "temporal_ln1.bias",
        "layers.temporal_encoder.fc2.weight": "temporal_fc2.weight",
        "layers.temporal_encoder.fc2.bias": "temporal_fc2.bias",
        "layers.temporal_encoder.ln2.weight": "temporal_ln2.weight",
        "layers.temporal_encoder.ln2.bias": "temporal_ln2.bias",
        "layers.fc1.weight": "fc1.weight",
        "layers.fc1.bias": "fc1.bias",
        "layers.ln1.weight": "ln1.weight",
        "layers.ln1.bias": "ln1.bias",
        "layers.fc2.weight": "fc2.weight",
        "layers.fc2.bias": "fc2.bias",
        "layers.ln2.weight": "ln2.weight",
        "layers.ln2.bias": "ln2.bias",
        "layers.fc3.weight": "fc3.weight",
        "layers.fc3.bias": "fc3.bias",
    }

    # Try direct mapping first
    for mlx_key, pytorch_key in layer_mapping.items():
        if mlx_key in mlx_weights:
            weight = mlx_weights[mlx_key]
            # Convert to torch tensor
            pytorch_state[pytorch_key] = torch.from_numpy(weight).float()
        else:
            # Try alternative key formats
            # MLX might save as "temporal_encoder.fc1.weight" directly
            simple_key = mlx_key.replace("layers.", "")
            if simple_key in mlx_weights:
                weight = mlx_weights[simple_key]
                pytorch_state[pytorch_key] = torch.from_numpy(weight).float()

    # If mapping failed, try automatic mapping (same keys)
    if not pytorch_state:
        print("⚠️  Layer mapping failed, trying direct key mapping...")
        for k, v in mlx_weights.items():
            pytorch_state[k] = torch.from_numpy(v).float()

    print(f"\nPyTorch {network_type} state_dict keys:")
    for k in pytorch_state.keys():
        print(f"  {k}: {pytorch_state[k].shape}")

    return pytorch_state


def convert_model(input_dir, output_dir):
    """Convert MLX model to PyTorch."""
    print("=" * 60)
    print("MLX → PyTorch Model Conversion")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load MLX weights
    print(f"\n1. Loading MLX model from {input_dir}...")
    actor_np, critic_np = load_mlx_model(input_dir)
    print(f"   ✓ Loaded actor: {len(actor_np)} parameters")
    print(f"   ✓ Loaded critic: {len(critic_np)} parameters")

    # Create PyTorch models
    print("\n2. Creating PyTorch models...")
    pytorch_actor = PyTorchActor()
    pytorch_critic = PyTorchCritic()
    print("   ✓ Actor architecture created")
    print("   ✓ Critic architecture created")

    # Map weights
    print("\n3. Mapping weights...")
    actor_state = map_mlx_to_pytorch(actor_np, "actor")
    critic_state = map_mlx_to_pytorch(critic_np, "critic")

    # Load into PyTorch models
    print("\n4. Loading weights into PyTorch models...")
    try:
        pytorch_actor.load_state_dict(actor_state, strict=False)
        print("   ✓ Actor weights loaded")
    except Exception as e:
        print(f"   ⚠️  Actor weight loading failed: {e}")
        print("   You may need to adjust the layer mapping in this script")

    try:
        pytorch_critic.load_state_dict(critic_state, strict=False)
        print("   ✓ Critic weights loaded")
    except Exception as e:
        print(f"   ⚠️  Critic weight loading failed: {e}")
        print("   You may need to adjust the layer mapping in this script")

    # Save PyTorch models
    print(f"\n5. Saving PyTorch models to {output_dir}...")
    torch.save(pytorch_actor.state_dict(), f"{output_dir}/actor.pt")
    torch.save(pytorch_critic.state_dict(), f"{output_dir}/critic.pt")

    # Also save full models (includes architecture)
    torch.save(pytorch_actor, f"{output_dir}/actor_full.pt")
    torch.save(pytorch_critic, f"{output_dir}/critic_full.pt")

    print("   ✓ Saved actor.pt")
    print("   ✓ Saved critic.pt")
    print("   ✓ Saved actor_full.pt")
    print("   ✓ Saved critic_full.pt")

    print("\n" + "=" * 60)
    print("✓ Conversion complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Test inference parity:")
    print(f"   python scripts/test_model_parity.py --mlx {input_dir} --pytorch {output_dir}")
    print(f"2. Upload to cloud:")
    print(f"   aws s3 cp {output_dir}/ s3://my-bucket/models/ --recursive")
    print(f"3. Deploy on AWS EC2")


def main():
    parser = argparse.ArgumentParser(description="Convert MLX model to PyTorch")
    parser.add_argument("--input", required=True, help="Input MLX model directory")
    parser.add_argument("--output", required=True, help="Output PyTorch model directory")
    args = parser.parse_args()

    convert_model(args.input, args.output)


if __name__ == "__main__":
    main()
