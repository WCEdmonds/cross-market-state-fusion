#!/usr/bin/env python3
"""
Transformer-based PPO strategy with self-attention temporal encoding.

Improvements over base RL strategy:
- Self-attention learns which past states are most predictive
- Position encoding captures time-relative patterns
- Longer lookback window (20 states vs 5)
- Better temporal pattern recognition (spikes, reversals, trends)

Expected performance gain: +10-20% over base RL strategy
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from collections import deque
from typing import List, Dict, Optional
from dataclasses import dataclass

from .base import Strategy, MarketState, Action
from .rl_mlx import Experience, RLStrategy


class TransformerTemporalEncoder(nn.Module):
    """
    Transformer-based temporal encoder using self-attention.

    Advantages over simple MLP encoder:
    1. Self-attention: Learns which past moments matter most
    2. Position encoding: Captures time-relative patterns
    3. Multi-head attention: Can attend to multiple patterns simultaneously
    4. Scalable: Can handle longer sequences (20+ states)

    Architecture:
        Input: (batch, seq_len, input_dim)
        ↓ Linear projection to d_model
        ↓ + Positional encoding
        ↓ TransformerEncoder layers (self-attention + FFN)
        ↓ Pool over sequence (take last state or mean)
        ↓ Linear projection to output_dim
        Output: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 18,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 20,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Input projection: map input_dim to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (sinusoidal)
        self.register_buffer("pos_encoding", self._create_positional_encoding(seq_len, d_model))

        # Transformer encoder layers
        self.transformer_layers = [
            TransformerEncoderLayer(
                dims=d_model,
                num_heads=n_heads,
                mlp_dims=d_model * 4,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]

        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)

        # Layer norm
        self.ln = nn.LayerNorm(d_model)

    def _create_positional_encoding(self, seq_len: int, d_model: int) -> mx.array:
        """
        Create sinusoidal positional encodings.

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        This allows the model to learn to attend by relative positions.
        """
        position = mx.arange(seq_len, dtype=mx.float32).reshape(-1, 1)
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * (-np.log(10000.0) / d_model)
        )

        pe = mx.zeros((seq_len, d_model))

        # Even indices: sine
        pe[:, 0::2] = mx.sin(position * div_term)

        # Odd indices: cosine
        if d_model % 2 == 0:
            pe[:, 1::2] = mx.cos(position * div_term)
        else:
            pe[:, 1::2] = mx.cos(position * div_term[:-1])

        return pe

    def register_buffer(self, name: str, value: mx.array):
        """Register a buffer (non-trainable parameter)."""
        setattr(self, f"_{name}", value)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) sequence of states

        Returns:
            Temporal features: (batch, output_dim)
        """
        # Project input to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        # Broadcast pos_encoding across batch dimension
        x = x + self._pos_encoding  # (batch, seq_len, d_model)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Apply layer norm
        x = self.ln(x)

        # Pool over sequence - take last state's representation
        # Alternative: mean pooling → x = mx.mean(x, axis=1)
        x = x[:, -1, :]  # (batch, d_model)

        # Project to output dimension
        out = self.output_proj(x)  # (batch, output_dim)

        return out


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Architecture:
        Input
        ↓
        Multi-Head Self-Attention
        ↓
        + Residual & LayerNorm
        ↓
        Feed-Forward Network (MLP)
        ↓
        + Residual & LayerNorm
        ↓
        Output
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiHeadAttention(dims, num_heads)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(dims, mlp_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims, dims),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(dims)
        self.ln2 = nn.LayerNorm(dims)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, dims)

        Returns:
            Output: (batch, seq_len, dims)
        """
        # Self-attention with residual connection
        attn_out = self.attention(x, x, x)  # Q, K, V all from x (self-attention)
        attn_out = self.dropout(attn_out)
        x = self.ln1(x + attn_out)

        # Feed-forward with residual connection
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)

        return x


class TransformerActor(nn.Module):
    """
    Policy network with Transformer-based temporal encoding.

    Architecture:
        Current state (18) + Transformer temporal features (32) = 50
        → 64 → LayerNorm → tanh → 64 → LayerNorm → tanh → 3 (softmax)
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_size: int = 64,
        output_dim: int = 3,
        seq_len: int = 20,
        temporal_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        # Transformer temporal encoder
        self.temporal_encoder = TransformerTemporalEncoder(
            input_dim=input_dim,
            d_model=64,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            output_dim=temporal_dim,
        )

        # Policy head
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def __call__(self, current_state: mx.array, state_sequence: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            current_state: (batch, 18) current features
            state_sequence: (batch, seq_len, 18) sequence of past states

        Returns:
            Action probabilities: (batch, 3)
        """
        # Encode temporal patterns with Transformer
        temporal_features = self.temporal_encoder(state_sequence)

        # Combine current state + temporal features
        combined = mx.concatenate([current_state, temporal_features], axis=-1)

        # Policy network
        h = mx.tanh(self.ln1(self.fc1(combined)))
        h = mx.tanh(self.ln2(self.fc2(h)))
        logits = self.fc3(h)
        probs = mx.softmax(logits, axis=-1)

        return probs


class TransformerCritic(nn.Module):
    """
    Value network with Transformer-based temporal encoding.

    Architecture:
        Current state (18) + Transformer temporal features (32) = 50
        → 96 → LayerNorm → tanh → 96 → LayerNorm → tanh → 1
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_size: int = 96,
        seq_len: int = 20,
        temporal_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()

        # Transformer temporal encoder
        self.temporal_encoder = TransformerTemporalEncoder(
            input_dim=input_dim,
            d_model=64,
            n_heads=n_heads,
            n_layers=n_layers,
            seq_len=seq_len,
            output_dim=temporal_dim,
        )

        # Value head
        combined_dim = input_dim + temporal_dim
        self.fc1 = nn.Linear(combined_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def __call__(self, current_state: mx.array, state_sequence: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            current_state: (batch, 18) current features
            state_sequence: (batch, seq_len, 18) sequence of past states

        Returns:
            Value estimate: (batch, 1)
        """
        # Encode temporal patterns
        temporal_features = self.temporal_encoder(state_sequence)

        # Combine
        combined = mx.concatenate([current_state, temporal_features], axis=-1)

        # Value network
        h = mx.tanh(self.ln1(self.fc1(combined)))
        h = mx.tanh(self.ln2(self.fc2(h)))
        value = self.fc3(h)

        return value


class TransformerRLStrategy(RLStrategy):
    """
    RL strategy with Transformer-based temporal encoding.

    Key improvements over base RLStrategy:
    - Self-attention learns which past moments matter
    - Longer lookback window (20 states vs 5)
    - Position encoding captures time-relative patterns
    - Multi-head attention for multiple pattern types

    Expected performance: +10-20% over base RL
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_size: int = 64,
        critic_hidden_size: int = 96,
        seq_len: int = 20,  # 20 states = 10 seconds at 0.5s ticks
        temporal_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        **kwargs,
    ):
        # Initialize base class WITHOUT creating networks
        # (we'll override with Transformer versions)
        super().__init__(
            input_dim=input_dim,
            hidden_size=hidden_size,
            critic_hidden_size=critic_hidden_size,
            history_len=seq_len,  # Map seq_len to history_len
            temporal_dim=temporal_dim,
            **kwargs,
        )

        # Override with Transformer networks
        self.actor = TransformerActor(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_dim=3,
            seq_len=seq_len,
            temporal_dim=temporal_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        self.critic = TransformerCritic(
            input_dim=input_dim,
            hidden_size=critic_hidden_size,
            seq_len=seq_len,
            temporal_dim=temporal_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        # Re-create optimizers with new networks
        self.actor_optimizer = optim.Adam(learning_rate=self.lr_actor)
        self.critic_optimizer = optim.Adam(learning_rate=self.lr_critic)

        # Eval on init
        mx.eval(self.actor.parameters(), self.critic.parameters())

        # Update history management for longer sequences
        self.seq_len = seq_len
        self._state_history: Dict[str, deque] = {}

    def _get_temporal_state(self, asset: str, current_features: np.ndarray) -> np.ndarray:
        """
        Get sequence of states for Transformer.

        Returns: (seq_len, input_dim) array
        """
        if asset not in self._state_history:
            self._state_history[asset] = deque(maxlen=self.seq_len)

        history = self._state_history[asset]

        # Add current state
        history.append(current_features.copy())

        # Pad if not enough history
        if len(history) < self.seq_len:
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (self.seq_len - len(history))
            sequence = np.stack(padding + list(history))
        else:
            sequence = np.stack(list(history))

        return sequence  # (seq_len, input_dim)

    def act(self, state: MarketState) -> Action:
        """Select action using Transformer policy."""
        features = state.to_features()

        # Get state sequence
        state_sequence = self._get_temporal_state(state.asset, features)

        # Convert to MLX arrays
        # Current state: (1, 18)
        features_mx = mx.array(features.reshape(1, -1))

        # State sequence: (1, seq_len, 18)
        sequence_mx = mx.array(state_sequence.reshape(1, self.seq_len, -1))

        # Get action probabilities and value
        probs = self.actor(features_mx, sequence_mx)
        value = self.critic(features_mx, sequence_mx)

        # Eval
        mx.eval(probs, value)

        probs_np = np.array(probs[0])
        value_np = float(np.array(value[0, 0]))

        if self.training:
            # Sample from distribution
            action_idx = np.random.choice(self.output_dim, p=probs_np)
        else:
            # Greedy
            action_idx = int(np.argmax(probs_np))

        # Store for experience collection
        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np
        self._last_temporal_state = state_sequence.flatten()  # Flatten for compatibility

        return Action(action_idx)

    def __str__(self):
        return f"TransformerRL(seq_len={self.seq_len}, n_heads=4, n_layers=2)"
