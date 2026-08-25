"""Memory-conscious transformer primitives for the next runtime milestone.

This module is an inference foundation rather than a complete model loader. It
accepts the existing memory-mapped QuantizedMatrix objects for linear layers and
keeps the autoregressive KV cache in float16 with a hard token limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantized import QuantizedMatrix


def rms_norm(vector: np.ndarray, weight: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if vector.ndim != 1 or weight.shape != vector.shape:
        raise ValueError("vector and weight must be one-dimensional with equal shape")
    mean_square = float(np.mean(vector * vector))
    return vector * (1.0 / np.sqrt(mean_square + epsilon)) * weight


def silu(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    # Clip only for numerical safety on very small devices without changing the
    # normal operating range of transformer activations.
    return vector / (1.0 + np.exp(-np.clip(vector, -40.0, 40.0)))


def apply_rope(vector: np.ndarray, position: int, theta: float = 10000.0) -> np.ndarray:
    """Apply rotary position embedding to an even-width single-token vector."""
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or vector.size % 2 != 0:
        raise ValueError("RoPE vector must be one-dimensional with an even width")
    result = vector.copy()
    pair_count = vector.size // 2
    for pair in range(pair_count):
        frequency = theta ** (-2.0 * pair / vector.size)
        angle = position * frequency
        cosine, sine = np.cos(angle), np.sin(angle)
        left, right = vector[2 * pair], vector[2 * pair + 1]
        result[2 * pair] = left * cosine - right * sine
        result[2 * pair + 1] = left * sine + right * cosine
    return result


@dataclass
class KVCache:
    """A fixed-capacity float16 key/value cache for one attention head."""

    max_tokens: int
    width: int

    def __post_init__(self) -> None:
        if self.max_tokens <= 0 or self.width <= 0:
            raise ValueError("max_tokens and width must be positive")
        self.keys = np.empty((self.max_tokens, self.width), dtype=np.float16)
        self.values = np.empty((self.max_tokens, self.width), dtype=np.float16)
        self.length = 0
        self.start = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        key = np.asarray(key, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        if key.shape != (self.width,) or value.shape != (self.width,):
            raise ValueError("key and value must match cache width")
        if self.length < self.max_tokens:
            index = (self.start + self.length) % self.max_tokens
            self.length += 1
        else:
            index = self.start
            self.start = (self.start + 1) % self.max_tokens
        self.keys[index] = key
        self.values[index] = value

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.length == 0:
            return self.keys[:0], self.values[:0]
        indices = (self.start + np.arange(self.length)) % self.max_tokens
        return self.keys[indices].astype(np.float32), self.values[indices].astype(np.float32)


class QuantizedTransformerBlock:
    """Single-head transformer block backed by memory-mapped linear matrices."""

    def __init__(
        self,
        *,
        q_proj: QuantizedMatrix,
        k_proj: QuantizedMatrix,
        v_proj: QuantizedMatrix,
        o_proj: QuantizedMatrix,
        gate_proj: QuantizedMatrix,
        up_proj: QuantizedMatrix,
        down_proj: QuantizedMatrix,
        input_norm: np.ndarray,
        post_attention_norm: np.ndarray,
        max_context_tokens: int = 256,
        rope_theta: float = 10000.0,
    ):
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.input_norm = np.asarray(input_norm, dtype=np.float32)
        self.post_attention_norm = np.asarray(post_attention_norm, dtype=np.float32)
        self.rope_theta = rope_theta
        self.position = 0
        self.cache = KVCache(max_context_tokens, q_proj.rows)
        hidden_size = self.input_norm.size
        if self.input_norm.shape != (hidden_size,) or self.post_attention_norm.shape != (hidden_size,):
            raise ValueError("normalization weights must match hidden size")
        for matrix in (q_proj, k_proj, v_proj, gate_proj, up_proj):
            if matrix.cols != hidden_size:
                raise ValueError("projection input width must match hidden size")
        if o_proj.cols != q_proj.rows or down_proj.cols != gate_proj.rows:
            raise ValueError("output projection widths do not match their inputs")
        if o_proj.rows != hidden_size or down_proj.rows != hidden_size:
            raise ValueError("output projection rows must match hidden size")

    @property
    def cache_memory_bytes(self) -> int:
        return self.cache.memory_bytes

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.shape != self.input_norm.shape:
            raise ValueError("hidden state shape does not match block")

        normalized = rms_norm(hidden, self.input_norm)
        query = apply_rope(self.q_proj.matvec(normalized), self.position, self.rope_theta)
        key = apply_rope(self.k_proj.matvec(normalized), self.position, self.rope_theta)
        value = self.v_proj.matvec(normalized)
        self.cache.append(key, value)
        keys, values = self.cache.arrays()

        scores = (keys @ query) / np.sqrt(float(query.size))
        scores -= np.max(scores)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities)
        attention = probabilities @ values
        residual = hidden + self.o_proj.matvec(attention)

        normalized = rms_norm(residual, self.post_attention_norm)
        gate = silu(self.gate_proj.matvec(normalized))
        up = self.up_proj.matvec(normalized)
        mlp = self.down_proj.matvec(gate * up)
        self.position += 1
        return residual + mlp
