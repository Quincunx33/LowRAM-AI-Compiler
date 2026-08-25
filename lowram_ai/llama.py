"""A minimal Llama-family GGUF text-generation runtime.

The runtime is intentionally narrow and transparent. It supports the standard
unfused tensor names emitted for Llama-family models and the tensor types already
implemented by :mod:`lowram_ai.gguf`. It uses one token at a time and a fixed
float16 KV cache to keep peak working memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .gguf import GGUFReader
from .memory import enforce_rss_budget
from .native import NativeKernel
from .tokenizer import Tokenizer
from .transformer import apply_rope, rms_norm, silu


@dataclass(frozen=True)
class LlamaConfig:
    architecture: str
    vocabulary_size: int
    hidden_size: int
    intermediate_size: int
    layer_count: int
    attention_heads: int
    key_value_heads: int
    head_dimension: int
    context_length: int
    rope_theta: float
    rope_dimension: int
    rms_epsilon: float
    bos_token_id: int | None
    eos_token_id: int | None

    @classmethod
    def from_reader(cls, reader: GGUFReader, tokenizer: Tokenizer) -> "LlamaConfig":
        metadata = reader.metadata
        architecture = str(metadata.get("general.architecture", ""))
        if architecture != "llama":
            raise ValueError(
                f"unsupported architecture {architecture!r}; the current runtime targets 'llama'"
            )
        prefix = architecture + "."
        required = {
            "hidden_size": prefix + "embedding_length",
            "intermediate_size": prefix + "feed_forward_length",
            "layer_count": prefix + "block_count",
            "attention_heads": prefix + "attention.head_count",
        }
        missing = [label for label, key in required.items() if key not in metadata]
        if missing:
            raise ValueError(f"GGUF is missing Llama metadata: {', '.join(missing)}")
        hidden_size = int(metadata[required["hidden_size"]])
        attention_heads = int(metadata[required["attention_heads"]])
        key_value_heads = int(metadata.get(prefix + "attention.head_count_kv", attention_heads))
        if hidden_size % attention_heads != 0:
            raise ValueError("embedding length must be divisible by attention head count")
        if attention_heads % key_value_heads != 0:
            raise ValueError("attention head count must be divisible by KV head count")
        head_dimension = hidden_size // attention_heads
        rope_dimension = int(metadata.get(prefix + "rope.dimension_count", head_dimension))
        return cls(
            architecture=architecture,
            vocabulary_size=int(metadata.get("general.vocab_size", len(tokenizer.tokens))),
            hidden_size=hidden_size,
            intermediate_size=int(metadata[required["intermediate_size"]]),
            layer_count=int(metadata[required["layer_count"]]),
            attention_heads=attention_heads,
            key_value_heads=key_value_heads,
            head_dimension=head_dimension,
            context_length=int(metadata.get(prefix + "context_length", 2048)),
            rope_theta=float(metadata.get(prefix + "rope.freq_base", 10000.0)),
            rope_dimension=rope_dimension,
            rms_epsilon=float(metadata.get(prefix + "attention.layer_norm_rms_epsilon", 1e-5)),
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
        )


class MultiHeadKVCache:
    """Fixed-size float16 cache for grouped-query attention."""

    def __init__(self, max_tokens: int, key_value_heads: int, head_dimension: int):
        if max_tokens <= 0 or key_value_heads <= 0 or head_dimension <= 0:
            raise ValueError("KV cache dimensions must be positive")
        self.keys = np.empty((max_tokens, key_value_heads, head_dimension), dtype=np.float16)
        self.values = np.empty_like(self.keys)
        self.length = 0
        self.start = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.keys.nbytes + self.values.nbytes)

    @property
    def max_tokens(self) -> int:
        return self.keys.shape[0]

    def append(self, keys: np.ndarray, values: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        expected = self.keys.shape[1:]
        if keys.shape != expected or values.shape != expected:
            raise ValueError(f"KV tensors must have shape {expected}")
        if self.length < self.max_tokens:
            index = (self.start + self.length) % self.max_tokens
            self.length += 1
        else:
            index = self.start
            self.start = (self.start + 1) % self.max_tokens
        self.keys[index] = keys
        self.values[index] = values

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        indices = (self.start + np.arange(self.length)) % self.max_tokens
        return self.keys[indices].astype(np.float32), self.values[indices].astype(np.float32)


class LlamaRuntime:
    """Load and run a narrow standard Llama-family GGUF model."""

    def __init__(
        self,
        reader: GGUFReader,
        *,
        max_context_tokens: int | None = None,
        max_ram_mb: int | None = None,
    ):
        self.reader = reader
        self.native: NativeKernel | None = None
        self.max_ram_mb = max_ram_mb
        self.tokenizer = Tokenizer.from_metadata(reader.metadata)
        self.config = LlamaConfig.from_reader(reader, self.tokenizer)
        self.max_context_tokens = min(
            self.config.context_length,
            max_context_tokens or self.config.context_length,
        )
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        self._caches = [
            MultiHeadKVCache(
                self.max_context_tokens,
                self.config.key_value_heads,
                self.config.head_dimension,
            )
            for _ in range(self.config.layer_count)
        ]
        self.position = 0
        if self.max_ram_mb is not None:
            if self.max_ram_mb <= 0:
                raise ValueError("max_ram_mb must be positive")
            estimated = self.estimated_model_bytes
            if estimated > self.max_ram_mb * 1024 * 1024:
                raise MemoryError(
                    f"model budget exceeded: estimated {estimated / 1024 / 1024:.1f} MiB "
                    f"> {self.max_ram_mb} MiB"
                )
            enforce_rss_budget(self.max_ram_mb)
        self.native = NativeKernel.try_open(reader.path)

    @classmethod
    def open(
        cls,
        path: str,
        *,
        max_context_tokens: int | None = None,
        max_ram_mb: int | None = None,
    ) -> "LlamaRuntime":
        return cls(
            GGUFReader(path),
            max_context_tokens=max_context_tokens,
            max_ram_mb=max_ram_mb,
        )

    def close(self) -> None:
        if self.native is not None:
            self.native.close()
            self.native = None
        self.reader.close()

    def __enter__(self) -> "LlamaRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def cache_memory_bytes(self) -> int:
        return sum(cache.memory_bytes for cache in self._caches)

    @property
    def weight_memory_bytes(self) -> int:
        return sum(self.reader.tensor_nbytes(info) for info in self.reader.iter_tensors())

    @property
    def estimated_model_bytes(self) -> int:
        # Include a conservative scratch allowance for activations and logits.
        scratch = (self.config.hidden_size + self.config.intermediate_size) * 4 * 6
        return self.weight_memory_bytes + self.cache_memory_bytes + scratch

    def _tensor_name(self, *candidates: str) -> str:
        for candidate in candidates:
            if candidate in self.reader.tensors:
                return candidate
        raise ValueError(f"GGUF is missing required tensor; tried: {', '.join(candidates)}")

    def _linear(self, name: str, vector: np.ndarray) -> np.ndarray:
        info = self.reader.tensor_info(name)
        if info.type_name not in {"F32", "F16", "Q4_0", "Q4_1", "Q4_K", "Q5_0", "Q6_K", "Q8_0"}:
            raise NotImplementedError(
                f"tensor {name} uses {info.type_name}; add its decoder before running this model"
            )
        if self.native is not None:
            native_result = self.native.matvec(info, vector)
            if native_result is not None:
                return native_result
        return self.reader.tensor_matvec(info, vector)

    def _apply_rope_heads(self, values: np.ndarray, head_count: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        result = values.copy()
        rotated = min(self.config.rope_dimension, self.config.head_dimension)
        if rotated % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        for head in range(head_count):
            start = head * self.config.head_dimension
            result[start : start + rotated] = apply_rope(
                values[start : start + rotated], self.position, self.config.rope_theta
            )
        return result

    def _block(self, layer: int, hidden: np.ndarray) -> np.ndarray:
        prefix = f"blk.{layer}."
        attn_norm_name = self._tensor_name(prefix + "attn_norm.weight")
        q_name = self._tensor_name(prefix + "attn_q.weight")
        k_name = self._tensor_name(prefix + "attn_k.weight")
        v_name = self._tensor_name(prefix + "attn_v.weight")
        o_name = self._tensor_name(prefix + "attn_output.weight")
        ff_norm_name = self._tensor_name(prefix + "ffn_norm.weight")
        gate_name = self._tensor_name(prefix + "ffn_gate.weight")
        up_name = self._tensor_name(prefix + "ffn_up.weight")
        down_name = self._tensor_name(prefix + "ffn_down.weight")

        normalized = rms_norm(
            hidden,
            self.reader.decode_tensor(attn_norm_name).reshape(-1),
            self.config.rms_epsilon,
        )
        query = self._linear(q_name, normalized)
        key = self._linear(k_name, normalized)
        value = self._linear(v_name, normalized)
        query = self._apply_rope_heads(query, self.config.attention_heads)
        key = self._apply_rope_heads(key, self.config.key_value_heads)
        self._caches[layer].append(
            key.reshape(self.config.key_value_heads, self.config.head_dimension),
            value.reshape(self.config.key_value_heads, self.config.head_dimension),
        )
        keys, values = self._caches[layer].arrays()

        query_heads = query.reshape(self.config.attention_heads, self.config.head_dimension)
        attention_heads = np.empty_like(query_heads)
        group_size = self.config.attention_heads // self.config.key_value_heads
        for head in range(self.config.attention_heads):
            kv_head = head // group_size
            scores = keys[:, kv_head, :] @ query_heads[head]
            scores /= np.sqrt(float(self.config.head_dimension))
            scores -= np.max(scores)
            probabilities = np.exp(scores)
            probabilities /= np.sum(probabilities)
            attention_heads[head] = probabilities @ values[:, kv_head, :]
        attention = attention_heads.reshape(-1)
        residual = hidden + self._linear(o_name, attention)

        normalized = rms_norm(
            residual,
            self.reader.decode_tensor(ff_norm_name).reshape(-1),
            self.config.rms_epsilon,
        )
        gate = silu(self._linear(gate_name, normalized))
        up = self._linear(up_name, normalized)
        mlp = self._linear(down_name, gate * up)
        return residual + mlp

    def forward_token(self, token_id: int) -> np.ndarray:
        enforce_rss_budget(self.max_ram_mb)
        if self.position >= self.max_context_tokens:
            raise RuntimeError(
                "context limit reached; start a new session or increase max_context_tokens"
            )
        if not 0 <= token_id < self.tokenizer.vocab_size:
            raise ValueError(f"token id out of range: {token_id}")
        embedding_name = self._tensor_name("token_embd.weight")
        hidden = self.reader.tensor_vector(embedding_name, token_id)
        for layer in range(self.config.layer_count):
            hidden = self._block(layer, hidden)
        output_norm_name = self._tensor_name("output_norm.weight")
        output_name = self._tensor_name("output.weight", "token_embd.weight")
        hidden = rms_norm(
            hidden,
            self.reader.decode_tensor(output_norm_name).reshape(-1),
            self.config.rms_epsilon,
        )
        logits = self._linear(output_name, hidden)
        self.position += 1
        enforce_rss_budget(self.max_ram_mb)
        return logits

    def reset(self) -> None:
        self._caches = [
            MultiHeadKVCache(
                self.max_context_tokens,
                self.config.key_value_heads,
                self.config.head_dimension,
            )
            for _ in range(self.config.layer_count)
        ]
        self.position = 0

    @staticmethod
    def _sample_next(
        logits: np.ndarray,
        generated: list[int],
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        rng: np.random.Generator,
    ) -> int:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least 1")
        scores = np.asarray(logits, dtype=np.float32).copy()
        if repetition_penalty > 1:
            for token_id in set(generated[-128:]):
                if scores[token_id] < 0:
                    scores[token_id] *= repetition_penalty
                else:
                    scores[token_id] /= repetition_penalty
        if temperature == 0:
            return int(np.argmax(scores))
        scores /= temperature
        if top_k > 0 and top_k < scores.size:
            threshold = np.partition(scores, -top_k)[-top_k]
            scores[scores < threshold] = -np.inf
        finite = np.isfinite(scores)
        finite_scores = scores[finite]
        if finite_scores.size == 0:
            return int(np.argmax(logits))
        probabilities = np.zeros_like(scores, dtype=np.float64)
        shifted = finite_scores - np.max(finite_scores)
        probabilities[finite] = np.exp(shifted)
        probabilities /= probabilities.sum()
        if top_p < 1:
            sorted_ids = np.argsort(probabilities)[::-1]
            cumulative = np.cumsum(probabilities[sorted_ids])
            remove = cumulative > top_p
            if remove.any():
                remove[0] = False
                probabilities[sorted_ids[remove]] = 0
                probabilities /= probabilities.sum()
        return int(rng.choice(scores.size, p=probabilities))

    def generate_ids(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        add_bos: bool | None = None,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        seed: int | None = 0,
    ) -> list[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt_ids = self.tokenizer.encode(prompt, add_bos=add_bos)
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens")
        generated = list(prompt_ids)
        rng = np.random.default_rng(seed)
        logits = np.empty(self.tokenizer.vocab_size, dtype=np.float32)
        for token_id in prompt_ids:
            logits = self.forward_token(token_id)
        for _ in range(max_new_tokens):
            next_id = self._sample_next(
                logits,
                generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                rng=rng,
            )
            generated.append(next_id)
            if self.config.eos_token_id is not None and next_id == self.config.eos_token_id:
                break
            logits = self.forward_token(next_id)
        return generated

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        seed: int | None = 0,
    ) -> str:
        return self.tokenizer.decode(
            self.generate_ids(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
            )
        )
