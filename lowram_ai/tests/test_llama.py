import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lowram_ai.llama import LlamaRuntime


def s(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def align(value: int, boundary: int = 32) -> int:
    return value + (boundary - value % boundary) % boundary


def metadata_entry(key: str, type_id: int, value: bytes) -> bytes:
    return s(key) + struct.pack("<I", type_id) + value


def make_llama_fixture(path: Path) -> None:
    vocab = ["<s>", "</s>", "▁A", "▁B", "▁C"]
    metadata = b"".join(
        [
            metadata_entry("general.architecture", 8, s("llama")),
            metadata_entry("general.vocab_size", 4, struct.pack("<I", len(vocab))),
            metadata_entry("llama.embedding_length", 4, struct.pack("<I", 4)),
            metadata_entry("llama.feed_forward_length", 4, struct.pack("<I", 8)),
            metadata_entry("llama.block_count", 4, struct.pack("<I", 1)),
            metadata_entry("llama.attention.head_count", 4, struct.pack("<I", 1)),
            metadata_entry("llama.attention.head_count_kv", 4, struct.pack("<I", 1)),
            metadata_entry("llama.context_length", 4, struct.pack("<I", 8)),
            metadata_entry("llama.rope.dimension_count", 4, struct.pack("<I", 4)),
            metadata_entry("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-5)),
            metadata_entry("tokenizer.ggml.model", 8, s("llama")),
            metadata_entry(
                "tokenizer.ggml.tokens",
                9,
                struct.pack("<I", 8) + struct.pack("<Q", len(vocab)) + b"".join(s(item) for item in vocab),
            ),
            metadata_entry("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
            metadata_entry("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 1)),
        ]
    )

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    rng = np.random.default_rng(2)
    embedding = np.zeros((4, 5), dtype=np.float32)
    embedding[:, 2] = 1.0
    embedding[:, 3] = 0.5
    embedding[:, 4] = -0.5
    tensors.append(("token_embd.weight", (4, 5), 0, embedding.T.astype(np.float32).tobytes()))
    tensors.append(("blk.0.attn_norm.weight", (4,), 0, np.ones(4, dtype=np.float32).tobytes()))
    for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
        tensors.append((f"blk.0.{name}.weight", (4, 4), 0, np.zeros((4, 4), dtype=np.float32).tobytes()))
    tensors.append(("blk.0.ffn_norm.weight", (4,), 0, np.ones(4, dtype=np.float32).tobytes()))
    for name in ("ffn_gate", "ffn_up"):
        tensors.append((f"blk.0.{name}.weight", (4, 8), 0, np.zeros((4, 8), dtype=np.float32).tobytes()))
    tensors.append(("blk.0.ffn_down.weight", (8, 4), 0, np.zeros((8, 4), dtype=np.float32).tobytes()))
    tensors.append(("output_norm.weight", (4,), 0, np.ones(4, dtype=np.float32).tobytes()))
    output = np.zeros((4, 5), dtype=np.float32)
    output[:, 2] = 2.0
    tensors.append(("output.weight", (4, 5), 0, output.T.astype(np.float32).tobytes()))

    descriptors = []
    relative_offset = 0
    for name, shape, type_id, payload in tensors:
        relative_offset = align(relative_offset)
        descriptors.append(
            s(name)
            + struct.pack("<I", len(shape))
            + b"".join(struct.pack("<Q", dimension) for dimension in shape)
            + struct.pack("<I", type_id)
            + struct.pack("<Q", relative_offset)
        )
        relative_offset += len(payload)

    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", 14)
        + metadata
        + b"".join(descriptors)
    )
    data_start = align(len(header))
    payload = bytearray(b"\x00" * (data_start - len(header) + relative_offset))
    for (_, _, _, tensor_payload), descriptor in zip(tensors, descriptors):
        # Descriptor offsets are recovered by reading the final uint64 field.
        tensor_offset = struct.unpack("<Q", descriptor[-8:])[0]
        payload[data_start - len(header) + tensor_offset : data_start - len(header) + tensor_offset + len(tensor_payload)] = tensor_payload
    path.write_bytes(header + payload)


class LlamaRuntimeTests(unittest.TestCase):
    def test_tiny_llama_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            make_llama_fixture(path)
            with LlamaRuntime.open(str(path), max_context_tokens=4, max_ram_mb=1024) as runtime:
                self.assertEqual(runtime.config.hidden_size, 4)
                self.assertEqual(runtime.config.layer_count, 1)
                self.assertEqual(runtime.tokenizer.encode("A"), [0, 2])
                generated = runtime.generate("A", max_new_tokens=2)
                self.assertEqual(generated, "A A A")
                self.assertLess(runtime.cache_memory_bytes, 1024)
                self.assertLess(runtime.estimated_model_bytes, 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
