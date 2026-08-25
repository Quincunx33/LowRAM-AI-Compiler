import tempfile
import unittest
from pathlib import Path

import numpy as np

from lowram_ai.quantized import QuantizedMatrix, quantize_npy_matrix
from lowram_ai.transformer import KVCache, QuantizedTransformerBlock, apply_rope, rms_norm


def make_quantized(directory: Path, name: str, shape: tuple[int, int], rng: np.random.Generator) -> QuantizedMatrix:
    source_path = directory / f"{name}.npy"
    output_path = directory / f"{name}.lrq"
    np.save(source_path, rng.normal(0, 0.12, size=shape).astype(np.float32))
    quantize_npy_matrix(source_path, output_path, bits=4, group_size=8, chunk_rows=2)
    return QuantizedMatrix(output_path)


class TransformerTests(unittest.TestCase):
    def test_primitives(self):
        vector = np.arange(8, dtype=np.float32)
        normalized = rms_norm(vector, np.ones(8, dtype=np.float32))
        self.assertEqual(normalized.shape, (8,))
        np.testing.assert_allclose(apply_rope(vector, 0), vector)

    def test_fixed_capacity_cache(self):
        cache = KVCache(max_tokens=2, width=4)
        for value in range(3):
            cache.append(np.full(4, value), np.full(4, value + 10))
        keys, values = cache.arrays()
        self.assertEqual(cache.length, 2)
        np.testing.assert_allclose(keys[:, 0], [1, 2])
        np.testing.assert_allclose(values[:, 0], [11, 12])

    def test_quantized_transformer_block_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            rng = np.random.default_rng(9)
            hidden, ffn = 8, 16
            matrices = {
                "q": make_quantized(path, "q", (hidden, hidden), rng),
                "k": make_quantized(path, "k", (hidden, hidden), rng),
                "v": make_quantized(path, "v", (hidden, hidden), rng),
                "o": make_quantized(path, "o", (hidden, hidden), rng),
                "gate": make_quantized(path, "gate", (ffn, hidden), rng),
                "up": make_quantized(path, "up", (ffn, hidden), rng),
                "down": make_quantized(path, "down", (hidden, ffn), rng),
            }
            try:
                block = QuantizedTransformerBlock(
                    q_proj=matrices["q"],
                    k_proj=matrices["k"],
                    v_proj=matrices["v"],
                    o_proj=matrices["o"],
                    gate_proj=matrices["gate"],
                    up_proj=matrices["up"],
                    down_proj=matrices["down"],
                    input_norm=np.ones(hidden, dtype=np.float32),
                    post_attention_norm=np.ones(hidden, dtype=np.float32),
                    max_context_tokens=2,
                )
                first = block.forward(np.ones(hidden, dtype=np.float32))
                second = block.forward(np.zeros(hidden, dtype=np.float32))
                third = block.forward(np.ones(hidden, dtype=np.float32))
                self.assertEqual(first.shape, (hidden,))
                self.assertEqual(second.shape, (hidden,))
                self.assertEqual(third.shape, (hidden,))
                self.assertEqual(block.cache.length, 2)
                self.assertEqual(block.position, 3)
                self.assertLess(block.cache_memory_bytes, 2 * 2 * hidden * 2 + 1)
            finally:
                for matrix in matrices.values():
                    matrix.close()


if __name__ == "__main__":
    unittest.main()
