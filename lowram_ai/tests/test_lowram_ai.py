import tempfile
import unittest
from pathlib import Path

import numpy as np

from lowram_ai.planner import build_budget_plan
from lowram_ai.quantized import QuantizedMatrix, quantize_npy_matrix


class LowRamAiTests(unittest.TestCase):
    def test_budget_plan_flags_large_model(self):
        plan = build_budget_plan(
            device_ram_mb=1024,
            parameters=3_000_000_000,
            layers=28,
            hidden_size=3072,
            requested_context_tokens=512,
            quantization_bits=4,
        )
        self.assertLess(plan.model_budget_mb, 1024)
        self.assertFalse(plan.fits_budget)
        self.assertTrue(any("smaller" in item.lower() for item in plan.recommendations))

    def test_budget_plan_accepts_small_profile(self):
        plan = build_budget_plan(
            device_ram_mb=1024,
            parameters=500_000_000,
            layers=16,
            hidden_size=2048,
            requested_context_tokens=256,
            quantization_bits=4,
        )
        self.assertTrue(plan.fits_budget)
        self.assertEqual(plan.batch_size, 1)

    def test_quantized_matrix_is_close_to_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            rng = np.random.default_rng(3)
            source = rng.normal(size=(17, 33)).astype(np.float32)
            source_path = tmp_path / "source.npy"
            output_path = tmp_path / "weights.lrq"
            np.save(source_path, source)
            info = quantize_npy_matrix(
                source_path, output_path, bits=4, group_size=8, chunk_rows=3
            )
            self.assertEqual(info["format"], "LRQ1")
            self.assertLess(output_path.stat().st_size, source_path.stat().st_size)

            vector = rng.normal(size=33).astype(np.float32)
            expected = source @ vector
            with QuantizedMatrix(output_path) as matrix:
                actual = matrix.matvec(vector)
            self.assertLess(float(np.mean(np.abs(expected - actual))), 0.5)

    def test_quantizer_handles_odd_column_count(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = np.arange(15, dtype=np.float32).reshape(3, 5)
            source_path = tmp_path / "odd.npy"
            output_path = tmp_path / "odd.lrq"
            np.save(source_path, source)
            quantize_npy_matrix(source_path, output_path, bits=4, group_size=4)
            with QuantizedMatrix(output_path) as matrix:
                result = matrix.matvec(np.ones(5, dtype=np.float32))
            self.assertEqual(result.shape, (3,))


if __name__ == "__main__":
    unittest.main()
