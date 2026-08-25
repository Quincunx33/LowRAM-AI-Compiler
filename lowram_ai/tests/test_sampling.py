import unittest

import numpy as np

from lowram_ai.llama import LlamaRuntime


class SamplingTests(unittest.TestCase):
    def test_greedy_is_deterministic(self):
        logits = np.array([0.1, 4.0, 2.0], dtype=np.float32)
        result = LlamaRuntime._sample_next(
            logits, [], temperature=0, top_k=40, top_p=0.9,
            repetition_penalty=1, rng=np.random.default_rng(1)
        )
        self.assertEqual(result, 1)

    def test_seeded_sampling_is_repeatable(self):
        logits = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        kwargs = dict(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
        first = LlamaRuntime._sample_next(logits, [], rng=np.random.default_rng(7), **kwargs)
        second = LlamaRuntime._sample_next(logits, [], rng=np.random.default_rng(7), **kwargs)
        self.assertEqual(first, second)

    def test_top_k_and_repetition_penalty(self):
        logits = np.array([2.0, 1.9, 1.0], dtype=np.float32)
        result = LlamaRuntime._sample_next(
            logits, [0], temperature=0, top_k=0, top_p=1.0,
            repetition_penalty=2.0, rng=np.random.default_rng(1)
        )
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
