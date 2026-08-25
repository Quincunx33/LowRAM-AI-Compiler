import unittest

from lowram_ai.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer.from_metadata(
            {
                "tokenizer.ggml.tokens": ["<s>", "</s>", "▁hello", "▁world", "!", "<0x3F>"],
                "tokenizer.ggml.bos_token_id": 0,
                "tokenizer.ggml.eos_token_id": 1,
                "tokenizer.ggml.unknown_token_id": 5,
                "tokenizer.ggml.model": "llama",
            }
        )

    def test_encode_and_decode(self):
        ids = self.tokenizer.encode("hello world!")
        self.assertEqual(ids, [0, 2, 3, 4])
        self.assertEqual(self.tokenizer.decode(ids), "hello world!")

    def test_unknown_character_uses_unk(self):
        ids = self.tokenizer.encode("?")
        self.assertEqual(ids, [0, 5])

    def test_special_tokens_are_not_decoded_as_text(self):
        self.assertEqual(self.tokenizer.decode([0, 2, 1]), "hello")


if __name__ == "__main__":
    unittest.main()
