import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lowram_ai.gguf import GGUFReader, align_offset, inspect_gguf


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def build_fixture(path: Path) -> tuple[np.ndarray, np.ndarray]:
    alignment = 32
    f32_values = np.array([1.0, -2.0, 3.5, 4.25], dtype=np.float32)
    q_values = np.concatenate((np.arange(-8, 8), np.arange(7, -9, -1))).astype(np.int8)
    scale = np.float16(0.5)
    nibbles = ((q_values[:16] + 8) | ((q_values[16:] + 8) << 4)).astype(np.uint8)
    q4_block = scale.tobytes() + nibbles.tobytes()

    metadata = b"".join(
        (
            gguf_string("general.architecture")
            + struct.pack("<I", 8)
            + gguf_string("llama"),
            gguf_string("general.alignment")
            + struct.pack("<I", 4)
            + struct.pack("<I", alignment),
            gguf_string("test.tags")
            + struct.pack("<I", 9)
            + struct.pack("<I", 8)
            + struct.pack("<Q", 2)
            + gguf_string("lowram")
            + gguf_string("mvp"),
            gguf_string("test.enabled") + struct.pack("<I", 7) + struct.pack("<?", True),
        )
    )
    descriptors = b"".join(
        (
            gguf_string("test.f32")
            + struct.pack("<I", 1)
            + struct.pack("<Q", 4)
            + struct.pack("<I", 0)
            + struct.pack("<Q", 0),
            gguf_string("test.q4")
            + struct.pack("<I", 1)
            + struct.pack("<Q", 32)
            + struct.pack("<I", 2)
            + struct.pack("<Q", 32),
        )
    )
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", 2)
        + struct.pack("<Q", 4)
        + metadata
        + descriptors
    )
    tensor_data_start = align_offset(len(header), alignment)
    payload = (
        b"\x00" * (tensor_data_start - len(header))
        + f32_values.tobytes()
        + b"\x00" * (32 - len(f32_values.tobytes()))
        + q4_block
    )
    path.write_bytes(header + payload)
    return f32_values, q_values.astype(np.float32) * np.float32(scale)


class GGUFReaderTests(unittest.TestCase):
    def test_metadata_and_tensor_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            expected_f32, expected_q4 = build_fixture(path)
            with GGUFReader(path) as reader:
                self.assertEqual(reader.version, 3)
                self.assertEqual(reader.alignment, 32)
                self.assertEqual(reader.metadata["general.architecture"], "llama")
                self.assertEqual(reader.metadata["test.tags"], ["lowram", "mvp"])
                self.assertTrue(reader.metadata["test.enabled"])
                self.assertEqual(reader.tensor_info("test.q4").type_name, "Q4_0")
                np.testing.assert_allclose(reader.decode_tensor("test.f32"), expected_f32)
                np.testing.assert_allclose(reader.decode_tensor("test.q4"), expected_q4)

    def test_inspection_does_not_decode_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            build_fixture(path)
            result = inspect_gguf(path)
            self.assertEqual(result["metadata"]["general.architecture"], "llama")
            self.assertEqual([item["type"] for item in result["tensors"]], ["F32", "Q4_0"])
            self.assertEqual(result["alignment"], 32)


if __name__ == "__main__":
    unittest.main()


def build_q4k_fixture(path: Path) -> np.ndarray:
    metadata = gguf_string("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", 32)
    packed_scales = bytes([1] * 12)
    q_bytes = bytes([0x21] * 128)
    block = np.float16(1.0).tobytes() + np.float16(0.5).tobytes() + packed_scales + q_bytes
    descriptor = (
        gguf_string("test.q4k")
        + struct.pack("<I", 1)
        + struct.pack("<Q", 256)
        + struct.pack("<I", 12)
        + struct.pack("<Q", 0)
    )
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 1) + metadata + descriptor
    data_start = align_offset(len(header), 32)
    path.write_bytes(header + b"\x00" * (data_start - len(header)) + block)
    expected = np.concatenate(
        [
            np.full(32, 0.5, dtype=np.float32),
            np.full(32, 1.5, dtype=np.float32),
            np.full(32, 0.5, dtype=np.float32),
            np.full(32, 1.5, dtype=np.float32),
            np.full(32, 1.0, dtype=np.float32),
            np.full(32, 2.0, dtype=np.float32),
            np.full(32, 1.0, dtype=np.float32),
            np.full(32, 2.0, dtype=np.float32),
        ]
    )
    return expected


class Q4KTests(unittest.TestCase):
    def test_q4k_block_decodes_with_correct_span_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q4k.gguf"
            expected = build_q4k_fixture(path)
            with GGUFReader(path) as reader:
                self.assertEqual(reader.tensor_nbytes("test.q4k"), 144)
                np.testing.assert_allclose(reader.decode_tensor("test.q4k"), expected)
