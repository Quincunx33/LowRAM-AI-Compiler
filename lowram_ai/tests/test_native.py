from __future__ import annotations

import ctypes
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np


class NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        library = root / "native" / "build" / "liblowram_kernel.so"
        if not library.exists():
            raise unittest.SkipTest("native library is not built")
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.lowram_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        cls.lib.lowram_open.restype = ctypes.c_void_p
        cls.lib.lowram_close.argtypes = [ctypes.c_void_p]
        cls.lib.lowram_matvec.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
            ctypes.c_uint64, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        cls.lib.lowram_matvec.restype = ctypes.c_int

    def _matvec(self, payload: bytes, type_id: int, expected: float) -> None:
        with tempfile.NamedTemporaryFile() as model:
            model.write(payload)
            model.flush()
            error = ctypes.create_string_buffer(256)
            handle = self.lib.lowram_open(model.name.encode(), error, len(error))
            self.assertTrue(handle, error.value.decode())
            try:
                vector = np.ones(32, dtype=np.float32)
                output = np.zeros(1, dtype=np.float32)
                status = self.lib.lowram_matvec(
                    handle, 0, type_id, 32, 1,
                    vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
                self.assertEqual(status, 0)
                self.assertAlmostEqual(float(output[0]), expected, places=5)
            finally:
                self.lib.lowram_close(handle)

    def test_q4_0_dot_product(self):
        # d=1.0, low nibbles=9 -> +1, high nibbles=7 -> -1; sum is zero.
        self._matvec(struct.pack("<H", 0x3C00) + bytes([0x79]) * 16, 2, 0.0)

    def test_q4_1_dot_product(self):
        # d=1.0, m=2.0, low q=1 -> 3 and high q=2 -> 4; 32 values sum to 112.
        payload = struct.pack("<HH", 0x3C00, 0x4000) + bytes([0x21]) * 16
        self._matvec(payload, 3, 112.0)

    def test_bf16_matvec(self):
        # 32 copies of bfloat16 1.5 multiplied by a vector of ones.
        self._matvec(struct.pack("<H", 0x3FC0) * 32, 30, 48.0)


if __name__ == "__main__":
    unittest.main()
