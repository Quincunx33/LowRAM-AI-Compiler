"""Optional ctypes bridge for the native low-RAM matvec kernel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np


_TYPE_IDS = {
    "F32": 0,
    "F16": 1,
    "Q5_0": 6,
    "Q8_0": 8,
    "Q4_K": 12,
    "Q6_K": 14,
}


class NativeKernel:
    def __init__(self, library_path: str | Path, model_path: str | Path):
        self.library_path = str(library_path)
        self.model_path = str(model_path)
        self._lib = ctypes.CDLL(self.library_path)
        self._lib.lowram_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        self._lib.lowram_open.restype = ctypes.c_void_p
        self._lib.lowram_close.argtypes = [ctypes.c_void_p]
        self._lib.lowram_close.restype = None
        self._lib.lowram_matvec.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.lowram_matvec.restype = ctypes.c_int
        error = ctypes.create_string_buffer(256)
        self._handle = self._lib.lowram_open(self.model_path.encode(), error, len(error))
        if not self._handle:
            raise OSError(error.value.decode(errors="replace") or "native model mapping failed")

    @classmethod
    def find_library(cls) -> Path | None:
        candidates = []
        override = os.environ.get("LOWRAM_KERNEL_PATH")
        if override:
            candidates.append(Path(override))
        root = Path(__file__).resolve().parent.parent
        candidates.extend(
            [
                root / "native" / "build" / "liblowram_kernel.so",
                root / "native" / "build" / "liblowram_kernel.dylib",
                root / "native" / "build" / "lowram_kernel.dll",
            ]
        )
        return next((path for path in candidates if path.exists()), None)

    @classmethod
    def try_open(cls, model_path: str | Path) -> "NativeKernel | None":
        library = cls.find_library()
        if library is None:
            return None
        try:
            return cls(library, model_path)
        except (OSError, ctypes.ArgumentError):
            return None

    def matvec(self, info: Any, vector: np.ndarray) -> np.ndarray | None:
        type_id = _TYPE_IDS.get(info.type_name)
        if type_id is None or len(info.shape) != 2:
            return None
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        output_width = info.element_count // info.shape[0]
        output = np.empty(output_width, dtype=np.float32)
        status = self._lib.lowram_matvec(
            self._handle,
            int(info.data_offset),
            type_id,
            int(info.shape[0]),
            int(output_width),
            vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status == 3:
            return None
        if status != 0:
            raise RuntimeError(f"native matvec failed with status {status}")
        return output

    def close(self) -> None:
        if self._handle:
            self._lib.lowram_close(self._handle)
            self._handle = None

    def __enter__(self) -> "NativeKernel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
