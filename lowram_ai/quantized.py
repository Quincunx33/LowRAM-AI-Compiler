"""Streaming quantization and memory-mapped inference for dense matrices.

This MVP uses a small, documented binary format (LRQ1). It is intentionally
narrow: dense float32/float16 matrices are quantized row-wise and can then be
used for matrix-vector products without materializing the full float matrix.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Iterator

import numpy as np


MAGIC = b"LRQ1"
HEADER_PREFIX = struct.Struct("<4sI")
SUPPORTED_BITS = (4, 8)


def _pack_signed(values: np.ndarray, bits: int) -> bytes:
    if bits == 8:
        return values.astype(np.int8, copy=False).tobytes(order="C")
    # Two signed 4-bit values per byte, stored as unsigned nibbles.
    unsigned = values.astype(np.int16, copy=False) + 8
    if unsigned.size % 2:
        unsigned = np.concatenate((unsigned, np.zeros(1, dtype=np.int16)))
    packed = (unsigned[0::2] & 0x0F) | ((unsigned[1::2] & 0x0F) << 4)
    return packed.astype(np.uint8, copy=False).tobytes(order="C")


def _unpack_signed(raw: np.ndarray, count: int, bits: int) -> np.ndarray:
    if bits == 8:
        return raw[:count].view(np.int8).astype(np.float32)
    values = np.empty(raw.size * 2, dtype=np.float32)
    values[0::2] = (raw & 0x0F).astype(np.float32) - 8
    values[1::2] = ((raw >> 4) & 0x0F).astype(np.float32) - 8
    return values[:count]


def _iter_rows(matrix: np.ndarray, chunk_rows: int) -> Iterator[np.ndarray]:
    for start in range(0, matrix.shape[0], chunk_rows):
        yield np.asarray(matrix[start : start + chunk_rows], dtype=np.float32)


def quantize_npy_matrix(
    input_path: str | Path,
    output_path: str | Path,
    *,
    bits: int = 4,
    group_size: int = 64,
    chunk_rows: int = 32,
) -> dict[str, object]:
    """Quantize a 2-D .npy matrix in bounded-memory chunks.

    Each row is split into groups. A float32 scale is stored for every group;
    zero-point is symmetric zero, making the runtime simple and predictable.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if group_size <= 0 or chunk_rows <= 0:
        raise ValueError("group_size and chunk_rows must be positive")

    source = np.load(input_path, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError("input .npy must contain a 2-D matrix")
    rows, cols = map(int, source.shape)
    groups_per_row = (cols + group_size - 1) // group_size
    packed_row_bytes = (cols + 1) // 2 if bits == 4 else cols
    packed_data_bytes = rows * packed_row_bytes
    scales_count = rows * groups_per_row

    header = {
        "format": "LRQ1",
        "rows": rows,
        "cols": cols,
        "bits": bits,
        "group_size": group_size,
        "groups_per_row": groups_per_row,
        "packed_row_bytes": packed_row_bytes,
        "scale_dtype": "float32",
        "scales_count": scales_count,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Use two bounded temporary streams so scales are not retained in RAM while
    # the packed weight section is assembled.
    temp_data = output.with_suffix(output.suffix + ".weights.tmp")
    temp_scales = output.with_suffix(output.suffix + ".scales.tmp")
    try:
        with temp_data.open("wb") as data_handle, temp_scales.open("wb") as scale_handle:
            for block in _iter_rows(source, chunk_rows):
                for row in block:
                    scales = np.empty(groups_per_row, dtype=np.float32)
                    quantized = np.empty(cols, dtype=np.int8)
                    limit = (1 << (bits - 1)) - 1
                    for group in range(groups_per_row):
                        start = group * group_size
                        end = min(cols, start + group_size)
                        max_abs = float(np.max(np.abs(row[start:end]))) if end > start else 0.0
                        scale = max(max_abs / limit, 1e-12)
                        scales[group] = scale
                        quantized[start:end] = np.clip(
                            np.rint(row[start:end] / scale), -limit - 1, limit
                        ).astype(np.int8)
                    data_handle.write(_pack_signed(quantized, bits))
                    scale_handle.write(scales.tobytes(order="C"))

        # Offsets are part of the header, so solve the small fixed point until
        # the header length and offsets agree exactly.
        header["packed_data_offset"] = 0
        header["scale_offset"] = 0
        for _ in range(8):
            header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
            packed_data_offset = HEADER_PREFIX.size + len(header_bytes)
            scale_offset = packed_data_offset + packed_data_bytes
            if (
                header["packed_data_offset"] == packed_data_offset
                and header["scale_offset"] == scale_offset
            ):
                break
            header["packed_data_offset"] = packed_data_offset
            header["scale_offset"] = scale_offset
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with output.open("wb") as handle:
            handle.write(HEADER_PREFIX.pack(MAGIC, len(header_bytes)))
            handle.write(header_bytes)
            with temp_data.open("rb") as data_handle:
                while chunk := data_handle.read(1024 * 1024):
                    handle.write(chunk)
            with temp_scales.open("rb") as scale_handle:
                while chunk := scale_handle.read(1024 * 1024):
                    handle.write(chunk)
    finally:
        temp_data.unlink(missing_ok=True)
        temp_scales.unlink(missing_ok=True)

    return {
        **header,
        "input_bytes": int(Path(input_path).stat().st_size),
        "output_bytes": int(output.stat().st_size),
        "compression_ratio": round(Path(input_path).stat().st_size / output.stat().st_size, 3),
    }


class QuantizedMatrix:
    """Read an LRQ1 file and compute matrix-vector products with low peak RAM."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        magic, header_len = HEADER_PREFIX.unpack(self._handle.read(HEADER_PREFIX.size))
        if magic != MAGIC:
            raise ValueError("not an LRQ1 quantized matrix")
        self.header = json.loads(self._handle.read(header_len).decode("utf-8"))
        self.rows = int(self.header["rows"])
        self.cols = int(self.header["cols"])
        self.bits = int(self.header["bits"])
        self.group_size = int(self.header["group_size"])
        self.groups_per_row = int(self.header["groups_per_row"])
        self.packed_row_bytes = int(self.header["packed_row_bytes"])
        self.packed_data_offset = int(self.header["packed_data_offset"])
        self.scale_offset = int(self.header["scale_offset"])
        self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if getattr(self, "_mapping", None) is not None:
            self._mapping.close()
            self._mapping = None
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "QuantizedMatrix":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _row(self, index: int) -> np.ndarray:
        if not 0 <= index < self.rows:
            raise IndexError(index)
        start = self.packed_data_offset + index * self.packed_row_bytes
        raw = np.frombuffer(
            self._mapping,
            dtype=np.uint8,
            count=self.packed_row_bytes,
            offset=start,
        ).copy()
        values = _unpack_signed(raw, self.cols, self.bits)
        scale_start = self.scale_offset + index * self.groups_per_row * 4
        scales = np.frombuffer(
            self._mapping,
            dtype=np.float32,
            count=self.groups_per_row,
            offset=scale_start,
        ).copy()
        for group, scale in enumerate(scales):
            begin = group * self.group_size
            end = min(self.cols, begin + self.group_size)
            values[begin:end] *= scale
        return values

    def matvec(self, vector: np.ndarray | list[float]) -> np.ndarray:
        vector_array = np.asarray(vector, dtype=np.float32)
        if vector_array.shape != (self.cols,):
            raise ValueError(f"vector must have shape ({self.cols},)")
        output = np.empty(self.rows, dtype=np.float32)
        for row in range(self.rows):
            output[row] = np.dot(self._row(row), vector_array)
        return output
