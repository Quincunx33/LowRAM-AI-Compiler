"""A small, dependency-light GGUF v2/v3 reader.

The reader parses metadata and tensor descriptors without loading tensor data.
Supported tensor decoders are F32, F16, Q4_0, Q4_1, and Q8_0; additional GGML
quantization schemes can be added behind the same TensorInfo interface.
"""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np


MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_ARRAY_ITEMS = 10_000_000

# GGUF metadata value type IDs from the official specification.
META_UINT8 = 0
META_INT8 = 1
META_UINT16 = 2
META_INT16 = 3
META_UINT32 = 4
META_INT32 = 5
META_FLOAT32 = 6
META_BOOL = 7
META_STRING = 8
META_ARRAY = 9
META_UINT64 = 10
META_INT64 = 11
META_FLOAT64 = 12

META_FORMATS: dict[int, str] = {
    META_UINT8: "B",
    META_INT8: "b",
    META_UINT16: "H",
    META_INT16: "h",
    META_UINT32: "I",
    META_INT32: "i",
    META_FLOAT32: "f",
    META_BOOL: "?",
    META_UINT64: "Q",
    META_INT64: "q",
    META_FLOAT64: "d",
}

# GGML tensor type IDs used by common Llama-family GGUF files.
GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    relative_offset: int
    data_offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"TYPE_{self.ggml_type}")

    @property
    def element_count(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count


class GGUFReader:
    """Parse a GGUF file while keeping tensor payloads file-backed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle: BinaryIO = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self.version = 0
        self.alignment = DEFAULT_ALIGNMENT
        self.tensor_data_offset = 0
        self._parse()
        self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read_exact(self, count: int) -> bytes:
        data = self._handle.read(count)
        if len(data) != count:
            raise ValueError(f"truncated GGUF file: wanted {count} bytes, got {len(data)}")
        return data

    def _read_struct(self, fmt: str) -> Any:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self._read_exact(size))[0]

    def _read_string(self) -> str:
        length = self._read_struct("Q")
        if length > MAX_STRING_BYTES:
            raise ValueError(f"GGUF string is too large: {length} bytes")
        try:
            return self._read_exact(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8 string in GGUF metadata") from error

    def _read_metadata_value(self, value_type: int) -> Any:
        if value_type == META_STRING:
            return self._read_string()
        if value_type == META_ARRAY:
            element_type = self._read_struct("I")
            length = self._read_struct("Q")
            if length > MAX_ARRAY_ITEMS:
                raise ValueError(f"GGUF metadata array is too large: {length} items")
            return [self._read_metadata_value(element_type) for _ in range(length)]
        fmt = META_FORMATS.get(value_type)
        if fmt is None:
            raise ValueError(f"unsupported GGUF metadata type: {value_type}")
        return self._read_struct(fmt)

    def _parse(self) -> None:
        if self._read_exact(4) != MAGIC:
            raise ValueError("not a GGUF file")
        self.version = self._read_struct("I")
        if self.version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {self.version}")
        tensor_count = self._read_struct("Q")
        metadata_count = self._read_struct("Q")
        if tensor_count > MAX_ARRAY_ITEMS:
            raise ValueError("GGUF tensor count is unreasonably large")
        if metadata_count > MAX_ARRAY_ITEMS:
            raise ValueError("GGUF metadata count is unreasonably large")

        for _ in range(metadata_count):
            key = self._read_string()
            value_type = self._read_struct("I")
            self.metadata[key] = self._read_metadata_value(value_type)
        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if self.alignment <= 0 or self.alignment % 8 != 0:
            raise ValueError(f"invalid GGUF alignment: {self.alignment}")

        descriptors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = self._read_string()
            dimension_count = self._read_struct("I")
            if dimension_count > 8:
                raise ValueError(f"unsupported tensor rank: {dimension_count}")
            shape = tuple(int(self._read_struct("Q")) for _ in range(dimension_count))
            ggml_type = int(self._read_struct("I"))
            relative_offset = int(self._read_struct("Q"))
            descriptors.append((name, shape, ggml_type, relative_offset))

        position = self._handle.tell()
        self.tensor_data_offset = position + (self.alignment - position % self.alignment) % self.alignment
        file_size = self.path.stat().st_size
        for name, shape, ggml_type, relative_offset in descriptors:
            data_offset = self.tensor_data_offset + relative_offset
            if data_offset < self.tensor_data_offset or data_offset >= file_size:
                raise ValueError(f"tensor offset outside file: {name}")
            self.tensors[name] = TensorInfo(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                relative_offset=relative_offset,
                data_offset=data_offset,
            )

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"tensor not found: {name}") from error

    def iter_tensors(self) -> Iterator[TensorInfo]:
        return iter(self.tensors.values())

    def tensor_bytes(self, tensor: str | TensorInfo, count: int) -> bytes:
        """Read an explicitly bounded byte range from a tensor payload."""
        info = self.tensor_info(tensor) if isinstance(tensor, str) else tensor
        if count < 0 or info.data_offset + count > self.path.stat().st_size:
            raise ValueError("tensor byte range exceeds file")
        if self._mapping is None:
            raise RuntimeError("GGUFReader is closed")
        return bytes(self._mapping[info.data_offset : info.data_offset + count])

    def decode_tensor(self, tensor: str | TensorInfo) -> np.ndarray:
        """Decode a supported tensor into float32.

        This method allocates the decoded tensor by design. Use metadata and
        tensor_bytes for inspection when the tensor is too large for RAM.
        """
        info = self.tensor_info(tensor) if isinstance(tensor, str) else tensor
        count = info.element_count
        type_name = info.type_name
        if type_name == "F32":
            raw = self.tensor_bytes(info, count * 4)
            values = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
        elif type_name == "F16":
            raw = self.tensor_bytes(info, count * 2)
            values = np.frombuffer(raw, dtype="<f2").astype(np.float32, copy=True)
        elif type_name in ("Q4_0", "Q4_1", "Q8_0"):
            values = self._decode_legacy_quant(info)
        else:
            raise NotImplementedError(f"tensor decoder not implemented for {type_name}")
        return values.reshape(info.shape)

    def _decode_legacy_quant(self, info: TensorInfo) -> np.ndarray:
        block_size = 32
        if info.element_count % block_size != 0:
            raise ValueError(f"{info.type_name} tensor size must be divisible by 32")
        if info.type_name == "Q4_0":
            bytes_per_block = 18
        elif info.type_name == "Q4_1":
            bytes_per_block = 20
        else:
            bytes_per_block = 34
        block_count = info.element_count // block_size
        raw = np.frombuffer(
            self.tensor_bytes(info, block_count * bytes_per_block), dtype=np.uint8
        )
        decoded = np.empty(info.element_count, dtype=np.float32)
        for block in range(block_count):
            source = raw[block * bytes_per_block : (block + 1) * bytes_per_block]
            if info.type_name == "Q4_0":
                scale = np.frombuffer(source[:2].tobytes(), dtype="<f2")[0].astype(np.float32)
                nibbles = source[2:]
                q = np.empty(32, dtype=np.float32)
                q[:16] = (nibbles & 0x0F).astype(np.float32) - 8
                q[16:] = (nibbles >> 4).astype(np.float32) - 8
                decoded[block * 32 : (block + 1) * 32] = q * scale
            elif info.type_name == "Q4_1":
                scale = np.frombuffer(source[:2].tobytes(), dtype="<f2")[0].astype(np.float32)
                minimum = np.frombuffer(source[2:4].tobytes(), dtype="<f2")[0].astype(np.float32)
                nibbles = source[4:]
                q = np.empty(32, dtype=np.float32)
                q[:16] = (nibbles & 0x0F).astype(np.float32)
                q[16:] = (nibbles >> 4).astype(np.float32)
                decoded[block * 32 : (block + 1) * 32] = q * scale + minimum
            else:
                scale = np.frombuffer(source[:2].tobytes(), dtype="<f2")[0].astype(np.float32)
                q = source[2:].view(np.int8).astype(np.float32)
                decoded[block * 32 : (block + 1) * 32] = q * scale
        return decoded


def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return offset + (alignment - offset % alignment) % alignment


def inspect_gguf(path: str | Path) -> dict[str, Any]:
    """Return JSON-friendly metadata and tensor descriptors without decoding data."""
    with GGUFReader(path) as reader:
        return {
            "path": str(path),
            "version": reader.version,
            "alignment": reader.alignment,
            "tensor_data_offset": reader.tensor_data_offset,
            "metadata": reader.metadata,
            "tensors": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "type": item.type_name,
                    "type_id": item.ggml_type,
                    "data_offset": item.data_offset,
                }
                for item in reader.iter_tensors()
            ],
        }
