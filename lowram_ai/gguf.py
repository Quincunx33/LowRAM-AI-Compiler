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

    def tensor_nbytes(self, tensor: str | TensorInfo) -> int:
        """Estimate the exact payload size for supported GGML tensor types."""
        info = self.tensor_info(tensor) if isinstance(tensor, str) else tensor
        count = info.element_count
        if info.type_name == "F32":
            return count * 4
        if info.type_name == "F16":
            return count * 2
        if info.type_name in ("Q4_0", "Q4_1", "Q8_0"):
            if count % 32 != 0:
                raise ValueError(f"{info.type_name} tensor size must be divisible by 32")
            return (count // 32) * self._legacy_block_bytes(info.type_name)
        if info.type_name == "Q4_K":
            if count % 256 != 0:
                raise ValueError("Q4_K tensor size must be divisible by 256")
            return (count // 256) * 144
        raise NotImplementedError(f"tensor size not implemented for {info.type_name}")

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"tensor not found: {name}") from error

    def iter_tensors(self) -> Iterator[TensorInfo]:
        return iter(self.tensors.values())

    def tensor_bytes(self, tensor: str | TensorInfo, count: int, offset: int = 0) -> bytes:
        """Read an explicitly bounded byte range from a tensor payload."""
        info = self.tensor_info(tensor) if isinstance(tensor, str) else tensor
        if offset < 0 or count < 0 or info.data_offset + offset + count > self.path.stat().st_size:
            raise ValueError("tensor byte range exceeds file")
        if self._mapping is None:
            raise RuntimeError("GGUFReader is closed")
        start = info.data_offset + offset
        return bytes(self._mapping[start : start + count])

    def _legacy_block_bytes(self, type_name: str) -> int:
        if type_name == "Q4_0":
            return 18
        if type_name == "Q4_1":
            return 20
        if type_name == "Q8_0":
            return 34
        raise NotImplementedError(f"tensor decoder not implemented for {type_name}")

    def _decode_legacy_blocks(self, raw: bytes, type_name: str, block_count: int) -> np.ndarray:
        bytes_per_block = self._legacy_block_bytes(type_name)
        raw_array = np.frombuffer(raw, dtype=np.uint8)
        decoded = np.empty(block_count * 32, dtype=np.float32)
        for block in range(block_count):
            source = raw_array[block * bytes_per_block : (block + 1) * bytes_per_block]
            if type_name == "Q4_0":
                scale = np.frombuffer(source[:2].tobytes(), dtype="<f2")[0].astype(np.float32)
                nibbles = source[2:]
                q = np.empty(32, dtype=np.float32)
                q[:16] = (nibbles & 0x0F).astype(np.float32) - 8
                q[16:] = (nibbles >> 4).astype(np.float32) - 8
                decoded[block * 32 : (block + 1) * 32] = q * scale
            elif type_name == "Q4_1":
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

    @staticmethod
    def _unpack_q4_k_scales(packed: np.ndarray, block: int) -> tuple[int, int]:
        """Unpack one 6-bit scale/min pair from a Q4_K 12-byte scale array."""
        if block < 4:
            return int(packed[block] & 0x3F), int(packed[block + 4] & 0x3F)
        scale = int((packed[block + 4] & 0x0F) | ((packed[block - 4] >> 6) << 4))
        minimum = int((packed[block + 4] >> 4) | ((packed[block] >> 6) << 4))
        return scale, minimum

    def _decode_q4_k_blocks(self, raw: bytes, block_count: int) -> np.ndarray:
        """Decode Q4_K blocks using the upstream ggml 256-value layout."""
        raw_array = np.frombuffer(raw, dtype=np.uint8)
        decoded = np.empty(block_count * 256, dtype=np.float32)
        for block_index in range(block_count):
            block = raw_array[block_index * 144 : (block_index + 1) * 144]
            d = np.frombuffer(block[:2].tobytes(), dtype="<f2")[0].astype(np.float32)
            dmin = np.frombuffer(block[2:4].tobytes(), dtype="<f2")[0].astype(np.float32)
            packed_scales = block[4:16]
            quants = block[16:]
            output_start = block_index * 256
            for sub_block in range(8):
                scale_q, min_q = self._unpack_q4_k_scales(packed_scales, sub_block)
                scale = d * scale_q
                minimum = dmin * min_q
                quant_start = (sub_block // 2) * 32
                quant_bytes = quants[quant_start : quant_start + 32]
                values = (
                    (quant_bytes & 0x0F) if sub_block % 2 == 0 else (quant_bytes >> 4)
                ).astype(np.float32)
                start = output_start + sub_block * 32
                decoded[start : start + 32] = values * scale - minimum
        return decoded

    def _decode_legacy_vector(self, info: TensorInfo, byte_offset: int, count: int) -> np.ndarray:
        if count % 32 != 0:
            raise ValueError(f"{info.type_name} vector length must be divisible by 32")
        bytes_per_block = self._legacy_block_bytes(info.type_name)
        return self._decode_legacy_blocks(
            self.tensor_bytes(info, (count // 32) * bytes_per_block, byte_offset),
            info.type_name,
            count // 32,
        )

    def tensor_vector(self, tensor: str | TensorInfo, index: int) -> np.ndarray:
        """Read one column/vector from a GGML matrix without loading its siblings."""
        info = self.tensor_info(tensor) if isinstance(tensor, str) else tensor
        if len(info.shape) != 2:
            raise ValueError("tensor_vector currently requires a 2-D tensor")
        width = info.shape[0]
        columns = info.element_count // width
        if not 0 <= index < columns:
            raise IndexError(index)
        type_name = info.type_name
        if type_name in ("F32", "F16"):
            item_size = 4 if type_name == "F32" else 2
            raw = self.tensor_bytes(info, width * item_size, index * width * item_size)
            dtype = "<f4" if type_name == "F32" else "<f2"
            return np.frombuffer(raw, dtype=dtype, count=width).astype(np.float32, copy=True)
        if type_name in ("Q4_0", "Q4_1", "Q8_0"):
            row_bytes = (width // 32) * self._legacy_block_bytes(type_name)
            raw = self.tensor_bytes(info, row_bytes, index * row_bytes)
            return self._decode_legacy_blocks(raw, type_name, width // 32)
        if type_name == "Q4_K":
            row_bytes = (width // 256) * 144
            raw = self.tensor_bytes(info, row_bytes, index * row_bytes)
            return self._decode_q4_k_blocks(raw, width // 256)
        raise NotImplementedError(f"tensor decoder not implemented for {type_name}")

    def tensor_matvec(self, tensor: str | TensorInfo, vector: np.ndarray | list[float]) -> np.ndarray:
        """Compute W.T @ vector for a GGML matrix shaped [input, output]."""
        info = self.tensor_info(tensor) if isinstance(tensor, str) else tensor
        if len(info.shape) != 2:
            raise ValueError("tensor_matvec currently requires a 2-D tensor")
        vector_array = np.asarray(vector, dtype=np.float32)
        input_width = info.shape[0]
        output_width = info.element_count // input_width
        if vector_array.shape != (input_width,):
            raise ValueError(f"vector must have shape ({input_width},)")
        output = np.empty(output_width, dtype=np.float32)
        for column in range(output_width):
            output[column] = np.dot(self.tensor_vector(info, column), vector_array)
        return output

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
        elif type_name == "Q4_K":
            if count % 256 != 0:
                raise ValueError("Q4_K tensor size must be divisible by 256")
            values = self._decode_q4_k_blocks(
                self.tensor_bytes(info, (count // 256) * 144), count // 256
            )
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
