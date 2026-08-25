"""Measure the prototype's peak RSS on a file-backed matrix workload."""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np

from lowram_ai.quantized import QuantizedMatrix, quantize_npy_matrix


def peak_rss_mb() -> float:
    # Linux/Windows report KiB; macOS reports bytes.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/lowram-memory"))
    parser.add_argument("--size", type=int, default=2048)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    source_path = args.workdir / "large.npy"
    output_path = args.workdir / "large.lrq"

    source = np.lib.format.open_memmap(
        source_path, mode="w+", dtype=np.float32, shape=(args.size, args.size)
    )
    rng = np.random.default_rng(17)
    for start in range(0, args.size, 64):
        source[start : start + 64] = rng.normal(
            0, 0.2, size=(min(64, args.size - start), args.size)
        ).astype(np.float32)
    source.flush()
    del source

    info = quantize_npy_matrix(source_path, output_path, bits=4, group_size=64, chunk_rows=16)
    after_quantize = peak_rss_mb()
    with QuantizedMatrix(output_path) as matrix:
        vector = np.ones(matrix.cols, dtype=np.float32)
        output = matrix.matvec(vector)
        checksum = float(output[:16].sum())
    after_inference = peak_rss_mb()

    print(json.dumps({
        "source_mb": round(source_path.stat().st_size / 1024 / 1024, 2),
        "quantized_mb": round(output_path.stat().st_size / 1024 / 1024, 2),
        "compression_ratio": info["compression_ratio"],
        "peak_rss_after_quantize_mb": round(after_quantize, 2),
        "peak_rss_after_inference_mb": round(after_inference, 2),
        "checksum": round(checksum, 5),
    }, indent=2))


if __name__ == "__main__":
    main()
