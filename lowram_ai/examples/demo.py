"""End-to-end demo for the LowRAM AI matrix prototype."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lowram_ai.quantized import QuantizedMatrix, quantize_npy_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("lowram_demo"))
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--cols", type=int, default=512)
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    source_path = args.workdir / "weights.npy"
    quantized_path = args.workdir / "weights.lrq"
    rng = np.random.default_rng(11)
    source = rng.normal(0, 0.25, size=(args.rows, args.cols)).astype(np.float32)
    np.save(source_path, source)
    info = quantize_npy_matrix(source_path, quantized_path, bits=4, group_size=64)

    vector = rng.normal(size=args.cols).astype(np.float32)
    expected = source @ vector
    with QuantizedMatrix(quantized_path) as matrix:
        actual = matrix.matvec(vector)
    error = np.mean(np.abs(expected - actual))
    print({**info, "mean_absolute_error": round(float(error), 6)})


if __name__ == "__main__":
    main()
