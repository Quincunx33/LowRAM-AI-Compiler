#!/usr/bin/env python3
"""Benchmark one local GGUF model without downloading or modifying it."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from lowram_ai.gguf import GGUFReader
from lowram_ai.llama import LlamaRuntime
from lowram_ai.native import NativeKernel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=64)
    parser.add_argument("--max-ram-mb", type=int, default=1024)
    args = parser.parse_args()

    rng = np.random.default_rng(123)
    comparisons = []
    seen_types = set()
    with GGUFReader(args.model) as reader:
        native = NativeKernel.try_open(args.model)
        if native is None:
            raise RuntimeError("native kernel not found; build native/build first")
        try:
            for info in reader.iter_tensors():
                if info.type_name not in {"Q4_K", "Q5_0", "Q6_K"} or len(info.shape) != 2:
                    continue
                vector = rng.standard_normal(info.shape[0], dtype=np.float32)
                python_result = reader.tensor_matvec(info, vector)
                native_result = native.matvec(info, vector)
                if native_result is None:
                    continue
                difference = np.abs(python_result - native_result)
                comparisons.append(
                    {
                        "tensor": info.name,
                        "type": info.type_name,
                        "shape": list(info.shape),
                        "max_abs_error": float(difference.max(initial=0.0)),
                        "mean_abs_error": float(difference.mean()),
                    }
                )
                seen_types.add(info.type_name)
                if {"Q4_K", "Q5_0", "Q6_K"}.issubset(seen_types) and len(comparisons) >= 3:
                    break
        finally:
            native.close()

    start = time.perf_counter()
    with LlamaRuntime.open(
        str(args.model), max_context_tokens=args.max_context, max_ram_mb=args.max_ram_mb
    ) as runtime:
        generated = runtime.generate(args.prompt, max_new_tokens=args.max_new_tokens)
        estimated_bytes = runtime.estimated_model_bytes
    elapsed = time.perf_counter() - start
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(json.dumps({
        "model": args.model.name,
        "model_bytes": args.model.stat().st_size,
        "estimated_runtime_bytes": estimated_bytes,
        "generation": generated,
        "new_tokens": args.max_new_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "tokens_per_second": round(args.max_new_tokens / elapsed, 4),
        "peak_rss_mb": round(rss_kb / 1024, 2),
        "native_matvec_comparisons": comparisons,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
