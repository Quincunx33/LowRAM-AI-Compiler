"""Command-line entry points for the LowRAM AI prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .gguf import GGUFReader, inspect_gguf
from .llama import LlamaRuntime
from .planner import build_budget_plan
from .quantized import QuantizedMatrix, quantize_npy_matrix


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lowram-ai",
        description="Plan and run memory-budgeted quantized inference prototypes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="estimate a safe inference memory budget")
    plan.add_argument("--device-ram-mb", type=_positive_int, default=1024)
    plan.add_argument("--parameters", type=_positive_int, required=True)
    plan.add_argument("--layers", type=_positive_int, required=True)
    plan.add_argument("--hidden-size", type=_positive_int, required=True)
    plan.add_argument("--context", type=_positive_int, default=512)
    plan.add_argument("--bits", type=int, choices=(2, 3, 4, 8, 16, 32), default=4)

    quantize = commands.add_parser("quantize", help="stream-quantize a 2-D .npy matrix")
    quantize.add_argument("input", type=Path)
    quantize.add_argument("output", type=Path)
    quantize.add_argument("--bits", type=int, choices=(4, 8), default=4)
    quantize.add_argument("--group-size", type=_positive_int, default=64)
    quantize.add_argument("--chunk-rows", type=_positive_int, default=32)

    inspect = commands.add_parser("inspect-gguf", help="inspect GGUF metadata and tensor descriptors")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--decode", type=str, help="decode one supported tensor and show a preview")

    generate = commands.add_parser("generate", help="generate text with a supported Llama GGUF model")
    generate.add_argument("model", type=Path)
    generate.add_argument("prompt", type=str)
    generate.add_argument("--max-new-tokens", type=_positive_int, default=32)
    generate.add_argument("--max-context", type=_positive_int, default=None)
    generate.add_argument("--max-ram-mb", type=_positive_int, default=None)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=40)
    generate.add_argument("--top-p", type=float, default=0.9)
    generate.add_argument("--repetition-penalty", type=float, default=1.05)
    generate.add_argument("--seed", type=int, default=0)

    run = commands.add_parser("run", help="run a memory-mapped matrix-vector product")
    run.add_argument("matrix", type=Path)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--repeat", type=_positive_int, default=1)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = build_budget_plan(
            device_ram_mb=args.device_ram_mb,
            parameters=args.parameters,
            layers=args.layers,
            hidden_size=args.hidden_size,
            requested_context_tokens=args.context,
            quantization_bits=args.bits,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "quantize":
        result = quantize_npy_matrix(
            args.input,
            args.output,
            bits=args.bits,
            group_size=args.group_size,
            chunk_rows=args.chunk_rows,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "inspect-gguf":
        if args.decode:
            with GGUFReader(args.path) as reader:
                tensor = reader.decode_tensor(args.decode)
                print(json.dumps({
                    "tensor": args.decode,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "preview": np.round(tensor.reshape(-1)[:16], 6).tolist(),
                }, indent=2))
        else:
            print(json.dumps(inspect_gguf(args.path), indent=2))
        return 0

    if args.command == "generate":
        with LlamaRuntime.open(
            str(args.model),
            max_context_tokens=args.max_context,
            max_ram_mb=args.max_ram_mb,
        ) as runtime:
            print(runtime.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                seed=args.seed,
            ))
        return 0

    if args.command == "run":
        rng = np.random.default_rng(args.seed)
        with QuantizedMatrix(args.matrix) as matrix:
            vector = rng.standard_normal(matrix.cols, dtype=np.float32)
            result = None
            for _ in range(args.repeat):
                result = matrix.matvec(vector)
            assert result is not None
            print(json.dumps({
                "rows": matrix.rows,
                "cols": matrix.cols,
                "bits": matrix.bits,
                "repeat": args.repeat,
                "output_preview": np.round(result[:8], 5).tolist(),
            }, indent=2))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
