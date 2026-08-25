# LowRAM AI Production-Oriented Test Report

## Release

The production-oriented milestone was pushed to [Quincunx33/LowRAM-AI-Compiler](https://github.com/Quincunx33/LowRAM-AI-Compiler) in commit [`bd5b28f`](https://github.com/Quincunx33/LowRAM-AI-Compiler/commit/bd5b28f).

## Implemented enhancements

| Area | Result |
|---|---|
| Native inference | Portable C++20 shared library with mmap-backed matvec kernels |
| Quantization | Q4_K, Q5_0, Q6_K, Q8_0, F16 and F32 paths |
| Runtime | Python fallback plus optional native acceleration |
| Tokenization | Llama-style pieces and GPT-2 byte-level BPE from GGUF metadata |
| Generation | Greedy and seeded temperature/top-k/top-p sampling with repetition penalty |
| Memory control | Estimated model footprint, fixed float16 KV cache, `--max-ram-mb` guard, RSS checks |
| Packaging | `pyproject.toml`, editable installation, `lowram-ai` console command |
| Portability | CMake native build with POSIX mmap and Windows mapping branches |
| Testing | Unit tests, native/Python numerical comparisons, real-model benchmark |

## Test results

The unit suite passed **17 tests**. The test coverage includes GGUF metadata and alignment, Q4_K decoding, tokenizer behavior, sampling, KV cache, transformer primitives, Llama generation, and memory-budget initialization.

The real model was `SmolLM2-135M-Instruct-Q4_K_M.gguf` from the public bartowski GGUF repository. The downloaded file was 105,454,432 bytes. With a 1,024 MB runtime ceiling and 64-token context, the runtime generated four new tokens from the prompt `Hello` as `Hello, and I'm`.

| Metric | Measured result |
|---|---:|
| Estimated runtime memory | 105,193,728 bytes |
| Peak RSS on sandbox CPU | approximately 177 MiB |
| Throughput | approximately 3.99 tokens/second |
| Maximum native/Python sampled matvec error | approximately 1.34e-5 |
| Native quantized types checked | Q4_K, Q5_0, Q6_K |

These are host-specific measurements. They are not a guarantee for every 1 GB Android device; the intended device must be benchmarked separately.

## Build and run

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release -j2
python3 -m pip install -e .
lowram-ai generate model.gguf "Hello" \
  --max-new-tokens 32 \
  --max-context 256 \
  --max-ram-mb 1024 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --repetition-penalty 1.05 \
  --seed 42
```

For repeatable validation:

```bash
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
PYTHONPATH=. python3 scripts/benchmark_real_model.py model.gguf \
  --prompt 'Hello' --max-new-tokens 4 --max-context 64 --max-ram-mb 1024
```

## Remaining production work

The current result is a production-oriented **MVP**, not a finished commercial SDK. Remaining work includes Android NDK packaging and on-device testing, SIMD/NEON/AVX2 kernels, more GGUF quantization formats, architecture-specific chat templates, cancellation and streaming APIs, fuzz testing for malformed GGUF files, and quality regression tests against a reference runtime.

## References

[1]: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md "Official GGUF specification"

[2]: https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md "llama.cpp model architecture guide"

[3]: https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF "SmolLM2-135M-Instruct GGUF model card"
