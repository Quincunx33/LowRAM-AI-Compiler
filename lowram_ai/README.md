# LowRAM AI MVP

LowRAM AI is a low-memory inference prototype for running quantized model components under a strict memory budget. It implements a streaming row-wise quantizer for dense `.npy` matrices, a compact `LRQ1` binary format, a bounded-memory GGUF reader, a memory-mapped tensor runtime, and a narrow Llama-family text-generation loop.

The project is intentionally architecture-specific at this stage. Its main constraint is explicit: **weights and KV cache must be accounted for before generation, and tensor payloads must not be fully expanded into RAM**.

## What is included

The `planner` module estimates a conservative model budget from device RAM, parameter count, layer count, hidden size, context length, and quantization level. It reserves memory for the operating system, host application, and runtime headroom.

The `quantized` module converts a 2-D float16/float32 NumPy matrix to 4-bit or 8-bit row-wise grouped weights. Quantization is performed in row chunks, and the output uses memory mapping at inference time. A matrix-vector product dequantizes one row at a time, which keeps the working set bounded by the row width rather than the complete matrix.

The `gguf` module parses GGUF v2/v3 headers, typed metadata, alignment, and tensor descriptors without reading tensor payloads into memory. The current decoder supports F32, F16, Q4_0, Q4_1, and Q8_0. The `transformer` module adds RMSNorm, RoPE, a fixed-capacity float16 KV cache, and a single-head transformer block that can use memory-mapped quantized projection matrices.

## Quick start

From the repository root:

```bash
python3 -m lowram_ai plan \
  --device-ram-mb 1024 \
  --parameters 500000000 \
  --layers 16 \
  --hidden-size 2048 \
  --context 256 \
  --bits 4
```

Quantize a matrix:

```bash
python3 -m lowram_ai quantize weights.npy weights.lrq \
  --bits 4 --group-size 64 --chunk-rows 32
```

Run a memory-mapped matrix-vector product:

```bash
python3 -m lowram_ai run weights.lrq --repeat 1
```

Inspect a GGUF file without loading tensor payloads:

```bash
python3 -m lowram_ai inspect-gguf model.gguf
python3 -m lowram_ai inspect-gguf model.gguf --decode token_embd.weight
```

Generate text with a supported Llama GGUF model and enforce a 1 GB ceiling:

```bash
python3 -m lowram_ai generate model.gguf "Hello" \
  --max-new-tokens 32 --max-context 256 --max-ram-mb 1024
```

Run the end-to-end demo:

```bash
PYTHONPATH=. python3 -m lowram_ai.examples.demo \
  --workdir /tmp/lowram-demo --rows 512 --cols 512
```

Run tests without third-party test tooling:

```bash
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
```

## Current 1 GB profile

A 1 GB device cannot normally dedicate its entire RAM to the model. The planner therefore uses a conservative application budget. A practical first target is a 300M–1.5B parameter model with 4-bit weights, batch size one, and a 256–512 token context. Larger models require distillation, more aggressive quantization, or partial offload and will not necessarily be smooth on a true 1 GB device.

The current runtime supports one Llama-family architecture with standard `blk.{layer}.*` tensor names, embedded tokenizer metadata, multi-head/grouped-query attention, greedy decoding, a fixed-capacity float16 KV cache, and a `--max-ram-mb` guard. Its next production steps are Q4_K/I-quants, optimized quantized kernels, sampling, chat templates, more architectures, and Android NDK or native mobile bindings.

## Scope and limitations

The custom `LRQ1` format stores symmetric per-group scales and supports 4-bit and 8-bit dense matrices. GGUF reading is metadata-first and supports selected legacy tensor types: F32, F16, Q4_0, Q4_1, and Q8_0. The Llama runtime requires standard tensor names and currently uses greedy generation. Numerical accuracy, latency, and peak RSS must still be measured on the intended device before any release claim.

## References

The binary layout follows the [official GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Llama tensor naming and reversed GGML/PyTorch dimension conventions follow the [llama.cpp model-architecture guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md). Q4_K block structure and dequantization ordering are cross-checked against the upstream [ggml common quantization definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h) and [scalar dequantization reference](https://docs.rs/ramvamp-core/latest/src/ramvamp_core/kernels/quants/dequant.rs.html).

## Native production path

The portable Python path is useful for correctness tests and unsupported formats. For practical CPU inference, build the C++20 library from `native/`; the Python runtime discovers `native/build/liblowram_kernel.so`, `.dylib`, or the Windows DLL automatically, or uses `LOWRAM_KERNEL_PATH` when explicitly set.

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release -j2
python3 -m pip install -e .
lowram-ai generate model.gguf "Hello" --max-new-tokens 32 --max-context 256 --max-ram-mb 1024
```

The native backend accelerates F32, F16, Q4_K, Q5_0, Q6_K, and Q8_0 matvec operations. Python remains the correctness fallback for formats without a native kernel. Use `scripts/benchmark_real_model.py` to compare native and Python outputs and to measure peak RSS and tokens per second on the target device.

## Real-model validation

The release was exercised with SmolLM2-135M-Instruct Q4_K_M. The tested GGUF file was 105,454,432 bytes; four generated tokens completed at approximately 3.99 tokens/second on the sandbox CPU, with approximately 177 MiB peak RSS. Sampled native/Python matvec comparisons had maximum absolute error below 1.4e-5. These are validation measurements, not a guarantee for every CPU or Android device.

## References

The binary layout follows the [official GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md). Llama tensor naming and model metadata follow the [llama.cpp model-architecture guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md). Q4_K block structure and dequantization ordering are cross-checked against the upstream [ggml common quantization definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h) and the [scalar Q4_K reference](https://docs.rs/ramvamp-core/latest/src/ramvamp_core/kernels/quants/dequant.rs.html).
