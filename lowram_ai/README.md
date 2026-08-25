# LowRAM AI MVP

LowRAM AI is a first prototype for running quantized model components under a strict memory budget. It currently implements a streaming row-wise quantizer for dense `.npy` matrices, a compact `LRQ1` binary format, a memory-mapped reader, and a matrix-vector runtime that does not materialize the full float32 matrix.

This is an inference primitive, not yet a complete LLM compiler. The purpose of this MVP is to prove the most important constraint: **weights can be stored compactly and consumed in small working buffers instead of being fully expanded in RAM**.

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

The current runtime is intentionally simple. The transformer block is a foundation for validation, not a full model architecture loader. Its next production steps are architecture-specific GGUF tensor mapping, tokenizer integration, multi-layer generation, operator fusion, optimized quantized linear and attention kernels, KV-cache paging/compression, and target backends for Android NDK, Linux, Windows, and iOS Metal/Core ML integration.

## Scope and limitations

The custom `LRQ1` format stores symmetric per-group scales and supports 4-bit and 8-bit dense matrices. GGUF reading is currently metadata-first and supports only selected legacy tensor types. The project does not yet import PyTorch, ONNX, or Safetensors models; it does not implement tokenization; and it does not provide a complete text-generation loop. Numerical accuracy and peak RSS should be measured again on the intended device before any release claim.
