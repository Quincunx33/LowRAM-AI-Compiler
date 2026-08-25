# LowRAM AI Compiler

A prototype memory-budgeted AI optimizer/runtime for deploying quantized model components on constrained devices, with an initial target of smooth operation within a 1 GB RAM device.

## MVP capabilities

This repository currently includes a conservative RAM budget planner, streaming row-wise 4-bit/8-bit quantization for dense `.npy` matrices, the compact `LRQ1` binary format, a memory-mapped matrix-vector runtime, and a bounded-memory GGUF v2/v3 reader. The reader can inspect typed metadata and tensor descriptors without loading the tensor payload into RAM, and it decodes F32, F16, Q4_0, Q4_1, and Q8_0 tensors.

The repository also contains the first transformer-block foundation: RMSNorm, RoPE, a fixed-capacity float16 KV cache, and quantized linear projections. This is not yet a complete text-generation engine; tokenizer integration, model-architecture mapping, and multi-layer generation remain future work.

## Quick start

```bash
python3 -m lowram_ai plan \
  --device-ram-mb 1024 \
  --parameters 500000000 \
  --layers 16 \
  --hidden-size 2048 \
  --context 256 \
  --bits 4
```

```bash
python3 -m lowram_ai quantize weights.npy weights.lrq \
  --bits 4 --group-size 64 --chunk-rows 32
python3 -m lowram_ai run weights.lrq --repeat 1
python3 -m lowram_ai inspect-gguf model.gguf
python3 -m lowram_ai inspect-gguf model.gguf --decode token_embd.weight
```

Run the end-to-end demo and tests:

```bash
PYTHONPATH=. python3 -m lowram_ai.examples.demo
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
```

See [`lowram_ai/README.md`](lowram_ai/README.md) for design details, limitations, memory assumptions, and the roadmap toward transformer inference on Android, Linux, Windows, and iOS targets. GGUF format references are recorded in [`GGUF_RESEARCH.md`](GGUF_RESEARCH.md).
