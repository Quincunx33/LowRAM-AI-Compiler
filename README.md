# LowRAM AI Compiler

A prototype memory-budgeted AI optimizer/runtime for deploying quantized model components on constrained devices, with an initial target of smooth operation within a 1 GB RAM device.

## MVP capabilities

This repository currently includes a conservative RAM budget planner, streaming row-wise 4-bit/8-bit quantization for dense `.npy` matrices, the compact `LRQ1` binary format, and a memory-mapped matrix-vector runtime that avoids materializing the complete float32 matrix in RAM.

The current MVP is the foundation for a complete transformer compiler. It does not yet import PyTorch, ONNX, GGUF, or Safetensors models and does not yet include tokenization or text generation.

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
```

Run the end-to-end demo and tests:

```bash
PYTHONPATH=. python3 -m lowram_ai.examples.demo
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
```

See [`lowram_ai/README.md`](lowram_ai/README.md) for design details, limitations, memory assumptions, and the roadmap toward transformer inference on Android, Linux, Windows, and iOS targets.
