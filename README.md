# LowRAM AI Compiler

A prototype memory-budgeted AI optimizer/runtime for deploying quantized model components on constrained devices, with an initial target of smooth operation within a 1 GB RAM device.

## MVP capabilities

This repository includes a conservative RAM budget planner, streaming row-wise 4-bit/8-bit quantization for dense `.npy` matrices, the compact `LRQ1` format, a bounded-memory GGUF v2/v3 reader, and a narrow Llama-family text-generation runtime. The reader can inspect typed metadata and tensor descriptors without loading the tensor payload into RAM, and it decodes F32, F16, Q4_0, Q4_1, Q4_K, and Q8_0 tensors.

The runtime includes embedded-vocabulary tokenization, standard Llama tensor mapping, RMSNorm, RoPE, grouped-query attention, a fixed-capacity float16 KV cache, greedy generation, and a configurable `--max-ram-mb` guard. It is a complete small-model generation MVP, not yet a production engine for every GGUF architecture.

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
python3 -m lowram_ai generate model.gguf "Hello" --max-new-tokens 32 --max-context 256 --max-ram-mb 1024
```

Run the end-to-end demo and tests:

```bash
PYTHONPATH=. python3 -m lowram_ai.examples.demo
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
```

See [`lowram_ai/README.md`](lowram_ai/README.md) for design details, limitations, memory assumptions, and the roadmap toward transformer inference on Android, Linux, Windows, and iOS targets. GGUF format references are recorded in [`GGUF_RESEARCH.md`](GGUF_RESEARCH.md).

## References

The GGUF layout and tensor metadata conventions follow the [official GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and the [llama.cpp model-architecture guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md). The supported quantization descriptions are cross-checked against the [Hugging Face GGUF documentation](https://huggingface.co/docs/hub/en/gguf) and upstream [ggml quantization structures](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h).

## Production-oriented native path

The Python implementation remains the portable correctness fallback. For real CPU inference, build the optional C++20 shared library and let the runtime discover it automatically:

```bash
sudo apt-get install cmake g++
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release -j2
python3 -m pip install -e .
lowram-ai generate model.gguf "Hello" --max-new-tokens 32 --max-context 256 --max-ram-mb 1024
```

The native library includes mmap-backed F32/F16, Q5_0, Q6_K, Q8_0, and Q4_K matvec kernels. The runtime falls back to Python for unsupported legacy affine kernels, so model correctness is not tied to native build availability.

A repeatable real-model benchmark is provided at `scripts/benchmark_real_model.py`. The release validation used SmolLM2-135M-Instruct Q4_K_M, a 105,454,432-byte GGUF file. On the sandbox CPU, the native path generated four new tokens at approximately 3.99 tokens/second with a measured peak RSS of approximately 177 MiB; native versus Python matvec comparisons stayed within 1.4e-5 maximum absolute error on sampled Q4_K/Q5_0/Q6_K tensors. These measurements are host-specific and must be repeated on the target device.
