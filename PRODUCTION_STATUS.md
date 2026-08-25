# LowRAM AI production upgrade status

## Implemented in this iteration

The native C++ runtime now includes direct Q4_0 and Q4_1 dot-product kernels, optional AVX2 acceleration for float32 and legacy 4-bit blocks, OpenMP parallelization across output columns, and `posix_madvise(..., POSIX_MADV_RANDOM)` for mmap-backed tensor access. Model payloads remain file-backed; the runtime does not decode the full model into RAM. The Python bridge keeps activation vectors contiguous and aligned while leaving the model mapping zero-copy.

The GGUF reader now decodes BF16 tensors in addition to its existing supported formats. The native bridge also handles BF16 matvec operations. The new native regression tests cover Q4_0, Q4_1, and BF16 numerical behavior.

An optional FastAPI server provides `/health`, `/v1/model`, and `/v1/generate`. It supports an environment-configured API key (`LOWRAM_API_KEY`) and a simple per-process rate limit (`LOWRAM_RATE_LIMIT`). Docker and Docker Compose definitions mount GGUF models read-only, and CI builds the native library, runs the test suite, and validates package metadata.

## Build and run

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j2
python3 -m pip install -e .
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
```

For the optional server:

```bash
python3 -m pip install -e '.[server]'
LOWRAM_MODEL=/models/model.gguf LOWRAM_MAX_RAM_MB=1024 LOWRAM_API_KEY=change-me lowram-ai-server
```

## Validation

The native library builds successfully with GCC/CMake. The complete suite passes **20 tests**, including native Q4_0, Q4_1, and BF16 cases. The local FastAPI integration test also successfully loaded the deterministic tiny GGUF model and returned valid responses from `/health`, `/v1/model`, and `/v1/generate`; generation returned `A A A` for the test prompt. These tests verify correctness on a small fixture; throughput and peak RSS must still be measured on the target low-RAM device with its actual model.

For Render, `LOWRAM_MODEL` must point to a model file that exists inside the service. The included `render.yaml` is deployment-ready, but a model must be provisioned through the chosen storage/download approach. A free ephemeral service filesystem is not a durable model store, and the service memory limit must be compatible with the selected GGUF model.

## Remaining work before a production release

The supplied roadmap contains several large projects that are not safe to claim as complete from this iteration: IQ4_NL/IQ3_S and Q5_K decoding, activation-aware quantization and pruning, full Mistral/Phi/Gemma/Qwen/DeepSeek/StableLM execution, MoE and YaRN implementations, GPU backends, mobile bindings, gRPC/WebSocket serving, distributed observability, and formal PyPI/mobile release pipelines. Each requires architecture-specific fixtures and target-device validation. Unsupported formats continue to fail explicitly rather than silently producing incorrect output.
