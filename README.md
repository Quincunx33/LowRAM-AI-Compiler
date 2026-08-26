# LowRAM AI Compiler

A memory-budgeted GGUF inference runtime and developer playground for running small instruction-tuned language models on constrained Linux devices. The project combines a portable Python runtime with optional C++20 native kernels, a FastAPI chat server, streaming responses, persistent browser chat history, and a sandboxed developer workspace.

> **Current recommended local model:** [SmolLM2-360M-Instruct Q4_K_M](https://github.com/Quincunx33/LowRAM-AI-Compiler/releases/tag/v0.4.0). It is provided as a GitHub Release asset rather than committed to the repository.

## What this project provides

The runtime reads GGUF models through mmap-backed tensor access, supports memory budgeting, and includes quantized matrix-vector operations for common tensor types. The optional native path provides C++20 kernels with SIMD/OpenMP support where available. The web application provides a dark, responsive chat playground with message streaming, conversation sessions, code-block copy controls, model settings, and a developer workspace for creating files, uploading files, running restricted Python, and creating or extracting ZIP archives.

The project is designed for experimentation and constrained-device deployments. It is not a replacement for a full production inference engine such as llama.cpp and does not support every GGUF architecture.

## Requirements

| Requirement | Recommended |
|---|---|
| Python | 3.10 or newer |
| Native build | CMake and a C++20 compiler; optional but recommended on Linux |
| RAM for SmolLM2-360M Q4 | 512 MB–1 GB, depending on context and process overhead |
| Operating systems | Linux first; other platforms may require build adjustments |
| Model format | GGUF with a supported Llama-family architecture |

## Install

Clone the repository and install the package with the server dependencies:

```bash
git clone https://github.com/Quincunx33/LowRAM-AI-Compiler.git
cd LowRAM-AI-Compiler
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[server]'
```

Build the optional native library on Linux:

```bash
sudo apt-get install cmake g++
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release -j2
```

The Python implementation remains the correctness fallback when the native library is unavailable or a tensor type is not implemented by the native path.

## Download a model

Download `SmolLM2-360M-Instruct-Q4_K_M.gguf` from the [v0.4.0 release](https://github.com/Quincunx33/LowRAM-AI-Compiler/releases/tag/v0.4.0). Create a local model directory and set `LOWRAM_MODEL` to the downloaded file:

```bash
mkdir -p models
# Place SmolLM2-360M-Instruct-Q4_K_M.gguf inside models/
export LOWRAM_MODEL="$PWD/models/SmolLM2-360M-Instruct-Q4_K_M.gguf"
```

Do not commit GGUF model binaries to Git history. Use GitHub Releases or another model store instead.

## Run the command-line tools

Plan a memory budget:

```bash
python3 -m lowram_ai plan \
  --device-ram-mb 1024 \
  --parameters 500000000 \
  --layers 16 \
  --hidden-size 2048 \
  --context 256 \
  --bits 4
```

Inspect a GGUF file and run generation:

```bash
python3 -m lowram_ai inspect-gguf "$LOWRAM_MODEL"
python3 -m lowram_ai generate "$LOWRAM_MODEL" "Explain memory mapping briefly." \
  --max-new-tokens 64 \
  --max-context 256 \
  --max-ram-mb 512
```

The included launcher is convenient for a local server:

```bash
./run_local_smollm2_360.sh
```

The launcher accepts environment overrides:

```bash
LOWRAM_MODEL=/absolute/path/model.gguf \\
LOWRAM_MAX_CONTEXT=256 \\
LOWRAM_MAX_RAM_MB=512 \\
LOWRAM_PORT=8766 \\
./run_local_smollm2_360.sh
```

## Run the web playground

Start the FastAPI server directly:

```bash
export LOWRAM_MODEL="$PWD/models/SmolLM2-360M-Instruct-Q4_K_M.gguf"
export LOWRAM_MAX_CONTEXT=256
export LOWRAM_MAX_RAM_MB=512
export LOWRAM_HOST=127.0.0.1
export LOWRAM_PORT=8766
python3 -m lowram_ai.api
```

Open `http://127.0.0.1:8766/` in a browser. The server exposes the responsive chat interface, streaming generation, conversation management, and developer workspace.

## HTTP API

The API accepts JSON requests. A minimal generation request is:

```bash
curl -X POST http://127.0.0.1:8766/v1/generate \\
  -H 'Content-Type: application/json' \\
  -d '{
    "prompt": "Explain what a quantized model is in one short paragraph.",
    "max_new_tokens": 64,
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "seed": 0
  }'
```

The response includes generated text, a conversation ID, history information, usage data, and the effective generation settings. Pass the returned `conversation_id` in later requests for bounded multi-turn context.

A streaming endpoint is available at `POST /v1/generate/stream` and returns Server-Sent Events. Each event contains a `delta`; the final event contains `done`, `conversation_id`, and usage information.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check whether the model is loaded |
| `GET /v1/model` | Read architecture, context, and memory estimates |
| `POST /v1/generate` | Generate one JSON response |
| `POST /v1/generate/stream` | Generate through SSE events |
| `POST /v1/conversations` | Create a conversation session |
| `DELETE /v1/conversations/{id}` | Delete a session |
| `GET /v1/workspace` | List workspace files and limits |
| `POST /v1/workspace/file` | Create or replace a text file |
| `POST /v1/workspace/upload` | Upload one or more files |
| `POST /v1/workspace/python` | Run restricted Python code |
| `POST /v1/workspace/zip` | Create a ZIP archive |
| `POST /v1/workspace/unzip` | Extract a ZIP archive safely |
| `GET /v1/workspace/download` | Download a workspace file |

For a deployed service, set `LOWRAM_API_KEY` and send the value in the `x-api-key` header. Also set `LOWRAM_RATE_LIMIT` to limit requests per minute:

```bash
export LOWRAM_API_KEY='replace-with-a-long-random-secret'
export LOWRAM_RATE_LIMIT=60
```

Never expose a production API key in browser JavaScript or commit it to Git. Keep it in the hosting provider’s secret/environment-variable settings.

## Developer workspace

The workspace is rooted in a controlled directory and rejects path traversal. It supports text-file creation, file reads, multi-file upload, safe ZIP creation/extraction, and restricted Python execution with a timeout. It is intended for small development experiments, not untrusted multi-tenant code execution or arbitrary system administration.

A Python workspace request looks like this:

```bash
curl -X POST http://127.0.0.1:8766/v1/workspace/python \\
  -H 'Content-Type: application/json' \\
  -d '{
    "filename": "main.py",
    "code": "print(2 + 2)"
  }'
```

## Tests and benchmarks

Run the unit test suite:

```bash
PYTHONPATH=. python3 -m unittest discover -s lowram_ai/tests -p 'test_*.py' -v
```

Run the end-to-end demo:

```bash
PYTHONPATH=. python3 -m lowram_ai.examples.demo
```

A real-model benchmark is available at `scripts/benchmark_real_model.py`. Measurements depend on the host CPU, compiler, model, context size, and operating-system memory behavior; do not treat repository benchmark numbers as guarantees for a specific phone or server.

## Memory and quality limitations

Quantization reduces model memory but also changes numerical precision. Context length consumes additional KV-cache memory, and the runtime resets the cache for each request while reconstructing bounded session history. A 512MB device should use a small model and a short context. A 1GB temporary host can run the 360M model more comfortably, but response quality remains below modern hosted models.

The small models included in this workflow are especially limited for Bengali, long reasoning, and complex code generation. If response quality is the priority, use a substantially larger model or a hosted inference service. If offline operation and low memory are the priority, reduce context and generation length and accept the quality trade-off.

## Deployment

The repository includes `Dockerfile`, `docker-compose.yml`, and `render.yaml` examples. A hosted deployment must provision the GGUF file through a release download, mounted storage, or a startup download step; the model should not be added to Git history. Set `LOWRAM_MODEL`, `LOWRAM_MAX_RAM_MB`, `LOWRAM_MAX_CONTEXT`, `LOWRAM_HOST`, `LOWRAM_PORT`, `LOWRAM_API_KEY`, and `LOWRAM_RATE_LIMIT` in the hosting environment.

Free hosting tiers may sleep when idle, have limited RAM/CPU, and use ephemeral filesystems. Treat them as demonstrations unless the provider offers persistent storage and an always-on service plan.

## License

This project is released under the Apache License 2.0.

## References

GGUF metadata and tensor-layout work follows the [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), the [llama.cpp model-architecture guide](https://github.com/ggml-org/ggml/blob/master/docs/development/HOWTO-add-model.md), and [Hugging Face GGUF documentation](https://huggingface.co/docs/hub/en/gguf).

Model links:

- [SmolLM2-360M-Instruct GGUF repository](https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF)
- [SmolLM2-360M Q4_K_M release asset](https://github.com/Quincunx33/LowRAM-AI-Compiler/releases/tag/v0.4.0)
