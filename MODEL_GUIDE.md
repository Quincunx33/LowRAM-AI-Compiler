# Recommended model configurations

## Temporary hosted playground

For the temporary hosted playground, the recommended model is **Llama 3.2 1B Instruct Q4_K_M**. It uses the runtime’s supported `llama` architecture and is substantially stronger than the 135M and 360M models for general instruction following and English chat.

The GGUF file is approximately 771 MiB. With a 512-token context, the runtime reported approximately 800 MB of model weights and 16 MB of KV cache. The temporary server should therefore use at least a 1 GB model budget, with additional headroom for Python, the operating system, and native buffers.

Start it with:

```bash
LOWRAM_MODEL=/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf \\
LOWRAM_MAX_RAM_MB=1024 LOWRAM_MAX_CONTEXT=512 \\
PYTHONPATH=. python3 -m lowram_ai.api
```

The included `run_local_smollm2_360.sh` launcher now defaults to this model name and a 1 GB budget. Place the model in the project `models/` directory or set `LOWRAM_MODEL` to its full path.

## Offline low-memory fallback

For a real 512 MB–1 GB device, **SmolLM2-360M-Instruct Q4_K_M** remains the safer fallback because its model file is approximately 259 MiB. It is faster and uses less RAM, but its answer quality is noticeably weaker.

Neither small local model should be expected to match Gemini or ChatGPT. Bengali quality may remain inconsistent, especially with a 128-token context. The hosted 1B option improves general responses but still runs on CPU and may be slow.

Model sources:

- [Llama 3.2 1B Instruct Q4_K_M GGUF](https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF)
- [SmolLM2 360M Instruct GGUF](https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF)
