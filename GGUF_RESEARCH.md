# GGUF implementation notes

## Sources

1. [Official GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
2. [Hugging Face GGUF documentation](https://huggingface.co/docs/hub/en/gguf)

## Findings used in the MVP

The official specification describes GGUF as a single-file, extensible, mmap-compatible binary format. The file has a header containing the magic bytes `GGUF`, a version, tensor count, and metadata key-value count. Metadata values use typed entries, including unsigned and signed integers, floats, booleans, strings, and arrays.

Tensor metadata follows the header and metadata. Each tensor has a name, dimensions, a ggml tensor type, and an offset relative to the aligned tensor-data region. The global alignment is read from `general.alignment`; when omitted, the documented default is 32 bytes. Tensor offsets are aligned to this boundary.

The implementation starts with format parsing and safe metadata inspection. It intentionally supports a narrow set of tensor decoders first: F32, F16, Q4_0, and Q8_0. Q4_K and other K/I-quants require additional block-layout and lookup-table logic and are reserved for the next milestone.

Hugging Face documents GGUF as carrying both tensor data and standardized metadata, unlike tensor-only formats. Its quantization overview describes Q4_0 as 4-bit blocks of 32 weights with a block scale, and Q8_0 as 8-bit blocks of 32 weights with a block scale. These layouts are suitable for a small, independently testable first decoder.

## Llama wiring findings

The llama.cpp model-development guide identifies a sequence of conversion, architecture definition, graph implementation, and backend validation. It notes that GGML tensor dimensions are typically in reverse order compared with PyTorch dimensions. Common Llama GGUF metadata keys include `llama.block_count`, `llama.embedding_length`, `llama.attention.head_count`, `llama.attention.head_count_kv`, `llama.feed_forward_length`, `llama.context_length`, `llama.rope.dimension_count`, and `llama.attention.layer_norm_rms_epsilon`.

Common tensor names include `token_embd.weight`, `output_norm.weight`, `output.weight`, `blk.{layer}.attn_norm.weight`, `blk.{layer}.attn_q.weight`, `blk.{layer}.attn_k.weight`, `blk.{layer}.attn_v.weight`, `blk.{layer}.attn_output.weight`, `blk.{layer}.ffn_norm.weight`, `blk.{layer}.ffn_gate.weight`, `blk.{layer}.ffn_up.weight`, and `blk.{layer}.ffn_down.weight`. Tokenizer information is stored in metadata arrays such as `tokenizer.ggml.tokens`, `tokenizer.ggml.scores`, `tokenizer.ggml.merges`, plus `tokenizer.ggml.bos_token_id`, `eos_token_id`, and `unknown_token_id`.

The first full-generation implementation will target a narrow Llama-family layout and will fail clearly when required keys or tensors are missing. It will use metadata-first loading and file-backed tensor decoding rather than loading the entire GGUF file.

## Q4_K follow-up

The upstream `ggml-common.h` defines `block_q4_K` as a 256-value super-block containing two fp16 super-scales, 12 bytes of packed 6-bit sub-block scales/minima, and 128 bytes of 4-bit quants, for 144 bytes total. The reference `dequantize_row_q4_K` implementation in upstream `ggml-quants.c` is the source of truth for the exact scale unpacking and sub-block ordering. Q4_K support is therefore the next quantization milestone after this complete legacy-Q4 generation path.

The focused scalar reference confirms Q4_K ordering: each 256-value super-block has four 64-value spans. Each span uses 32 packed bytes; the low nibbles produce one contiguous 32-value sub-block and the high nibbles produce the next. The weight formula is `w = d * scale_q * q - dmin * min_q`. This means the decoder must process sub-blocks in pairs per 64-value span, not map each 32-byte region to one sub-block by parity across the whole block.

## Real-model compatibility audit

The selected public `bartowski/SmolLM2-135M-Instruct-Q4_K_M.gguf` file is approximately 101 MiB and reports `general.architecture=llama`, 30 layers, embedding length 576, feed-forward length 1536, 9 attention heads, 3 KV heads, and a 8192-token context. Its tokenizer metadata reports `tokenizer.ggml.model=gpt2`, with 49,152 tokens and embedded merges. Its tensor mix includes Q4_K, Q5_0, Q6_K, Q8_0, F32, and F16-compatible entries; therefore a real run requires Q5_0/Q6_K decoders and GPT-2 BPE tokenization in addition to the initial legacy-Q4 path.

The current `GGUFReader` retains a `Path` field, parses descriptors first, and owns a Python mmap that is closed via the context manager. The native backend can therefore open the same path independently and must be closed before the reader mapping is released.
