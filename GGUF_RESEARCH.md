# GGUF implementation notes

## Sources

1. [Official GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
2. [Hugging Face GGUF documentation](https://huggingface.co/docs/hub/en/gguf)

## Findings used in the MVP

The official specification describes GGUF as a single-file, extensible, mmap-compatible binary format. The file has a header containing the magic bytes `GGUF`, a version, tensor count, and metadata key-value count. Metadata values use typed entries, including unsigned and signed integers, floats, booleans, strings, and arrays.

Tensor metadata follows the header and metadata. Each tensor has a name, dimensions, a ggml tensor type, and an offset relative to the aligned tensor-data region. The global alignment is read from `general.alignment`; when omitted, the documented default is 32 bytes. Tensor offsets are aligned to this boundary.

The implementation starts with format parsing and safe metadata inspection. It intentionally supports a narrow set of tensor decoders first: F32, F16, Q4_0, and Q8_0. Q4_K and other K/I-quants require additional block-layout and lookup-table logic and are reserved for the next milestone.

Hugging Face documents GGUF as carrying both tensor data and standardized metadata, unlike tensor-only formats. Its quantization overview describes Q4_0 as 4-bit blocks of 32 weights with a block scale, and Q8_0 as 8-bit blocks of 32 weights with a block scale. These layouts are suitable for a small, independently testable first decoder.
