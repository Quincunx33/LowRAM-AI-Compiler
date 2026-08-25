"""Dependency-light tokenizers backed entirely by GGUF metadata.

The implementation supports common Llama SentencePiece-style pieces and GPT-2
byte-level BPE vocabularies such as the real SmolLM2 GGUF test model. It avoids
external tokenizer files and keeps the vocabulary/merge tables immutable after
initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _gpt2_byte_encoder() -> dict[int, str]:
    bytes_list = list(range(ord("!"), ord("~") + 1))
    bytes_list += list(range(ord("¡"), ord("¬") + 1))
    bytes_list += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(bytes_list)
    missing = 0
    for byte in range(256):
        if byte not in bytes_list:
            bytes_list.append(byte)
            codepoints.append(256 + missing)
            missing += 1
    return dict(zip(bytes_list, map(chr, codepoints)))


@dataclass
class Tokenizer:
    tokens: list[str]
    scores: list[float] | None = None
    bos_id: int | None = None
    eos_id: int | None = None
    unk_id: int | None = None
    model: str = "llama"
    merges: list[str] | None = None
    add_bos_token: bool = True
    add_space_prefix: bool = False

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "Tokenizer":
        raw_tokens = metadata.get("tokenizer.ggml.tokens")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValueError("GGUF is missing tokenizer.ggml.tokens")
        raw_scores = metadata.get("tokenizer.ggml.scores")
        scores = [float(score) for score in raw_scores] if isinstance(raw_scores, list) else None
        raw_merges = metadata.get("tokenizer.ggml.merges")
        merges = [str(item) for item in raw_merges] if isinstance(raw_merges, list) else None
        return cls(
            tokens=[str(token) for token in raw_tokens],
            scores=scores,
            bos_id=_optional_int(metadata.get("tokenizer.ggml.bos_token_id")),
            eos_id=_optional_int(metadata.get("tokenizer.ggml.eos_token_id")),
            unk_id=_optional_int(metadata.get("tokenizer.ggml.unknown_token_id")),
            model=str(metadata.get("tokenizer.ggml.model", "llama")),
            merges=merges,
            add_bos_token=bool(metadata.get("tokenizer.ggml.add_bos_token", True)),
            add_space_prefix=bool(metadata.get("tokenizer.ggml.add_space_prefix", False)),
        )

    def __post_init__(self) -> None:
        self.piece_to_id: dict[str, int] = {}
        for index, piece in enumerate(self.tokens):
            self.piece_to_id.setdefault(piece, index)
        self._pieces = sorted(
            (
                piece
                for piece in self.piece_to_id
                if piece and not piece.startswith("<0x") and not piece.startswith("<|")
            ),
            key=lambda piece: (-len(piece), piece),
        )
        self._byte_tokens = {
            piece.upper(): index
            for index, piece in enumerate(self.tokens)
            if len(piece) == 6 and piece[:3] == "<0x" and piece[-1] == ">"
        }
        self._gpt2_encoder = _gpt2_byte_encoder()
        self._gpt2_decoder = {value: byte for byte, value in self._gpt2_encoder.items()}
        self._merge_ranks: dict[tuple[str, str], int] = {}
        if self.merges:
            for rank, merge in enumerate(self.merges):
                pair = merge.split(" ", 1)
                if len(pair) == 2:
                    self._merge_ranks[(pair[0], pair[1])] = rank

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def _normalized_text(self, text: str) -> str:
        return "▁" + text.replace(" ", "▁")

    def _gpt2_bpe(self, text: str) -> list[str]:
        symbols = list(text)
        while len(symbols) > 1:
            ranked_pairs = [
                (self._merge_ranks[pair], index, pair)
                for index, pair in enumerate(zip(symbols, symbols[1:]))
                if pair in self._merge_ranks
            ]
            if not ranked_pairs:
                break
            _, index, _ = min(ranked_pairs)
            symbols[index : index + 2] = [symbols[index] + symbols[index + 1]]
        return symbols

    def encode(self, text: str, *, add_bos: bool | None = None) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        add_bos = self.add_bos_token if add_bos is None else add_bos
        result: list[int] = []
        if add_bos and self.bos_id is not None:
            result.append(self.bos_id)
        if self.model == "gpt2":
            encoded = "".join(self._gpt2_encoder[byte] for byte in text.encode("utf-8"))
            for piece in self._gpt2_bpe(encoded):
                token_id = self.piece_to_id.get(piece)
                if token_id is not None:
                    result.append(token_id)
                elif self.unk_id is not None:
                    result.append(self.unk_id)
                else:
                    raise ValueError(f"tokenizer cannot encode BPE piece: {piece!r}")
            return result

        normalized = self._normalized_text(text)
        position = 0
        while position < len(normalized):
            match = next((piece for piece in self._pieces if normalized.startswith(piece, position)), None)
            if match is not None:
                result.append(self.piece_to_id[match])
                position += len(match)
                continue
            if normalized[position] == "▁":
                position += 1
                continue
            byte_id = self._byte_tokens.get(f"<0x{ord(normalized[position]):02X}>")
            if byte_id is not None:
                result.append(byte_id)
            elif self.unk_id is not None:
                result.append(self.unk_id)
            else:
                raise ValueError(f"tokenizer cannot encode character: {normalized[position]!r}")
            position += 1
        return result

    def decode(self, token_ids: list[int]) -> str:
        special_ids = {item for item in (self.bos_id, self.eos_id, self.unk_id) if item is not None}
        if self.model == "gpt2":
            encoded = "".join(
                self.tokens[token_id]
                for token_id in token_ids
                if 0 <= token_id < len(self.tokens) and token_id not in special_ids
            )
            raw = bytes(self._gpt2_decoder[character] for character in encoded if character in self._gpt2_decoder)
            return raw.decode("utf-8", errors="replace")

        pieces: list[str] = []
        byte_buffer = bytearray()
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.tokens):
                raise ValueError(f"token id out of range: {token_id}")
            if token_id in special_ids:
                continue
            piece = self.tokens[token_id]
            if len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"):
                try:
                    byte_buffer.append(int(piece[3:5], 16))
                    continue
                except ValueError:
                    pass
            if byte_buffer:
                pieces.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()
            pieces.append(piece.replace("▁", " "))
        if byte_buffer:
            pieces.append(byte_buffer.decode("utf-8", errors="replace"))
        return "".join(pieces).lstrip()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
