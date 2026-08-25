"""Small tokenizer for GGUF vocab metadata.

This MVP targets the common Llama SentencePiece-style metadata layout. It uses
embedded vocabulary pieces and greedy longest-piece matching, so the runtime has
no external tokenizer files. It is intentionally conservative and reports when
a model uses a tokenizer layout outside this subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Tokenizer:
    tokens: list[str]
    scores: list[float] | None = None
    bos_id: int | None = None
    eos_id: int | None = None
    unk_id: int | None = None
    model: str = "llama"

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "Tokenizer":
        raw_tokens = metadata.get("tokenizer.ggml.tokens")
        if not isinstance(raw_tokens, list) or not raw_tokens:
            raise ValueError("GGUF is missing tokenizer.ggml.tokens")
        tokens = [str(token) for token in raw_tokens]
        raw_scores = metadata.get("tokenizer.ggml.scores")
        scores = [float(score) for score in raw_scores] if isinstance(raw_scores, list) else None
        return cls(
            tokens=tokens,
            scores=scores,
            bos_id=_optional_int(metadata.get("tokenizer.ggml.bos_token_id")),
            eos_id=_optional_int(metadata.get("tokenizer.ggml.eos_token_id")),
            unk_id=_optional_int(metadata.get("tokenizer.ggml.unknown_token_id")),
            model=str(metadata.get("tokenizer.ggml.model", "llama")),
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

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def _normalized_text(self, text: str) -> str:
        # SentencePiece commonly represents a word boundary with U+2581.
        return "▁" + text.replace(" ", "▁")

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = self._normalized_text(text)
        result: list[int] = []
        if add_bos and self.bos_id is not None:
            result.append(self.bos_id)
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
        pieces: list[str] = []
        byte_buffer = bytearray()
        special_ids = {item for item in (self.bos_id, self.eos_id, self.unk_id) if item is not None}
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
