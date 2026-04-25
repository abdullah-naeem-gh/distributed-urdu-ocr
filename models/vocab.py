"""
Vocabulary for Urdu OCR
========================
Character-level vocabulary with special tokens, serializable to JSON
for deployment.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class Vocabulary:
    """
    Character-level vocabulary for Urdu text.

    Maps individual characters to integer indices.
    Supports save/load for deployment without retraining.
    """

    def __init__(self):
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self._built = False

        # Always have special tokens at fixed positions
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.char2idx[token] = idx
            self.idx2char[idx] = token

    @property
    def size(self) -> int:
        return len(self.char2idx)

    def build_from_texts(self, texts: List[str],
                         chars_file: Optional[str] = None) -> "Vocabulary":
        """
        Build vocabulary from a list of text strings.
        Optionally seed from a chars.txt file (one char per line).
        """
        chars = set()

        # Seed from chars file if provided (e.g., UHWR/chars.txt)
        if chars_file is not None:
            path = Path(chars_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        ch = line.strip()
                        if ch:
                            chars.add(ch)

        # Extract all unique characters from texts
        for text in texts:
            if text:
                for ch in str(text):
                    chars.add(ch)

        # Sort for deterministic ordering
        sorted_chars = sorted(chars)

        # Assign indices starting after special tokens
        idx = len(SPECIAL_TOKENS)
        for ch in sorted_chars:
            if ch not in self.char2idx:
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch
                idx += 1

        self._built = True
        return self

    def encode(self, text: str, add_sos: bool = True,
               add_eos: bool = True) -> List[int]:
        """Convert text string to list of token indices."""
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        for ch in text:
            indices.append(self.char2idx.get(ch, UNK_IDX))
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int],
               strip_special: bool = True) -> str:
        """Convert list of token indices back to text string."""
        chars = []
        for idx in indices:
            token = self.idx2char.get(idx, UNK_TOKEN)
            if strip_special and token in SPECIAL_TOKENS:
                if token == EOS_TOKEN:
                    break  # Stop at EOS
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls()
        vocab.char2idx = data["char2idx"]
        vocab.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        vocab._built = True
        return vocab

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size}, special={len(SPECIAL_TOKENS)})"
