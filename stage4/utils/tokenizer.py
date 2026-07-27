"""
Simple word-level tokenizer for language modeling.
"""

import re
import json
from typing import List, Dict, Optional
from collections import Counter


class SimpleTokenizer:
    """
    Simple word-level tokenizer.

    Converts text to token IDs and back.
    Uses a fixed vocabulary built from training data.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # Beginning of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize tokenizer.

        Args:
            vocab: Optional pre-built vocabulary mapping tokens to IDs
        """
        if vocab is not None:
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            self.token_to_id = {}
            self.id_to_token = {}

        self.vocab_size = len(self.token_to_id)

    @classmethod
    def from_text(cls, text: str, max_vocab_size: int = 1000, min_freq: int = 2) -> 'SimpleTokenizer':
        """
        Build tokenizer vocabulary from text.

        Args:
            text: Training text to build vocabulary from
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a word to be included

        Returns:
            SimpleTokenizer with built vocabulary
        """
        # Tokenize text into words
        words = cls._tokenize_text(text)

        # Count word frequencies
        word_counts = Counter(words)

        # Start with special tokens
        vocab = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.BOS_TOKEN: 2,
            cls.EOS_TOKEN: 3,
        }

        # Add most common words up to max_vocab_size
        for word, count in word_counts.most_common():
            if len(vocab) >= max_vocab_size:
                break
            if count >= min_freq and word not in vocab:
                vocab[word] = len(vocab)

        return cls(vocab)

    @staticmethod
    def _tokenize_text(text: str) -> List[str]:
        """
        Tokenize text into words.

        Simple tokenization: lowercase, split on whitespace and punctuation.
        """
        # Lowercase
        text = text.lower()

        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:\'\"])', r' \1 ', text)

        # Split on whitespace
        words = text.split()

        # Filter empty strings
        words = [w.strip() for w in words if w.strip()]

        return words

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode
            add_bos: Add beginning of sequence token
            add_eos: Add end of sequence token

        Returns:
            List of token IDs
        """
        words = self._tokenize_text(text)

        tokens = []
        if add_bos:
            tokens.append(self.token_to_id[self.BOS_TOKEN])

        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                tokens.append(self.token_to_id[self.UNK_TOKEN])

        if add_eos:
            tokens.append(self.token_to_id[self.EOS_TOKEN])

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special: Skip special tokens in output

        Returns:
            Decoded text
        """
        special_ids = {
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.BOS_TOKEN],
            self.token_to_id[self.EOS_TOKEN],
        }

        words = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                if skip_special and token_id in special_ids:
                    continue
                words.append(self.id_to_token[token_id])
            else:
                words.append(self.UNK_TOKEN)

        # Join words with spaces, fixing punctuation spacing
        text = ' '.join(words)

        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?;:\'\"])', r'\1', text)

        return text

    def get_pad_id(self) -> int:
        """Get padding token ID."""
        return self.token_to_id[self.PAD_TOKEN]

    def get_bos_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.token_to_id[self.BOS_TOKEN]

    def get_eos_id(self) -> int:
        """Get end of sequence token ID."""
        return self.token_to_id[self.EOS_TOKEN]

    def save(self, path: str):
        """Save tokenizer vocabulary to file."""
        with open(path, 'w') as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load tokenizer vocabulary from file."""
        with open(path, 'r') as f:
            vocab = json.load(f)
        return cls(vocab)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self):
        return f"SimpleTokenizer(vocab_size={self.vocab_size})"
