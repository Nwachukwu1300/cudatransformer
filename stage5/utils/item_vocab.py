"""
Item vocabulary for sequential recommendation.

This is the Stage 4 vocabulary approach (see stage4/utils/tokenizer.py::SimpleTokenizer)
pointed at item IDs instead of words. Instead of mapping words -> integer IDs, it maps
movie IDs -> integer IDs. Everything downstream (embedding lookup, transformer, output
projection, checkpointing) is identical, because to the model a token is just an integer.

The on-disk format and the save/load/encode/decode interface intentionally mirror
SimpleTokenizer so that stage4/utils/checkpoint.py works unchanged for Stage 5.
"""

import json
from typing import List, Dict, Optional, Iterable
from collections import Counter


class ItemVocab:
    """
    Vocabulary mapping item (movie) IDs to contiguous integer token IDs.

    Special tokens occupy the first four IDs, exactly as SimpleTokenizer does,
    so the two are structurally interchangeable from the model's point of view.
    """

    # Special tokens (same layout as SimpleTokenizer)
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"  # Beginning of a user's history
    EOS_TOKEN = "<EOS>"  # End of a user's history

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Args:
            vocab: Optional pre-built mapping from token string to integer ID.
                   Item tokens are stored as the string form of the movie ID
                   (e.g. "1193"), since JSON object keys must be strings.
        """
        if vocab is not None:
            self.token_to_id = vocab
            self.id_to_token = {v: k for k, v in vocab.items()}
        else:
            self.token_to_id = {}
            self.id_to_token = {}

        self.vocab_size = len(self.token_to_id)

    @classmethod
    def from_interactions(
        cls,
        movie_ids: Iterable[int],
        max_vocab_size: Optional[int] = None,
        min_freq: int = 1,
    ) -> "ItemVocab":
        """
        Build a vocabulary from a stream of interacted movie IDs.

        Items are added most-frequent-first (like the word tokenizer), so the most
        popular movies get the lowest IDs. This is purely conventional and does not
        affect the math.

        Args:
            movie_ids: Iterable of movie IDs (ints) across all interactions.
            max_vocab_size: Optional cap on total vocabulary size (incl. specials).
            min_freq: Minimum number of interactions for a movie to be included.
        """
        counts = Counter(int(m) for m in movie_ids)

        vocab = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
            cls.BOS_TOKEN: 2,
            cls.EOS_TOKEN: 3,
        }

        for movie_id, count in counts.most_common():
            if max_vocab_size is not None and len(vocab) >= max_vocab_size:
                break
            if count >= min_freq:
                key = str(movie_id)
                if key not in vocab:
                    vocab[key] = len(vocab)

        return cls(vocab)

    def encode(
        self,
        movie_ids: List[int],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Encode a list of movie IDs to token IDs.

        Unknown movies map to <UNK>.
        """
        tokens = []
        if add_bos:
            tokens.append(self.token_to_id[self.BOS_TOKEN])

        for movie_id in movie_ids:
            key = str(int(movie_id))
            if key in self.token_to_id:
                tokens.append(self.token_to_id[key])
            else:
                tokens.append(self.token_to_id[self.UNK_TOKEN])

        if add_eos:
            tokens.append(self.token_to_id[self.EOS_TOKEN])

        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> List[int]:
        """
        Decode token IDs back to movie IDs.

        Returns a list of movie IDs (ints). Special tokens are skipped by default.
        """
        special_ids = {
            self.token_to_id[self.PAD_TOKEN],
            self.token_to_id[self.UNK_TOKEN],
            self.token_to_id[self.BOS_TOKEN],
            self.token_to_id[self.EOS_TOKEN],
        }

        movie_ids = []
        for token_id in token_ids:
            if skip_special and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id)
            if token is not None and token not in (
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ):
                movie_ids.append(int(token))

        return movie_ids

    def id_to_movie(self, token_id: int) -> Optional[int]:
        """Map a single token ID back to its movie ID (None for special tokens)."""
        token = self.id_to_token.get(token_id)
        if token is None or token in (
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ):
            return None
        return int(token)

    def get_pad_id(self) -> int:
        return self.token_to_id[self.PAD_TOKEN]

    def get_bos_id(self) -> int:
        return self.token_to_id[self.BOS_TOKEN]

    def get_eos_id(self) -> int:
        return self.token_to_id[self.EOS_TOKEN]

    def save(self, path: str):
        """Save vocabulary to JSON (same format as SimpleTokenizer)."""
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ItemVocab":
        """Load vocabulary from JSON."""
        with open(path, "r") as f:
            vocab = json.load(f)
        return cls(vocab)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self):
        return f"ItemVocab(vocab_size={self.vocab_size})"
