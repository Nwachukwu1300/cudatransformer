"""
Data loading utilities for language model training.
"""

import numpy as np
from typing import List, Tuple, Iterator
import os


def load_text_file(path: str) -> str:
    """Load text from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def create_sequences(
    token_ids: List[int],
    seq_len: int,
    stride: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/target sequences for language modeling.

    For language modeling, input is tokens[:-1] and target is tokens[1:].

    Args:
        token_ids: List of token IDs from encoded text
        seq_len: Length of each sequence
        stride: Stride between sequences (default: seq_len for no overlap)

    Returns:
        inputs: Array of shape (num_sequences, seq_len)
        targets: Array of shape (num_sequences, seq_len)
    """
    if stride is None:
        stride = seq_len

    inputs = []
    targets = []

    for i in range(0, len(token_ids) - seq_len, stride):
        inp = token_ids[i:i + seq_len]
        tgt = token_ids[i + 1:i + seq_len + 1]

        if len(inp) == seq_len and len(tgt) == seq_len:
            inputs.append(inp)
            targets.append(tgt)

    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)


class TextDataLoader:
    """
    DataLoader for text sequences.
    """

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ):
        """
        Initialize dataloader.

        Args:
            inputs: Input sequences of shape (num_sequences, seq_len)
            targets: Target sequences of shape (num_sequences, seq_len)
            batch_size: Number of sequences per batch
            shuffle: Whether to shuffle data each epoch
        """
        assert len(inputs) == len(targets), "inputs and targets must have same length"

        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(inputs)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        indices = np.arange(self.num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield self.inputs[batch_idx], self.targets[batch_idx]

    def __len__(self) -> int:
        """Number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def prepare_language_model_data(
    text_path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    train_split: float = 0.9
) -> Tuple[TextDataLoader, TextDataLoader]:
    """
    Prepare data loaders for language model training.

    Args:
        text_path: Path to text file
        tokenizer: Tokenizer to use
        seq_len: Sequence length
        batch_size: Batch size
        train_split: Fraction of data for training

    Returns:
        train_loader, val_loader
    """
    # Load text
    text = load_text_file(text_path)

    # Encode entire text
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)

    print(f"  Total tokens: {len(token_ids)}")

    # Create sequences
    inputs, targets = create_sequences(token_ids, seq_len, stride=seq_len // 2)

    print(f"  Total sequences: {len(inputs)}")

    # Split into train/val
    num_train = int(len(inputs) * train_split)
    train_inputs, train_targets = inputs[:num_train], targets[:num_train]
    val_inputs, val_targets = inputs[num_train:], targets[num_train:]

    print(f"  Train sequences: {len(train_inputs)}")
    print(f"  Val sequences: {len(val_inputs)}")

    # Create data loaders
    train_loader = TextDataLoader(train_inputs, train_targets, batch_size, shuffle=True)
    val_loader = TextDataLoader(val_inputs, val_targets, batch_size, shuffle=False)

    return train_loader, val_loader
