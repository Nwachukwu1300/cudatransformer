"""
Data utilities for transformer training.

Generate toy sequence data and provide a simple dataloader.
"""

import numpy as np
from typing import Tuple, Iterator


def generate_repeat_data(
    num_samples: int,
    seq_len: int,
    pattern_len: int = 2,
    vocab_size: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate repeat pattern data.

    Each sequence is a repeating pattern. The model learns to predict
    the next token in the pattern.

    Example with pattern [1, 2]:
        Input:  [1, 2, 1, 2, 1, 2, 1, 2]
        Target: [2, 1, 2, 1, 2, 1, 2, 1]  (next token at each position)

    Args:
        num_samples: Number of sequences to generate
        seq_len: Length of each sequence
        pattern_len: Length of the repeating pattern
        vocab_size: Size of vocabulary (tokens are 1 to vocab_size-1, 0 reserved)

    Returns:
        inputs: Array of shape (num_samples, seq_len)
        targets: Array of shape (num_samples, seq_len)
    """
    inputs = np.zeros((num_samples, seq_len), dtype=np.int64)
    targets = np.zeros((num_samples, seq_len), dtype=np.int64)

    for i in range(num_samples):
        # Generate random pattern (tokens 1 to vocab_size-1)
        pattern = np.random.randint(1, vocab_size, size=pattern_len)

        # Fill sequence with repeating pattern
        for j in range(seq_len):
            inputs[i, j] = pattern[j % pattern_len]
            # Target is the next token in the pattern
            targets[i, j] = pattern[(j + 1) % pattern_len]

    return inputs, targets


def generate_copy_data(
    num_samples: int,
    seq_len: int,
    copy_len: int = 3,
    vocab_size: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate copy task data.

    The model learns to copy a sequence of tokens.
    First `copy_len` tokens are random, followed by zeros,
    then the model should predict the copied tokens.

    Example:
        Input:  [3, 1, 4, 0, 0, 0, 0, 0]
        Target: [1, 4, 0, 0, 0, 3, 1, 4]  (shifted for next-token)

    Args:
        num_samples: Number of sequences to generate
        seq_len: Length of each sequence
        copy_len: Number of tokens to copy
        vocab_size: Size of vocabulary

    Returns:
        inputs: Array of shape (num_samples, seq_len)
        targets: Array of shape (num_samples, seq_len)
    """
    inputs = np.zeros((num_samples, seq_len), dtype=np.int64)
    targets = np.zeros((num_samples, seq_len), dtype=np.int64)

    for i in range(num_samples):
        # Random tokens for copying
        tokens = np.random.randint(1, vocab_size, size=copy_len)

        # Place tokens at start
        inputs[i, :copy_len] = tokens

        # Target is next token prediction
        for j in range(seq_len - 1):
            targets[i, j] = inputs[i, j + 1] if j + 1 < copy_len else 0

        # After delimiter, copy the sequence
        if seq_len > copy_len + 1:
            copy_start = copy_len + 1
            for j in range(min(copy_len, seq_len - copy_start)):
                targets[i, copy_start + j - 1] = tokens[j]

    return inputs, targets


class SequenceDataLoader:
    """
    Simple DataLoader for sequence data.

    Iterates over batches of input/target sequence pairs.
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
            inputs: Input sequences of shape (num_samples, seq_len)
            targets: Target sequences of shape (num_samples, seq_len)
            batch_size: Number of samples per batch
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
