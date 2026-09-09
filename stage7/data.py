"""
Data pipeline for Stage 7 -- reuses the EXACT Stage 4 tokenizer and sequence
builder so the JAX model trains on identical data to the Stage 4 language model.

We deliberately reuse (not reimplement):
  - stage4/utils/tokenizer.py :: SimpleTokenizer  (loaded from the saved
    language_model_tokenizer.json -> identical vocab and token IDs)
  - stage4/utils/data.py       :: create_sequences (seq_len=64, stride=32)

The only new code is a small batch iterator that yields int32 arrays (JAX's
default int type) with reproducible per-epoch shuffling.
"""

import os
import importlib.util

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# Stage 4 defaults (see stage4/train_language_model.py and stage4_results.txt).
DATA_PATH = os.path.join(_ROOT, "data", "tinystories_train.txt")
TOKENIZER_PATH = os.path.join(_ROOT, "stage4", "checkpoints", "language_model_tokenizer.json")
MAX_CHARS = 5 * 1024 * 1024   # Stage 4 reads only the first 5 MB
SEQ_LEN = 64
STRIDE = 32                   # seq_len // 2, 50% overlap (Stage 4 value)
TRAIN_SPLIT = 0.9             # sequential 90/10 split


def _load_by_path(rel_path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Load the Stage 4 tokenizer + sequence builder by file path (their only imports
# are stdlib/numpy, so this is safe and avoids any package-name collisions).
_tok_mod = _load_by_path(os.path.join("stage4", "utils", "tokenizer.py"), "stage4_tokenizer")
_data_mod = _load_by_path(os.path.join("stage4", "utils", "data.py"), "stage4_data")
SimpleTokenizer = _tok_mod.SimpleTokenizer
create_sequences = _data_mod.create_sequences


def load_dataset(data_path=DATA_PATH, tokenizer_path=TOKENIZER_PATH,
                 max_chars=MAX_CHARS, seq_len=SEQ_LEN, stride=STRIDE,
                 train_split=TRAIN_SPLIT):
    """Return ((train_inputs, train_targets), (val_inputs, val_targets), tokenizer).

    Reproduces the Stage 4 pipeline exactly: first `max_chars` of TinyStories,
    encoded with the saved word-level tokenizer (no BOS/EOS), windowed into
    (input, next-token target) pairs, split sequentially 90/10. Arrays are int32.
    """
    tokenizer = SimpleTokenizer.load(tokenizer_path)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read(max_chars)

    # Same as Stage 4: encode with no special tokens injected into the stream.
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)

    inputs, targets = create_sequences(token_ids, seq_len, stride)  # int64 (n, 64)
    inputs = inputs.astype(np.int32)
    targets = targets.astype(np.int32)

    n_train = int(len(inputs) * train_split)
    train = (inputs[:n_train], targets[:n_train])
    val = (inputs[n_train:], targets[n_train:])
    return train, val, tokenizer


def iterate_batches(inputs, targets, batch_size, shuffle, rng=None):
    """Yield (input_batch, target_batch) int32 arrays.

    Mirrors Stage 4's TextDataLoader (no drop_last: the final partial batch is
    kept). Shuffling uses a passed-in np.random.Generator for reproducibility.
    """
    n = len(inputs)
    idx = np.arange(n)
    if shuffle:
        (rng if rng is not None else np.random).shuffle(idx)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        yield inputs[b], targets[b]


def num_batches(n, batch_size):
    return (n + batch_size - 1) // batch_size
