"""
MovieLens 1M loading and per-user sequence construction for Stage 5.

The only conceptual difference from Stage 4's text data pipeline is that sequences
are built PER USER: each user's movies, sorted by timestamp, form one sequence.
We never slide a window across a user boundary, because "the next movie this user
watched" is only meaningful within a single user's history. Otherwise this mirrors
stage4/utils/data.py exactly (same (input, target) next-step shift, same batching).
"""

import os
import zipfile
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download_movielens(dest_dir: str) -> str:
    """
    Download and extract MovieLens 1M if not already present.

    Returns the path to the extracted ml-1m directory.
    """
    os.makedirs(dest_dir, exist_ok=True)
    extracted_dir = os.path.join(dest_dir, "ml-1m")

    if os.path.exists(os.path.join(extracted_dir, "ratings.dat")):
        return extracted_dir

    zip_path = os.path.join(dest_dir, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print(f"  Downloading MovieLens 1M from {MOVIELENS_1M_URL} ...")
        urllib.request.urlretrieve(MOVIELENS_1M_URL, zip_path)

    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    if not os.path.exists(os.path.join(extracted_dir, "ratings.dat")):
        raise FileNotFoundError(
            f"MovieLens 1M ratings.dat not found under {extracted_dir} after extraction."
        )
    return extracted_dir


def load_movielens_ratings(ratings_path: str) -> List[Tuple[int, int, int]]:
    """
    Parse ratings.dat (userId::movieId::rating::timestamp).

    Returns a list of (user_id, movie_id, timestamp) tuples. Ratings themselves are
    not used as a filter: this is implicit-feedback sequential recommendation, so
    every interaction (regardless of rating value) is part of the user's sequence.
    """
    interactions = []
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id, movie_id, _rating, timestamp = parts
            interactions.append((int(user_id), int(movie_id), int(timestamp)))
    return interactions


def load_movie_titles(movies_path: str) -> Dict[int, str]:
    """Parse movies.dat (movieId::title::genres) -> {movie_id: title}."""
    titles = {}
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 2:
                continue
            movie_id, title = parts[0], parts[1]
            titles[int(movie_id)] = title
    return titles


def build_user_sequences(
    interactions: List[Tuple[int, int, int]]
) -> Dict[int, List[int]]:
    """
    Group interactions by user and sort each user's movies by timestamp.

    Returns {user_id: [movie_id, ...]} in chronological order.
    """
    by_user = defaultdict(list)
    for user_id, movie_id, timestamp in interactions:
        by_user[user_id].append((timestamp, movie_id))

    user_sequences = {}
    for user_id, items in by_user.items():
        items.sort(key=lambda t: t[0])  # sort by timestamp
        user_sequences[user_id] = [movie_id for _ts, movie_id in items]

    return user_sequences


def create_user_sequences(
    user_sequences: Dict[int, List[int]],
    vocab,
    seq_len: int,
    stride: int = None,
    min_seq_len: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn per-user movie histories into (input, target) next-item pairs.

    For each user, we encode their movie IDs to token IDs and slide a window of
    length seq_len over that single user's history. input = tokens[i:i+seq_len],
    target = tokens[i+1:i+seq_len+1] -- identical to language-model next-token
    framing, just confined to one user. Short histories (< seq_len+1) are left-padded
    with <PAD> so they still yield one training example.

    Args:
        user_sequences: {user_id: [movie_id, ...]} chronological.
        vocab: ItemVocab used to encode movie IDs.
        seq_len: Window length.
        stride: Window stride (default seq_len // 2, like Stage 4).
        min_seq_len: Skip users with fewer than this many interactions.

    Returns:
        inputs, targets: int64 arrays of shape (num_sequences, seq_len).
    """
    if stride is None:
        stride = max(1, seq_len // 2)

    pad_id = vocab.get_pad_id()
    inputs = []
    targets = []

    for movie_ids in user_sequences.values():
        if len(movie_ids) < min_seq_len:
            continue

        tokens = vocab.encode(movie_ids, add_bos=False, add_eos=False)

        # A user with fewer than seq_len+1 tokens still gives one left-padded example.
        if len(tokens) < seq_len + 1:
            padded = [pad_id] * (seq_len + 1 - len(tokens)) + tokens
            inputs.append(padded[:seq_len])
            targets.append(padded[1:seq_len + 1])
            continue

        for i in range(0, len(tokens) - seq_len, stride):
            inp = tokens[i:i + seq_len]
            tgt = tokens[i + 1:i + seq_len + 1]
            if len(inp) == seq_len and len(tgt) == seq_len:
                inputs.append(inp)
                targets.append(tgt)

    return (
        np.array(inputs, dtype=np.int64),
        np.array(targets, dtype=np.int64),
    )
