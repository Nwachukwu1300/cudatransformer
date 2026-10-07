"""
Stage 5 inference: shared loading + prediction logic.

This is the single source of truth for loading the trained recommender
checkpoint and running next-item prediction. Both `recommend.py` (the CLI
deliverable) and `webapp/backend/app.py` (the public web app) import from
here, so there is exactly one place that knows how to load the model and run
a forward pass -- no duplicated checkpoint-loading or path-juggling logic.

Uses the same `importlib.util`-based module loading as the rest of this repo
(stage directories are not real installable Python packages), rather than
introducing a different import style just for this file.
"""

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_stage2_path = os.path.join(_ROOT, "stage2")
_stage3_path = os.path.join(_ROOT, "stage3")
_stage4_path = os.path.join(_ROOT, "stage4")

if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor  # noqa: E402,F401  (re-exported for convenience)


def _load_module(base_path, rel_path, name):
    full_path = os.path.join(base_path, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Same transformer model as Stage 4.
_decoder_mod = _load_module(_stage3_path, "models/decoder_lm.py", "stage3_decoder")
DecoderLanguageModel = _decoder_mod.DecoderLanguageModel

# Reused Stage 4 checkpoint loader.
_checkpoint_mod = _load_module(_stage4_path, "utils/checkpoint.py", "stage4_checkpoint")
load_checkpoint = _checkpoint_mod.load_checkpoint

# Stage 5 item vocabulary + data helpers.
_item_vocab_mod = _load_module(_HERE, "utils/item_vocab.py", "stage5_item_vocab")
ItemVocab = _item_vocab_mod.ItemVocab
_ml_data_mod = _load_module(_HERE, "utils/data.py", "stage5_data")
load_movie_titles = _ml_data_mod.load_movie_titles
load_movielens_ratings = _ml_data_mod.load_movielens_ratings
build_user_sequences = _ml_data_mod.build_user_sequences

DEFAULT_CHECKPOINT_PATH = os.path.join(_HERE, "checkpoints", "recommender")
DEFAULT_MOVIES_PATH = os.path.join(_HERE, "data", "ml-1m", "movies.dat")


def load_recommender(checkpoint_path=None):
    """
    Load the trained recommender checkpoint.

    Args:
        checkpoint_path: Base path for the checkpoint, without extension.
                          Defaults to stage5/checkpoints/recommender.

    Returns:
        (model, vocab, config, training_info) -- same shape as
        stage4/utils/checkpoint.py::load_checkpoint.
    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    return load_checkpoint(checkpoint_path, DecoderLanguageModel, ItemVocab)


def predict_next_items(model, vocab, history_movie_ids, top_k=5):
    """
    Predict the next items for a user given their movie history.

    Encode the history, run one forward pass, read the logits at the last
    position, and return the highest-scoring items (excluding movies already
    seen and special tokens).

    Args:
        model: A loaded DecoderLanguageModel.
        vocab: The matching ItemVocab.
        history_movie_ids: List of movie IDs (ints), most recent last.
        top_k: Number of recommendations to return.

    Returns:
        List of (movie_id, score) tuples, highest score first.
    """
    token_ids = vocab.encode(history_movie_ids, add_bos=False, add_eos=False)
    if len(token_ids) == 0:
        return []
    if len(token_ids) > model.max_seq_len:
        token_ids = token_ids[-model.max_seq_len:]

    input_array = np.array([token_ids], dtype=np.int64)
    logits = model(input_array)
    last_logits = logits.data[0, -1, :]

    seen = set(vocab.encode(history_movie_ids))
    special = {vocab.get_pad_id(), vocab.get_bos_id(),
               vocab.get_eos_id(), vocab.token_to_id[vocab.UNK_TOKEN]}

    ranked = np.argsort(last_logits)[::-1]
    results = []
    for token_id in ranked:
        token_id = int(token_id)
        if token_id in special or token_id in seen:
            continue
        movie_id = vocab.id_to_movie(token_id)
        if movie_id is None:
            continue
        results.append((movie_id, float(last_logits[token_id])))
        if len(results) >= top_k:
            break
    return results
