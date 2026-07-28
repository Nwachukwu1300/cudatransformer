"""
Stage 5 Deliverable: Next-Item Recommendation

Load the trained recommender checkpoint, take a user's movie history, and predict
the next movie they're likely to pick. This is the recommendation analogue of Stage
4's generate.py -- same model, same forward pass, same "read the logits at the last
position" mechanic. The only difference is that the vocabulary maps movie IDs (not
words), so the predicted token IDs decode back to movies.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

_base_path = os.path.dirname(os.path.abspath(__file__))
_stage2_path = os.path.join(_base_path, "stage2")
_stage3_path = os.path.join(_base_path, "stage3")
_stage4_path = os.path.join(_base_path, "stage4")
_stage5_path = os.path.join(_base_path, "stage5")

if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor  # noqa: E402


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
_item_vocab_mod = _load_module(_stage5_path, "utils/item_vocab.py", "stage5_item_vocab")
ItemVocab = _item_vocab_mod.ItemVocab
_ml_data_mod = _load_module(_stage5_path, "utils/data.py", "stage5_data")


def predict_next_items(model, vocab, history_movie_ids, top_k=5):
    """
    Predict the next items for a user given their movie history.

    Encode the history, run one forward pass, read the logits at the last position,
    and return the highest-scoring items (excluding movies already seen and specials).
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


def main():
    parser = argparse.ArgumentParser(
        description="Predict the next movie for a user from their history"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(_stage5_path, "checkpoints", "recommender"),
        help="Path to checkpoint (without extension)",
    )
    parser.add_argument(
        "--history", type=str, default=None,
        help="Comma-separated movie IDs, most recent last. "
             "If omitted, a real MovieLens user's history is used.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--user-id", type=int, default=None,
                        help="Use this MovieLens user's real history as the prompt.")
    args = parser.parse_args()

    print("=" * 70)
    print("Next-Item Recommendation - CUDA Transformer Engine")
    print("=" * 70)
    print()

    print(f"Loading checkpoint from: {args.checkpoint}")
    model, vocab, config, training_info = load_checkpoint(
        args.checkpoint, DecoderLanguageModel, ItemVocab
    )
    print(f"  Model: {config['num_layers']} layers, {config['embed_dim']} dim, "
          f"{config['num_heads']} heads")
    print(f"  Vocab size: {config['vocab_size']} (movies + specials)")
    if training_info:
        print(f"  Final train loss: {training_info.get('final_train_loss', 'N/A')}")
    print()

    # Movie titles for readable output.
    ml_dir = os.path.join(_stage5_path, "data", "ml-1m")
    movie_titles = {}
    movies_path = os.path.join(ml_dir, "movies.dat")
    if os.path.exists(movies_path):
        movie_titles = _ml_data_mod.load_movie_titles(movies_path)

    # Build the input history.
    if args.history:
        history = [int(x) for x in args.history.split(",") if x.strip()]
        source = "provided --history"
    else:
        interactions = _ml_data_mod.load_movielens_ratings(
            os.path.join(ml_dir, "ratings.dat")
        )
        user_sequences = _ml_data_mod.build_user_sequences(interactions)
        if args.user_id is not None and args.user_id in user_sequences:
            user_id = args.user_id
        else:
            user_id = sorted(user_sequences.keys())[0]
        # Hold out the last movie so we can show the actual next pick for context.
        history = user_sequences[user_id][:-1]
        actual_next = user_sequences[user_id][-1]
        source = f"MovieLens user {user_id}"
        print(f"Actual next movie for user {user_id}: "
              f"{movie_titles.get(actual_next, f'movie {actual_next}')}")

    history_window = history[-(model.max_seq_len - 1):]

    print(f"\nUser history ({source}), most recent last:")
    for m in history_window[-10:]:
        print(f"  - {movie_titles.get(m, f'movie {m}')}")

    preds = predict_next_items(model, vocab, history_window, top_k=args.top_k)

    print(f"\nTop-{args.top_k} predicted next movies:")
    print("-" * 70)
    for rank, (movie_id, score) in enumerate(preds, 1):
        print(f"  {rank}. {movie_titles.get(movie_id, f'movie {movie_id}')}  "
              f"(score={score:.2f})")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
