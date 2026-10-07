"""
Stage 5 Deliverable: Next-Item Recommendation

Load the trained recommender checkpoint, take a user's movie history, and predict
the next movie they're likely to pick. This is the recommendation analogue of Stage
4's generate.py -- same model, same forward pass, same "read the logits at the last
position" mechanic. The only difference is that the vocabulary maps movie IDs (not
words), so the predicted token IDs decode back to movies.

This is a thin CLI wrapper -- the actual loading and prediction logic lives in
stage5/inference.py, which is also what webapp/backend/app.py imports, so there
is exactly one source of truth shared by both the CLI and the web app.
"""

import argparse
import importlib.util
import os
import sys

_base_path = os.path.dirname(os.path.abspath(__file__))
_stage5_path = os.path.join(_base_path, "stage5")


def _load_module(base_path, rel_path, name):
    full_path = os.path.join(base_path, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_inference_mod = _load_module(_stage5_path, "inference.py", "stage5_inference")
load_recommender = _inference_mod.load_recommender
predict_next_items = _inference_mod.predict_next_items
load_movie_titles = _inference_mod.load_movie_titles
load_movielens_ratings = _inference_mod.load_movielens_ratings
build_user_sequences = _inference_mod.build_user_sequences


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
    model, vocab, config, training_info = load_recommender(args.checkpoint)
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
        movie_titles = load_movie_titles(movies_path)

    # Build the input history.
    if args.history:
        history = [int(x) for x in args.history.split(",") if x.strip()]
        source = "provided --history"
    else:
        interactions = load_movielens_ratings(
            os.path.join(ml_dir, "ratings.dat")
        )
        user_sequences = build_user_sequences(interactions)
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
