"""
Stage 5: Train the Sequential Recommender (the recommendation pivot)

Trains a GPT-style decoder to predict the next movie a user will interact with,
given their chronological history. This is the SAME model class used in Stage 4
for language modeling -- stage3/models/decoder_lm.py::DecoderLanguageModel -- with
NO changes to the transformer. Only the input representation differs: movie item
IDs take the place of word tokens.

Uses only the Stage 2/3 autograd engine - no PyTorch or TensorFlow.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

# Get absolute paths
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


# ---- Reused unchanged from Stages 2/3/4 --------------------------------------
_loss_mod = _load_module(_stage2_path, "nn/loss.py", "stage2_loss")
_adam_mod = _load_module(_stage2_path, "optim/adam.py", "stage2_adam")
CrossEntropyLoss = _loss_mod.CrossEntropyLoss
Adam = _adam_mod.Adam

# The EXACT SAME transformer model used for the Stage 4 language model.
_decoder_mod = _load_module(_stage3_path, "models/decoder_lm.py", "stage3_decoder")
DecoderLanguageModel = _decoder_mod.DecoderLanguageModel

# Stage 4 data loader (generic batch iterator) and checkpoint utils, reused as-is.
_stage4_data_mod = _load_module(_stage4_path, "utils/data.py", "stage4_data")
TextDataLoader = _stage4_data_mod.TextDataLoader
_checkpoint_mod = _load_module(_stage4_path, "utils/checkpoint.py", "stage4_checkpoint")
save_checkpoint = _checkpoint_mod.save_checkpoint

# ---- New Stage 5 pieces: item vocabulary + MovieLens data --------------------
_item_vocab_mod = _load_module(_stage5_path, "utils/item_vocab.py", "stage5_item_vocab")
ItemVocab = _item_vocab_mod.ItemVocab
_ml_data_mod = _load_module(_stage5_path, "utils/data.py", "stage5_data")


def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch. Identical to the Stage 4 loop."""
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()

        logits = model(inputs)

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.data)
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion):
    """Evaluate on a held-out set. Identical to the Stage 4 loop."""
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in dataloader:
        logits = model(inputs)

        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        total_loss += float(loss.data)
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def predict_next_items(model, vocab, history_movie_ids, top_k=5):
    """
    Predict the next items for a user given their movie history.

    Same mechanic as language-model generation: encode the sequence, run one forward
    pass, read the logits at the LAST position, and take the highest-scoring items.

    Returns a list of (movie_id, score) for the top_k predictions, excluding movies
    already in the history and special tokens.
    """
    token_ids = vocab.encode(history_movie_ids, add_bos=False, add_eos=False)
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
    parser = argparse.ArgumentParser(description="Stage 5: train sequential recommender")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-hidden", type=int, default=512)
    parser.add_argument("--max-users", type=int, default=None,
                        help="Cap number of users (for a fast smoke run).")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny fast run: few users, 1 epoch. Overrides other sizes.")
    args = parser.parse_args()

    if args.smoke:
        args.max_users = args.max_users or 200
        args.epochs = 1

    print("=" * 70)
    print("Stage 5: Sequential Recommender Training")
    print("CUDA Transformer Engine - The Recommendation Pivot")
    print("=" * 70)
    print()
    print("NOTE: The transformer architecture is UNCHANGED from Stage 3/4.")
    print("      We reuse DecoderLanguageModel exactly; only the vocabulary and")
    print("      data differ -- movie item IDs replace word tokens.")
    print()

    np.random.seed(42)

    DATA_DIR = os.path.join(_stage5_path, "data")
    CHECKPOINT_PATH = os.path.join(_stage5_path, "checkpoints", "recommender")

    # 1. Load MovieLens 1M
    print("Loading MovieLens 1M...")
    ml_dir = _ml_data_mod.download_movielens(DATA_DIR)
    interactions = _ml_data_mod.load_movielens_ratings(
        os.path.join(ml_dir, "ratings.dat")
    )
    movie_titles = _ml_data_mod.load_movie_titles(os.path.join(ml_dir, "movies.dat"))
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Movies with titles: {len(movie_titles):,}")

    user_sequences = _ml_data_mod.build_user_sequences(interactions)
    print(f"  Users: {len(user_sequences):,}")

    if args.max_users is not None:
        keep = sorted(user_sequences.keys())[:args.max_users]
        user_sequences = {u: user_sequences[u] for u in keep}
        print(f"  (capped to first {len(user_sequences)} users)")

    # 2. Build the item vocabulary (movie IDs take the place of word tokens)
    all_movie_ids = (m for seq in user_sequences.values() for m in seq)
    vocab = ItemVocab.from_interactions(all_movie_ids)
    VOCAB_SIZE = vocab.vocab_size
    print(f"  Built item vocab with vocab_size={VOCAB_SIZE} "
          f"({VOCAB_SIZE - 4} movies + 4 special tokens)")

    # 3. Build per-user next-item training sequences
    inputs, targets = _ml_data_mod.create_user_sequences(
        user_sequences, vocab, seq_len=args.seq_len, stride=args.seq_len // 2
    )
    print(f"  Created {len(inputs):,} training sequences (per-user windows)")

    # Split into train/val (90/10)
    perm = np.random.permutation(len(inputs))
    inputs, targets = inputs[perm], targets[perm]
    num_train = int(len(inputs) * 0.9)
    train_inputs, train_targets = inputs[:num_train], targets[:num_train]
    val_inputs, val_targets = inputs[num_train:], targets[num_train:]

    train_loader = TextDataLoader(train_inputs, train_targets, args.batch_size, shuffle=True)
    val_loader = TextDataLoader(val_inputs, val_targets, args.batch_size, shuffle=False)
    print(f"  Train sequences: {len(train_inputs):,}")
    print(f"  Val sequences: {len(val_inputs):,}")
    print()

    # 4. Create the model -- SAME class and hyperparameters as Stage 4
    print("Creating model (DecoderLanguageModel, unchanged from Stage 3/4)...")
    model = DecoderLanguageModel(
        vocab_size=VOCAB_SIZE,
        max_seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_hidden_dim=args.ff_hidden,
    )
    print(model)
    print()

    criterion = CrossEntropyLoss()
    optimizer = Adam(list(model.parameters()), lr=args.lr)

    print("Training Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print()

    # 5. Training loop
    print("Starting training...")
    print("-" * 70)

    results = []
    start_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)
        epoch_time = time.time() - epoch_start

        results.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        best_val_loss = min(best_val_loss, val_loss)

        print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}  ({epoch_time:.1f}s)")

    training_time = time.time() - start_time

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final Train Loss: {results[-1]['train_loss']:.4f}")
    print(f"Final Val Loss: {results[-1]['val_loss']:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Training Time: {training_time:.1f} seconds")
    print()

    # 6. Save checkpoint (same approach as Stage 4)
    print("Saving checkpoint...")
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    model_config = {
        "vocab_size": VOCAB_SIZE,
        "max_seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "ff_hidden_dim": args.ff_hidden,
    }
    training_info = {
        "final_train_loss": results[-1]["train_loss"],
        "final_val_loss": results[-1]["val_loss"],
        "best_val_loss": best_val_loss,
        "training_time": training_time,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "dataset": "MovieLens 1M",
    }
    save_checkpoint(model, vocab, CHECKPOINT_PATH, model_config, training_info)
    print()

    # 7. Example predictions: give the model a user's history, predict next items
    print("Generating example predictions...")
    print("-" * 70)

    example_user_ids = sorted(user_sequences.keys())[:4]
    examples = []
    for user_id in example_user_ids:
        full_history = user_sequences[user_id]
        # Hold out the last movie as the "actual next" for context.
        history = full_history[:-1]
        actual_next = full_history[-1]
        # Use a recent window of the history as input.
        history_window = history[-(args.seq_len - 1):]
        preds = predict_next_items(model, vocab, history_window, top_k=5)
        examples.append((user_id, history_window, actual_next, preds))

        print(f"\nUser {user_id} (history of {len(full_history)} movies)")
        print("  Recent history:")
        for m in history_window[-5:]:
            print(f"    - {movie_titles.get(m, f'movie {m}')}")
        print("  Top-5 predicted next movies:")
        for movie_id, score in preds:
            print(f"    * {movie_titles.get(movie_id, f'movie {movie_id}')}  "
                  f"(score={score:.2f})")
        print(f"  Actual next movie: {movie_titles.get(actual_next, f'movie {actual_next}')}")

    print()

    # 8. Write results file
    results_path = os.path.join(_base_path, "stage5_results.txt")
    with open(results_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Stage 5 Training Results: Sequential Recommender\n")
        f.write("CUDA Transformer Engine - The Recommendation Pivot\n")
        f.write("=" * 70 + "\n\n")

        f.write("Architecture note:\n")
        f.write("  The transformer is UNCHANGED from Stage 3/4. This model is the\n")
        f.write("  exact same DecoderLanguageModel class used for the tiny language\n")
        f.write("  model; only the vocabulary (movie item IDs instead of words) and\n")
        f.write("  the training data (user interaction sequences) are different.\n\n")

        f.write("Model Architecture:\n")
        f.write(f"  Vocab size: {VOCAB_SIZE} ({VOCAB_SIZE - 4} movies + 4 specials)\n")
        f.write(f"  Max seq len: {args.seq_len}\n")
        f.write(f"  Embed dim: {args.embed_dim}\n")
        f.write(f"  Num layers: {args.num_layers}\n")
        f.write(f"  Num heads: {args.num_heads}\n")
        f.write(f"  FF hidden: {args.ff_hidden}\n")
        f.write(f"  Total parameters: {model.count_parameters():,}\n\n")

        f.write("Dataset:\n")
        f.write("  Source: MovieLens 1M (GroupLens)\n")
        f.write(f"  Users: {len(user_sequences):,}\n")
        f.write(f"  Interactions: {len(interactions):,}\n")
        f.write(f"  Train sequences: {len(train_inputs):,}\n")
        f.write(f"  Val sequences: {len(val_inputs):,}\n")
        f.write("  Sequences are per-user, ordered by timestamp.\n\n")

        f.write("Training Progress:\n")
        for r in results:
            f.write(f"  Epoch {r['epoch']:3d}: Train Loss={r['train_loss']:.4f}, "
                    f"Val Loss={r['val_loss']:.4f}\n")

        f.write("\nFinal Results:\n")
        f.write(f"  Final Train Loss: {results[-1]['train_loss']:.4f}\n")
        f.write(f"  Final Val Loss: {results[-1]['val_loss']:.4f}\n")
        f.write(f"  Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"  Training Time: {training_time:.1f} seconds\n\n")

        f.write("Example Predictions (user history -> next item):\n")
        f.write("-" * 60 + "\n")
        for user_id, history_window, actual_next, preds in examples:
            f.write(f"\nUser {user_id}\n")
            f.write("  Recent history:\n")
            for m in history_window[-5:]:
                f.write(f"    - {movie_titles.get(m, f'movie {m}')}\n")
            f.write("  Top-5 predicted next movies:\n")
            for movie_id, score in preds:
                f.write(f"    * {movie_titles.get(movie_id, f'movie {movie_id}')}  "
                        f"(score={score:.2f})\n")
            f.write(f"  Actual next movie: "
                    f"{movie_titles.get(actual_next, f'movie {actual_next}')}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Results saved to {results_path}")
    print()
    print("Stage 5 Complete!")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}_*")
    print("Run 'python3 recommend.py' to get predictions from the trained model.")


if __name__ == "__main__":
    main()
