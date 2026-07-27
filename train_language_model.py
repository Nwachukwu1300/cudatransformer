"""
Stage 4: Train Tiny Language Model

Train a GPT-style decoder on simple stories for text generation.
Uses only the Stage 2/3 autograd engine - no PyTorch or TensorFlow.
"""

import numpy as np
import time
import sys
import os
import importlib.util

# Get absolute paths
_base_path = os.path.dirname(os.path.abspath(__file__))
_stage2_path = os.path.join(_base_path, 'stage2')
_stage3_path = os.path.join(_base_path, 'stage3')
_stage4_path = os.path.join(_base_path, 'stage4')

# Add to path
if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor

# Import stage2 modules using importlib
def _load_module(base_path, rel_path, name):
    full_path = os.path.join(base_path, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_loss_mod = _load_module(_stage2_path, 'nn/loss.py', 'stage2_loss')
_adam_mod = _load_module(_stage2_path, 'optim/adam.py', 'stage2_adam')

CrossEntropyLoss = _loss_mod.CrossEntropyLoss
Adam = _adam_mod.Adam

# Import stage3 model
_decoder_mod = _load_module(_stage3_path, 'models/decoder_lm.py', 'stage3_decoder')
DecoderLanguageModel = _decoder_mod.DecoderLanguageModel

# Import stage4 utilities
_tokenizer_mod = _load_module(_stage4_path, 'utils/tokenizer.py', 'stage4_tokenizer')
_data_mod = _load_module(_stage4_path, 'utils/data.py', 'stage4_data')
_checkpoint_mod = _load_module(_stage4_path, 'utils/checkpoint.py', 'stage4_checkpoint')

SimpleTokenizer = _tokenizer_mod.SimpleTokenizer
load_text_file = _data_mod.load_text_file
create_sequences = _data_mod.create_sequences
TextDataLoader = _data_mod.TextDataLoader
save_checkpoint = _checkpoint_mod.save_checkpoint


def train_epoch(model, dataloader, optimizer, criterion, epoch):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)

        # Reshape for cross entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute loss
        loss = criterion(logits_flat, targets_flat)

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        total_loss += float(loss.data)
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion):
    """Evaluate on validation set."""
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


def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8):
    """
    Generate text continuation from a prompt.

    Args:
        model: Trained language model
        tokenizer: Tokenizer
        prompt: Text prompt to continue
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)

    Returns:
        Generated text
    """
    # Encode prompt
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

    # Generate tokens one at a time
    for _ in range(max_length):
        # Truncate to max sequence length
        if len(token_ids) > model.max_seq_len:
            input_ids = token_ids[-model.max_seq_len:]
        else:
            input_ids = token_ids

        # Forward pass
        input_array = np.array([input_ids], dtype=np.int64)
        logits = model(input_array)

        # Get logits for last position
        last_logits = logits.data[0, -1, :]

        # Apply temperature
        if temperature > 0:
            last_logits = last_logits / temperature

        # Softmax to get probabilities
        max_logit = np.max(last_logits)
        exp_logits = np.exp(last_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        # Sample from distribution
        next_token = np.random.choice(len(probs), p=probs)

        # Stop if EOS token
        if next_token == tokenizer.get_eos_id():
            break

        token_ids.append(next_token)

    # Decode generated tokens
    return tokenizer.decode(token_ids, skip_special=True)


def main():
    print("=" * 70)
    print("Stage 4: Tiny Language Model Training")
    print("CUDA Transformer Engine - Language Generation")
    print("=" * 70)
    print()

    # Set random seed
    np.random.seed(42)

    # Configuration
    DATA_PATH = os.path.join(_base_path, 'data', 'tinystories_train.txt')
    CHECKPOINT_PATH = os.path.join(_stage4_path, 'checkpoints', 'language_model')

    # Read only a subset of TinyStories (5MB to keep training reasonable)
    MAX_CHARS = 5 * 1024 * 1024  # 5MB

    # Model configuration - small model for fast training
    VOCAB_SIZE = 2000  # Will be set from tokenizer - larger for TinyStories
    MAX_SEQ_LEN = 64
    EMBED_DIM = 128
    NUM_LAYERS = 4
    NUM_HEADS = 4
    FF_HIDDEN = 512

    # Training configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 15  # Fewer epochs since we have more data
    LEARNING_RATE = 0.001

    print("Loading and tokenizing data...")
    print(f"  Data file: {DATA_PATH}")
    print(f"  Using first {MAX_CHARS // (1024*1024)}MB of data")

    # Load text (subset for training)
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read(MAX_CHARS)
    print(f"  Loaded {len(text)} characters")

    # Build tokenizer
    tokenizer = SimpleTokenizer.from_text(text, max_vocab_size=2000, min_freq=2)
    VOCAB_SIZE = tokenizer.vocab_size
    print(f"  Built tokenizer with vocab_size={VOCAB_SIZE}")

    # Encode text
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    print(f"  Encoded to {len(token_ids)} tokens")

    # Create sequences
    inputs, targets = create_sequences(token_ids, MAX_SEQ_LEN, stride=MAX_SEQ_LEN // 2)
    print(f"  Created {len(inputs)} training sequences")

    # Split into train/val (90/10)
    num_train = int(len(inputs) * 0.9)
    train_inputs, train_targets = inputs[:num_train], targets[:num_train]
    val_inputs, val_targets = inputs[num_train:], targets[num_train:]

    train_loader = TextDataLoader(train_inputs, train_targets, BATCH_SIZE, shuffle=True)
    val_loader = TextDataLoader(val_inputs, val_targets, BATCH_SIZE, shuffle=False)

    print(f"  Train sequences: {len(train_inputs)}")
    print(f"  Val sequences: {len(val_inputs)}")
    print()

    # Create model
    print("Creating model...")
    model = DecoderLanguageModel(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_hidden_dim=FF_HIDDEN
    )
    print(model)
    print()

    # Create optimizer and loss
    criterion = CrossEntropyLoss()
    optimizer = Adam(list(model.parameters()), lr=LEARNING_RATE)

    print("Training Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 70)

    results = []
    start_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_loss = evaluate(model, val_loader, criterion)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Print progress every 5 epochs or first/last
        if epoch % 5 == 0 or epoch == 1 or epoch == NUM_EPOCHS:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

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

    # Save checkpoint
    print("Saving checkpoint...")
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    model_config = {
        'vocab_size': VOCAB_SIZE,
        'max_seq_len': MAX_SEQ_LEN,
        'embed_dim': EMBED_DIM,
        'num_layers': NUM_LAYERS,
        'num_heads': NUM_HEADS,
        'ff_hidden_dim': FF_HIDDEN
    }

    training_info = {
        'final_train_loss': results[-1]['train_loss'],
        'final_val_loss': results[-1]['val_loss'],
        'best_val_loss': best_val_loss,
        'training_time': training_time,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }

    save_checkpoint(model, tokenizer, CHECKPOINT_PATH, model_config, training_info)
    print()

    # Generate some examples
    print("Generating example completions...")
    print("-" * 70)

    prompts = [
        "Once upon a time",
        "The king walked into",
        "A little girl named",
        "The cat sat",
        "One day a"
    ]

    generations = []
    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_length=30, temperature=0.7)
        generations.append((prompt, generated))
        print(f"\nPrompt: \"{prompt}\"")
        print(f"Generated: \"{generated}\"")

    print()

    # Save results
    results_path = os.path.join(_base_path, 'stage4_results.txt')
    with open(results_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Stage 4 Training Results: Tiny Language Model\n")
        f.write("CUDA Transformer Engine\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model Architecture:\n")
        f.write(f"  Vocab size: {VOCAB_SIZE}\n")
        f.write(f"  Max seq len: {MAX_SEQ_LEN}\n")
        f.write(f"  Embed dim: {EMBED_DIM}\n")
        f.write(f"  Num layers: {NUM_LAYERS}\n")
        f.write(f"  Num heads: {NUM_HEADS}\n")
        f.write(f"  FF hidden: {FF_HIDDEN}\n")
        f.write(f"  Total parameters: {model.count_parameters():,}\n\n")

        f.write("Dataset:\n")
        f.write(f"  Source: TinyStories (Eldan & Li, 2023)\n")
        f.write(f"  Used: First 5MB subset of training data\n")
        f.write(f"  Total tokens: {len(token_ids)}\n")
        f.write(f"  Train sequences: {len(train_inputs)}\n")
        f.write(f"  Val sequences: {len(val_inputs)}\n\n")

        f.write("Training Progress:\n")
        for r in results:
            f.write(f"  Epoch {r['epoch']:3d}: Train Loss={r['train_loss']:.4f}, "
                    f"Val Loss={r['val_loss']:.4f}\n")

        f.write(f"\nFinal Results:\n")
        f.write(f"  Final Train Loss: {results[-1]['train_loss']:.4f}\n")
        f.write(f"  Final Val Loss: {results[-1]['val_loss']:.4f}\n")
        f.write(f"  Best Val Loss: {best_val_loss:.4f}\n")
        f.write(f"  Training Time: {training_time:.1f} seconds\n\n")

        f.write("Example Generations:\n")
        f.write("-" * 50 + "\n")
        for prompt, generated in generations:
            f.write(f"\nPrompt: \"{prompt}\"\n")
            f.write(f"Generated: \"{generated}\"\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Results saved to {results_path}")
    print()
    print("Stage 4 Complete!")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}_*")
    print("Run 'python3 generate.py' to generate more text from the trained model.")


if __name__ == "__main__":
    main()
