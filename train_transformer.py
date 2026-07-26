"""
Stage 3: Transformer Decoder Training

Train a small transformer decoder on a toy sequence prediction task.
Uses only the Stage 2 autograd engine and Stage 1 kernels.
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

# Add to path
if _stage2_path not in sys.path:
    sys.path.insert(0, _stage2_path)

from tensor import Tensor

# Import stage2 modules using importlib to avoid nn name collision
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

# Import stage3 modules
_decoder_mod = _load_module(_stage3_path, 'models/decoder_lm.py', 'stage3_decoder')
_data_mod = _load_module(_stage3_path, 'utils/data.py', 'stage3_data')
_attn_mod = _load_module(_stage3_path, 'nn/attention.py', 'stage3_attention')

DecoderLanguageModel = _decoder_mod.DecoderLanguageModel
generate_repeat_data = _data_mod.generate_repeat_data
SequenceDataLoader = _data_mod.SequenceDataLoader
MultiHeadAttention = _attn_mod.MultiHeadAttention
scaled_dot_product_attention = _attn_mod.scaled_dot_product_attention


def numerical_gradient_check(
    model: DecoderLanguageModel,
    inputs: np.ndarray,
    targets: np.ndarray,
    param_name: str,
    param: Tensor,
    eps: float = 1e-4,
    num_checks: int = 5
) -> bool:
    """
    Numerical gradient check for a specific parameter.

    Args:
        model: The model to check
        inputs: Input token IDs
        targets: Target token IDs
        param_name: Name of parameter being checked
        param: The parameter tensor
        eps: Finite difference epsilon
        num_checks: Number of random indices to check

    Returns:
        True if gradients match within tolerance
    """
    criterion = CrossEntropyLoss()
    batch_size, seq_len = inputs.shape

    # Forward and backward to get analytical gradient
    for p in model.parameters():
        p.zero_grad()

    logits = model(inputs)
    logits_flat = logits.reshape(batch_size * seq_len, model.vocab_size)
    targets_flat = targets.reshape(-1)
    loss = criterion(logits_flat, targets_flat)
    loss.backward()

    if param.grad is None:
        print(f"  {param_name}: No gradient computed")
        return False

    analytical_grad = param.grad.copy()

    # Check random indices
    flat_size = param.data.size
    check_indices = np.random.choice(flat_size, min(num_checks, flat_size), replace=False)

    max_rel_error = 0.0
    for flat_idx in check_indices:
        idx = np.unravel_index(flat_idx, param.data.shape)

        # f(x + eps)
        old_val = param.data[idx]
        param.data[idx] = old_val + eps

        for p in model.parameters():
            p.zero_grad()
        logits = model(inputs)
        logits_flat = logits.reshape(batch_size * seq_len, model.vocab_size)
        loss_plus = criterion(logits_flat, targets_flat)

        # f(x - eps)
        param.data[idx] = old_val - eps

        for p in model.parameters():
            p.zero_grad()
        logits = model(inputs)
        logits_flat = logits.reshape(batch_size * seq_len, model.vocab_size)
        loss_minus = criterion(logits_flat, targets_flat)

        # Restore
        param.data[idx] = old_val

        # Numerical gradient
        numerical_grad = (float(loss_plus.data) - float(loss_minus.data)) / (2 * eps)
        analytical = analytical_grad[idx]

        # Relative error
        rel_error = abs(numerical_grad - analytical) / (abs(numerical_grad) + abs(analytical) + 1e-8)
        max_rel_error = max(max_rel_error, rel_error)

    passed = max_rel_error < 0.01  # 1% tolerance
    return passed, max_rel_error


def run_gradient_checks(model: DecoderLanguageModel, inputs: np.ndarray, targets: np.ndarray):
    """Run gradient checks on attention parameters."""
    print("\nRunning numerical gradient checks on attention...")
    print("-" * 50)

    # Find first attention layer
    attn = model.decoder.layers[0].attention

    params_to_check = [
        ('W_q.weight', attn.W_q.weight),
        ('W_k.weight', attn.W_k.weight),
        ('W_v.weight', attn.W_v.weight),
        ('W_o.weight', attn.W_o.weight),
    ]

    all_passed = True
    for name, param in params_to_check:
        passed, max_error = numerical_gradient_check(model, inputs, targets, name, param)
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status} (max rel error: {max_error:.6f})")
        all_passed = all_passed and passed

    print("-" * 50)
    if all_passed:
        print("All gradient checks PASSED!")
    else:
        print("Some gradient checks FAILED. Review implementation.")

    return all_passed


def train_epoch(
    model: DecoderLanguageModel,
    dataloader: SequenceDataLoader,
    optimizer: Adam,
    criterion: CrossEntropyLoss,
    epoch: int,
    debug: bool = False
) -> float:
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()

        # Forward pass (debug shapes on first batch of first epoch)
        show_debug = debug and batch_idx == 0
        if show_debug:
            print(f"\nShape check (epoch {epoch}, batch 0):")

        logits = model(inputs, debug=show_debug)

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


def evaluate(model: DecoderLanguageModel, dataloader: SequenceDataLoader) -> float:
    """Evaluate prediction accuracy."""
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        logits = model(inputs)
        predictions = np.argmax(logits.data, axis=-1)

        # Count correct predictions (exclude padding if any)
        mask = targets != 0
        correct += np.sum((predictions == targets) & mask)
        total += np.sum(mask)

    return 100.0 * correct / total if total > 0 else 0.0


def main():
    print("=" * 70)
    print("Stage 3: Transformer Decoder Training")
    print("CUDA Transformer Engine - Transformer Architecture")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Hyperparameters (small model for toy task)
    VOCAB_SIZE = 10
    MAX_SEQ_LEN = 16
    EMBED_DIM = 64
    NUM_LAYERS = 2
    NUM_HEADS = 4
    FF_HIDDEN = 256

    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001

    NUM_TRAIN = 500
    NUM_TEST = 100
    PATTERN_LEN = 2

    print("Model Configuration:")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Max seq len: {MAX_SEQ_LEN}")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  FF hidden: {FF_HIDDEN}")
    print()

    print("Training Configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    # Generate toy data (repeat pattern task)
    print("Generating toy data (repeat pattern task)...")
    train_inputs, train_targets = generate_repeat_data(
        NUM_TRAIN, MAX_SEQ_LEN, pattern_len=PATTERN_LEN, vocab_size=VOCAB_SIZE
    )
    test_inputs, test_targets = generate_repeat_data(
        NUM_TEST, MAX_SEQ_LEN, pattern_len=PATTERN_LEN, vocab_size=VOCAB_SIZE
    )

    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Test samples: {NUM_TEST}")
    print(f"  Pattern length: {PATTERN_LEN}")
    print(f"  Example input:  {train_inputs[0][:8]}")
    print(f"  Example target: {train_targets[0][:8]}")
    print()

    train_loader = SequenceDataLoader(train_inputs, train_targets, BATCH_SIZE, shuffle=True)
    test_loader = SequenceDataLoader(test_inputs, test_targets, BATCH_SIZE, shuffle=False)

    # Create model
    print("Creating Transformer Decoder model...")
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

    # Training loop
    print("Starting training...")
    print("-" * 70)

    results = []
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # Debug shapes on first epoch
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, epoch,
            debug=(epoch == 1)
        )

        test_acc = evaluate(model, test_loader)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_acc': test_acc
        })

        # Print progress every 10 epochs or first epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Test Acc={test_acc:.1f}%")

    training_time = time.time() - start_time

    # Run gradient checks
    print()
    grad_check_inputs = train_inputs[:2]
    grad_check_targets = train_targets[:2]
    grad_checks_passed = run_gradient_checks(model, grad_check_inputs, grad_check_targets)

    # Final results
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Initial Loss: {results[0]['train_loss']:.4f}")
    print(f"Final Loss: {results[-1]['train_loss']:.4f}")
    print(f"Final Test Accuracy: {results[-1]['test_acc']:.1f}%")
    print(f"Training Time: {training_time:.1f} seconds")
    print()

    # Verify loss decreased
    loss_decreased = results[-1]['train_loss'] < results[0]['train_loss']
    if loss_decreased:
        print("SUCCESS: Loss decreased during training!")
    else:
        print("WARNING: Loss did not decrease. Check implementation.")

    # Save results
    save_results(results, training_time, model, loss_decreased, grad_checks_passed)


def save_results(results, training_time, model, loss_decreased, grad_checks_passed):
    """Save training results to file."""
    results_file = 'stage3_results.txt'

    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Stage 3 Training Results: Transformer Decoder\n")
        f.write("CUDA Transformer Engine\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model Architecture:\n")
        f.write(f"  Vocab size: {model.vocab_size}\n")
        f.write(f"  Max seq len: {model.max_seq_len}\n")
        f.write(f"  Embed dim: {model.embed_dim}\n")
        f.write(f"  Num layers: {model.num_layers}\n")
        f.write(f"  Num heads: {model.num_heads}\n")
        f.write(f"  FF hidden: {model.ff_hidden_dim}\n")
        f.write(f"  Total parameters: {model.count_parameters():,}\n\n")

        f.write("Training Progress:\n")
        for r in results:
            f.write(f"  Epoch {r['epoch']:3d}: Loss={r['train_loss']:.4f}, "
                    f"Test Acc={r['test_acc']:.1f}%\n")

        f.write(f"\nFinal Results:\n")
        f.write(f"  Initial Loss: {results[0]['train_loss']:.4f}\n")
        f.write(f"  Final Loss: {results[-1]['train_loss']:.4f}\n")
        f.write(f"  Final Test Accuracy: {results[-1]['test_acc']:.1f}%\n")
        f.write(f"  Training Time: {training_time:.1f} seconds\n\n")

        f.write("Verification:\n")
        f.write(f"  Loss decreased: {'YES' if loss_decreased else 'NO'}\n")
        f.write(f"  Gradient checks passed: {'YES' if grad_checks_passed else 'NO'}\n\n")

        f.write("=" * 70 + "\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
