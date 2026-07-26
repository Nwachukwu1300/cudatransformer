"""
Training script for MNIST MLP.

Train a multi-layer perceptron on MNIST using custom autograd engine.
"""

import numpy as np
import time
import sys
import os

# Add stage2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stage2'))

from tensor import Tensor
from nn.loss import CrossEntropyLoss
from optim.adam import Adam
from models.mlp import MLP
from utils.data import load_mnist, DataLoader


def evaluate(model: MLP, test_loader: DataLoader) -> float:
    """
    Evaluate model on test set.

    Args:
        model: MLP model
        test_loader: DataLoader for test data

    Returns:
        Test accuracy as a percentage
    """
    correct = 0
    total = 0

    for batch_images, batch_labels in test_loader:
        # Flatten images
        batch_size = batch_images.shape[0]
        batch_images_flat = batch_images.reshape(batch_size, -1)

        # Forward pass (no gradient tracking for evaluation)
        x = Tensor(batch_images_flat, requires_grad=False)
        logits = model(x)

        # Get predictions
        predictions = np.argmax(logits.data, axis=1)

        # Count correct predictions
        correct += np.sum(predictions == batch_labels)
        total += batch_size

    accuracy = 100.0 * correct / total
    return accuracy


def train_epoch(
    model: MLP,
    train_loader: DataLoader,
    optimizer: Adam,
    criterion: CrossEntropyLoss,
    epoch: int
) -> float:
    """
    Train model for one epoch.

    Args:
        model: MLP model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        epoch: Current epoch number

    Returns:
        Average loss for the epoch
    """
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (batch_images, batch_labels) in enumerate(train_loader):
        # Flatten images
        batch_size = batch_images.shape[0]
        batch_images_flat = batch_images.reshape(batch_size, -1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        x = Tensor(batch_images_flat, requires_grad=False)
        logits = model(x)

        # Compute loss
        loss = criterion(logits, batch_labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track loss
        total_loss += loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        num_batches += 1

        # Print progress
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}], "
                  f"Loss: {avg_loss:.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training function."""
    print("=" * 70)
    print("Stage 2: MNIST MLP Training")
    print("CUDA Transformer Engine - Deep Learning Engine")
    print("=" * 70)
    print()

    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999

    # Load MNIST data
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        data_dir='./data/mnist',
        normalize=True
    )
    print(f"Train samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print()

    # Create data loaders
    train_loader = DataLoader(train_images, train_labels,
                             batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_images, test_labels,
                            batch_size=batch_size, shuffle=False)

    # Create model
    print("Creating MLP model...")
    model = MLP(input_size=784, hidden1_size=256,
                hidden2_size=128, num_classes=10)
    print(model)
    print()

    # Create loss function and optimizer
    criterion = CrossEntropyLoss()
    optimizer = Adam(list(model.parameters()),
                    lr=learning_rate,
                    beta1=beta1,
                    beta2=beta2)

    # Training configuration
    print("Training Configuration:")
    print(f"  Optimizer: Adam (lr={learning_rate}, beta1={beta1}, beta2={beta2})")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Loss: Cross Entropy")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 70)

    best_test_acc = 0.0
    training_results = []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)

        # Evaluate
        test_acc = evaluate(model, test_loader)

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # Store results
        training_results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_acc': test_acc
        })

        # Print epoch summary
        print(f"  Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")

    training_time = time.time() - start_time

    # Print final results
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Final Train Loss: {training_results[-1]['train_loss']:.4f}")
    print(f"Training Time: {training_time:.1f} seconds")
    print()

    # Save results to file
    results_file = 'stage2_results.txt'
    print(f"Saving results to {results_file}...")

    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Stage 2 Training Results: MNIST MLP\n")
        f.write("CUDA Transformer Engine - Deep Learning Engine\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model Architecture:\n")
        f.write("- Input: 784 (28x28 flattened)\n")
        f.write("- Hidden 1: 256 + ReLU\n")
        f.write("- Hidden 2: 128 + ReLU\n")
        f.write("- Output: 10 (classes)\n")
        f.write(f"- Total Parameters: {model.count_parameters():,}\n\n")

        f.write("Training Configuration:\n")
        f.write(f"- Optimizer: Adam (lr={learning_rate}, beta1={beta1}, beta2={beta2})\n")
        f.write(f"- Batch Size: {batch_size}\n")
        f.write(f"- Epochs: {num_epochs}\n")
        f.write("- Loss: Cross Entropy\n\n")

        f.write("Training Progress:\n")
        for result in training_results:
            f.write(f"Epoch {result['epoch']}: "
                   f"Train Loss={result['train_loss']:.4f}, "
                   f"Test Acc={result['test_acc']:.2f}%\n")

        f.write("\nFinal Results:\n")
        f.write(f"- Best Test Accuracy: {best_test_acc:.2f}%\n")
        f.write(f"- Final Train Loss: {training_results[-1]['train_loss']:.4f}\n")
        f.write(f"- Training Time: {training_time:.1f} seconds\n\n")

        f.write("=" * 70 + "\n")

    print(f"Results saved to {results_file}")
    print()
    print("Stage 2 Complete!")


if __name__ == "__main__":
    main()
