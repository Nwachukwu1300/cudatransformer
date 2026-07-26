"""
MNIST data loading utilities.
"""

import numpy as np
import os
import gzip
import urllib.request
from typing import Tuple


def download_mnist(data_dir: str = './data/mnist') -> None:
    """
    Download MNIST dataset if not already present.

    Args:
        data_dir: Directory to store MNIST data
    """
    os.makedirs(data_dir, exist_ok=True)

    # Try multiple mirrors in case one is down
    mirrors = [
        'https://storage.googleapis.com/cvdf-datasets/mnist/',
        'http://yann.lecun.com/exdb/mnist/',
    ]

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            success = False
            for base_url in mirrors:
                try:
                    urllib.request.urlretrieve(base_url + filename, filepath)
                    print(f"Downloaded {filename} from {base_url}")
                    success = True
                    break
                except Exception as e:
                    print(f"  Failed from {base_url}: {e}")
                    continue

            if not success:
                raise RuntimeError(f"Failed to download {filename} from all mirrors")


def load_mnist_images(filepath: str) -> np.ndarray:
    """
    Load MNIST images from gz file.

    Args:
        filepath: Path to .gz file containing images

    Returns:
        Images array of shape (num_images, 28, 28)
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # Read image data
        buf = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)

    return data


def load_mnist_labels(filepath: str) -> np.ndarray:
    """
    Load MNIST labels from gz file.

    Args:
        filepath: Path to .gz file containing labels

    Returns:
        Labels array of shape (num_labels,)
    """
    with gzip.open(filepath, 'rb') as f:
        # Read magic number and count
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read label data
        buf = f.read(num_labels)
        data = np.frombuffer(buf, dtype=np.uint8)

    return data


def load_mnist(data_dir: str = './data/mnist', normalize: bool = True) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """
    Load MNIST dataset.

    Args:
        data_dir: Directory containing MNIST data
        normalize: If True, normalize pixel values to [0, 1]

    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        Images are float32 arrays of shape (num_images, 28, 28)
        Labels are int64 arrays of shape (num_labels,)
    """
    # Download if necessary
    download_mnist(data_dir)

    # Load training data
    train_images = load_mnist_images(
        os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    )
    train_labels = load_mnist_labels(
        os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    )

    # Load test data
    test_images = load_mnist_images(
        os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    )
    test_labels = load_mnist_labels(
        os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    )

    # Convert to float32
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    # Normalize if requested
    if normalize:
        train_images /= 255.0
        test_images /= 255.0

    return (train_images, train_labels), (test_images, test_labels)


class DataLoader:
    """
    Simple data loader for batching.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True
    ):
        """
        Initialize data loader.

        Args:
            images: Images array
            labels: Labels array
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(images)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        """Iterate over batches."""
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]
            yield batch_images, batch_labels

    def __len__(self):
        """Return number of batches."""
        return self.num_batches
