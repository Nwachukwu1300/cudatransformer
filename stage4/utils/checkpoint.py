"""
Checkpoint saving and loading utilities.
"""

import numpy as np
import json
import os
from typing import Dict, Any


def save_checkpoint(
    model,
    tokenizer,
    path: str,
    model_config: Dict[str, Any],
    training_info: Dict[str, Any] = None
):
    """
    Save model checkpoint.

    Saves:
    - Model weights as .npz file
    - Model config as .json file
    - Tokenizer vocabulary as .json file
    - Training info as .json file (optional)

    Args:
        model: The DecoderLanguageModel to save
        tokenizer: The tokenizer to save
        path: Base path for checkpoint (without extension)
        model_config: Model configuration dict
        training_info: Optional training information (loss, epochs, etc.)
    """
    # Create checkpoint directory if needed
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Collect all model parameters
    param_dict = {}
    param_idx = 0

    for param in model.parameters():
        param_dict[f'param_{param_idx}'] = param.data
        param_idx += 1

    # Save weights
    weights_path = f"{path}_weights.npz"
    np.savez(weights_path, **param_dict)
    print(f"  Saved weights to {weights_path}")

    # Save model config
    config_path = f"{path}_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"  Saved config to {config_path}")

    # Save tokenizer
    tokenizer_path = f"{path}_tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"  Saved tokenizer to {tokenizer_path}")

    # Save training info if provided
    if training_info is not None:
        info_path = f"{path}_training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        print(f"  Saved training info to {info_path}")


def load_checkpoint(path: str, model_class, tokenizer_class):
    """
    Load model from checkpoint.

    Args:
        path: Base path for checkpoint (without extension)
        model_class: The model class to instantiate
        tokenizer_class: The tokenizer class to use

    Returns:
        model: Loaded model with weights
        tokenizer: Loaded tokenizer
        model_config: Model configuration
        training_info: Training information (or None)
    """
    # Load model config
    config_path = f"{path}_config.json"
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    print(f"  Loaded config from {config_path}")

    # Load tokenizer
    tokenizer_path = f"{path}_tokenizer.json"
    tokenizer = tokenizer_class.load(tokenizer_path)
    print(f"  Loaded tokenizer from {tokenizer_path}")

    # Create model
    model = model_class(
        vocab_size=model_config['vocab_size'],
        max_seq_len=model_config['max_seq_len'],
        embed_dim=model_config['embed_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        ff_hidden_dim=model_config.get('ff_hidden_dim')
    )

    # Load weights
    weights_path = f"{path}_weights.npz"
    weights = np.load(weights_path)
    print(f"  Loaded weights from {weights_path}")

    # Restore weights to model parameters
    param_idx = 0
    for param in model.parameters():
        key = f'param_{param_idx}'
        if key in weights:
            param.data = weights[key]
        param_idx += 1

    print(f"  Restored {param_idx} parameters")

    # Load training info if available
    training_info = None
    info_path = f"{path}_training_info.json"
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            training_info = json.load(f)
        print(f"  Loaded training info from {info_path}")

    return model, tokenizer, model_config, training_info
