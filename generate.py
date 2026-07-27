"""
Text Generation Script

Load trained language model from checkpoint and generate text completions.
"""

import numpy as np
import sys
import os
import argparse
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

# Import modules using importlib
def _load_module(base_path, rel_path, name):
    full_path = os.path.join(base_path, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import stage3 model
_decoder_mod = _load_module(_stage3_path, 'models/decoder_lm.py', 'stage3_decoder')
DecoderLanguageModel = _decoder_mod.DecoderLanguageModel

# Import stage4 utilities
_tokenizer_mod = _load_module(_stage4_path, 'utils/tokenizer.py', 'stage4_tokenizer')
_checkpoint_mod = _load_module(_stage4_path, 'utils/checkpoint.py', 'stage4_checkpoint')

SimpleTokenizer = _tokenizer_mod.SimpleTokenizer
load_checkpoint = _checkpoint_mod.load_checkpoint


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
    parser = argparse.ArgumentParser(description='Generate text from trained language model')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_stage4_path, 'checkpoints', 'language_model'),
                        help='Path to checkpoint (without extension)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt to continue')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (0.0 = greedy, higher = more random)')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 70)
    print("Text Generation - CUDA Transformer Engine")
    print("=" * 70)
    print()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    model, tokenizer, config, training_info = load_checkpoint(
        args.checkpoint, DecoderLanguageModel, SimpleTokenizer
    )
    print(f"  Model: {config['num_layers']} layers, {config['embed_dim']} dim, "
          f"{config['num_heads']} heads")
    print(f"  Vocab size: {config['vocab_size']}")
    if training_info:
        print(f"  Final val loss: {training_info.get('final_val_loss', 'N/A')}")
    print()

    # Generate text
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Default prompts
        prompts = [
            "Once upon a time",
            "The king walked into",
            "A little girl named",
            "The cat sat",
            "One day a"
        ]

    print(f"Generating with temperature={args.temperature}, max_length={args.max_length}")
    print("-" * 70)

    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        for i in range(args.num_samples):
            generated = generate_text(
                model, tokenizer, prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            if args.num_samples > 1:
                print(f"  [{i+1}] {generated}")
            else:
                print(f"Generated: \"{generated}\"")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
