# Stage 4: Train the Tiny Language Model

**Goal:** prove the Stage 3 transformer actually learns language, not just a
synthetic task.

## What's here

[`utils/tokenizer.py`](utils/tokenizer.py) — `SimpleTokenizer`, a word-level
tokenizer with save/load, built specifically so its interface could be mirrored
by Stage 5's `ItemVocab` without any other code needing to change.

[`utils/data.py`](utils/data.py) — `create_sequences`, a sliding-window
sequence builder (seq_len=64, stride=32, 50% overlap) over the token stream.

[`utils/checkpoint.py`](utils/checkpoint.py) — save/load for model weights,
config, and tokenizer together.

`checkpoints/` (gitignored — large files) holds the trained weights, tokenizer,
config, and training info used by [`generate.py`](../generate.py).

The model itself is **Stage 3's `DecoderLanguageModel`, unchanged** — this
folder is data plumbing and training orchestration around it, not a new
architecture.

## Deliverable

A script that takes a text prompt and generates a completion.

```bash
# Train (CPU, ~2 hours for the full 15-epoch run)
python3 train_language_model.py

# Generate from the trained checkpoint
python3 generate.py --prompt "The king walked into"
```

## Results

1,311,488-param decoder (vocab 2,000, seq 64, embed 128, 4 layers, 4 heads),
trained on the first 5MB of TinyStories, 15 epochs:

| Metric | Value |
|---|---|
| Train loss | 3.40 → **1.69** |
| Val loss | 2.87 → 2.66 (best 2.45 at epoch 6) |
| Training time | ~2.07 hours (CPU) |

Example completions:

> **Prompt:** "The king walked into" → *"the king walked into the forest. he
> was not scared of the lion. he was too scared to go away. the king saw the
> queen and wanted the best."*

> **Prompt:** "Once upon a time" → *"once upon a time, there was a little girl
> named sue. sue loved to play with her toys. one day, sue's friend, tom, came
> to play"*

All five example generations: [`stage4_results.txt`](../stage4_results.txt).

Full numbers for every stage: [BENCHMARKS.md](../BENCHMARKS.md).

## Next

Stage 5 retrains this exact same model on movie-watching sequences instead of
text, to prove the architecture generalizes. See
[stage5/README.md](../stage5/README.md).
