# Stage 5: The Recommendation Pivot

**Goal:** prove the same architecture generalizes to a different domain, by
changing only the input representation.

## What's here

[`utils/item_vocab.py`](utils/item_vocab.py) — `ItemVocab`, mirroring Stage 4's
`SimpleTokenizer` interface exactly (same special-token layout, same
`encode`/`decode`/`save`/`load`), so movie IDs can stand in for word tokens
without any model or checkpoint code changing.

[`utils/data.py`](utils/data.py) — builds sequences **per user**, ordered by
timestamp, and never slides a window across a user boundary (the one genuinely
domain-specific decision in this stage — a statement about how data is
segmented, not about the model).

`checkpoints/`, `data/` (both gitignored — large files) hold the trained
weights and the MovieLens 1M dataset.

The model is, again, **Stage 3's `DecoderLanguageModel`, completely
unchanged** — only the vocabulary (movie IDs instead of words) and data
segmentation differ. See [stage5_writeup.md](../stage5_writeup.md) for the full
argument that next-item and next-token prediction are structurally the same
task, and the root [README.md](../README.md#architecture) for the diagram.

## Deliverable

A script that takes a user's item history and predicts the next item.

```bash
# Train (CPU, ~1.8 hours for the full 15-epoch run)
python3 train_recommender.py

# Predict from the trained checkpoint
python3 recommend.py --user-id 1
```

## Results

Same hyperparameters as Stage 4, bigger vocabulary (3,710 movie IDs → 1,749,248
params). 6,040 users, 1.0M interactions, 15 epochs:

| Metric | Value |
|---|---|
| Train loss | 6.46 → **4.20** |
| Val loss | 5.77 → 5.03 |
| Training time | ~1.82 hours (CPU) |

Example prediction — user who watched *A Bug's Life, Antz, The Hunchback of
Notre Dame, Hercules, Mulan*:

| Rank | Predicted next movie | Score |
|---|---|---|
| 1 | The Lion King (1994) | 11.68 |
| 2 | The Sword in the Stone (1963) | 10.02 |
| 3 | The Swan Princess (1994) | 9.51 |

*(Actual next movie: Pocahontas — a plausible miss, same genre cluster.)*

All example predictions: [`stage5_results.txt`](../stage5_results.txt).

Full numbers for every stage: [BENCHMARKS.md](../BENCHMARKS.md).

## Stop point

Per [BUILD_STAGES.md](../BUILD_STAGES.md), this is the end of the core
project: a transformer built from CUDA kernels up, trained end to end, proven
on two different applications, with benchmarks from Stage 1. Stages 6–8 are
optional extensions.

## Next

Stage 6 replaces this architecture's attention mechanism with a faster, more
memory-efficient tiled implementation. See
[stage6/README.md](../stage6/README.md).
