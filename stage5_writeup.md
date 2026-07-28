# Stage 5 Writeup: Next-Item Prediction *is* Next-Token Prediction

## The claim

Stages 3 and 4 built a transformer decoder and trained it to predict the next **word**
in a sentence. Stage 5 trains the *same* model to predict the next **movie** a user will
watch. No line of the transformer changed. This document explains why that works — why
next-item recommendation and next-token language modeling are not two problems that
happen to share a tool, but literally the **same problem** wearing different labels.

## Both tasks are "given a sequence of symbols, predict the next symbol"

A language model sees a sequence of token IDs — `[the, king, walked, into, ...]` mapped to
integers `[42, 913, 7, 88, ...]` — and learns a distribution over the vocabulary for the
symbol at each next position. A sequential recommender sees a sequence of item IDs — a
user's movies in the order they watched them, `[Toy Story, Aladdin, Mulan, ...]` mapped to
integers `[6, 231, 480, ...]` — and learns a distribution over the catalog for the item at
each next position.

Strip away the human-facing labels ("word" vs "movie") and both tasks are identical:

> You are given a sequence of discrete symbols drawn from a fixed vocabulary. Model the
> probability of the next symbol conditioned on all the symbols before it.

That is autoregressive sequence modeling. The transformer never knew or cared that its
integers meant words. It only ever saw integers, an embedding table, attention over
positions, and a softmax over a vocabulary. Swapping the *meaning* of the integers — words
become movies — changes nothing about the computation. The training objective is the same
too: cross-entropy between the predicted next-symbol distribution and the actual next
symbol, with a causal mask so position *t* may only attend to positions ≤ *t*. In both
domains the causal mask encodes the same real-world constraint: you can only use the past
to predict the future. A user's future movie choices must be predicted from earlier ones,
exactly as a sentence's next word must be predicted from earlier words.

## Where the two tasks meet in the code

The model class is `stage3/models/decoder_lm.py::DecoderLanguageModel`. Its `forward`
signature is `forward(token_ids: np.ndarray) -> logits` where `token_ids` has shape
`(batch, seq_len)` and the output has shape `(batch, seq_len, vocab_size)`. That signature
is domain-agnostic: it accepts a batch of integer sequences and returns a score for every
vocabulary entry at every position. Nothing inside it inspects what the integers *mean*.

The pipeline is:

```
integer IDs -> token embedding lookup -> + positional embedding
            -> N transformer blocks (causal self-attention + FFN, pre-norm residual)
            -> final layernorm -> linear projection to vocab_size -> logits
```

Every stage of that pipeline is oblivious to semantics:

- **Embedding lookup** is `embedding_matrix[token_ids]` — it gathers row *i* for symbol *i*.
  Whether row 480 stands for the word "castle" or the movie *Mulan* is irrelevant; it is
  just the 480th learnable vector. The Stage 5 embedding table has 3,710 rows (3,706 movies
  + 4 special tokens) instead of 2,000 (words); the lookup operation is byte-for-byte the
  same.
- **Attention** computes similarities between positions in embedding space. It learns which
  earlier items matter for predicting the next one — the same mechanism that learns which
  earlier words matter for the next word.
- **Output projection + softmax** produce a distribution over the vocabulary. Argmax /
  top-k over that distribution is "most likely next word" in Stage 4 and "most likely next
  movie" in Stage 5 — same operation, different lookup table on the way out.

## What actually changed (and why it isn't the architecture)

Only two things were swapped, and both live *outside* the model:

1. **The vocabulary.** Stage 4 used `SimpleTokenizer` to map words → IDs. Stage 5 uses
   `stage5/utils/item_vocab.py::ItemVocab` to map movie IDs → IDs. `ItemVocab` was written
   to mirror the tokenizer's interface exactly (same special-token layout, same
   `encode`/`decode`/`save`/`load`), which is *why the Stage 4 checkpoint code
   (`save_checkpoint`/`load_checkpoint`) works for Stage 5 without a single edit. From the
   model's perspective a "vocabulary" is just a bijection between symbols and the integers
   `0..V-1`; it does not matter what the symbols are.

2. **The data shaping.** Stage 4 slid a window over one long stream of text. Stage 5 builds
   sequences **per user**, ordered by timestamp (`stage5/utils/data.py`), and never slides a
   window across a user boundary — "the next movie" is only meaningful *within* one user's
   history. This is the one genuinely domain-specific decision, and notice that it is a
   statement about how the *data* is segmented, not about the model. The text pipeline made
   the analogous choice implicitly (a document is one coherent stream); recommendation just
   makes the unit of "a coherent sequence" a single user.

The transformer itself — `stage3/nn/embedding.py`, `stage3/nn/attention.py`,
`stage3/nn/transformer.py`, and `stage3/models/decoder_lm.py` — was reused **unchanged**.
The loss (`CrossEntropyLoss`) and optimizer (`Adam`) were reused unchanged. Even the batch
iterator (`TextDataLoader`) was reused unchanged, because it only ever operated on integer
arrays. The only new files are the item vocabulary, the MovieLens loader, and the two
driver scripts (`train_recommender.py`, `recommend.py`).

## Why this matters

This is the payoff of building the engine from the kernels up rather than calling a library.
Because the transformer was written to consume abstract integer sequences — not "text" —
pointing it at a completely different domain took new *data plumbing* and nothing else. The
same 1.7M-parameter decoder that continues *"the king walked into..."* also answers *"a user
who watched Toy Story, Aladdin, and Mulan will next watch..."*. Next-token prediction and
next-item prediction were never two problems. They are one problem — autoregressive
next-symbol modeling over discrete sequences — and one architecture solves both.

The exact final training loss and worked examples for the recommender are in
`stage5_results.txt`.
