# Build Stages: CUDA Transformer Engine

Read PROJECT_CONTEXT.md first. Work through these stages in order. Each stage lists a goal, what to build, and a deliverable that proves the stage works. Do not start a stage until the previous one has a working deliverable.

## Stage 1: CUDA kernel fundamentals

Goal: Get comfortable writing and testing GPU kernels.

Build:
- A vector addition kernel
- A matrix multiplication kernel
- A softmax kernel
- A reduction kernel (sum or max)

Learn and apply: threads, blocks, shared memory, global memory, memory coalescing.

Deliverable: A benchmarking script that runs each kernel on GPU and a naive CPU version, and prints the speedup for each. Save the results in a results file.

## Stage 2: A small deep learning engine

Goal: Build the core of a framework, using the kernels from Stage 1.

Build:
- A Tensor class that tracks operations for backpropagation
- Linear layer
- LayerNorm
- ReLU and GELU
- Cross entropy loss
- SGD optimizer, then Adam

Wire these together into a forward pass, loss calculation, backward pass, and weight update loop.

Deliverable: Train a small MLP on MNIST using only the kernels and classes built in Stage 1 and 2. Report final accuracy.

## Stage 3: The transformer

Goal: Replace the MLP with a transformer, built on the same engine.

Build:
- Token embeddings
- Positional embeddings
- Multi head attention
- Feed forward network
- Residual connections and LayerNorm

Deliverable: A transformer that can do next token prediction on a small dataset. Confirm it runs a full forward and backward pass without errors.

## Stage 4: Train the tiny language model

Goal: Prove the transformer actually learns language.

Dataset: TinyStories or WikiText 2.

Train a small GPT style decoder using the Stage 3 transformer.

Deliverable: A script that takes a text prompt and generates a completion. Example: input "The king walked into" produces a coherent continuation. Save a few example generations to show the model works.

## Stage 5: The recommendation pivot

Goal: Prove the same architecture generalizes to a different domain.

Dataset: MovieLens or a similar sequential interaction dataset.

Change only the input representation: instead of word tokens, use item IDs (movies, products, or songs) as the vocabulary. The transformer architecture from Stage 3 stays the same.

Deliverable: A script that takes a user's item history and predicts the next item. Include a short writeup explaining why next item prediction and next token prediction are the same task structurally.

## Stop point

After Stage 5, the core project is complete. You have a transformer built from CUDA kernels up, trained end to end, proven on two different applications, with benchmarks from Stage 1.

## Optional stages (only after Stage 5 works)

Do not start these until Stage 5 has a working deliverable, and check with me before starting any of them.

### Stage 6: Simplified tiled attention

Replace standard attention with a simplified tiled version that reduces memory reads and writes, similar in spirit to FlashAttention. Benchmark speed and memory use against the Stage 3 attention implementation. This does not need to be a full production FlashAttention implementation, a working simplified version with a clear writeup is enough.

### Stage 7: TPU training comparison

Port training to a TPU using JAX and Flax. Benchmark training speed, memory use, and cost against the GPU version from earlier stages. Report the comparison.

### Stage 8: Polish and documentation

Add architecture diagrams, a benchmark summary, a writeup of the CUDA optimizations, and clear README documentation across the repository. Structure the repo so each stage has its own folder with its own deliverable clearly shown.
