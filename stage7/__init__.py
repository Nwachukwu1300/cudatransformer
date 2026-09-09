"""
Stage 7: TPU training comparison.

A JAX/Flax REIMPLEMENTATION of the Stage 3 transformer architecture -- NOT a
port or wrapper of the Stage 1-6 CUDA kernels or the NumPy autograd engine. Our
CUDA kernels cannot run on a TPU, so for this stage only we rebuild the exact
same architecture in Flax so that a GPU-vs-TPU training comparison is fair
(identical code, only the hardware changes).

Only two things are shared with earlier stages:
  - the architecture definition (mirrors stage3/), and
  - the Stage 4 training recipe: data/tinystories_train.txt (first 5 MB) and the
    saved word-level tokenizer (stage4/checkpoints/language_model_tokenizer.json).

Contents:
  model.py           - Flax linen module (DecoderLM) mirroring the Stage 3 transformer
  data.py            - loads the Stage 4 data/tokenizer and builds training batches
  train_benchmark.py - device-agnostic training + benchmark; writes stage7_results.txt
  COLAB_README.md    - how to run the identical script on Colab GPU and TPU runtimes
"""
