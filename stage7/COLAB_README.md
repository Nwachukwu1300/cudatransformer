# Stage 7 on Google Colab (GPU and TPU)

JAX/Flax training can't be benchmarked on a real GPU or TPU from this Mac (CPU
only — confirmed via `jax.devices()`). Run the same script on Colab's GPU
runtime, then again on its TPU runtime, to fill in both rows — the same way
Stage 2 and Stage 6's GPU numbers were produced.

## What you need on Colab

The script only needs three things from the repo, not the full 2.2 GB
`data/` folder:

- `stage7/` (this folder)
- `stage4/utils/tokenizer.py` and `stage4/utils/data.py` (reused as-is)
- `stage4/checkpoints/language_model_tokenizer.json` (33 KB — the saved vocab)
- The **first 5 MB** of `data/tinystories_train.txt` (Stage 4 only ever reads
  `max_chars = 5*1024*1024`, so there's no need to upload the whole file)

`stage4/checkpoints/` and `data/` are gitignored (large/generated files), so a
plain `git clone` will NOT bring them — upload them separately as below.

## Steps

1. Colab menu → **Runtime → Change runtime type → GPU** (a T4 or L4 is fine).
2. Get the code:
   ```
   !git clone <your-repo-url>
   %cd cudatransformer
   ```
3. Install Flax/Optax (JAX with CUDA is already preinstalled on Colab GPU runtimes):
   ```
   !pip -q install -U flax optax
   ```
4. Upload the tokenizer and a 5 MB slice of the training data (from your local
   machine, since both are gitignored):
   ```python
   from google.colab import files
   import os
   os.makedirs("stage4/checkpoints", exist_ok=True)
   os.makedirs("data", exist_ok=True)
   uploaded = files.upload()  # pick language_model_tokenizer.json, then move it:
   os.rename("language_model_tokenizer.json",
             "stage4/checkpoints/language_model_tokenizer.json")
   uploaded = files.upload()  # pick a <=5MB slice of tinystories_train.txt
   os.rename("tinystories_train.txt", "data/tinystories_train.txt")
   ```
   To make the 5 MB slice locally first (avoids uploading the full 2.2 GB file):
   ```bash
   head -c 5242880 data/tinystories_train.txt > tinystories_train_5mb.txt
   # upload this instead, then on Colab:
   # os.rename("tinystories_train_5mb.txt", "data/tinystories_train.txt")
   ```
5. Run the full training benchmark:
   ```
   !python stage7/train_benchmark.py
   ```
   This trains 15 epochs (Stage 4's recipe), then appends a `gpu` row to
   `stage7_results.json` and rewrites `stage7_results.txt` with the comparison
   table (GPU row + the Stage 4 CPU/NumPy reference row).
6. Download both files back:
   ```python
   from google.colab import files
   files.download("stage7_results.txt")
   files.download("stage7_results.json")
   ```

## Then do the same on TPU

1. **Runtime → Change runtime type → TPU**.
2. Same install command; JAX-on-TPU is preinstalled on Colab TPU runtimes too:
   ```
   !pip -q install -U flax optax
   ```
   (If devices don't show up, **Runtime → Restart runtime** after installing.)
3. Re-upload the tokenizer + 5 MB data slice (Colab TPU runtimes are separate
   VMs — files from the GPU session are gone).
4. Run the identical command:
   ```
   !python stage7/train_benchmark.py
   ```
   This appends a `tpu` row to the same `stage7_results.json` structure. If you
   bring back the GPU run's `stage7_results.json` and drop it in the working
   directory first, the TPU run will accumulate into it rather than starting
   fresh, and the final `stage7_results.txt` will show both rows plus the
   "Both GPU and TPU rows recorded. Stage 7 complete." line.
5. Download `stage7_results.txt` / `stage7_results.json` again and merge them
   back into the repo (overwrite the local copies from the CPU-smoke run).

## Kaggle TPU fallback

If Colab TPU is unavailable, Kaggle's free TPU v3-8 works the same way:
`pip install -U "jax[tpu]" flax optax`, upload the same three inputs, run
`python stage7/train_benchmark.py`. jax-on-TPU can be version-sensitive on
Kaggle — restart the session if devices don't show up after install.

## Sanity checks the script already does for you

- **Architecture:** asserts the Flax model has exactly 1,311,488 parameters
  (Stage 3/4's count) before training starts — a mismatch means the
  architecture drifted and the comparison wouldn't be fair.
- **Training:** the results file's "Verification" section checks that the
  final loss is actually lower than the initial loss for every recorded row.
- **Cost:** `train_hours x public on-demand rate` — a rough estimate, not
  what you actually paid (Colab/Kaggle free tiers are usually $0).

## Local smoke test (already done on this Mac, CPU-only)

```
python3 -m venv .venv_stage7 && source .venv_stage7/bin/activate
pip install jax jaxlib flax optax
python stage7/train_benchmark.py --smoke
```
Confirms the model/data pipeline is correct before spending Colab GPU/TPU
time: params=1,311,488, loss dropped 8.59 → 4.99 over 100 CPU steps.
