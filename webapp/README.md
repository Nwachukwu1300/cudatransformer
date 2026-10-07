# Web App: the recommender, shipped

**Goal:** take the model trained in [stage5/](../stage5/) and expose it as a
real, public product — a web app anyone can use, with its own API.

This is deliberately **not** a `stageN/` folder. The numbered stages are the
story of building the engine (kernels → autograd → transformer → two trained
applications). This is what comes after that: productization. Nothing here
trains or changes a model; it serves one that already exists.

## What's here

```
webapp/
├── backend/
│   ├── app.py            FastAPI app — loads the model once, serves the API + frontend
│   ├── schemas.py        pydantic request/response models
│   ├── search_index.py   in-memory title search over the MovieLens catalog
│   └── catalog.py        year + genre metadata parsed from movies.dat
├── frontend/
│   ├── index.html        single page — search, history, results
│   ├── style.css         light "engineering document" theme
│   └── app.js            search-as-you-type, history list, recommend call
├── requirements_webapp.txt
└── test_webapp.py        11 backend smoke tests

../render.yaml            Render Blueprint — lives at the repo root, since
                          Render only auto-detects it there
```

The actual inference lives in
[`stage5/inference.py`](../stage5/inference.py) — the single source of truth
that both this app and the [`recommend.py`](../recommend.py) CLI import, so
the two can't silently diverge. One of the tests cross-checks the API's output
against the CLI's for the same input.

## The API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check. Also used by the frontend to pre-warm a sleeping instance. |
| `GET` | `/api/search?q=toy&limit=10` | Title search for the autocomplete. Case-insensitive, prefix matches ranked first. |
| `POST` | `/api/recommend` | `{"history": [1, 588, 1907], "top_k": 5}` → ranked next-movie predictions. |

`history` is a list of MovieLens movie IDs, **most recent last** — matching
`predict_next_items`' contract. Capped at 63 items (the model's `max_seq_len`
is 64).

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"history": [1, 588, 1907], "top_k": 3}'
```
```json
{"recommendations": [
  {"movie_id": 364,  "title": "Lion King, The (1994)", "score": 10.40},
  {"movie_id": 2687, "title": "Tarzan (1999)",         "score": 8.55},
  {"movie_id": 2355, "title": "Bug's Life, A (1998)",  "score": 8.51}
]}
```

Scores are raw logits, not probabilities — they rank, they don't calibrate.
The UI labels them accordingly.

## Run it locally

```bash
python3 -m venv .venv_webapp && source .venv_webapp/bin/activate
python3 -m pip install -r webapp/requirements_webapp.txt
uvicorn webapp.backend.app:app --reload --app-dir .
```

Then open <http://localhost:8000>. The frontend is served by the same process,
so there's no separate dev server and no CORS to configure.

Tests:
```bash
python3 -m pytest webapp/test_webapp.py
```

## How it's deployed

Render, free tier, native Python runtime — no Dockerfile, because the entire
dependency footprint is numpy + fastapi + uvicorn (no system packages, no GPU,
no compiled extensions beyond standard wheels).
[`render.yaml`](../render.yaml) defines the service; Render reads it when the
repo is connected.

Three things worth knowing if you redeploy this:

- **`render.yaml` sits at the repo root, not in here.** Render's Blueprint flow
  only auto-detects it at the root of the repository.
- **It runs from the repo root**, not `webapp/`. `backend/app.py` loads
  `stage5/inference.py` by path, which loads `stage2`/`stage3`/`stage4`
  modules in turn — the service needs the whole tree.
- **uvicorn must bind to `$PORT`**, which Render injects. Hardcoding a port is
  the most common reason a first deploy never passes its health check.

### The checkpoint files are committed on purpose

`stage4/checkpoints/` and most of `stage5/checkpoints/` are gitignored —
they're large and regenerable by retraining. But the four **recommender**
artifacts are committed deliberately:

```
stage5/checkpoints/recommender_config.json
stage5/checkpoints/recommender_tokenizer.json
stage5/checkpoints/recommender_weights.npz        (6.7MB)
stage5/checkpoints/recommender_training_info.json
```

The deployed service needs the model at runtime, and at ~6.8MB total it's far
cheaper to ship them in the repo than to stand up separate artifact hosting
(S3, Git LFS, a Render disk) for a model this small. The `.gitignore` carve-out
uses `stage5/checkpoints/*` rather than a trailing slash — a trailing-slash
directory ignore stops git descending into the directory at all, which
silently defeats the `!` negation lines below it.

## Related

- [stage5/README.md](../stage5/README.md) — how this model was trained, and its results
- [stage5_writeup.md](../stage5_writeup.md) — why next-movie prediction and next-word prediction are the same problem
- [Root README](../README.md) — the whole from-scratch engine this sits on top of
