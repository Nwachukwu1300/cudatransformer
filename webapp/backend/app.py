"""
FastAPI app serving the Stage 5 recommender.

Loads the trained checkpoint once at startup (via stage5/inference.py, the
same module recommend.py uses) and serves two API endpoints plus the static
frontend. See webapp/README.md for how to run this locally and how it's
deployed.
"""

import importlib.util
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import (
    HealthResponse,
    Recommendation,
    RecommendRequest,
    RecommendResponse,
    SearchResponse,
    SearchResult,
)
from .catalog import load_movie_meta
from .search_index import SearchIndex

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_WEBAPP_DIR = os.path.dirname(_BACKEND_DIR)
_ROOT = os.path.dirname(_WEBAPP_DIR)
_STAGE5_DIR = os.path.join(_ROOT, "stage5")
_FRONTEND_DIR = os.path.join(_WEBAPP_DIR, "frontend")


def _load_module(base_path, rel_path, name):
    full_path = os.path.join(base_path, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_inference_mod = _load_module(_STAGE5_DIR, "inference.py", "stage5_inference")

# Mutable module-level state, populated once at startup (see `lifespan` below)
# and read (never mutated) on every request. The model is stateless at
# inference time, so sharing one loaded copy across requests is safe.
state = {}

DEV_MODE = os.environ.get("ENV", "production") == "development"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, vocab, config, training_info = _inference_mod.load_recommender()
    state["model"] = model
    state["vocab"] = vocab
    state["config"] = config

    movies_path = os.path.join(_STAGE5_DIR, "data", "ml-1m", "movies.dat")
    movie_titles = _inference_mod.load_movie_titles(movies_path)
    state["movie_titles"] = movie_titles
    state["movie_meta"] = load_movie_meta(movies_path)
    state["search_index"] = SearchIndex.build(movie_titles)

    print(f"Recommender loaded: {config['num_layers']} layers, "
          f"{config['vocab_size']} vocab, {len(movie_titles)} titles indexed.")
    yield
    state.clear()


app = FastAPI(title="CUDA Transformer Engine — Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if DEV_MODE else [],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.get("/api/search", response_model=SearchResponse)
def search(q: str = "", limit: int = 10):
    limit = max(1, min(limit, 25))
    index: SearchIndex = state["search_index"]
    matches = index.search(q, limit=limit)
    return SearchResponse(
        results=[SearchResult(movie_id=mid, title=title) for mid, title in matches]
    )


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    model = state["model"]
    vocab = state["vocab"]
    movie_titles = state["movie_titles"]
    movie_meta = state["movie_meta"]

    preds = _inference_mod.predict_next_items(model, vocab, req.history, top_k=req.top_k)

    recommendations = []
    for movie_id, score in preds:
        fallback = movie_titles.get(movie_id, f"movie {movie_id}")
        meta = movie_meta.get(movie_id)
        recommendations.append(
            Recommendation(
                movie_id=movie_id,
                title=fallback,
                score=score,
                display=meta.display if meta else fallback,
                year=meta.year if meta else "",
                genres=meta.genres if meta else [],
            )
        )

    return RecommendResponse(recommendations=recommendations)


# Mounted last so it never shadows the /api/* routes above.
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
