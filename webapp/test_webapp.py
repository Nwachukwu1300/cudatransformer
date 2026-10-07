"""
Smoke tests for the webapp FastAPI backend.

Run with: pytest webapp/test_webapp.py (from the repo root, with
webapp/requirements_webapp.txt installed).
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from webapp.backend.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs FastAPI's lifespan startup
    # (loading the model/search index) -- a bare TestClient(app) does not,
    # and every request would fail with KeyError on `state["model"]`.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_search_finds_known_title(client):
    resp = client.get("/api/search", params={"q": "Toy Story"})
    assert resp.status_code == 200
    titles = [r["title"] for r in resp.json()["results"]]
    assert any("Toy Story (1995)" == t for t in titles)


def test_search_is_case_insensitive(client):
    resp_lower = client.get("/api/search", params={"q": "toy story"})
    resp_upper = client.get("/api/search", params={"q": "TOY STORY"})
    assert resp_lower.status_code == 200
    assert resp_upper.status_code == 200
    titles_lower = {r["movie_id"] for r in resp_lower.json()["results"]}
    titles_upper = {r["movie_id"] for r in resp_upper.json()["results"]}
    assert titles_lower == titles_upper
    assert len(titles_lower) > 0


def test_search_finds_non_ascii_title(client):
    # Movie ID 73 in movies.dat: "Misérables, Les (1995)". Regression test
    # for the latin-1 decoding in stage5/utils/data.py::load_movie_titles --
    # if that ever breaks, this title would either fail to load or come back
    # mangled, and this search would silently return nothing.
    resp = client.get("/api/search", params={"q": "Mis"})
    assert resp.status_code == 200
    results = resp.json()["results"]
    matching = [r for r in results if r["movie_id"] == 73]
    assert len(matching) == 1
    assert matching[0]["title"] == "Misérables, Les (1995)"


def test_search_empty_query_returns_empty(client):
    resp = client.get("/api/search", params={"q": ""})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_recommend_returns_top_k(client):
    resp = client.post("/api/recommend", json={"history": [1, 34, 587], "top_k": 3})
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    assert len(recs) == 3
    for rec in recs:
        assert "movie_id" in rec
        assert "title" in rec
        assert "score" in rec


def test_recommend_matches_cli(client):
    # Cross-check against the exact same known-good values recommend.py
    # produces for this history (verified manually against
    # `python3 recommend.py --history 1,34,587 --top-k 3`).
    resp = client.post("/api/recommend", json={"history": [1, 34, 587], "top_k": 3})
    assert resp.status_code == 200
    recs = resp.json()["recommendations"]
    titles = [r["title"] for r in recs]
    assert titles == [
        "League of Their Own, A (1992)",
        "Austin Powers: International Man of Mystery (1997)",
        "Aladdin (1992)",
    ]


def test_recommend_handles_unknown_movie_id_gracefully(client):
    # A movie ID far outside the real MovieLens range -- should fall through
    # to ItemVocab's UNK handling, not raise a 500.
    resp = client.post("/api/recommend", json={"history": [999999999], "top_k": 3})
    assert resp.status_code == 200
    assert "recommendations" in resp.json()


def test_recommend_rejects_empty_history(client):
    resp = client.post("/api/recommend", json={"history": [], "top_k": 3})
    assert resp.status_code == 422


def test_recommend_rejects_oversized_history(client):
    resp = client.post("/api/recommend", json={"history": list(range(100)), "top_k": 3})
    assert resp.status_code == 422
