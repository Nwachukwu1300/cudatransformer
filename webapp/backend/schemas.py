"""
Pydantic request/response models for the recommender API.
"""

from typing import List

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    movie_id: int
    title: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class RecommendRequest(BaseModel):
    # Most recent last, matching stage5/inference.py::predict_next_items's
    # contract. max_length=63 leaves room under the model's max_seq_len=64
    # (predict_next_items also truncates internally, this is a defense-in-depth
    # API-level contract, not a replacement for that truncation).
    history: List[int] = Field(..., min_length=1, max_length=63)
    top_k: int = Field(default=5, ge=1, le=20)


class Recommendation(BaseModel):
    movie_id: int
    title: str
    score: float
    # Presentation-only metadata from movies.dat, shown under each result.
    # `display` is the title with the trailing "(year)" stripped.
    display: str = ""
    year: str = ""
    genres: List[str] = []


class RecommendResponse(BaseModel):
    recommendations: List[Recommendation]


class HealthResponse(BaseModel):
    status: str
