"""
In-memory title search over the MovieLens movie catalog.

Builds one list of (lowercased_title, movie_id, title) at startup from
stage5/utils/data.py::load_movie_titles (which already handles movies.dat's
latin-1 encoding correctly), then does a simple case-insensitive substring
match per query -- no fuzzy-matching library needed for ~3,883 titles.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SearchIndex:
    # (lowercased_title, movie_id, original_title), sorted by title for
    # stable, readable ordering when scores tie.
    entries: List[Tuple[str, int, str]]

    @classmethod
    def build(cls, movie_titles: dict) -> "SearchIndex":
        """movie_titles: {movie_id: title}, as returned by load_movie_titles."""
        entries = sorted(
            ((title.lower(), movie_id, title) for movie_id, title in movie_titles.items()),
            key=lambda e: e[2],
        )
        return cls(entries=entries)

    def search(self, query: str, limit: int = 10) -> List[Tuple[int, str]]:
        """
        Case-insensitive search: titles that start with the query are ranked
        before titles that merely contain it. Returns [(movie_id, title), ...].
        """
        q = query.strip().lower()
        if not q:
            return []

        # Full scan every time: ~3,883 titles is sub-millisecond in Python,
        # and an early break risks under-filling results (e.g. few prefix
        # matches but plenty of substring matches later in the sorted list).
        prefix_matches = []
        substring_matches = []
        for lowered, movie_id, title in self.entries:
            if lowered.startswith(q):
                prefix_matches.append((movie_id, title))
            elif q in lowered:
                substring_matches.append((movie_id, title))

        return (prefix_matches + substring_matches)[:limit]
