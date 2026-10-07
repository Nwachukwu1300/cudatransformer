"""
Movie catalog metadata for the UI.

stage5/utils/data.py::load_movie_titles gives {movie_id: title}, which is all
the model path needs. The UI additionally shows the release year and genres
under each recommendation, so this parses the third `::` field of movies.dat
too. Kept here rather than in stage5/ because it is purely presentational --
nothing in the model or the CLI deliverable depends on it.
"""

import re
from typing import Dict, List, NamedTuple

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


class MovieMeta(NamedTuple):
    title: str          # full title as stored, e.g. "Toy Story (1995)"
    display: str        # title without the trailing year, e.g. "Toy Story"
    year: str           # "1995", or "" if the title has no trailing year
    genres: List[str]   # e.g. ["Animation", "Children's", "Comedy"]


def load_movie_meta(movies_path: str) -> Dict[int, MovieMeta]:
    """Parse movies.dat (movieId::title::genre|genre|...) -> {movie_id: MovieMeta}."""
    meta: Dict[int, MovieMeta] = {}
    with open(movies_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 2:
                continue
            movie_id, title = parts[0], parts[1]
            genres = parts[2].split("|") if len(parts) > 2 and parts[2] else []

            match = _YEAR_RE.search(title)
            year = match.group(1) if match else ""
            display = _YEAR_RE.sub("", title).strip() if match else title

            meta[int(movie_id)] = MovieMeta(
                title=title, display=display, year=year, genres=genres
            )
    return meta
