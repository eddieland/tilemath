"""Union tiles into vector geometries."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

def union(inputtiles: Iterable[str], parsenames: bool) -> list[dict[str, Any]]:
    """Union a collection of tiles into vector geometries.

    Args:
        inputtiles: Iterable of tile identifiers or file paths.
        parsenames: Whether to parse tile names from input strings.

    Returns:
        List of GeoJSON feature objects representing the unioned geometries.
    """
