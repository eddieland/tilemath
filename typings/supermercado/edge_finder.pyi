"""Find edge tiles in a tileset."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

def findedges(inputtiles: Iterable[str], parsenames: bool) -> list[dict[str, Any]]:
    """Find edge tiles in a collection of tiles.

    Args:
        inputtiles: Iterable of tile identifiers or file paths.
        parsenames: Whether to parse tile names from input strings.

    Returns:
        List of GeoJSON feature objects representing edge tiles.
    """
