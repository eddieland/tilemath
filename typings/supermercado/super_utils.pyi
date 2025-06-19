"""Utility functions for supermercado."""

from __future__ import annotations

from collections.abc import Generator, Iterable, Sequence
from typing import Any

import numpy as np

def parseString(tilestring: str, matcher: str) -> tuple[int, int, int] | None:
    """Parse a tile string using a regex matcher.

    Args:
        tilestring: String containing tile coordinates.
        matcher: Regex pattern to extract coordinates.

    Returns:
        Tuple of (x, y, z) coordinates or None if parsing fails.
    """

def get_range(xyz: Sequence[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    """Get the bounding range of a collection of tiles.

    Args:
        xyz: Sequence of (x, y, z) tile coordinates.

    Returns:
        Tuple of (xmin, xmax, ymin, ymax) bounds.
    """

def burnXYZs(
    tiles: Sequence[tuple[int, int, int]], xmin: int, xmax: int, ymin: int, ymax: int, pad: int = 1
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Burn tile coordinates into a raster array.

    Args:
        tiles: Sequence of (x, y, z) tile coordinates.
        xmin: Minimum x coordinate.
        xmax: Maximum x coordinate.
        ymin: Minimum y coordinate.
        ymax: Maximum y coordinate.
        pad: Padding around the bounds.

    Returns:
        Raster array with burned tiles.
    """

def tile_parser(tiles: Iterable[str], parsenames: bool = False) -> list[tuple[int, int, int]]:
    """Parse tile identifiers from strings.

    Args:
        tiles: Iterable of tile identifier strings.
        parsenames: Whether to parse tile names from strings.

    Returns:
        List of (x, y, z) tile coordinates.
    """

def get_idx() -> str:
    """Get a tile index pattern.

    Returns:
        Regex pattern for tile indices.
    """

def get_zoom(tiles: Sequence[tuple[int, int, int]]) -> int:
    """Get the zoom level from a collection of tiles.

    Args:
        tiles: Sequence of (x, y, z) tile coordinates.

    Returns:
        Zoom level of the tiles.
    """

def filter_features(features: Iterable[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
    """Filter GeoJSON features.

    Args:
        features: Iterable of GeoJSON feature objects.

    Yields:
        Filtered GeoJSON features.
    """

class Unprojecter:
    """Utility class for unprojecting coordinates from web mercator."""

    R2D: float
    A: float

    def __init__(self) -> None:
        """Initialize the unprojector."""

    def xy_to_lng_lat(self, coordinates: Sequence[tuple[float, float]]) -> Generator[tuple[float, float], None, None]:
        """Convert web mercator coordinates to longitude/latitude.

        Args:
            coordinates: Sequence of (x, y) web mercator coordinates.

        Yields:
            Longitude/latitude coordinate pairs.
        """

    def unproject(self, feature: dict[str, Any]) -> dict[str, Any]:
        """Unproject a GeoJSON feature from web mercator to geographic coordinates.

        Args:
            feature: GeoJSON feature with web mercator coordinates.

        Returns:
            GeoJSON feature with geographic coordinates.
        """
