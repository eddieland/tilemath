"""Burn vector geometries into raster tiles."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypedDict

import numpy as np
from affine import Affine

class TileRange(TypedDict):
    """Tile range bounds."""

    x: dict[str, int]
    y: dict[str, int]

class Bounds(TypedDict):
    """Geographic bounds."""

    west: float
    south: float
    east: float
    north: float

def project_geom(geom: dict[str, Any]) -> dict[str, Any]:
    """Project a GeoJSON geometry to web mercator coordinates.

    Args:
        geom: GeoJSON geometry object.

    Returns:
        Projected geometry with coordinates in web mercator.
    """

def find_extrema(features: Iterable[dict[str, Any]]) -> Bounds:
    """Find the geographic bounds of a collection of GeoJSON features.

    Args:
        features: Iterable of GeoJSON feature objects.

    Returns:
        Geographic bounds containing all features.
    """

def tile_extrema(bounds: Bounds, zoom: int) -> TileRange:
    """Get the tile range that covers the given geographic bounds.

    Args:
        bounds: Geographic bounds.
        zoom: Zoom level.

    Returns:
        Tile range covering the bounds.
    """

def make_transform(tilerange: TileRange, zoom: int) -> Affine:
    """Create an affine transform for the given tile range.

    Args:
        tilerange: Tile range bounds.
        zoom: Zoom level.

    Returns:
        Affine transformation matrix.
    """

def burn(polys: Sequence[dict[str, Any]], zoom: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Burn vector polygons into a raster and return tile coordinates.

    Args:
        polys: Sequence of GeoJSON polygon features.
        zoom: Zoom level for tile generation.

    Returns:
        Array of tile coordinates (x, y, z) where polygons intersect tiles.
    """
