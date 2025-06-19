"""Command line interface for supermercado."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cligj
from supermercado import burntiles as burntiles
from supermercado import edge_finder as edge_finder
from supermercado import super_utils as super_utils
from supermercado import uniontiles as uniontiles

def cli() -> None:
    """Main CLI entry point."""

def edges(inputtiles: Iterable[str], parsenames: bool) -> None:
    """Find edge tiles in a tileset.

    Args:
        inputtiles: Iterable of tile identifiers or file paths.
        parsenames: Whether to parse tile names from input strings.
    """

def union(inputtiles: Iterable[str], parsenames: bool) -> None:
    """Union tiles into vector geometries.

    Args:
        inputtiles: Iterable of tile identifiers or file paths.
        parsenames: Whether to parse tile names from input strings.
    """

@cligj.features_in_arg
@cligj.sequence_opt
def burn(features: Iterable[dict[str, Any]], sequence: bool, zoom: int) -> None:
    """Burn vector features into raster tiles.

    Args:
        features: Iterable of GeoJSON feature objects.
        sequence: Whether features are in a sequence format.
        zoom: Zoom level for tile generation.
    """
