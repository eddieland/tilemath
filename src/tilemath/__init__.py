"""tilemath is a Python library for tile and coordinate utilities.

Currently, all functionality is provided through the `tilemath.mercantile` module, which offers a modern, type-safe
implementation of spherical mercator coordinate and tile utilities. This library is designed as an
API-compatible replacement for the original mercantile library with full type safety and minimal dependencies.

As the library evolves, additional modules and a higher-level API may be added to provide more specialized tile
mathematics and coordinate transformation utilities.

Basic Usage:
    >>> import tilemath.mercantile as mercantile
    >>>
    >>> # Convert longitude/latitude to tile coordinates
    >>> tile = mercantile.tile(-122.4194, 37.7749, 12)  # San Francisco
    >>> print(f"Tile: {tile.x}, {tile.y}, {tile.z}")
    Tile: 655, 1582, 12
    >>>
    >>> # Get bounding box for a tile
    >>> bbox = mercantile.bounds(tile)
    >>> print(f"Bounds: {bbox}")
    Bounds: LngLatBbox(west=-122.431640625, south=37.77071473849203, east=-122.4072265625, north=37.78808138412046)
    >>>
    >>> # Convert tile to quadkey
    >>> quadkey = mercantile.quadkey(tile)
    >>> print(f"Quadkey: {quadkey}")
    Quadkey: 023010233033

Type Safety:
    >>> from tilemath.mercantile import Tile, LngLatBbox
    >>>
    >>> # All functions have proper type hints
    >>> def process_tile(tile: Tile) -> LngLatBbox:
    ...     return mercantile.bounds(tile)
    >>>
    >>> # Type checkers will catch errors
    >>> tile = Tile(x=1, y=2, z=3)
    >>> bbox: LngLatBbox = process_tile(tile)

Available Functions:
    The mercantile module provides comprehensive tile utilities including:
    - tile(), bounds(), xy(), lnglat() for coordinate conversions
    - quadkey(), quadkey_to_tile() for Microsoft quadkey support
    - tiles() for generating tiles from bounding boxes
    - children(), parent(), neighbors() for tile relationships
    - feature() for GeoJSON representation
    - And many more specialized functions
"""
