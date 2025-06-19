"""Web mercator XYZ tile utilities - Type stubs"""

from collections.abc import Generator, Sequence
from typing import Any, NamedTuple

__version__: str

__all__: list[str]

R2D: float
RE: float
CE: float
EPSILON: float
LL_EPSILON: float

class Tile(NamedTuple):
    """An XYZ web mercator tile.

    Args:
        x: X index of the tile.
        y: Y index of the tile.
        z: Zoom level z.

    Note:
        Constructor validates that x and y are within valid range (0, 2**z - 1).
        Issues FutureWarning for out-of-range coordinates in current version.
    """

    x: int
    y: int
    z: int

class LngLat(NamedTuple):
    """A longitude and latitude pair.

    Args:
        lng: Longitude in decimal degrees east.
        lat: Latitude in decimal degrees north.
    """

    lng: float
    lat: float

class LngLatBbox(NamedTuple):
    """A geographic bounding box.

    Args:
        west: Western boundary in decimal degrees.
        south: Southern boundary in decimal degrees.
        east: Eastern boundary in decimal degrees.
        north: Northern boundary in decimal degrees.
    """

    west: float
    south: float
    east: float
    north: float

class Bbox(NamedTuple):
    """A web mercator bounding box.

    Args:
        left: Left boundary in meters.
        bottom: Bottom boundary in meters.
        right: Right boundary in meters.
        top: Top boundary in meters.
    """

    left: float
    bottom: float
    right: float
    top: float

class MercantileError(Exception):
    """Base exception for mercantile library."""

class InvalidLatitudeError(MercantileError):
    """Raised when math errors occur beyond ~85 degrees N or S."""

class InvalidZoomError(MercantileError):
    """Raised when a zoom level is invalid."""

class ParentTileError(MercantileError):
    """Raised when a parent tile cannot be determined."""

class QuadKeyError(MercantileError):
    """Raised when errors occur in computing or parsing quad keys.

    Note:
        In current version, this derives from ValueError but will change
        to not derive from ValueError in mercantile 2.0.
    """

class TileArgParsingError(MercantileError):
    """Raised when errors occur in parsing a function's tile arg(s)."""

class TileError(MercantileError):
    """Raised when a tile can't be determined."""

def ul(*tile: Tile | int) -> LngLat:
    """Returns the upper left longitude and latitude of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        LngLat: Upper left corner coordinates.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """

def bounds(*tile: Tile | int) -> LngLatBbox:
    """Returns the bounding box of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        LngLatBbox: Geographic bounding box of the tile.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """

def truncate_lnglat(lng: float, lat: float) -> tuple[float, float]:
    """Truncate longitude and latitude to valid web mercator limits.

    Args:
        lng: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.

    Returns:
        Tuple[float, float]: Truncated (lng, lat) coordinates.
    """

def xy(lng: float, lat: float, truncate: bool = False) -> tuple[float, float]:
    """Convert longitude and latitude to web mercator x, y.

    Args:
        lng: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.
        truncate: Whether to truncate inputs to web mercator limits.

    Returns:
        Tuple[float, float]: Web mercator coordinates (x, y) in meters.
            y will be inf at the North Pole (lat >= 90) and -inf at the
            South Pole (lat <= -90).
    """

def lnglat(x: float, y: float, truncate: bool = False) -> LngLat:
    """Convert web mercator x, y to longitude and latitude.

    Args:
        x: Web mercator x coordinate in meters.
        y: Web mercator y coordinate in meters.
        truncate: Whether to truncate outputs to web mercator limits.

    Returns:
        LngLat: Longitude and latitude coordinates.
    """

def neighbors(*tile: Tile | int, **kwargs: Any) -> list[Tile]:
    """Get the neighbors of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List[Tile]: Up to eight neighboring tiles. Invalid tiles (e.g.,
            Tile(-1, -1, z)) are omitted from the result.

    Note:
        Makes no guarantees regarding neighbor tile ordering.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """

def xy_bounds(*tile: Tile | int) -> Bbox:
    """Get the web mercator bounding box of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        Bbox: Web mercator bounding box in meters.

    Note:
        Epsilon is subtracted from the right limit and added to the bottom
        limit for precision handling.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """

def tile(lng: float, lat: float, zoom: int, truncate: bool = False) -> Tile:
    """Get the tile containing a longitude and latitude.

    Args:
        lng: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.
        zoom: Web mercator zoom level.
        truncate: Whether to truncate inputs to web mercator limits.

    Returns:
        Tile: The tile containing the given coordinates.

    Raises:
        InvalidLatitudeError: If latitude is beyond valid range and truncate=False.
    """

def quadkey(*tile: Tile | int) -> str:
    """Get the quadkey of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        str: Quadkey string representation of the tile.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """

def quadkey_to_tile(qk: str) -> Tile:
    """Get the tile corresponding to a quadkey.

    Args:
        qk: Quadkey string.

    Returns:
        Tile: The tile corresponding to the quadkey.

    Raises:
        QuadKeyError: If quadkey contains invalid digits.

    Note:
        Issues DeprecationWarning about QuadKeyError inheritance change in v2.0.
    """

def tiles(
    west: float, south: float, east: float, north: float, zooms: int | Sequence[int], truncate: bool = False
) -> Generator[Tile, None, None]:
    """Get the tiles overlapped by a geographic bounding box.

    Args:
        west: Western boundary in decimal degrees.
        south: Southern boundary in decimal degrees.
        east: Eastern boundary in decimal degrees.
        north: Northern boundary in decimal degrees.
        zooms: One or more zoom levels.
        truncate: Whether to truncate inputs to web mercator limits.

    Yields:
        Tile: Tiles that overlap the bounding box.

    Note:
        A small epsilon is used on the south and east parameters so that this
        function yields exactly one tile when given the bounds of that same tile.
        Handles antimeridian crossing by splitting into two bounding boxes.
    """

def parent(*tile: Tile | int, **kwargs: Any) -> Tile | None:
    """Get the parent of a tile.

    The parent is the tile of one zoom level lower that contains the
    given "child" tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        **kwargs: Keyword arguments including:
            zoom: Target zoom level of the returned parent tile.
                Defaults to one lower than the input tile.

    Returns:
        Optional[Tile]: Parent tile, or None if input tile is at zoom level 0.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
        InvalidZoomError: If zoom is not an integer less than input tile zoom.
        ParentTileError: If parent of non-integer tile is requested.
    """

def children(*tile: Tile | int, **kwargs: Any) -> list[Tile]:
    """Get the children of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        **kwargs: Keyword arguments including:
            zoom: Target zoom level for returned children. If unspecified,
                returns immediate (zoom + 1) children.

    Returns:
        List[Tile]: Child tiles ordered top-left, top-right, bottom-right,
            bottom-left. For deeper zoom levels, returns all children in
            depth-first clockwise winding order.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
        InvalidZoomError: If zoom is not an integer greater than input tile zoom.
    """

def simplify(tiles: Sequence[Tile]) -> list[Tile]:
    """Reduces the size of the tileset as much as possible by merging leaves into parents.

    Args:
        tiles: Sequence of tiles to merge.

    Returns:
        List[Tile]: Simplified tileset with merged tiles.

    Note:
        Removes child tiles when their parent is already present and merges
        complete sets of 4 children into their parent tile.
    """

def rshift(val: int, n: int) -> int:
    """Right shift operation with proper handling of large integers.

    Args:
        val: Integer value to shift.
        n: Number of positions to shift right.

    Returns:
        int: Right-shifted value.
    """

def bounding_tile(*bbox: float, **kwds: Any) -> Tile:
    """Get the smallest tile containing a geographic bounding box.

    Args:
        *bbox: Bounding box as west, south, east, north values in decimal degrees.
            Can also accept 2 values which will be duplicated.
        **kwds: Keyword arguments including:
            truncate: Whether to truncate inputs to web mercator limits.

    Returns:
        Tile: Smallest tile containing the bounding box.

    Note:
        When the bbox spans lines of lng 0 or lat 0, the bounding tile
        will be Tile(x=0, y=0, z=0).

    Raises:
        InvalidLatitudeError: If latitude values are invalid and truncate=False.
    """

def feature(
    *tile: Tile | int,
    fid: str | None = None,
    props: dict[str, Any] | None = None,
    projected: str = "geographic",
    buffer: float | None = None,
    precision: int | None = None,
) -> dict[str, Any]:
    """Get the GeoJSON feature corresponding to a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        fid: Feature id. If None, uses string representation of tile.
        props: Optional extra feature properties to include.
        projected: Coordinate system for output. Use 'mercator' for
            web mercator coordinates, 'geographic' for lat/lng.
        buffer: Optional buffer distance for the GeoJSON polygon.
        precision: Number of decimal places for coordinate truncation.
            Must be >= 0 if specified.

    Returns:
        Dict[str, Any]: GeoJSON Feature dict with tile geometry and properties.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """

def geojson_bounds(obj: dict[str, Any] | list[Any] | tuple[Any, ...]) -> LngLatBbox:
    """Returns the bounding box of a GeoJSON object.

    Args:
        obj: A GeoJSON geometry, feature, feature collection, or coordinate array.

    Returns:
        LngLatBbox: Geographic bounding box of the GeoJSON object.
    """

def minmax(zoom: int) -> tuple[int, int]:
    """Minimum and maximum tile coordinates for a zoom level.

    Args:
        zoom: Web mercator zoom level.

    Returns:
        Tuple[int, int]: (minimum, maximum) tile coordinates where minimum
            is always 0 and maximum is (2 ** zoom - 1).

    Raises:
        InvalidZoomError: If zoom level is not a positive integer.
    """
