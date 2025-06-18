"""Mercantile-compatible API for tilemath.

All functions and classes in this module are designed to be 100% API compatible with the original mercantile library.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

__all__ = [
    "Bbox",
    "LngLat",
    "LngLatBbox",
    "Tile",
    "bounding_tile",
    "bounds",
    "children",
    "feature",
    "lnglat",
    "neighbors",
    "parent",
    "quadkey",
    "quadkey_to_tile",
    "simplify",
    "tile",
    "tiles",
    "ul",
    "xy_bounds",
    "minmax",
    # Constants
    # "R2D",
    # "RE",
    # "CE",
    # "EPSILON",
    # "LL_EPSILON",
]

R2D: Final[float] = 180.0 / 3.14159265358979323846  # Conversion factor from radians to degrees
RE: Final[float] = 6378137.0  # Radius of the Earth in meters (WGS84)
CE: Final[float] = 2 * 3.14159265358979323846 * RE  # Circumference of the Earth in meters (WGS84)
EPSILON: Final[float] = 1e-12  # Small value for precision handling
LL_EPSILON: Final[float] = 1e-6  # Small value for precision handling in longitude/latitude


# Exception classes matching mercantile
class MercantileError(Exception):
    """Base exception for mercantile library."""


class InvalidLatitudeError(MercantileError):
    """Raised when math errors occur beyond ~85 degrees N or S."""


class InvalidZoomError(MercantileError):
    """Raised when a zoom level is invalid."""


class ParentTileError(MercantileError):
    """Raised when a parent tile cannot be determined."""


class QuadKeyError(MercantileError):
    """Raised when errors occur in computing or parsing quad keys."""


class TileArgParsingError(MercantileError):
    """Raised when errors occur in parsing a function's tile arg(s)."""


class TileError(MercantileError):
    """Raised when a tile can't be determined."""


@dataclass(frozen=True)
class Tile:
    """An XYZ web mercator tile."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        """Finish initializing a Tile instance."""

    def __iter__(self) -> Iterator[int]:
        """Make Tile iterable like a namedtuple."""
        return iter((self.x, self.y, self.z))

    def __getitem__(self, index: int) -> int:
        """Make Tile indexable like a namedtuple."""
        return (self.x, self.y, self.z)[index]

    def __len__(self) -> int:
        """Return length like a namedtuple."""
        return 3


@dataclass(frozen=True)
class LngLat:
    """A longitude and latitude pair in decimal degrees."""

    lng: float
    lat: float

    def __post_init__(self) -> None:
        """Finish initializing a LngLat instance."""

    def __iter__(self) -> Iterator[float]:
        """Make LngLat iterable like a namedtuple."""
        return iter((self.lng, self.lat))

    def __getitem__(self, index: int) -> float:
        """Make LngLat indexable like a namedtuple."""
        return (self.lng, self.lat)[index]

    def __len__(self) -> int:
        """Return length like a namedtuple."""
        return 2


@dataclass(frozen=True)
class LngLatBbox:
    """A geographic bounding box."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        """Finish initializing a LngLatBbox instance.

        Raises:
            TileError: If the bounding box is invalid (e.g., west >= east or south >= north).
        """
        if self.west >= self.east or self.south >= self.north:
            raise TileError(f"Invalid bounding box: ({self.west}, {self.south}, {self.east}, {self.north})")

    def __iter__(self) -> Iterator[float]:
        """Make LngLatBbox iterable like a namedtuple."""
        return iter((self.west, self.south, self.east, self.north))

    def __getitem__(self, index: int) -> float:
        """Make LngLatBbox indexable like a namedtuple."""
        return (self.west, self.south, self.east, self.north)[index]

    def __len__(self) -> int:
        """Return length like a namedtuple."""
        return 4


@dataclass(frozen=True)
class Bbox:
    """A web mercator bounding box."""

    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        """Finish initializing a Bbox instance."""

    def __iter__(self) -> Iterator[float]:
        """Make Bbox iterable like a namedtuple."""
        return iter((self.left, self.bottom, self.right, self.top))

    def __getitem__(self, index: int) -> float:
        """Make Bbox indexable like a namedtuple."""
        return (self.left, self.bottom, self.right, self.top)[index]

    def __len__(self) -> int:
        """Return length like a namedtuple."""
        return 4


TileOrXyz: TypeAlias = "Tile | tuple[int, int, int]"


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
    raise NotImplementedError("tile function not yet implemented")


def bounds(*tile: TileOrXyz) -> LngLatBbox:
    """Returns the bounding box of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        LngLatBbox: Geographic bounding box of the tile.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    raise NotImplementedError("bounds function not yet implemented")


def quadkey(*tile: TileOrXyz) -> str:
    """Get the quadkey of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        str: Quadkey string representation of the tile.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    raise NotImplementedError("quadkey function not yet implemented")


def minmax(zoom: int) -> tuple[int, int]:
    """Minimum and maximum tile coordinates for a zoom level.

    Args:
        zoom: Web mercator zoom level.

    Returns:
        tuple[int, int]: (minimum, maximum) tile coordinates where minimum
            is always 0 and maximum is (2 ** zoom - 1).

    Raises:
        InvalidZoomError: If zoom level is not a positive integer.
    """
    if zoom < 0:
        raise InvalidZoomError("Zoom level must be a non-negative integer")

    raise NotImplementedError("minmax function not yet implemented")


def ul(*tile: TileOrXyz) -> LngLat:
    """Returns the upper left longitude and latitude of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        LngLat: Upper left corner coordinates.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    raise NotImplementedError("ul function not yet implemented")


def truncate_lnglat(lng: float, lat: float) -> tuple[float, float]:
    """Truncate longitude and latitude to valid web mercator limits.

    Args:
        lng: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.

    Returns:
        Truncated (lng, lat) coordinates.
    """
    if lng > 180.0:
        lng = 180.0
    elif lng < -180.0:
        lng = -180.0
    if lat > 90.0:
        lat = 90.0
    elif lat < -90.0:
        lat = -90.0
    return lng, lat


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
    raise NotImplementedError("xy function not yet implemented")


def lnglat(x: float, y: float, truncate: bool = False) -> LngLat:
    """Convert web mercator x, y to longitude and latitude.

    Args:
        x: Web mercator x coordinate in meters.
        y: Web mercator y coordinate in meters.
        truncate: Whether to truncate outputs to web mercator limits.

    Returns:
        LngLat: Longitude and latitude coordinates.
    """
    raise NotImplementedError("lnglat function not yet implemented")


def neighbors(*tile: TileOrXyz) -> list[Tile]:
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
    raise NotImplementedError("neighbors function not yet implemented")


def xy_bounds(*tile: TileOrXyz) -> Bbox:
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
    raise NotImplementedError("xy_bounds function not yet implemented")


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
    raise NotImplementedError("quadkey_to_tile function not yet implemented")


def tiles(
    west: float,
    south: float,
    east: float,
    north: float,
    zooms: int | Sequence[int],
    truncate: bool = False,
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
    raise NotImplementedError("tiles function not yet implemented")


def parent(
    *tile: TileOrXyz,
    zoom: int | None = None,
) -> Tile | None:
    """Get the parent of a tile.

    The parent is the tile of one zoom level lower that contains the given "child" tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        zoom: Target zoom level of the returned parent tile. Defaults to one lower than the input tile.

    Returns:
        Parent tile, or None if input tile is at zoom level 0.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
        InvalidZoomError: If zoom is not an integer less than input tile zoom.
        ParentTileError: If parent of non-integer tile is requested.
    """
    raise NotImplementedError("parent function not yet implemented")


def children(
    *tile: TileOrXyz,
    zoom: int | None = None,
) -> list[Tile]:
    """Get the children of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        zoom: Target zoom level for returned children. If unspecified, returns immediate (zoom + 1) children.

    Returns:
        Child tiles ordered top-left, top-right, bottom-right,
            bottom-left. For deeper zoom levels, returns all children in
            depth-first clockwise winding order.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
        InvalidZoomError: If zoom is not an integer greater than input tile zoom.
    """
    raise NotImplementedError("children function not yet implemented")


def simplify(*tiles: Sequence[TileOrXyz]) -> list[Tile]:
    """Reduces the size of the tileset as much as possible by merging leaves into parents.

    Args:
        tiles: Sequences of tiles to merge.

    Returns:
        Simplified tileset with merged tiles.

    Note:
        Removes child tiles when their parent is already present and merges
        complete sets of 4 children into their parent tile.
    """
    raise NotImplementedError("simplify function not yet implemented")


def bounding_tile(
    *bbox: LngLatBbox | LngLat | tuple[float, float, float, float] | tuple[float, float],
    **kwds: Any,
) -> Tile:
    """Get the smallest tile containing a geographic bounding box.

    Args:
        *bbox: Bounding box as west, south, east, north values in decimal degrees.
            Can also accept 2 values which will be duplicated.
        **kwds: Keyword arguments including:
            truncate: Whether to truncate inputs to web mercator limits.

    Returns:
        Smallest tile containing the bounding box.

    Note:
        When the bbox spans lines of lng 0 or lat 0, the bounding tile
        will be Tile(x=0, y=0, z=0).

    Raises:
        InvalidLatitudeError: If latitude values are invalid and truncate=False.
    """
    raise NotImplementedError("bounding_tile function not yet implemented")


def feature(
    *tile: TileOrXyz,
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
        GeoJSON Feature dict with tile geometry and properties.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    raise NotImplementedError("feature function not yet implemented")
