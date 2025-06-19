"""Mercantile-compatible API for tilemath.

All functions and classes in this module are designed to be 100% API compatible with the original mercantile library.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, TypeAlias

TileXyz: TypeAlias = "tuple[int, int, int]"

#: Matches mercantile's ``*tile`` argument type, which is a strange polymorphic type that can either be a Tile
#: instance, a coordinate triple, or a splatted coordinate triple.
TileArg: TypeAlias = "Tile | TileXyz | int"

#: # Conversion factor from radians to degrees.
R2D: Final[float] = 180.0 / 3.14159265358979323846
#: Radius of the Earth in meters (WGS84).
RE: Final[float] = 6378137.0
#: Circumference of the Earth in meters (WGS84).
CE: Final[float] = 2 * 3.14159265358979323846 * RE
#: # Small value for precision handling.
EPSILON: Final[float] = 1e-12
#: # Small value for precision handling in longitude/latitude.
LL_EPSILON: Final[float] = 1e-6

# Web mercator latitude limits (approximately ±85.05°)
MAX_LAT = 85.0511287798066
MIN_LAT = -85.0511287798066


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


# Dataclases which are namedtuple-like, immutable, and hashable, similar to mercantile's Tile/LngLat/LngLatBbox/Bbox.


@dataclass(frozen=True)
class Tile:  # noqa: UP006
    """An XYZ web mercator tile."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        """Finish initializing a Tile instance."""
        lo, hi = minmax(self.z)
        if not lo <= self.x <= hi or not lo <= self.y <= hi:
            warnings.warn("Tile x and y should be within the range (0, 2 ** zoom)", FutureWarning, stacklevel=2)

    def __iter__(self) -> Iterator[int]:
        """Make Tile iterable like a namedtuple."""
        return iter((self.x, self.y, self.z))

    def __getitem__(self, index: int) -> int:
        """Make Tile indexable like a namedtuple."""
        return (self.x, self.y, self.z)[index]

    def __len__(self) -> int:
        """Return length like a namedtuple."""
        return 3

    def __eq__(self, value: Any) -> bool:
        """Check equality with another Tile or similar object."""
        if isinstance(value, Tile):
            return self.x == value.x and self.y == value.y and self.z == value.z

        # Handle tuples, namedtuples, and other sequences
        try:
            if len(value) == 3:
                return (self.x, self.y, self.z) == tuple(value)
        except (TypeError, AttributeError):
            pass

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare Tile instances for sorting."""
        if not isinstance(other, Tile):
            return NotImplemented

        return (self.x, self.y, self.z) < (other.x, other.y, other.z)


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

    def __eq__(self, value: Any) -> bool:
        """Check equality with another LngLat or similar object."""
        if not isinstance(value, LngLat):
            return NotImplemented

        return self.lng == value.lng and self.lat == value.lat

    def __lt__(self, other: object) -> bool:
        """Compare LngLat instances for sorting."""
        if not isinstance(other, LngLat):
            return NotImplemented

        return (self.lng, self.lat) < (other.lng, other.lat)


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

    def __eq__(self, value: Any) -> bool:
        """Check equality with another LngLatBbox, tuple, or similar object."""
        if isinstance(value, LngLatBbox):
            return (
                self.west == value.west
                and self.south == value.south
                and self.east == value.east
                and self.north == value.north
            )

        # Handle tuples, namedtuples, and other sequences
        try:
            if len(value) == 4:
                return (self.west, self.south, self.east, self.north) == tuple(value)
        except (TypeError, AttributeError):
            pass

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare LngLatBbox instances for sorting."""
        if not isinstance(other, LngLatBbox):
            return NotImplemented

        return (self.west, self.south, self.east, self.north) < (other.west, other.south, other.east, other.north)


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

    def __eq__(self, value: Any) -> bool:
        """Check equality with another Bbox or similar object."""
        if isinstance(value, Bbox):
            return (
                self.left == value.left
                and self.bottom == value.bottom
                and self.right == value.right
                and self.top == value.top
            )

        # Handle tuples, namedtuples, and other sequences
        try:
            if len(value) == 4:
                return (self.left, self.bottom, self.right, self.top) == tuple(value)
        except (TypeError, AttributeError):
            pass

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        """Compare Bbox instances for sorting."""
        if not isinstance(other, Bbox):
            return NotImplemented

        return (self.left, self.bottom, self.right, self.top) < (other.left, other.bottom, other.right, other.top)


TileOrXyz: TypeAlias = "Tile | TileXyz"


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
    x, y = _xy(lng, lat, truncate=truncate)
    z2 = math.pow(2, zoom)

    if x <= 0:
        xtile = 0
    elif x >= 1:
        xtile = int(z2 - 1)
    else:
        # To address loss of precision in round-tripping between tile
        # and lng/lat, points within EPSILON of the right side of a tile
        # are counted in the next tile over.
        xtile = int(math.floor((x + EPSILON) * z2))

    if y <= 0:
        ytile = 0
    elif y >= 1:
        ytile = int(z2 - 1)
    else:
        ytile = int(math.floor((y + EPSILON) * z2))

    return Tile(x=xtile, y=ytile, z=zoom)


def bounds(*tile: TileArg) -> LngLatBbox:
    """Returns the bounding box of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        LngLatBbox: Geographic bounding box of the tile.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    as_tile = _parse_tile_arg(*tile)

    # Get the tile bounds in normalized coordinates [0, 1]
    z2 = 2.0**as_tile.z
    x_min = as_tile.x / z2
    x_max = (as_tile.x + 1) / z2
    y_min = as_tile.y / z2
    y_max = (as_tile.y + 1) / z2

    # Convert normalized x coordinates to longitude
    west = (x_min - 0.5) * 360.0
    east = (x_max - 0.5) * 360.0

    # Convert normalized y coordinates to latitude using inverse mercator projection
    # Note: In tile coordinates, y=0 is north, y=1 is south
    # So y_min corresponds to north, y_max corresponds to south
    north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_min))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_max))))

    return LngLatBbox(west=west, south=south, east=east, north=north)


def quadkey(*tile: TileArg) -> str:
    """Get the quadkey of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        str: Quadkey string representation of the tile.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    as_tile = _parse_tile_arg(*tile)

    x, y, z = as_tile.x, as_tile.y, as_tile.z

    if z == 0:
        return ""

    # Process bits from most significant to least significant
    result: list[str] = []
    for i in range(z - 1, -1, -1):
        digit = ((x >> i) & 1) | (((y >> i) & 1) << 1)
        result.append(str(digit))

    return "".join(result)


def geojson_bounds(obj: dict[str, Any] | list[Any] | tuple[Any, ...]) -> LngLatBbox:
    """Returns the bounding box of a GeoJSON object.

    Args:
        obj: A GeoJSON geometry, feature, feature collection, or coordinate array.

    Returns:
        LngLatBbox: Geographic bounding box of the GeoJSON object.
    """
    if not obj:
        raise ValueError("Cannot calculate bounds of empty object")

    min_lng = float("inf")
    max_lng = float("-inf")
    min_lat = float("inf")
    max_lat = float("-inf")

    has_coords = False

    for lng, lat in _coords(obj):
        has_coords = True
        min_lng = min(min_lng, lng)
        max_lng = max(max_lng, lng)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)

    if not has_coords:
        raise ValueError("No coordinates found in object")

    return LngLatBbox(
        west=min_lng,
        south=min_lat,
        east=max_lng,
        north=max_lat,
    )


@lru_cache(maxsize=32)
def minmax(zoom: int) -> tuple[int, int]:
    """Minimum and maximum tile coordinates for a zoom level.

    Args:
        zoom: Web mercator zoom level.

    Returns:
        (minimum, maximum) tile coordinates where minimum is always 0 and maximum is (2 ** zoom - 1).

    Raises:
        InvalidZoomError: If zoom level is not a positive integer.
    """
    if zoom < 0:
        msg = f"zoom must be a positive integer ({zoom=!r} is invalid)"
        raise InvalidZoomError(msg)

    # For web mercator tiles, minimum is always 0
    # Maximum is 2^zoom - 1 (since we have 2^zoom tiles in each dimension)
    return (0, 2**zoom - 1)


def ul(*tile: TileArg) -> LngLat:
    """Returns the upper left longitude and latitude of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        LngLat: Upper left corner coordinates.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    as_tile = _parse_tile_arg(*tile)

    # Get the tile bounds in normalized coordinates [0, 1]
    z2 = 2.0**as_tile.z
    x_norm = as_tile.x / z2
    y_norm = as_tile.y / z2

    # Convert normalized x to longitude
    lng = (x_norm - 0.5) * 360.0

    # Convert normalized y to latitude using inverse mercator projection
    # From the _xy function: y = 0.5 - 0.25 * ln((1 + sin(lat)) / (1 - sin(lat))) / π
    # Rearranging: ln((1 + sin(lat)) / (1 - sin(lat))) = 4π(0.5 - y)
    # Therefore: (1 + sin(lat)) / (1 - sin(lat)) = e^(4π(0.5 - y))
    # Solving: sin(lat) = (e^(4π(0.5 - y)) - 1) / (e^(4π(0.5 - y)) + 1)
    # Which simplifies to: sin(lat) = tanh(2π(0.5 - y))

    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_norm))))

    return LngLat(lng=lng, lat=lat)


def truncate_lnglat(lng: float, lat: float) -> tuple[float, float]:
    """Truncate longitude and latitude to valid web mercator limits.

    Args:
        lng: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.

    Returns:
        Truncated (lng, lat) coordinates.
    """
    # Clamp lng to [-180, 180]
    # Note: This is a simplification; in practice, lng should be clamped to the range [-180, 180) for web mercator.
    # However, we use [-180, 180] to match mercantile's behavior and avoid issues with extreme longitudes.
    # This means that at the antimeridian, the x coordinate will be inf or -inf.
    if lng > 180.0:
        lng = 180.0
    elif lng < -180.0:
        lng = -180.0

    # Clamp lat to [-90, 90]
    # Note: This is a simplification; in practice, lat should be clamped to the range [-85.051128, 85.051128] for web mercator.
    # However, we use [-90, 90] to match mercantile's behavior and avoid issues with extreme latitudes.
    # This means that at the poles, the y coordinate will be inf or -inf.
    # This is acceptable for web mercator, as it uses a spherical model.
    # In practice, web mercator tiles are not defined at the poles.
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
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)
    else:
        # Check for invalid inputs and issue FutureWarnings
        # This is equivalent to mercantile's behavior where it raises warnings for out-of-bounds values.
        # They mentioned an intention to raise errors in a future version, but that future version doesn't exist yet.

        if lng < -180.0 or lng > 180.0:
            warnings.warn(
                f"Invalid longitude {lng} is outside valid range [-180, 180]. "
                "This will raise an error in a future version. "
                "Use truncate=True to automatically clamp values.",
                FutureWarning,
                stacklevel=2,
            )

        if lat < -90.0 or lat > 90.0:
            warnings.warn(
                f"Invalid latitude {lat} is outside valid range [-90, 90]. "
                "This will raise an error in a future version. "
                "Use truncate=True to automatically clamp values.",
                FutureWarning,
                stacklevel=2,
            )

    # Convert longitude to x (simple linear transformation)
    x = lng * RE * math.pi / 180.0

    # Convert latitude to y using mercator projection
    # Handle edge cases for poles
    if lat >= 90.0:
        y = float("inf")
    elif lat <= -90.0:
        y = float("-inf")
    else:
        # Standard mercator formula: y = R * ln(tan(π/4 + φ/2))
        # where φ is latitude in radians
        lat_rad = lat * math.pi / 180.0
        y = RE * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))

    return x, y


def lnglat(x: float, y: float, truncate: bool = False) -> LngLat:
    """Convert web mercator x, y to longitude and latitude.

    Args:
        x: Web mercator x coordinate in meters.
        y: Web mercator y coordinate in meters.
        truncate: Whether to truncate outputs to web mercator limits.

    Returns:
        LngLat: Longitude and latitude coordinates.
    """
    # Convert x back to longitude (inverse of x = lng * RE * π / 180)
    lng = x * 180.0 / (RE * math.pi)

    # Convert y back to latitude using inverse mercator projection
    # From xy function: y = RE * ln(tan(π/4 + lat_rad/2))
    # Inverse: lat_rad = 2 * (atan(exp(y/RE)) - π/4)
    # Simplified: lat_rad = 2 * atan(exp(y/RE)) - π/2
    if math.isinf(y):
        # Handle poles
        if y == float("inf"):
            lat = 90.0
        else:  # y == float("-inf")
            lat = -90.0
    else:
        try:
            lat = math.degrees(2.0 * math.atan(math.exp(y / RE)) - math.pi / 2.0)
        except (OverflowError, ValueError):
            # Handle extreme values
            if y > 0:
                lat = 90.0
            else:
                lat = -90.0

    if truncate:
        lng, lat = truncate_lnglat(lng, lat)

    return LngLat(lng=lng, lat=lat)


def neighbors(
    *tile: TileArg,
    include_center: bool = False,
) -> list[Tile]:
    """Get the neighbors of a tile.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).
        include_center: Whether to include the center tile itself in the result.

    Returns:
        Up to eight neighboring tiles (nine if include_center=True). Invalid tiles (e.g., Tile(-1, -1, z)) are omitted
        from the result.

    Note:
        Makes no guarantees regarding neighbor tile ordering.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    as_tile = _parse_tile_arg(*tile)

    x, y, z = as_tile.x, as_tile.y, as_tile.z

    # Get the valid tile coordinate range for this zoom level
    tile_min, tile_max = minmax(z)

    result: list[Tile] = []

    # Check all 8 neighboring positions
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            # Skip the center tile (the input tile itself)
            if (dx == 0 and dy == 0) and not include_center:
                continue

            neighbor_x = x + dx
            neighbor_y = y + dy

            # Check if the neighbor coordinates are valid
            if tile_min <= neighbor_x <= tile_max and tile_min <= neighbor_y <= tile_max:
                result.append(Tile(neighbor_x, neighbor_y, z))

    return result


def xy_bounds(*tile: TileArg) -> Bbox:
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
    as_tile = _parse_tile_arg(*tile)

    # Get the tile bounds in normalized coordinates [0, 1]
    z2 = 2.0**as_tile.z
    x_min = as_tile.x / z2
    x_max = (as_tile.x + 1) / z2
    y_min = as_tile.y / z2
    y_max = (as_tile.y + 1) / z2

    # Convert normalized coordinates to web mercator meters
    # x conversion: x_meters = (x_norm - 0.5) * circumference
    left = (x_min - 0.5) * CE
    right = (x_max - 0.5) * CE

    # y conversion using inverse mercator projection
    # From _xy: y_norm = 0.5 - 0.25 * ln((1 + sin(lat)) / (1 - sin(lat))) / π
    # Inverse: ln((1 + sin(lat)) / (1 - sin(lat))) = 4π(0.5 - y_norm)
    # Therefore: sin(lat) = tanh(2π(0.5 - y_norm))
    # Then: lat = asin(tanh(2π(0.5 - y_norm)))
    # Finally: y_meters = RE * ln(tan(π/4 + lat/2))

    # For y_min (top of tile in normalized coords, northernmost latitude)
    lat_north = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_min))))
    if lat_north >= 90.0:
        top = float("inf")
    elif lat_north <= -90.0:
        top = float("-inf")
    else:
        lat_north_rad = math.radians(lat_north)
        top = RE * math.log(math.tan(math.pi / 4.0 + lat_north_rad / 2.0))

    # For y_max (bottom of tile in normalized coords, southernmost latitude)
    lat_south = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_max))))
    if lat_south >= 90.0:
        bottom = float("inf")
    elif lat_south <= -90.0:
        bottom = float("-inf")
    else:
        lat_south_rad = math.radians(lat_south)
        bottom = RE * math.log(math.tan(math.pi / 4.0 + lat_south_rad / 2.0))

    # Apply epsilon adjustments as mentioned in the docstring
    # Subtract epsilon from right limit and add to bottom limit for precision
    if not math.isinf(right):
        right -= EPSILON
    if not math.isinf(bottom):
        bottom += EPSILON

    return Bbox(left=left, bottom=bottom, right=right, top=top)


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
    if not qk:
        return Tile(0, 0, 0)

    x = 0
    y = 0
    z = len(qk)

    for i, digit in enumerate(qk):
        if digit not in "0123":
            raise QuadKeyError(f"Invalid quadkey digit: {digit}")

        # Convert digit to integer
        d = int(digit)

        # Extract x and y bits from the digit
        # Digit encoding: bit 0 = x bit, bit 1 = y bit
        x_bit = d & 1
        y_bit = (d >> 1) & 1

        # Set the bit at the appropriate position
        # Most significant bit first (i=0 corresponds to z-1 bit position)
        bit_pos = z - 1 - i
        x |= x_bit << bit_pos
        y |= y_bit << bit_pos

    return Tile(x, y, z)


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
        Tiles that overlap the bounding box.

    Note:
        A small epsilon is used on the south and east parameters so that this function yields exactly one tile when
        given the bounds of that same tile.

        Handles antimeridian crossing by splitting into two bounding boxes.
    """
    # Normalize zooms to a sequence
    if isinstance(zooms, int):
        zoom_levels = [zooms]
    else:
        zoom_levels = list(zooms)

    # Validate zoom levels
    for zoom in zoom_levels:
        if zoom < 0:
            raise InvalidZoomError(f"zoom must be a positive integer ({zoom=!r} is invalid)")

    # Handle antimeridian crossing (west > east)
    if west > east:
        # Split into two bounding boxes
        # First: from west to 180
        yield from tiles(west, south, 180.0, north, zoom_levels, truncate)
        # Second: from -180 to east
        yield from tiles(-180.0, south, east, north, zoom_levels, truncate)
        return

    # Clamp latitudes to web mercator limits to avoid InvalidLatitudeError
    # This matches mercantile's behavior for global tiles
    north_clamped = min(north, MAX_LAT)
    south_clamped = max(south, MIN_LAT)

    # Apply epsilon adjustments for precision as mentioned in docstring
    # This ensures that when given exact tile bounds, we get exactly one tile
    south_adj = south_clamped + LL_EPSILON
    east_adj = east - LL_EPSILON

    # Handle edge case where north is at or beyond MAX_LAT
    # Don't apply epsilon to north when it's already at the limit
    if north >= MAX_LAT:
        north_adj = MAX_LAT
    else:
        north_adj = north_clamped

    for zoom in zoom_levels:
        # Get the tile coordinates for the corners of the bounding box
        try:
            # Southwest corner
            sw_tile = tile(west, south_adj, zoom, truncate=truncate)
            # Northeast corner
            ne_tile = tile(east_adj, north_adj, zoom, truncate=truncate)
        except InvalidLatitudeError:
            if truncate:
                # This shouldn't happen with truncate=True, but handle it anyway
                continue
            else:
                # Try again with more aggressive clamping
                try:
                    sw_tile = tile(west, max(south_adj, MIN_LAT + LL_EPSILON), zoom, truncate=True)
                    ne_tile = tile(east_adj, min(north_adj, MAX_LAT - LL_EPSILON), zoom, truncate=True)
                except InvalidLatitudeError:
                    # Skip this zoom level if we still can't compute tiles
                    continue

        # Get tile coordinate bounds
        min_x = sw_tile.x
        max_x = ne_tile.x
        min_y = ne_tile.y  # Note: y increases southward in tile coordinates
        max_y = sw_tile.y

        # Ensure we stay within valid tile bounds for this zoom level
        tile_min, tile_max = minmax(zoom)
        min_x = max(min_x, tile_min)
        max_x = min(max_x, tile_max)
        min_y = max(min_y, tile_min)
        max_y = min(max_y, tile_max)

        # Generate all tiles in the range
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                yield Tile(x=x, y=y, z=zoom)


def parent(
    *tile: TileArg,
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
    as_tile = _parse_tile_arg(*tile)

    # If zoom level is 0, there's no parent
    if as_tile.z == 0:
        return None

    # Determine target zoom level and validate
    if zoom is None:
        target_zoom = as_tile.z - 1
    else:
        if not isinstance(zoom, int):
            raise InvalidZoomError(f"zoom must be an integer and less than {as_tile.z}")
        if zoom >= as_tile.z:
            raise InvalidZoomError(f"zoom must be an integer and less than {as_tile.z}")
        target_zoom = zoom

    # Check for non-integer tile coordinates after zoom validation
    if not (isinstance(as_tile.x, int) and isinstance(as_tile.y, int) and isinstance(as_tile.z, int)):
        raise ParentTileError("the parent of a non-integer tile is undefined")

    # Calculate zoom difference
    zoom_diff = as_tile.z - target_zoom

    # Parent coordinates are obtained by right-shifting by zoom_diff
    parent_x = as_tile.x >> zoom_diff
    parent_y = as_tile.y >> zoom_diff

    return Tile(parent_x, parent_y, target_zoom)


def children(
    *tile: TileArg,
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
    as_tile = _parse_tile_arg(*tile)

    # Determine target zoom level
    if zoom is None:
        target_zoom = as_tile.z + 1
    else:
        if zoom <= as_tile.z:
            raise InvalidZoomError(f"zoom must be an integer and greater than {as_tile.z}")

        target_zoom = zoom

    # Calculate zoom difference
    zoom_diff = target_zoom - as_tile.z

    # For immediate children (zoom_diff = 1), we have 4 children
    # For deeper levels, we need to generate all children recursively

    if zoom_diff == 1:
        # Four immediate children in clockwise order: top-left, top-right, bottom-right, bottom-left
        x = as_tile.x
        y = as_tile.y

        return [
            Tile(2 * x, 2 * y, target_zoom),  # top-left (a)
            Tile(2 * x + 1, 2 * y, target_zoom),  # top-right (b)
            Tile(2 * x + 1, 2 * y + 1, target_zoom),  # bottom-right (c)
            Tile(2 * x, 2 * y + 1, target_zoom),  # bottom-left (d)
        ]
    else:
        # For deeper zoom levels, recursively get children
        # Start with immediate children and then get their children
        immediate_children = children(as_tile, zoom=as_tile.z + 1)
        result = []

        for child in immediate_children:
            if target_zoom == child.z:
                result.append(child)
            else:
                result.extend(children(child, zoom=target_zoom))

        return result


def simplify(*tiles: Sequence[Tile] | Tile | Sequence[TileXyz] | TileXyz) -> list[Tile]:
    """Reduces the size of the tileset as much as possible by merging leaves into parents.

    Args:
        tiles: Sequences of tiles to merge.

    Returns:
        Simplified tileset with merged tiles.

    Note:
        Removes child tiles when their parent is already present and merges
        complete sets of 4 children into their parent tile.
    """
    if not tiles:
        return []

    as_tiles = [Tile(*t) if isinstance(t, tuple) else t for t in tiles]
    raise NotImplementedError("simplify function not yet implemented")


"""
# Reference implementation

def bounding_tile(*bbox, **kwds):
    \"""Get the smallest tile containing a geographic bounding box

    NB: when the bbox spans lines of lng 0 or lat 0, the bounding tile
    will be Tile(x=0, y=0, z=0).

    Parameters
    ----------
    bbox : sequence of float
        west, south, east, north bounding values in decimal degrees.

    Returns
    -------
    Tile

    \"""
    if len(bbox) == 2:
        bbox += bbox

    w, s, e, n = bbox

    truncate = bool(kwds.get("truncate"))

    if truncate:
        w, s = truncate_lnglat(w, s)
        e, n = truncate_lnglat(e, n)

    e = e - LL_EPSILON
    s = s + LL_EPSILON

    try:
        tmin = tile(w, n, 32)
        tmax = tile(e, s, 32)
    except InvalidLatitudeError:
        return Tile(0, 0, 0)

    cell = tmin[:2] + tmax[:2]
    z = _getBboxZoom(*cell)

    if z == 0:
        return Tile(0, 0, 0)

    x = rshift(cell[0], (32 - z))
    y = rshift(cell[1], (32 - z))

    return Tile(x, y, z)
"""


def bounding_tile(
    *bbox: LngLatBbox | LngLat | float,
    truncate: bool = False,
) -> Tile:
    """Get the smallest tile containing a geographic bounding box.

    Uses a direct mathematical approach with bit manipulation for optimal performance.
    When the bbox spans lines of lng 0 or lat 0, the bounding tile will be Tile(x=0, y=0, z=0).

    Args:
        *bbox: Bounding box as west, south, east, north values in decimal degrees.
            Can also accept 2 values which will be duplicated.
        truncate: Whether to truncate inputs to web mercator limits.

    Returns:
        Smallest tile containing the bounding box.

    Raises:
        InvalidLatitudeError: If latitude values are invalid and truncate=False.
        TileError: If the bounding box is invalid.
    """
    west: float
    south: float
    east: float
    north: float

    try:
        # Unpack *bbox into a single argument to simplify parsing
        if len(bbox) == 1:
            bbox_arg = bbox[0]
        else:
            bbox_arg = bbox  # The splatted args become a tuple

        # Parse the flexible bbox argument structure
        if isinstance(bbox_arg, LngLatBbox | LngLat):
            if len(bbox_arg) == 4:  # LngLatBbox
                west, south, east, north = bbox_arg
            else:  # LngLat - duplicate coordinates for point
                west, south = bbox_arg
                east, north = west, south
        # Check for other sequence-like objects (tuple, list)
        elif isinstance(bbox_arg, Sequence):
            if len(bbox_arg) == 4:
                west, south, east, north = bbox_arg  # type: ignore
            elif len(bbox_arg) == 2:
                west, south = bbox_arg  # type: ignore
                east, north = west, south
            else:
                raise ValueError("Bbox sequence must have 2 or 4 elements")
        else:
            raise ValueError("Invalid bbox argument type")

    except (TypeError, ValueError, IndexError) as e:
        raise TileError(f"Invalid bounding box arguments: {bbox!r}") from e

    if south > north:
        raise TileError(f"Invalid bounding box: south ({south}) > north ({north})")

    # Handle coordinate truncation if requested
    if truncate:
        west, south = truncate_lnglat(west, south)
        east, north = truncate_lnglat(east, north)

    # Apply epsilon adjustments to handle edge cases in tile calculations
    # This prevents issues with coordinates that fall exactly on tile boundaries
    east = east - LL_EPSILON
    south = south + LL_EPSILON

    try:
        # Calculate tiles at high zoom level (32) for precise coordinate mapping
        # Use northwest and southeast corners for proper tile boundary detection
        tmin = tile(west, north, 32, truncate=truncate)
        tmax = tile(east, south, 32, truncate=truncate)
    except InvalidLatitudeError:
        if truncate:
            # If truncation is enabled but we still get an error, return world tile
            return Tile(0, 0, 0)
        # Propagate the error if truncate is False
        raise

    # Combine tile coordinates for bbox calculation
    # cell contains [min_x, min_y, max_x, max_y] at zoom 32
    cell = tmin[:2] + tmax[:2]

    # Calculate the appropriate zoom level using bit manipulation
    # This finds the zoom where the bbox fits within a single tile
    zoom = _get_bbox_zoom(*cell)

    if zoom == 0:
        return Tile(0, 0, 0)

    # Right-shift coordinates from zoom 32 down to the calculated zoom level
    # This effectively "zooms out" the high-precision coordinates
    x = _rshift(cell[0], (32 - zoom))
    y = _rshift(cell[1], (32 - zoom))

    return Tile(x, y, zoom)


def feature(
    *tile: TileArg,
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


def _xy(lng: float, lat: float, truncate: bool = False) -> tuple[float, float]:
    """Convert longitude and latitude to normalized tile coordinates.

    Converts geographic coordinates to normalized coordinates in the range [0, 1] using spherical mercator projection.

    Args:
        lng: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.
        truncate: Whether to truncate inputs to web mercator limits.

    Returns:
        Normalized coordinates (x, y) in range [0, 1].

    Raises:
        InvalidLatitudeError: If latitude cannot be projected (e.g., at poles).
    """
    if truncate:
        lng, lat = truncate_lnglat(lng, lat)

    x = lng / 360.0 + 0.5
    sinlat = math.sin(math.radians(lat))

    try:
        y = 0.5 - 0.25 * math.log((1.0 + sinlat) / (1.0 - sinlat)) / math.pi
    except (ValueError, ZeroDivisionError) as e:
        msg = f"Y can not be computed: lat={lat!r}"
        raise InvalidLatitudeError(msg) from e
    else:
        return x, y


def _coords(obj: dict[str, Any] | list[Any] | tuple[Any, ...]) -> Generator[tuple[float, float], None, None]:
    """Iterate over all coordinates in a GeoJSON-like object or coordinate tuple.

    This function handles GeoJSON geometries, features, feature collections, and coordinate arrays, yielding each
    coordinate as a tuple of (longitude, latitude).

    Args:
        obj: A GeoJSON geometry, feature, feature collection, or coordinate array.

    Yields:
        Each coordinate as a tuple of (longitude, latitude).
    """
    # Extract coordinates based on object type
    if isinstance(obj, tuple | list):
        coordinates = obj
    else:
        if "features" in obj:
            # FeatureCollection
            for feat in obj["features"]:
                yield from _coords(feat["geometry"]["coordinates"])
            return
        elif "geometry" in obj:
            # Feature
            coordinates = obj["geometry"]["coordinates"]
        elif "coordinates" in obj:
            # Geometry
            coordinates = obj["coordinates"]
        else:
            return

    # Process coordinates recursively
    if not coordinates:
        return

    # Check if this is a coordinate pair (base case)
    if len(coordinates) >= 2 and all(isinstance(x, int | float) for x in coordinates[:2]):
        yield (float(coordinates[0]), float(coordinates[1]))
        return

    # Recursively process nested coordinate structures
    for coord in coordinates:
        yield from _coords(coord)


def _parse_tile_arg(*tile: TileArg) -> Tile:
    """Parse tile arguments into a Tile instance.

    Args:
        *tile: Either a Tile instance or 3 ints (X, Y, Z).

    Returns:
        Tile: Parsed tile instance.

    Raises:
        TileArgParsingError: If tile arguments are invalid.
    """
    if len(tile) == 1:
        arg = tile[0]
        if isinstance(arg, Tile):
            return arg

        if isinstance(arg, Sequence) and len(arg) == 3:
            return Tile(*arg)

        msg = f"Invalid tile sequence argument: {arg!r}"
        raise TileArgParsingError(msg)

    if len(tile) != 3 or not all(isinstance(t, int) for t in tile):
        msg = f"Invalid tile arguments: {tile!r}"
        raise TileArgParsingError(msg)

    return Tile(*tile)  # type: ignore


def _get_bbox_zoom(
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> int:
    """Calculate the appropriate zoom level for a bounding box using bit manipulation.

    Determines the highest zoom level where the bounding box fits within a single tile
    by examining bit patterns of the tile coordinates at zoom 32.

    Args:
        min_x: Minimum x coordinate at zoom 32.
        min_y: Minimum y coordinate at zoom 32.
        max_x: Maximum x coordinate at zoom 32.
        max_y: Maximum y coordinate at zoom 32.

    Returns:
        Zoom level where the bounding box fits in a single tile.

    Raises:
        ValueError: If coordinates are invalid or out of expected range.
    """
    # Maximum zoom level supported by the tile system
    MAX_ZOOM = 28

    # Iterate through zoom levels from 0 to MAX_ZOOM
    for zoom in range(0, MAX_ZOOM):
        # Create a bit mask for the current zoom level
        # At zoom z, we need to check if coordinates differ in the (32-z-1)th bit
        # This determines if they fall in different tiles at zoom level z
        mask = 1 << (32 - (zoom + 1))

        # Check if the bounding box spans multiple tiles at this zoom level
        # If any coordinate pair differs in the masked bit, they're in different tiles
        x_spans_tiles = (min_x & mask) != (max_x & mask)
        y_spans_tiles = (min_y & mask) != (max_y & mask)

        if x_spans_tiles or y_spans_tiles:
            # Return the current zoom level as the bbox spans multiple tiles
            return zoom

    # If we reach here, the bbox fits in a single tile even at maximum zoom
    return MAX_ZOOM


def _rshift(val: int, n: int) -> int:
    """Right shift a value by n bits, handling 32-bit overflow."""
    return (val % 0x100000000) >> n
