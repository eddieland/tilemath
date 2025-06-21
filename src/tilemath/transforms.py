"""Transformation functions for converting between tile and Web Mercator coordinates.

Provides functionality to create and apply transformation matrices for converting coordinates between tile space and
Web Mercator projection.

Core Concepts:
    Affine Transform:
        A 3x3 matrix that preserves straight lines and parallelism while allowing for:
        - Translation (moving coordinates)
        - Scaling (changing coordinate units)
        - Rotation
        - Shear

        The matrix structure used here is:
        | scale_x   0     translate_x |
        |   0    -scale_y translate_y |
        |   0       0         1       |

    Tile Coordinates:
        A coordinate system where:
        - (0,0) is the top-left of the tile
        - Values range from 0 to 1 across each tile
        - Y increases downward

    Web Mercator:
        A map projection coordinate system where:
        - Coordinates are in meters
        - Y increases upward
        - (0,0) is at the equator/prime meridian

This module provides two primary paradigms for representing transformations:
    1. Using a `Transform` class that wraps an affine matrix and provides methods for transforming points, bounds, and
       batches of points.
    2. Using standalone functions that accept an affine matrix and perform transformations directly.

    The `Transform` class can be used to create transformations from tile ranges, apply them to points, and compose
       multiple transformations together.

    The standalone functions can be used for more functional-style transformations without needing to instantiate a class.

Example:
    >>> tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
    >>> transform = make_transform(tilerange, zoom=1)
    >>> x, y = transform_point(transform, 0.5, 0.5)
    >>> my_transform = Transform.from_tilerange(tilerange, zoom=1)
    >>> bounds = my_transform.transform_bounds(0, 0, 1, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from tilemath import mercantile


@dataclass(frozen=True)
class Transform:
    """A wrapper for an affine transformation matrix which can be used to convert tile coordinates to Web Mercator.

    Example:
        >>> tilerange = TileRange.single_tile(x=0, y=0)
        >>> transform = Transform.from_tilerange(tilerange, zoom=1)
        >>> x, y = transform.transform_point(0.5, 0.5)
    """

    matrix: NDArray[np.float64]

    @classmethod
    def from_tilerange(cls, tilerange: TileRange, zoom: int) -> Self:
        """Creates a transformation from tile coordinates to Web Mercator."""
        return cls(matrix=make_transform(tilerange, zoom))

    @classmethod
    def identity(cls) -> Self:
        """Creates an identity transformation."""
        return cls(matrix=np.eye(3, dtype=np.float64))

    def inverse(self) -> Self:
        """Returns the inverse transformation."""
        # Once make_inverse_transform is implemented, could use that instead
        return self.__class__(matrix=np.linalg.inv(self.matrix))  # type: ignore[arg-type]

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        """Transform a single point."""
        return transform_point(self.matrix, x, y)

    def transform_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform multiple points efficiently."""
        return transform_points_batch(self.matrix, points)

    def transform_bounds(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> tuple[float, float, float, float]:
        """Transform a bounding box."""
        return transform_bounds(self.matrix, left=left, bottom=bottom, right=right, top=top)

    def __matmul__(self, other: Transform) -> Transform:
        """Compose two transformations using the @ operator."""
        return Transform(matrix=self.matrix @ other.matrix)

    @property
    def scale_x(self) -> float:
        """The x scaling factor."""
        return float(self.matrix[0, 0])

    @property
    def scale_y(self) -> float:
        """The y scaling factor."""
        return float(-self.matrix[1, 1])  # Note: negated due to y-axis flip

    @property
    def translation(self) -> tuple[float, float]:
        """The translation components (tx, ty)."""
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])


def make_transform(tilerange: TileRange, zoom: int) -> NDArray[np.float64]:
    """Creates a transformation matrix for converting tile coordinates to Web Mercator.

    Args:
        tilerange: Dictionary containing x and y coordinate ranges for tiles.
        zoom: The zoom level of the tiles.

    Returns:
        A 3x3 transformation matrix.

    Example:
        >>> tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        >>> transform = make_transform(tilerange, zoom=1)
        >>> transform.shape
        (3, 3)
    """
    # Get bounds in Web Mercator
    ulx, uly = mercantile.xy(*mercantile.ul(tilerange.x.min, tilerange.y.min, zoom))
    lrx, lry = mercantile.xy(*mercantile.ul(tilerange.x.max, tilerange.y.max, zoom))

    # Calculate pixel resolution
    xcell = (lrx - ulx) / float(tilerange.x.max - tilerange.x.min)
    ycell = (uly - lry) / float(tilerange.y.max - tilerange.y.min)

    return np.array(
        [
            [xcell, 0.0, ulx],
            [0.0, -ycell, uly],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_inverse_transform(tilerange: TileRange, zoom: int) -> NDArray[np.float64]:
    """Creates an inverse transformation matrix for converting Web Mercator coordinates to tile coordinates.

    Args:
        tilerange: Dictionary containing x and y coordinate ranges for tiles.
        zoom: The zoom level of the tiles.

    Returns:
        NDArray[np.float64]: A 3x3 inverse transformation matrix.

    Example:
        >>> tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        >>> inverse_transform = make_inverse_transform(tilerange, zoom=1)
        >>> inverse_transform.shape
        (3, 3)
    """
    forward_transform = make_transform(tilerange, zoom)
    return np.linalg.inv(forward_transform)  # type: ignore[return-value]


def transform_point(affine: NDArray[np.float64], x: float, y: float) -> tuple[float, float]:
    """Transforms a single point from tile coordinates to Web Mercator coordinates.

    Args:
        affine: 3x3 transformation matrix from make_transform().
        x: The x coordinate in tile space.
        y: The y coordinate in tile space.

    Returns:
        The transformed (x, y) coordinates in Web Mercator projection.

    Example:
        >>> tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        >>> transform = make_transform(tilerange, zoom=1)
        >>> x, y = transform_point(transform, 0.5, 0.5)
    """
    point = np.array([x, y, 1.0])
    transformed = affine @ point
    return float(transformed[0]), float(transformed[1])


def transform_points_batch(affine: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transforms multiple points from tile to Web Mercator coordinates efficiently.

    Args:
        affine: 3x3 transformation matrix from make_transform().
        points: Array of shape (n, 2) containing pairs of (x, y) coordinates.

    Returns:
        Array of shape (n, 2) containing transformed coordinates.

    Example:
        >>> tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        >>> transform = make_transform(tilerange, zoom=1)
        >>> points = np.array([[0.5, 0.5], [0.7, 0.3]])
        >>> result = transform_points_batch(transform, points)
        >>> result.shape
        (2, 2)
    """
    n_points = points.shape[0]
    homogeneous_points = np.column_stack([points, np.ones(n_points)])
    transformed = homogeneous_points @ affine.T
    return transformed[:, :2]


def transform_bounds(
    affine: NDArray[np.float64],
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> tuple[float, float, float, float]:
    """Transform a bounding box from tile coordinates to Web Mercator.

    Args:
        affine: 3x3 transformation matrix
        left: Left bound in tile coordinates
        bottom: Bottom bound in tile coordinates
        right: Right bound in tile coordinates
        top: Top bound in tile coordinates

    Returns:
        (left, bottom, right, top) in Web Mercator coordinates

    Example:
        >>> tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        >>> transform = make_transform(tilerange, zoom=1)
        >>> bounds = transform_bounds(transform, 0, 0, 1, 1)
        >>> bounds
        (ulx, uly, lrx, lry)
    """
    # Create an array of the bounding box corners in homogeneous coordinates
    bounds = np.array(
        [
            [left, bottom, 1.0],
            [right, bottom, 1.0],
            [right, top, 1.0],
            [left, top, 1.0],
        ]
    )

    # Apply the affine transformation
    transformed_bounds = bounds @ affine.T

    # Extract the transformed coordinates
    return (
        transformed_bounds[:, 0].min(),
        transformed_bounds[:, 1].min(),
        transformed_bounds[:, 0].max(),
        transformed_bounds[:, 1].max(),
    )


@dataclass(frozen=True)
class TileRangeAxis:
    """Range of tile coordinates along a single axis."""

    min: int
    max: int

    @classmethod
    def from_tuple(cls, bounds: tuple[int, int]) -> TileRangeAxis:
        """Create from (min, max) tuple."""
        return cls(min=bounds[0], max=bounds[1])


@dataclass(frozen=True)
class TileRange:
    """Range of tile coordinates in both x and y directions.

    This class can be used with :func:`make_transform` to generate affine transformation matrices for converting tile
    coordinates to Web Mercator coordinates, and vice versa.
    """

    x: TileRangeAxis
    y: TileRangeAxis

    @classmethod
    def from_dict(cls, d: TileRangeDict) -> Self:
        """Create from dict format {"x": {"min": 0, "max": 1}, "y": {"min": 0, "max": 1}}.

        Args:
            d: Dictionary containing x and y bounds.

        Returns:
            An instance of TileRange initialized with the provided bounds.
        """
        return cls(
            x=TileRangeAxis(min=d["x"]["min"], max=d["x"]["max"]),
            y=TileRangeAxis(min=d["y"]["min"], max=d["y"]["max"]),
        )

    @classmethod
    def from_bounds(
        cls,
        *,
        xmin: int,
        xmax: int,
        ymin: int,
        ymax: int,
    ) -> Self:
        """Create from individual bounds values.

        Args:
            xmin: Minimum x coordinate.
            xmax: Maximum x coordinate.
            ymin: Minimum y coordinate.
            ymax: Maximum y coordinate.

        Returns:
            An instance of TileRange initialized with the provided bounds.
        """
        return cls(x=TileRangeAxis(min=xmin, max=xmax), y=TileRangeAxis(min=ymin, max=ymax))

    @classmethod
    def from_tuples(
        cls,
        *,
        x: tuple[int, int],
        y: tuple[int, int],
    ) -> Self:
        """Create from (min, max) tuples for each axis.

        Args:
            x: Tuple containing (min, max) for x axis.
            y: Tuple containing (min, max) for y axis.

        Returns:
            An instance of TileRange initialized with the provided bounds.
        """
        return cls(x=TileRangeAxis.from_tuple(x), y=TileRangeAxis.from_tuple(y))

    @classmethod
    def single_tile(
        cls,
        x: int,
        y: int,
    ) -> Self:
        """Create a range for a single tile.

        Args:
            x: X coordinate of the tile.
            y: Y coordinate of the tile.

        Returns:
            An instance of TileRange representing a single tile.
        """
        return cls(x=TileRangeAxis(min=x, max=x + 1), y=TileRangeAxis(min=y, max=y + 1))


class TileRangeDictAxis(TypedDict):
    """TypedDict for TileRangeAxis representation."""

    min: int
    max: int


class TileRangeDict(TypedDict):
    """TypedDict for TileRange representation."""

    x: TileRangeDictAxis
    y: TileRangeDictAxis
