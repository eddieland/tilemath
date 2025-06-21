"""Shared fixtures and test configuration for transforms tests."""

from __future__ import annotations

from typing import Final

import pytest

from tilemath.transforms import TileRange

# Test constants as defined in the testing plan
SAMPLE_TILERANGES: Final[dict[str, TileRange]] = {
    "single_tile": TileRange.single_tile(x=0, y=0),
    "small_range": TileRange.from_bounds(xmin=0, xmax=2, ymin=0, ymax=2),
    "large_range": TileRange.from_bounds(xmin=100, xmax=200, ymin=50, ymax=150),
    "edge_case": TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1),
}

ZOOM_LEVELS: Final[list[int]] = [0, 1, 5, 10, 15, 18]  # Cover range from global to street level

KNOWN_COORDINATES: Final[list[tuple[float, float]]] = [
    (0.0, 0.0),  # Origin
    (0.5, 0.5),  # Center
    (1.0, 1.0),  # Corner
    (0.25, 0.75),  # Arbitrary point
]


@pytest.fixture
def sample_tileranges() -> dict[str, TileRange]:
    """Fixture providing sample tile ranges for testing."""
    return SAMPLE_TILERANGES


@pytest.fixture
def zoom_levels() -> list[int]:
    """Fixture providing zoom levels for testing."""
    return ZOOM_LEVELS


@pytest.fixture
def known_coordinates() -> list[tuple[float, float]]:
    """Fixture providing known coordinates for testing."""
    return KNOWN_COORDINATES
