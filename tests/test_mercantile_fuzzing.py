"""Fuzzing tests to compare tilemath.mercantile with upstream mercantile library."""

from __future__ import annotations

from collections.abc import Callable

import mercantile as upstream_mercantile
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy, composite

from tilemath.mercantile import (
    bounding_tile,
    bounds,
    children,
    lnglat,
    neighbors,
    parent,
    quadkey,
    quadkey_to_tile,
    tile,
    tiles,
    truncate_lnglat,
    ul,
    xy,
    xy_bounds,
)


@composite
def valid_coordinates(draw: Callable[[SearchStrategy[float]], float]) -> tuple[float, float]:
    """Generate valid longitude/latitude coordinates.

    This strategy generates latitude values between -85.051128779807 and 85.051128779807, which are the valid bounds
    for Web Mercator projection, and longitude values between -180 and 180.

    It ensures that the generated coordinates are valid and do not include NaN or infinity.
    """
    lng = draw(
        st.floats(
            min_value=-180.0,
            max_value=180.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    lat = draw(
        st.floats(
            min_value=-85.051128779807,
            max_value=85.051128779807,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return lng, lat


@composite
def valid_zoom(draw: Callable[[SearchStrategy[int]], int]) -> int:
    """Generate valid zoom levels."""
    return draw(st.integers(min_value=0, max_value=24))


@composite
def valid_tile_coords(draw: Callable[[SearchStrategy[int]], int]) -> tuple[int, int, int]:
    """Generate valid tile coordinates for a given zoom level."""
    zoom = draw(valid_zoom())
    max_coord = 2**zoom
    x = draw(st.integers(min_value=0, max_value=max_coord - 1))
    y = draw(st.integers(min_value=0, max_value=max_coord - 1))
    return x, y, zoom


@composite
def valid_bbox(
    draw: Callable[[SearchStrategy[tuple[float, float]]], tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Generate valid bounding boxes."""
    lng1, lat1 = draw(valid_coordinates())
    lng2, lat2 = draw(valid_coordinates())

    west = min(lng1, lng2)
    east = max(lng1, lng2)
    south = min(lat1, lat2)
    north = max(lat1, lat2)

    # Ensure the bbox is not degenerate
    assume(east > west)
    assume(north > south)

    return west, south, east, north


class TestTileFunctions:
    """Test core tile conversion functions."""

    @given(valid_coordinates(), valid_zoom())
    @settings(max_examples=1000)
    def test_tile_conversion(self, coords: tuple[float, float], zoom: int) -> None:
        """Test that tile() produces identical results to upstream mercantile."""
        lng, lat = coords

        our_result = tile(lng, lat, zoom)
        upstream_result = upstream_mercantile.tile(lng, lat, zoom)

        assert our_result.x == upstream_result.x
        assert our_result.y == upstream_result.y
        assert our_result.z == upstream_result.z

    @given(valid_tile_coords())
    @settings(max_examples=1000)
    def test_bounds_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that bounds() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        our_result = bounds(x, y, z)
        upstream_result = upstream_mercantile.bounds(x, y, z)

        assert abs(our_result.west - upstream_result.west) < 1e-10
        assert abs(our_result.south - upstream_result.south) < 1e-10
        assert abs(our_result.east - upstream_result.east) < 1e-10
        assert abs(our_result.north - upstream_result.north) < 1e-10

    @given(valid_coordinates())
    @settings(max_examples=1000)
    def test_xy_conversion(self, coords: tuple[float, float]) -> None:
        """Test that xy() produces identical results to upstream mercantile."""
        lng, lat = coords

        our_result = xy(lng, lat)
        upstream_result = upstream_mercantile.xy(lng, lat)

        # Allow for small floating point differences
        assert abs(our_result[0] - upstream_result[0]) < 5e-9
        assert abs(our_result[1] - upstream_result[1]) < 5e-8

    @given(
        st.floats(min_value=-20037508.342789244, max_value=20037508.342789244, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-20037508.342789244, max_value=20037508.342789244, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=1000)
    def test_lnglat_conversion(self, x: float, y: float) -> None:
        """Test that lnglat() produces identical results to upstream mercantile."""
        our_result = lnglat(x, y)
        upstream_result = upstream_mercantile.lnglat(x, y)

        assert abs(our_result.lng - upstream_result.lng) < 1e-10
        assert abs(our_result.lat - upstream_result.lat) < 1e-10

    @given(valid_tile_coords())
    @settings(max_examples=500)
    def test_parent_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that parent() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        # Skip zoom 0 tiles as they have no parent
        assume(z > 0)

        our_result = parent(x, y, z)
        assert our_result is not None, "Parent should not be None for zoom > 0"

        upstream_result = upstream_mercantile.parent(x, y, z)
        assert upstream_result is not None, "Upstream parent should not be None for zoom > 0"

        our_x, our_y, our_z = our_result
        upstream_x, upstream_y, upstream_z = upstream_result

        assert our_x == upstream_x
        assert our_y == upstream_y
        assert our_z == upstream_z

    @given(valid_tile_coords())
    @settings(max_examples=200)
    def test_children_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that children() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        # Skip high zoom levels to avoid too many children
        assume(z < 20)

        our_result = list(children(x, y, z))
        upstream_result = list(upstream_mercantile.children(x, y, z))

        assert len(our_result) == len(upstream_result)

        # Sort both lists for comparison
        our_sorted = sorted(our_result, key=lambda t: (t.x, t.y, t.z))
        upstream_sorted = sorted(upstream_result, key=lambda t: (t.x, t.y, t.z))

        for our_tile, upstream_tile in zip(our_sorted, upstream_sorted, strict=False):
            assert our_tile.x == upstream_tile.x
            assert our_tile.y == upstream_tile.y
            assert our_tile.z == upstream_tile.z

    @given(valid_tile_coords())
    @settings(max_examples=500)
    def test_neighbors_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that neighbors() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        our_result = list(neighbors(x, y, z))
        upstream_result = list(upstream_mercantile.neighbors(x, y, z))

        # Sort both lists for comparison
        our_sorted = sorted(our_result, key=lambda t: (t.x, t.y, t.z))
        upstream_sorted = sorted(upstream_result, key=lambda t: (t.x, t.y, t.z))

        assert len(our_sorted) == len(upstream_sorted)

        for our_tile, upstream_tile in zip(our_sorted, upstream_sorted, strict=False):
            assert our_tile.x == upstream_tile.x
            assert our_tile.y == upstream_tile.y
            assert our_tile.z == upstream_tile.z


class TestQuadKeyFunctions:
    """Test quadkey-related functions."""

    @given(valid_tile_coords())
    @settings(max_examples=1000)
    def test_quadkey_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that quadkey() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        our_result = quadkey(x, y, z)
        upstream_result = upstream_mercantile.quadkey(x, y, z)

        assert our_result == upstream_result

    @given(valid_tile_coords())
    @settings(max_examples=1000)
    def test_quadkey_to_tile_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that quadkey_to_tile() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        # First get a quadkey from the tile
        qk = quadkey(x, y, z)

        our_result = quadkey_to_tile(qk)
        upstream_result = upstream_mercantile.quadkey_to_tile(qk)

        assert our_result.x == upstream_result.x
        assert our_result.y == upstream_result.y
        assert our_result.z == upstream_result.z


class TestBoundingFunctions:
    """Test bounding box and tile range functions."""

    @given(valid_bbox())
    @settings(max_examples=200)
    def test_bounding_tile_conversion(self, bbox: tuple[float, float, float, float]) -> None:
        """Test that bounding_tile() produces identical results to upstream mercantile."""
        west, south, east, north = bbox

        our_result = bounding_tile(west, south, east, north)
        upstream_result = upstream_mercantile.bounding_tile(west, south, east, north)

        assert our_result.x == upstream_result.x
        assert our_result.y == upstream_result.y
        assert our_result.z == upstream_result.z

    @given(valid_bbox(), valid_zoom())
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.filter_too_much],
    )
    def test_tiles_conversion(self, bbox: tuple[float, float, float, float], zoom: int) -> None:
        """Test that tiles() produces identical results to upstream mercantile."""
        west, south, east, north = bbox

        # Limit to reasonable zoom levels to avoid generating too many tiles
        assume(zoom <= 10)

        # Ensure the bbox isn't too large at high zoom levels
        if zoom > 5:
            assume(east - west < 10.0)
            assume(north - south < 10.0)

        our_result = list(tiles(west, south, east, north, zoom))
        upstream_result = list(upstream_mercantile.tiles(west, south, east, north, zoom))

        # Sort both lists for comparison
        our_sorted = sorted(our_result, key=lambda t: (t.x, t.y, t.z))
        upstream_sorted = sorted(upstream_result, key=lambda t: (t.x, t.y, t.z))

        assert len(our_sorted) == len(upstream_sorted)

        for our_tile, upstream_tile in zip(our_sorted, upstream_sorted, strict=False):
            assert our_tile.x == upstream_tile.x
            assert our_tile.y == upstream_tile.y
            assert our_tile.z == upstream_tile.z


class TestUtilityFunctions:
    """Test utility functions."""

    @given(valid_coordinates())
    @settings(max_examples=1000)
    def test_truncate_lnglat_conversion(self, coords: tuple[float, float]) -> None:
        """Test that truncate_lnglat() produces identical results to upstream mercantile."""
        lng, lat = coords

        our_result = truncate_lnglat(lng, lat)
        our_lng, our_lat = our_result

        upstream_result: tuple[float, float] = upstream_mercantile.truncate_lnglat(lng, lat)
        upstream_lng, upstream_lat = upstream_result

        assert abs(our_lng - upstream_lng) < 1e-10
        assert abs(our_lat - upstream_lat) < 1e-10

    @given(valid_tile_coords())
    @settings(max_examples=1000)
    def test_ul_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that ul() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        our_result = ul(x, y, z)
        upstream_result = upstream_mercantile.ul(x, y, z)

        assert abs(our_result.lng - upstream_result.lng) < 1e-10
        assert abs(our_result.lat - upstream_result.lat) < 1e-10

    @given(valid_tile_coords())
    @settings(max_examples=1000)
    def test_xy_bounds_conversion(self, tile_coords: tuple[int, int, int]) -> None:
        """Test that xy_bounds() produces identical results to upstream mercantile."""
        x, y, z = tile_coords

        our_result = xy_bounds(x, y, z)
        upstream_result = upstream_mercantile.xy_bounds(x, y, z)

        # Allow for small floating point differences
        assert abs(our_result.left - upstream_result.left) < 5e-8
        assert abs(our_result.bottom - upstream_result.bottom) < 5e-8
        assert abs(our_result.right - upstream_result.right) < 5e-8
        assert abs(our_result.top - upstream_result.top) < 5e-8


class TestRoundTripConsistency:
    """Test round-trip consistency between coordinate systems."""

    @given(valid_coordinates(), valid_zoom())
    @settings(max_examples=1000)
    def test_tile_bounds_roundtrip(self, coords: tuple[float, float], zoom: int) -> None:
        """Test that tile->bounds->tile roundtrip is consistent with upstream."""
        lng, lat = coords

        # Our implementation
        our_tile = tile(lng, lat, zoom)
        our_bounds = bounds(our_tile.x, our_tile.y, our_tile.z)

        # Upstream implementation
        upstream_tile = upstream_mercantile.tile(lng, lat, zoom)
        upstream_bounds = upstream_mercantile.bounds(upstream_tile.x, upstream_tile.y, upstream_tile.z)

        our_x, our_y, our_z = our_tile
        upstream_x, upstream_y, upstream_z = upstream_tile

        # Both should produce the same tile
        assert our_x == upstream_x
        assert our_y == upstream_y
        assert our_z == upstream_z

        our_roundtrip = tile(our_bounds.west, our_bounds.south, zoom)
        upstream_roundtrip = upstream_mercantile.tile(upstream_bounds.west, upstream_bounds.south, zoom)

        our_roundtrip_x, our_roundtrip_y, our_roundtrip_z = our_roundtrip
        upstream_roundtrip_x, upstream_roundtrip_y, upstream_roundtrip_z = upstream_roundtrip

        # Roundtrip should be consistent
        assert our_roundtrip_x == upstream_roundtrip_x
        assert our_roundtrip_y == upstream_roundtrip_y
        assert our_roundtrip_z == upstream_roundtrip_z

    @given(valid_coordinates())
    @settings(max_examples=1000)
    def test_xy_lnglat_roundtrip(self, coords: tuple[float, float]) -> None:
        """Test that xy->lnglat roundtrip is consistent with upstream."""
        lng, lat = coords

        # Our implementation
        our_xy = xy(lng, lat)
        our_roundtrip = lnglat(our_xy[0], our_xy[1])

        # Upstream implementation
        upstream_xy = upstream_mercantile.xy(lng, lat)
        upstream_roundtrip = upstream_mercantile.lnglat(upstream_xy[0], upstream_xy[1])

        # Both should produce the same results (allow for small floating point differences)
        assert abs(our_xy[0] - upstream_xy[0]) < 5e-8
        assert abs(our_xy[1] - upstream_xy[1]) < 5e-8

        # Roundtrip should be consistent
        assert abs(our_roundtrip.lng - upstream_roundtrip.lng) < 1e-10
        assert abs(our_roundtrip.lat - upstream_roundtrip.lat) < 1e-10
