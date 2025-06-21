"""Unit tests for src/tilemath/transforms.py."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Final

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from tests.helpers.transform_helpers import (
    assert_coordinate_validity,
    assert_matrix_properties,
    assert_transform_properties,
    create_reference_points,
)
from tilemath import mercantile
from tilemath.transforms import (
    TileRange,
    TileRangeAxis,
    TileRangeDict,
    Transform,
    make_transform,
    transform_bounds,
    transform_point,
    transform_points_batch,
)

SINGLE_TILE_RANGE: Final = TileRange.single_tile(x=0, y=0)
SMALL_TILE_RANGE: Final = TileRange.from_bounds(xmin=0, xmax=2, ymin=0, ymax=2)
MED_TILE_RANGE: Final = TileRange.from_bounds(xmin=10, xmax=20, ymin=10, ymax=20)
EDGE_CASE_TILE_RANGE: Final = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)

SAMPLE_TILERANGES: Final[Mapping[str, TileRange]] = {
    "single_tile": SINGLE_TILE_RANGE,
    "small_range": SMALL_TILE_RANGE,
    "medium_range": MED_TILE_RANGE,
    "edge_case": EDGE_CASE_TILE_RANGE,
}

ZOOM_LEVELS: Final[Sequence[int]] = [0, 1, 5, 10, 15, 18]  # Cover range from global to street level

KNOWN_COORDINATES: Final[Sequence[tuple[float, float]]] = [
    (0.0, 0.0),  # Origin
    (0.5, 0.5),  # Center
    (1.0, 1.0),  # Corner
    (0.25, 0.75),  # Arbitrary point
]


class TestTileRangeAxis:
    """Test TileRangeAxis class functionality."""

    def test_creation_basic(self) -> None:
        """Test basic creation with min/max values."""
        axis = TileRangeAxis(min=0, max=10)
        assert axis.min == 0
        assert axis.max == 10

    def test_creation_from_tuple(self) -> None:
        """Test creation from tuple."""
        axis = TileRangeAxis.from_tuple((5, 15))
        assert axis.min == 5
        assert axis.max == 15

    def test_invalid_ranges(self) -> None:
        """Test validation of invalid ranges (min > max)."""
        # Note: TileRangeAxis doesn't currently validate min <= max
        # This test documents current behavior
        axis = TileRangeAxis(min=10, max=5)
        assert axis.min == 10
        assert axis.max == 5

    @given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_property_range_ordering(self, min_val: int, max_val: int) -> None:
        """Property test: min should always be <= max."""
        assume(min_val <= max_val)  # Only test valid ranges
        axis = TileRangeAxis(min=min_val, max=max_val)
        assert axis.min <= axis.max


class TestTileRange:
    """Test TileRange class functionality."""

    def test_from_bounds(self) -> None:
        """Test creation from individual bounds."""
        tilerange = TileRange.from_bounds(xmin=0, xmax=2, ymin=1, ymax=3)
        assert tilerange.x.min == 0
        assert tilerange.x.max == 2
        assert tilerange.y.min == 1
        assert tilerange.y.max == 3

    def test_from_dict(self) -> None:
        """Test creation from dictionary format."""
        d: TileRangeDict = {"x": {"min": 0, "max": 2}, "y": {"min": 1, "max": 3}}
        tilerange = TileRange.from_dict(d)
        assert tilerange.x.min == 0
        assert tilerange.x.max == 2
        assert tilerange.y.min == 1
        assert tilerange.y.max == 3

    def test_from_tuples(self) -> None:
        """Test creation from tuple pairs."""
        tilerange = TileRange.from_tuples(x=(0, 2), y=(1, 3))
        assert tilerange.x.min == 0
        assert tilerange.x.max == 2
        assert tilerange.y.min == 1
        assert tilerange.y.max == 3

    def test_single_tile(self) -> None:
        """Test single tile creation."""
        tilerange = TileRange.single_tile(x=5, y=7)
        assert tilerange.x.min == 5
        assert tilerange.x.max == 6
        assert tilerange.y.min == 7
        assert tilerange.y.max == 8

    def test_equivalence_between_creation_methods(self) -> None:
        """Test that different creation methods produce equivalent results."""
        # Create same range using different methods
        bounds_range = TileRange.from_bounds(xmin=0, xmax=2, ymin=1, ymax=3)
        dict_range = TileRange.from_dict({"x": {"min": 0, "max": 2}, "y": {"min": 1, "max": 3}})
        tuple_range = TileRange.from_tuples(x=(0, 2), y=(1, 3))

        assert bounds_range == dict_range
        assert dict_range == tuple_range
        assert bounds_range == tuple_range

    @given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_single_tile_properties(self, x: int, y: int) -> None:
        """Property test: single tile should have width/height of 1."""
        tilerange = TileRange.single_tile(x=x, y=y)
        assert tilerange.x.max - tilerange.x.min == 1
        assert tilerange.y.max - tilerange.y.min == 1


class TestTransform:
    """Test Transform class functionality."""

    @pytest.mark.parametrize(
        ("tilerange", "zoom"),
        product(SAMPLE_TILERANGES.values(), ZOOM_LEVELS),
    )
    def test_from_tilerange(self, tilerange: TileRange, zoom: int) -> None:
        """Test transform creation from tile ranges."""
        if zoom <= 1 and (tilerange.x.max > 5 or tilerange.y.max > 5):
            pytest.skip("Skipping due to potential overflow at low zoom levels")

        transform = Transform.from_tilerange(tilerange, zoom)
        assert_transform_properties(transform)

    def test_identity_transform(self) -> None:
        """Test identity transformation properties."""
        identity = Transform.identity()
        assert_transform_properties(identity)

        # Test that identity preserves points
        for x, y in KNOWN_COORDINATES:
            tx, ty = identity.transform_point(x, y)
            assert abs(tx - x) < 1e-10
            assert abs(ty - y) < 1e-10

    def test_inverse_transform(self) -> None:
        """Test that inverse transforms work correctly."""
        transform = Transform.from_tilerange(SMALL_TILE_RANGE, zoom=5)
        inverse = transform.inverse()

        assert_transform_properties(inverse)

        # Test that transform @ inverse ≈ identity
        composition = transform @ inverse
        identity_matrix = np.eye(3, dtype=np.float64)
        assert_allclose(composition.matrix, identity_matrix, atol=1e-10)

    def test_transform_composition(self) -> None:
        """Test transform composition using @ operator."""
        transform1 = Transform.from_tilerange(SINGLE_TILE_RANGE, zoom=1)
        transform2 = Transform.from_tilerange(SMALL_TILE_RANGE, zoom=2)

        composed = transform1 @ transform2
        assert_transform_properties(composed)

        # Test that composition matrix equals matrix multiplication
        expected = transform1.matrix @ transform2.matrix
        assert_allclose(composed.matrix, expected)

    def test_properties_scale_translation(self) -> None:
        """Test scale_x, scale_y, and translation properties."""
        tilerange = TileRange.from_bounds(xmin=0, xmax=1, ymin=0, ymax=1)
        transform = Transform.from_tilerange(tilerange, zoom=1)

        # Properties should be accessible and reasonable
        scale_x = transform.scale_x
        scale_y = transform.scale_y
        translation = transform.translation

        assert isinstance(scale_x, float)
        assert isinstance(scale_y, float)
        assert isinstance(translation, tuple)
        assert len(translation) == 2
        assert scale_x > 0  # Should be positive scaling
        assert scale_y > 0  # Should be positive scaling

    def test_transform_point(self) -> None:
        """Test single point transformation."""
        transform = Transform.from_tilerange(SINGLE_TILE_RANGE, zoom=1)

        for x, y in KNOWN_COORDINATES:
            tx, ty = transform.transform_point(x, y)
            assert_coordinate_validity(tx, ty)

    def test_transform_points_batch(self) -> None:
        """Test batch point transformation."""
        transform = Transform.from_tilerange(SMALL_TILE_RANGE, zoom=5)

        points = np.array(KNOWN_COORDINATES, dtype=np.float64)
        transformed = transform.transform_points(points)

        assert transformed.shape == points.shape
        assert transformed.dtype == np.float64

        # Check that all transformed points are valid
        for i in range(transformed.shape[0]):
            assert_coordinate_validity(transformed[i, 0], transformed[i, 1])

    def test_transform_bounds(self) -> None:
        """Test bounding box transformation."""
        transform = Transform.from_tilerange(EDGE_CASE_TILE_RANGE, zoom=3)

        left, bottom, right, top = transform.transform_bounds(left=0.0, bottom=0.0, right=1.0, top=1.0)

        assert_coordinate_validity(left, bottom)
        assert_coordinate_validity(right, top)
        assert left < right
        assert bottom < top

    @given(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    )
    def test_point_transform_consistency(self, x: float, y: float) -> None:
        """Property test: single and batch transforms should be consistent."""
        transform = Transform.from_tilerange(SINGLE_TILE_RANGE, zoom=1)

        # Single point transform
        single_x, single_y = transform.transform_point(x, y)

        # Batch transform with single point
        points = np.array([[x, y]], dtype=np.float64)
        batch_result = transform.transform_points(points)
        batch_x, batch_y = batch_result[0, 0], batch_result[0, 1]

        # Results should be identical
        assert abs(single_x - batch_x) < 1e-10
        assert abs(single_y - batch_y) < 1e-10

    @pytest.mark.parametrize(
        ("case", "tilerange"),
        SAMPLE_TILERANGES.items(),
    )
    def test_inverse_roundtrip(self, case: str, tilerange: TileRange) -> None:
        """Test that transform -> inverse -> transform gives identity."""
        transform = Transform.from_tilerange(tilerange, zoom=5)
        inverse = transform.inverse()

        for x, y in KNOWN_COORDINATES:
            # Forward then inverse
            fx, fy = transform.transform_point(x, y)
            bx, by = inverse.transform_point(fx, fy)

            # Should get back original coordinates (within tolerance)
            assert abs(bx - x) < 1e-10, f"Roundtrip failed for {case}: x {x} -> {fx} -> {bx}"
            assert abs(by - y) < 1e-10, f"Roundtrip failed for {case}: y {y} -> {fy} -> {by}"


class TestMakeTransform:
    """Test make_transform standalone function."""

    def test_matrix_shape(self) -> None:
        """Test that output matrix has correct shape (3x3)."""
        for name, tilerange in SAMPLE_TILERANGES.items():
            for zoom in ZOOM_LEVELS:
                matrix = make_transform(tilerange, zoom)
                assert matrix.shape == (3, 3), f"Wrong shape for {name} at zoom {zoom}"
                assert matrix.dtype == np.float64

    def test_matrix_properties(self) -> None:
        """Test mathematical properties of transformation matrix."""
        for tilerange in SAMPLE_TILERANGES.values():
            matrix = make_transform(tilerange, zoom=5)
            assert_matrix_properties(matrix)

    def test_determinant_nonzero(self) -> None:
        """Test that transformation matrices are invertible."""
        for name, tilerange in SAMPLE_TILERANGES.items():
            for zoom in ZOOM_LEVELS:
                # Skip problematic combinations that cause overflow
                if zoom == 0 and (tilerange.x.max > 5 or tilerange.y.max > 5):
                    continue
                matrix = make_transform(tilerange, zoom)
                det = np.linalg.det(matrix)
                assert not np.isnan(det), f"Determinant is NaN for {name} at zoom {zoom}"
                assert abs(det) > 1e-15, f"Matrix not invertible for {name} at zoom {zoom}: det={det}"

    @given(st.integers(min_value=0, max_value=18))
    def test_zoom_level_scaling(self, zoom: int) -> None:
        """Property test: higher zoom should give smaller scale factors."""
        tilerange = SAMPLE_TILERANGES["single_tile"]

        if zoom < 18:  # Compare with next zoom level
            matrix1 = make_transform(tilerange, zoom)
            matrix2 = make_transform(tilerange, zoom + 1)

            # Scale factors are on diagonal
            scale1_x = abs(matrix1[0, 0])
            scale1_y = abs(matrix1[1, 1])
            scale2_x = abs(matrix2[0, 0])
            scale2_y = abs(matrix2[1, 1])

            # Higher zoom should have smaller scale (more detailed)
            assert scale2_x < scale1_x, f"Scale should decrease with zoom: {scale1_x} vs {scale2_x}"
            assert scale2_y < scale1_y, f"Scale should decrease with zoom: {scale1_y} vs {scale2_y}"


class TestTransformPoint:
    """Test transform_point standalone function."""

    def test_origin_transform(self) -> None:
        """Test transformation of origin point."""
        tilerange = SAMPLE_TILERANGES["single_tile"]
        matrix = make_transform(tilerange, zoom=1)

        x, y = transform_point(matrix, 0.0, 0.0)
        assert_coordinate_validity(x, y)

    def test_corner_points(self) -> None:
        """Test transformation of tile corners."""
        tilerange = SAMPLE_TILERANGES["edge_case"]
        matrix = make_transform(tilerange, zoom=2)

        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        for corner_x, corner_y in corners:
            x, y = transform_point(matrix, corner_x, corner_y)
            assert_coordinate_validity(x, y)

    def test_consistency_with_mercantile(self) -> None:
        """Test consistency with mercantile library calculations."""
        # Test single tile at zoom 1
        tilerange = TileRange.single_tile(x=0, y=0)
        matrix = make_transform(tilerange, zoom=1)

        # Transform tile corners and compare with mercantile
        ul_x, ul_y = transform_point(matrix, 0.0, 0.0)
        lr_x, lr_y = transform_point(matrix, 1.0, 1.0)

        # Get mercantile bounds for tile (0, 0, 1)
        merc_ul = mercantile.ul(0, 0, 1)
        merc_lr = mercantile.ul(1, 1, 1)  # Lower-right of tile (0,0) is upper-left of tile (1,1)

        # Convert to Web Mercator
        merc_ul_xy = mercantile.xy(merc_ul.lng, merc_ul.lat)
        merc_lr_xy = mercantile.xy(merc_lr.lng, merc_lr.lat)

        # Should be approximately equal
        assert abs(ul_x - merc_ul_xy[0]) < 1e-6
        assert abs(ul_y - merc_ul_xy[1]) < 1e-6
        assert abs(lr_x - merc_lr_xy[0]) < 1e-6
        assert abs(lr_y - merc_lr_xy[1]) < 1e-6

    @given(
        st.floats(min_value=0, max_value=1, allow_nan=False),
        st.floats(min_value=0, max_value=1, allow_nan=False),
    )
    def test_point_in_tile_bounds(self, x: float, y: float) -> None:
        """Property test: points in [0,1] should transform to valid coordinates."""
        tilerange = SAMPLE_TILERANGES["single_tile"]
        matrix = make_transform(tilerange, zoom=5)

        tx, ty = transform_point(matrix, x, y)
        assert_coordinate_validity(tx, ty)


class TestTransformPointsBatch:
    """Test transform_points_batch standalone function."""

    def test_empty_array(self) -> None:
        """Test handling of empty point arrays."""
        tilerange = SAMPLE_TILERANGES["single_tile"]
        matrix = make_transform(tilerange, zoom=1)

        empty_points = np.array([], dtype=np.float64).reshape(0, 2)
        result = transform_points_batch(matrix, empty_points)

        assert result.shape == (0, 2)
        assert result.dtype == np.float64

    def test_single_point_consistency(self) -> None:
        """Test that batch transform of single point matches point transform."""
        tilerange = SAMPLE_TILERANGES["small_range"]
        matrix = make_transform(tilerange, zoom=3)

        for x, y in KNOWN_COORDINATES:
            # Single point transform
            single_x, single_y = transform_point(matrix, x, y)

            # Batch transform
            points = np.array([[x, y]], dtype=np.float64)
            batch_result = transform_points_batch(matrix, points)
            batch_x, batch_y = batch_result[0, 0], batch_result[0, 1]

            assert abs(single_x - batch_x) < 1e-15
            assert abs(single_y - batch_y) < 1e-15

    def test_large_batch(self) -> None:
        """Test correctness with large point batches."""
        matrix = make_transform(MED_TILE_RANGE, zoom=10)

        # Create large batch of random points
        np.random.seed(42)  # For reproducibility
        n_points = 1000
        points = np.random.rand(n_points, 2).astype(np.float64)

        result = transform_points_batch(matrix, points)

        assert result.shape == (n_points, 2)
        assert result.dtype == np.float64

        # Check that all points are valid
        for i in range(n_points):
            assert_coordinate_validity(result[i, 0], result[i, 1])

    def test_invalid_input_shapes(self) -> None:
        """Test error handling for invalid input shapes."""
        matrix = make_transform(SINGLE_TILE_RANGE, zoom=1)

        # Test wrong number of columns
        with pytest.raises((ValueError, IndexError)):
            invalid_points = np.array([[1.0], [2.0]], dtype=np.float64)  # Only 1 column
            transform_points_batch(matrix, invalid_points)

        # Test 1D array
        with pytest.raises((ValueError, IndexError)):
            invalid_points = np.array([1.0, 2.0], dtype=np.float64)  # 1D array
            transform_points_batch(matrix, invalid_points)

    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=1, allow_nan=False),
                st.floats(min_value=0, max_value=1, allow_nan=False),
            ),
            min_size=1,
            max_size=100,
        )
    )
    def test_batch_consistency(self, points: list[tuple[float, float]]) -> None:
        """Property test: batch results should match individual transforms."""
        matrix = make_transform(SINGLE_TILE_RANGE, zoom=2)

        # Convert to numpy array
        points_array = np.array(points, dtype=np.float64)

        # Batch transform
        batch_result = transform_points_batch(matrix, points_array)

        # Individual transforms
        for i, (x, y) in enumerate(points):
            single_x, single_y = transform_point(matrix, x, y)
            batch_x, batch_y = batch_result[i, 0], batch_result[i, 1]

            assert abs(single_x - batch_x) < 1e-15
            assert abs(single_y - batch_y) < 1e-15


class TestTransformBounds:
    """Test transform_bounds standalone function."""

    def test_unit_square(self) -> None:
        """Test transformation of unit square bounds."""
        matrix = make_transform(SINGLE_TILE_RANGE, zoom=1)

        left, bottom, right, top = transform_bounds(matrix, 0.0, 0.0, 1.0, 1.0)

        assert_coordinate_validity(left, bottom)
        assert_coordinate_validity(right, top)
        assert left < right
        assert bottom < top

    def test_bounds_ordering(self) -> None:
        """Test that transformed bounds maintain proper ordering."""
        matrix = make_transform(SMALL_TILE_RANGE, zoom=5)

        # Test various bound rectangles
        test_bounds = [
            (0.0, 0.0, 1.0, 1.0),
            (0.25, 0.25, 0.75, 0.75),
            (0.1, 0.2, 0.9, 0.8),
        ]

        for left_in, bottom_in, right_in, top_in in test_bounds:
            left_out, bottom_out, right_out, top_out = transform_bounds(matrix, left_in, bottom_in, right_in, top_in)

            assert left_out < right_out, f"Left >= Right: {left_out} >= {right_out}"
            assert bottom_out < top_out, f"Bottom >= Top: {bottom_out} >= {top_out}"

    def test_area_preservation_properties(self) -> None:
        """Test mathematical properties of area transformation."""
        matrix = make_transform(EDGE_CASE_TILE_RANGE, zoom=3)

        # Transform unit square
        left, bottom, right, top = transform_bounds(matrix, 0.0, 0.0, 1.0, 1.0)

        # Calculate transformed area
        transformed_area = (right - left) * (top - bottom)

        # Area should be positive and reasonable
        assert transformed_area > 0

        # For a single tile, area should be related to the determinant of the transform
        det = abs(np.linalg.det(matrix[:2, :2]))  # 2x2 submatrix for area scaling
        expected_area = det  # Unit square area * determinant

        # Should be approximately equal (within reasonable tolerance for large numbers)
        relative_error = abs(transformed_area - expected_area) / max(transformed_area, expected_area)
        assert relative_error < 1e-10, f"Area mismatch: {transformed_area} vs {expected_area}"

    @given(
        st.floats(min_value=0, max_value=0.4, allow_nan=False),  # left
        st.floats(min_value=0.6, max_value=0.99, allow_nan=False),  # right
        st.floats(min_value=0, max_value=0.4, allow_nan=False),  # bottom
        st.floats(min_value=0.6, max_value=0.99, allow_nan=False),  # top
    )
    def test_bounds_properties(self, left: float, bottom: float, right: float, top: float) -> None:
        """Property test: transformed bounds should maintain relationships."""
        assume(left < right and bottom < top)  # Only test valid bounds
        assume(right - left > 0.01 and top - bottom > 0.01)  # Avoid tiny rectangles

        matrix = make_transform(SINGLE_TILE_RANGE, zoom=2)

        t_left, t_bottom, t_right, t_top = transform_bounds(matrix, left, bottom, right, top)

        # Transformed bounds should maintain ordering
        assert t_left < t_right, f"Left >= Right: {t_left} >= {t_right}"
        assert t_bottom < t_top, f"Bottom >= Top: {t_bottom} >= {t_top}"

        # All coordinates should be valid
        assert_coordinate_validity(t_left, t_bottom)
        assert_coordinate_validity(t_right, t_top)


class TestIntegration:
    """Integration and cross-validation tests."""

    def test_mercantile_consistency(self) -> None:
        """Test consistency with mercantile library for known cases."""
        # Test several tiles at different zoom levels
        test_cases = [
            (0, 0, 1),
            (1, 1, 2),
            (10, 15, 5),
        ]

        for x, y, z in test_cases:
            tilerange = TileRange.single_tile(x=x, y=y)
            matrix = make_transform(tilerange, zoom=z)

            # Transform tile corners
            ul_transformed = transform_point(matrix, 0.0, 0.0)
            lr_transformed = transform_point(matrix, 1.0, 1.0)

            # Get mercantile bounds
            merc_ul = mercantile.ul(x, y, z)
            merc_lr = mercantile.ul(x + 1, y + 1, z)

            # Convert to Web Mercator
            merc_ul_xy = mercantile.xy(merc_ul.lng, merc_ul.lat)
            merc_lr_xy = mercantile.xy(merc_lr.lng, merc_lr.lat)

            # Should match within reasonable tolerance
            assert abs(ul_transformed[0] - merc_ul_xy[0]) < 1e-6
            assert abs(ul_transformed[1] - merc_ul_xy[1]) < 1e-6
            assert abs(lr_transformed[0] - merc_lr_xy[0]) < 1e-6
            assert abs(lr_transformed[1] - merc_lr_xy[1]) < 1e-6

    def test_roundtrip_accuracy(self) -> None:
        """Test accuracy of forward/inverse transformation roundtrips."""
        for name, tilerange in SAMPLE_TILERANGES.items():
            transform = Transform.from_tilerange(tilerange, zoom=8)
            inverse = transform.inverse()

            test_points = create_reference_points()
            for x, y in test_points:
                # Forward then inverse
                fx, fy = transform.transform_point(x, y)
                bx, by = inverse.transform_point(fx, fy)

                # Should recover original coordinates
                assert abs(bx - x) < 1e-12, f"X roundtrip error for {name}: {x} -> {fx} -> {bx}"
                assert abs(by - y) < 1e-12, f"Y roundtrip error for {name}: {y} -> {fy} -> {by}"

    def test_different_zoom_levels(self) -> None:
        """Test behavior across different zoom levels."""
        tilerange = SAMPLE_TILERANGES["single_tile"]

        # Test that transforms at different zoom levels are consistent
        for zoom in ZOOM_LEVELS:
            transform = Transform.from_tilerange(tilerange, zoom)
            assert_transform_properties(transform)

            # Transform center point
            cx, cy = transform.transform_point(0.5, 0.5)
            assert_coordinate_validity(cx, cy)

    def test_edge_cases_extreme_coordinates(self) -> None:
        """Test handling of extreme coordinate values."""
        tilerange = SAMPLE_TILERANGES["medium_range"]
        transform = Transform.from_tilerange(tilerange, zoom=15)

        # Test extreme but valid coordinates
        extreme_points = [
            (0.0, 0.0),
            (1.0, 1.0),
            (1e-10, 1e-10),
            (1.0 - 1e-10, 1.0 - 1e-10),
        ]

        for x, y in extreme_points:
            tx, ty = transform.transform_point(x, y)
            assert_coordinate_validity(tx, ty)


class TestErrorHandling:
    """Error handling and edge cases."""

    def test_invalid_matrix_shapes(self) -> None:
        """Test error handling for invalid matrix shapes."""
        # Test with wrong matrix shape
        invalid_matrix = np.array([[1, 0], [0, 1]], dtype=np.float64)  # 2x2 instead of 3x3

        with pytest.raises((ValueError, IndexError)):
            transform_point(invalid_matrix, 0.5, 0.5)

    def test_singular_matrices(self) -> None:
        """Test handling of non-invertible matrices."""
        # Create singular matrix (determinant = 0)
        singular_matrix = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float64)

        transform = Transform(matrix=singular_matrix)

        # Should raise error when trying to invert
        with pytest.raises((np.linalg.LinAlgError, ValueError)):
            transform.inverse()

    def test_numerical_precision_limits(self) -> None:
        """Test behavior at numerical precision limits."""
        transform = Transform.from_tilerange(SINGLE_TILE_RANGE, zoom=18)  # High zoom for small scales

        # Test very small differences
        x1, y1 = transform.transform_point(0.5, 0.5)
        x2, y2 = transform.transform_point(0.5 + 1e-15, 0.5 + 1e-15)

        # Results should be valid (though possibly identical due to precision)
        assert_coordinate_validity(x1, y1)
        assert_coordinate_validity(x2, y2)


class TestMathematicalProperties:
    """Property-based testing with Hypothesis for mathematical properties."""

    @given(st.integers(min_value=2, max_value=10))
    def test_transform_linearity(self, zoom: int) -> None:
        """Test that transformations preserve linear relationships."""

        tilerange = TileRange.single_tile(x=0, y=0)
        matrix = make_transform(tilerange, zoom)

        # Test linearity: T(a*p1 + b*p2) = a*T(p1) + b*T(p2)
        p1 = (0.25, 0.25)
        p2 = (0.75, 0.75)
        a, b = 0.3, 0.7

        # Linear combination of points
        combined_point = (a * p1[0] + b * p2[0], a * p1[1] + b * p2[1])

        # Transform combined point
        combined_transformed = transform_point(matrix, combined_point[0], combined_point[1])

        # Transform individual points and combine
        p1_transformed = transform_point(matrix, p1[0], p1[1])
        p2_transformed = transform_point(matrix, p2[0], p2[1])
        linear_combination = (
            a * p1_transformed[0] + b * p2_transformed[0],
            a * p1_transformed[1] + b * p2_transformed[1],
        )

        # Should be equal (affine transforms preserve linear combinations)
        # Use relative tolerance for large coordinate values
        rel_tol = 1e-10
        abs_tol = 1e-6
        assert abs(combined_transformed[0] - linear_combination[0]) < abs_tol + rel_tol * abs(combined_transformed[0])
        assert abs(combined_transformed[1] - linear_combination[1]) < abs_tol + rel_tol * abs(combined_transformed[1])

    @given(
        st.floats(min_value=0, max_value=1, allow_nan=False),
        st.floats(min_value=0, max_value=1, allow_nan=False),
    )
    def test_coordinate_bounds_preservation(self, x: float, y: float) -> None:
        """Test that coordinate transformations preserve expected bounds."""
        matrix = make_transform(SINGLE_TILE_RANGE, zoom=5)

        tx, ty = transform_point(matrix, x, y)
        assert_coordinate_validity(tx, ty)

        # Transformed coordinates should be within reasonable Web Mercator bounds
        # (approximately ±20037508 meters for the full world)
        assert abs(tx) < 25000000  # Allow some margin
        assert abs(ty) < 25000000

    @given(st.integers(min_value=2, max_value=10))
    def test_scaling_relationships(self, scale_factor: int) -> None:
        """Test relationships between different scale factors."""
        base_range = TileRange.single_tile(x=0, y=0)
        scaled_range = TileRange.from_bounds(xmin=0, xmax=scale_factor, ymin=0, ymax=scale_factor)

        base_matrix = make_transform(base_range, zoom=5)
        scaled_matrix = make_transform(scaled_range, zoom=5)

        # Both matrices should be valid
        assert_matrix_properties(base_matrix)
        assert_matrix_properties(scaled_matrix)

        # Scale factors should be positive
        base_scale_x = abs(base_matrix[0, 0])
        scaled_scale_x = abs(scaled_matrix[0, 0])

        assert base_scale_x > 0
        assert scaled_scale_x > 0
