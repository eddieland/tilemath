"""Helper functions for transform testing."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from numpy.typing import NDArray

from tilemath.transforms import TileRange, Transform


def create_test_transform_matrix() -> NDArray[np.float64]:
    """Create a simple test transformation matrix.

    Returns:
        A 3x3 identity transformation matrix for testing.
    """
    return np.eye(3, dtype=np.float64)


def assert_transform_properties(transform: Transform, tolerance: float = 1e-13) -> None:
    """Assert mathematical properties of transforms.

    Args:
        transform: The transform to validate.
        tolerance: Numerical tolerance for comparisons.
    """
    matrix = transform.matrix

    # Check matrix shape
    assert matrix.shape == (3, 3), f"Expected 3x3 matrix, got {matrix.shape}"

    # Check homogeneous coordinate row
    assert abs(matrix[2, 0]) < tolerance, f"Expected matrix[2,0] ≈ 0, got {matrix[2, 0]}"
    assert abs(matrix[2, 1]) < tolerance, f"Expected matrix[2,1] ≈ 0, got {matrix[2, 1]}"
    assert abs(matrix[2, 2] - 1.0) < tolerance, f"Expected matrix[2,2] ≈ 1, got {matrix[2, 2]}"

    # Check that matrix is invertible
    det = np.linalg.det(matrix)
    assert abs(det) > tolerance, f"Matrix should be invertible, determinant = {det}"


def create_reference_points() -> list[tuple[float, float]]:
    """Create reference points for cross-validation.

    Returns:
        List of (x, y) coordinate pairs for testing.
    """
    return [
        (0.0, 0.0),  # Origin
        (0.5, 0.5),  # Center
        (1.0, 1.0),  # Corner
        (0.25, 0.75),  # Arbitrary point
        (0.1, 0.9),  # Near corners
        (0.9, 0.1),  # Near corners
    ]


def assert_matrix_properties(matrix: NDArray[np.float64]) -> None:
    """Assert standard properties of transformation matrices.

    Args:
        matrix: The transformation matrix to validate.
    """
    assert matrix.shape == (3, 3), f"Expected 3x3 matrix, got {matrix.shape}"
    assert matrix[2, 0] == 0.0 and matrix[2, 1] == 0.0 and matrix[2, 2] == 1.0
    assert np.linalg.det(matrix) != 0  # Should be invertible


def assert_coordinate_validity(x: float, y: float) -> None:
    """Assert that coordinates are valid Web Mercator values.

    Args:
        x: X coordinate to validate.
        y: Y coordinate to validate.
    """
    assert not np.isnan(x) and not np.isnan(y), f"Coordinates should not be NaN: ({x}, {y})"
    assert not np.isinf(x) and not np.isinf(y), f"Coordinates should not be infinite: ({x}, {y})"


@st.composite
def tilerange_strategy(draw: st.DrawFn) -> TileRange:
    """Generate random tile ranges for property testing.

    Args:
        draw: Hypothesis draw function.

    Returns:
        A randomly generated TileRange.
    """
    min_x = draw(st.integers(min_value=0, max_value=1000))
    max_x = draw(st.integers(min_value=min_x + 1, max_value=min_x + 100))
    min_y = draw(st.integers(min_value=0, max_value=1000))
    max_y = draw(st.integers(min_value=min_y + 1, max_value=min_y + 100))
    return TileRange.from_bounds(xmin=min_x, xmax=max_x, ymin=min_y, ymax=max_y)
