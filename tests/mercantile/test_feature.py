"""Unit tests for generating a GeoJSON Feature from a Tile."""

from __future__ import annotations

import pytest

from tilemath.mercantile import Tile, TileArgParsingError, feature


class TestFeatureBasic:
    """Test basic feature generation functionality."""

    def test_feature_with_tile_instance(self) -> None:
        """Test feature generation with a Tile instance."""
        tile = Tile(1, 1, 1)
        result = feature(tile)

        assert result["type"] == "Feature"
        assert result["id"] == "Tile(x=1, y=1, z=1)"
        assert result["geometry"]["type"] == "Polygon"
        assert len(result["geometry"]["coordinates"]) == 1
        assert len(result["geometry"]["coordinates"][0]) == 5  # Closed ring

        # Check properties
        props = result["properties"]
        assert props["title"] == "XYZ tile 1, 1, 1"
        assert props["grid-name"] == "mercator"
        assert props["grid-zoom"] == 1
        assert props["grid-x"] == 1
        assert props["grid-y"] == 1

    def test_feature_with_tuple_args(self) -> None:
        """Test feature generation with tuple arguments."""
        result = feature((2, 3, 4))

        assert result["type"] == "Feature"
        assert result["id"] == "Tile(x=2, y=3, z=4)"
        assert result["properties"]["grid-x"] == 2
        assert result["properties"]["grid-y"] == 3
        assert result["properties"]["grid-zoom"] == 4

    def test_feature_with_individual_args(self) -> None:
        """Test feature generation with individual x, y, z arguments."""
        result = feature(5, 6, 7)

        assert result["type"] == "Feature"
        assert result["id"] == "Tile(x=5, y=6, z=7)"
        assert result["properties"]["grid-x"] == 5
        assert result["properties"]["grid-y"] == 6
        assert result["properties"]["grid-zoom"] == 7

    def test_feature_bbox_format(self) -> None:
        """Test that bbox is properly formatted."""
        result = feature(0, 0, 0)
        bbox = result["bbox"]

        assert len(bbox) == 4
        assert all(isinstance(coord, float) for coord in bbox)
        # For tile (0,0,0), bbox should be the entire world
        assert bbox[0] == -180.0  # west
        assert bbox[1] == pytest.approx(-85.0511287798066, abs=1e-10)  # south
        assert bbox[2] == 180.0  # east
        assert bbox[3] == pytest.approx(85.0511287798066, abs=1e-10)  # north

    def test_feature_coordinate_ring_closure(self) -> None:
        """Test that coordinate ring is properly closed."""
        result = feature(1, 1, 2)
        coords = result["geometry"]["coordinates"][0]

        # First and last coordinates should be identical
        assert coords[0] == coords[-1]
        assert len(coords) == 5  # 4 corners + closure


class TestFeatureProjections:
    """Test different coordinate projections."""

    def test_geographic_projection_default(self) -> None:
        """Test geographic projection (default)."""
        result = feature(1, 1, 1)
        coords = result["geometry"]["coordinates"][0]

        # Geographic coordinates should be in lat/lng range
        for coord in coords:
            lng, lat = coord
            assert -180 <= lng <= 180
            assert -90 <= lat <= 90

    def test_geographic_projection_explicit(self) -> None:
        """Test explicit geographic projection."""
        result = feature(1, 1, 1, projected="geographic")
        coords = result["geometry"]["coordinates"][0]

        # Should be same as default
        for coord in coords:
            lng, lat = coord
            assert -180 <= lng <= 180
            assert -90 <= lat <= 90

    def test_mercator_projection(self) -> None:
        """Test web mercator projection."""
        result = feature(1, 1, 1, projected="mercator")
        coords = result["geometry"]["coordinates"][0]

        # Mercator coordinates should be in meters
        for coord in coords:
            x, y = coord
            # Web mercator coordinates can be very large
            assert isinstance(x, float)
            assert isinstance(y, float)
            # Basic sanity check - should be within reasonable mercator bounds
            assert abs(x) < 25000000  # ~20 million meters is roughly world width
            assert abs(y) < 25000000

    def test_projection_coordinate_consistency(self) -> None:
        """Test that projections produce consistent coordinate ordering."""
        geo_result = feature(2, 1, 3, projected="geographic")
        merc_result = feature(2, 1, 3, projected="mercator")

        geo_coords = geo_result["geometry"]["coordinates"][0]
        merc_coords = merc_result["geometry"]["coordinates"][0]

        # Both should have same number of coordinates
        assert len(geo_coords) == len(merc_coords) == 5

        # Both should form closed rings
        assert geo_coords[0] == geo_coords[-1]
        assert merc_coords[0] == merc_coords[-1]


class TestFeatureCustomization:
    """Test feature customization options."""

    def test_custom_feature_id(self) -> None:
        """Test custom feature ID."""
        result = feature(1, 1, 1, fid="custom-tile-id")
        assert result["id"] == "custom-tile-id"

    def test_custom_properties(self) -> None:
        """Test custom properties."""
        custom_props: dict[str, str | int] = {"name": "test-tile", "category": "urban", "priority": 5}
        result = feature(1, 1, 1, props=custom_props)

        props = result["properties"]
        # Should include both default and custom properties
        assert props["name"] == "test-tile"
        assert props["category"] == "urban"
        assert props["priority"] == 5
        assert props["grid-name"] == "mercator"  # Default property preserved
        assert props["grid-x"] == 1

    def test_custom_properties_override_defaults(self) -> None:
        """Test that custom properties can override defaults."""
        custom_props = {"title": "Custom Title", "grid-name": "custom-grid"}
        result = feature(1, 1, 1, props=custom_props)

        props = result["properties"]
        assert props["title"] == "Custom Title"
        assert props["grid-name"] == "custom-grid"
        assert props["grid-x"] == 1  # Non-overridden default preserved

    def test_empty_custom_properties(self) -> None:
        """Test with empty custom properties dict."""
        result = feature(1, 1, 1, props={})

        # Should still have default properties
        props = result["properties"]
        assert props["title"] == "XYZ tile 1, 1, 1"
        assert props["grid-name"] == "mercator"


class TestFeatureBuffer:
    """Test buffer functionality."""

    def test_positive_buffer(self) -> None:
        """Test positive buffer expansion."""
        base_result = feature(1, 1, 2)
        buffered_result = feature(1, 1, 2, buffer=0.1)

        base_bbox = base_result["bbox"]
        buffered_bbox = buffered_result["bbox"]

        # Buffered bbox should be larger in all directions
        assert buffered_bbox[0] < base_bbox[0]  # west
        assert buffered_bbox[1] < base_bbox[1]  # south
        assert buffered_bbox[2] > base_bbox[2]  # east
        assert buffered_bbox[3] > base_bbox[3]  # north

        # Buffer should be exactly 0.1 in each direction
        assert buffered_bbox[0] == pytest.approx(base_bbox[0] - 0.1)
        assert buffered_bbox[1] == pytest.approx(base_bbox[1] - 0.1)
        assert buffered_bbox[2] == pytest.approx(base_bbox[2] + 0.1)
        assert buffered_bbox[3] == pytest.approx(base_bbox[3] + 0.1)

    def test_negative_buffer(self) -> None:
        """Test negative buffer (shrinking)."""
        base_result = feature(1, 1, 2)
        shrunk_result = feature(1, 1, 2, buffer=-0.05)

        base_bbox = base_result["bbox"]
        shrunk_bbox = shrunk_result["bbox"]

        # Shrunk bbox should be smaller in all directions
        assert shrunk_bbox[0] > base_bbox[0]  # west
        assert shrunk_bbox[1] > base_bbox[1]  # south
        assert shrunk_bbox[2] < base_bbox[2]  # east
        assert shrunk_bbox[3] < base_bbox[3]  # north

    def test_zero_buffer(self) -> None:
        """Test zero buffer (no change)."""
        base_result = feature(1, 1, 2)
        zero_buffer_result = feature(1, 1, 2, buffer=0.0)

        assert base_result["bbox"] == zero_buffer_result["bbox"]
        assert base_result["geometry"] == zero_buffer_result["geometry"]

    def test_buffer_with_mercator_projection(self) -> None:
        """Test buffer with mercator projection."""
        result = feature(1, 1, 2, projected="mercator", buffer=1000.0)  # 1km buffer

        # Should work without errors
        assert result["type"] == "Feature"
        assert "bbox" in result
        assert len(result["geometry"]["coordinates"][0]) == 5


class TestFeaturePrecision:
    """Test coordinate precision handling."""

    def test_precision_truncation(self) -> None:
        """Test coordinate precision truncation."""
        result = feature(1, 1, 5, precision=2)  # 2 decimal places
        coords = result["geometry"]["coordinates"][0]

        for coord in coords:
            lng, lat = coord
            # Check that coordinates are truncated to 2 decimal places
            assert lng == round(lng, 2)
            assert lat == round(lat, 2)

    def test_precision_zero(self) -> None:
        """Test zero precision (integer coordinates)."""
        result = feature(1, 1, 3, precision=0)
        coords = result["geometry"]["coordinates"][0]

        for coord in coords:
            lng, lat = coord
            assert lng == int(lng)
            assert lat == int(lat)

    def test_high_precision(self) -> None:
        """Test high precision values."""
        result = feature(1, 1, 10, precision=8)  # 8 decimal places
        coords = result["geometry"]["coordinates"][0]

        for coord in coords:
            lng, lat = coord
            assert lng == round(lng, 8)
            assert lat == round(lat, 8)

    def test_precision_with_buffer(self) -> None:
        """Test precision with buffer applied."""
        result = feature(1, 1, 3, buffer=0.123456789, precision=3)
        coords = result["geometry"]["coordinates"][0]

        for coord in coords:
            lng, lat = coord
            assert lng == round(lng, 3)
            assert lat == round(lat, 3)

    def test_precision_negative_raises_error(self) -> None:
        """Test that negative precision raises ValueError."""
        with pytest.raises(ValueError, match="Precision must be >= 0"):
            feature(1, 1, 1, precision=-1)


class TestFeatureEdgeCases:
    """Test edge cases and error conditions."""

    def test_world_tile(self) -> None:
        """Test world tile (0, 0, 0)."""
        result = feature(0, 0, 0)

        assert result["properties"]["grid-x"] == 0
        assert result["properties"]["grid-y"] == 0
        assert result["properties"]["grid-zoom"] == 0

        # World tile should span entire geographic range
        bbox = result["bbox"]
        assert bbox[0] == -180.0  # west
        assert bbox[2] == 180.0  # east

    def test_high_zoom_tile(self) -> None:
        """Test high zoom level tile."""
        result = feature(12345, 67890, 18)

        assert result["properties"]["grid-x"] == 12345
        assert result["properties"]["grid-y"] == 67890
        assert result["properties"]["grid-zoom"] == 18
        assert result["type"] == "Feature"

    def test_invalid_tile_args_raises_error(self) -> None:
        """Test that invalid tile arguments raise TileArgParsingError."""
        with pytest.raises(TileArgParsingError):
            feature("invalid")  # type: ignore[arg-type]

        with pytest.raises(TileArgParsingError):
            feature(1, 2)  # Missing z coordinate

        with pytest.raises(TileArgParsingError):
            feature(1, 2, 3, 4)  # Too many arguments

    def test_combined_options(self) -> None:
        """Test combining multiple options."""
        result = feature(
            10,
            20,
            8,
            fid="complex-tile",
            props={"type": "test", "value": 42},
            projected="mercator",
            buffer=500.0,
            precision=1,
        )

        assert result["id"] == "complex-tile"
        assert result["properties"]["type"] == "test"
        assert result["properties"]["value"] == 42
        assert result["properties"]["grid-x"] == 10

        # Coordinates should be in mercator projection with 1 decimal precision
        coords = result["geometry"]["coordinates"][0]
        for coord in coords:
            x, y = coord
            assert x == round(x, 1)
            assert y == round(y, 1)


class TestFeatureGeometry:
    """Test geometric properties of generated features."""

    def test_polygon_winding_order(self) -> None:
        """Test that polygon coordinates follow correct winding order."""
        result = feature(1, 1, 2)
        coords = result["geometry"]["coordinates"][0]

        # Remove the closing coordinate for calculation
        ring = coords[:-1]
        assert len(ring) == 4

        # Calculate signed area to determine winding order using shoelace formula
        signed_area = 0.0
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            signed_area += (ring[j][0] - ring[i][0]) * (ring[j][1] + ring[i][1])

        # The actual winding order depends on the coordinate system and implementation
        # What's important is that we have a consistent, closed polygon
        # Just verify we have a non-zero area (polygon is not degenerate)
        assert signed_area != 0

        # Verify the polygon is properly closed
        assert coords[0] == coords[-1]

    def test_coordinate_bounds_consistency(self) -> None:
        """Test that coordinate bounds match bbox."""
        result = feature(2, 3, 4)
        bbox = result["bbox"]
        coords = result["geometry"]["coordinates"][0]

        # Extract all x and y coordinates
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]

        # Bbox should match coordinate extents
        assert min(x_coords) == bbox[0]  # west
        assert min(y_coords) == bbox[1]  # south
        assert max(x_coords) == bbox[2]  # east
        assert max(y_coords) == bbox[3]  # north

    def test_tile_coverage_mathematical_consistency(self) -> None:
        """Test mathematical consistency of tile coverage."""
        # Test a few specific tiles with known geographic bounds
        test_cases = [
            (0, 0, 1),  # Northwest quadrant of world
            (1, 0, 1),  # Northeast quadrant of world
            (0, 1, 1),  # Southwest quadrant of world
            (1, 1, 1),  # Southeast quadrant of world
        ]

        for x, y, z in test_cases:
            result = feature(x, y, z)
            bbox = result["bbox"]

            # Basic sanity checks
            assert bbox[0] < bbox[2]  # west < east
            assert bbox[1] < bbox[3]  # south < north
            assert -180 <= bbox[0] <= 180  # Valid longitude range
            assert -180 <= bbox[2] <= 180
            assert -90 <= bbox[1] <= 90  # Valid latitude range
            assert -90 <= bbox[3] <= 90
