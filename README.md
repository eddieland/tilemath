# tilemath

A modern, type-safe Python library for spherical mercator coordinate and tile utilities. Designed as an API-compatible replacement for [mercantile](https://github.com/mapbox/mercantile) with full type safety and minimal dependencies.

## Features

- **Full Type Safety**: Complete type annotations for all functions and classes
- **API Compatible**: Drop-in replacement for mercantile with the same interface
- **Minimal Dependencies**: Zero runtime dependencies for core functionality
- **Modern Python**: Built for Python 3.10+ with modern language features
- **Fast**: Optimized implementations of coordinate transformations and tile operations

## Installation

```bash
pip install tilemath
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add tilemath
```

## Quick Start

```python
import tilemath.mercantile as mercantile

# Convert longitude/latitude to tile coordinates
tile = mercantile.tile(-122.4194, 37.7749, 12)  # San Francisco
print(f"Tile: {tile.x}, {tile.y}, {tile.z}")

# Get bounding box for a tile
bbox = mercantile.bounds(tile)
print(f"Bounds: {bbox}")

# Convert tile to quadkey
quadkey = mercantile.quadkey(tile)
print(f"Quadkey: {quadkey}")
```

## API Reference

The library provides the same API as mercantile, including:

- `tile(lng, lat, zoom)` - Get tile containing a longitude/latitude
- `bounds(tile)` - Get bounding box of a tile
- `xy(lng, lat)` - Convert longitude/latitude to web mercator coordinates
- `lnglat(x, y)` - Convert web mercator coordinates to longitude/latitude
- `quadkey(tile)` - Convert tile to Microsoft quadkey
- `quadkey_to_tile(quadkey)` - Convert quadkey to tile
- `tiles(west, south, east, north, zooms)` - Generate tiles for a bounding box
- `children(tile)` - Get child tiles
- `parent(tile)` - Get parent tile
- `neighbors(tile)` - Get neighboring tiles

## Type Safety

Unlike the original mercantile, tilemath provides complete type annotations:

```python
from tilemath.mercantile import Tile, Bbox

# All functions have proper type hints
def process_tile(tile: Tile) -> Bbox:
    return bounds(tile)

# Type checkers will catch errors
tile = Tile(x=1, y=2, z=3)
bbox: Bbox = process_tile(tile)
```

## Requirements

- Python 3.10 or higher
- No runtime dependencies

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and development workflows.

### Setup

```bash
# Clone the repository
git clone https://github.com/eddieland/tilemath.git
cd tilemath

# Install dependencies
make install
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_mercantile_upstream.py -v
```

### Code Quality

```bash
# Run linting and type checking
make lint

# Run everything (install, lint, test)
make
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

This project is inspired by and aims to be compatible with [mercantile](https://github.com/mapbox/mercantile) by Sean Gillies and contributors.
