# Contributing to tilemath

Thank you for your interest in contributing to tilemath! This guide will help you set up your development environment and understand our workflow.

## Prerequisites

### Installing uv

This project uses [uv](https://docs.astral.sh/uv/) for Python version management and dependency handling. If you don't have uv installed:

**macOS/Linux (curl method):**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**macOS (Homebrew):**
```shell
brew update && brew install uv
```

See the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms and methods.

### Python Setup

Install a compatible Python version:
```shell
uv python install 3.13  # or 3.11, 3.12
```

## Getting Started

1. **Fork and clone the repository:**
   ```shell
   # Fork the repo on GitHub first, then:
   git clone https://github.com/eddieland/tilemath.git
   cd tilemath
   ```

2. **Set up your development environment:**
   ```shell
   make install  # Installs all dependencies including dev tools
   ```

## Development Workflow

### Available Commands

The project includes a Makefile with convenient shortcuts:

```shell
# Complete development cycle (sync, lint, test):
make

# Individual operations:
make install    # Install/sync all dependencies
make lint       # Run ruff formatting and basedpyright type checking
make test       # Execute test suite
make build      # Create distribution packages
make clean      # Remove build artifacts
make upgrade    # Update dependencies to latest compatible versions
```

### Running Tests

```shell
# All tests:
make test

# Specific test with output:
uv run pytest -s src/tilemath/some_test.py

# All tests with verbose output:
uv run pytest -v
```

### Code Quality

Before submitting changes:

1. **Run the linter:** `make lint`
2. **Run tests:** `make test`
3. **Ensure zero errors:** Both commands must pass without warnings

### Dependency Management

```shell
# Add runtime dependency:
uv add package_name

# Add development dependency:
uv add --dev package_name

# Update all dependencies:
uv sync --upgrade

# Update specific package:
uv lock --upgrade-package package_name
```

### Local Development Installation

To use your development version as a local tool:
```shell
uv tool install --editable .
```

## IDE Configuration

### Recommended Extensions (VSCode/Cursor/Windsurf)

- [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Based Pyright](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) for type checking

### Virtual Environment

If you need shell access to the environment:
```shell
uv venv
source .venv/bin/activate
```

## Submitting Changes

1. **Create a feature branch:**
   ```shell
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the project's coding standards

3. **Test thoroughly:**
   ```shell
   make  # Runs sync, lint, and test
   ```

4. **Commit and push:**
   ```shell
   git add .
   git commit -m "Description of your changes"
   git push origin feature/your-feature-name
   ```

5. **Open a pull request** on GitHub

## Release Process

### For Maintainers

Releases are automated through GitHub Actions and PyPI trusted publishing:

#### Initial PyPI Setup (one-time)

1. **Create PyPI account** at [pypi.org](https://pypi.org/)

2. **Verify project name availability** at `https://pypi.org/project/PROJECT_NAME`

3. **Configure trusted publishing:**
   - Visit [PyPI publishing settings](https://pypi.org/manage/account/publishing/)
   - Add repository as "pending" trusted publisher
   - Specify: project name, repository owner/name, workflow file `publish.yml`

#### Creating Releases

1. **Ensure CI passes:** Check that all tests pass in GitHub Actions

2. **Create GitHub release:**
   - Go to repository → Releases → "Create a new release"
   - Create new tag (e.g., `v0.1.0`, `v1.2.3`)
   - Add release notes describing changes
   - Publish release

3. **Automated publishing:** GitHub Actions will automatically build and upload to PyPI

#### Subsequent Releases

Simply repeat the release creation process with a new version tag. The automation handles the rest.

## Resources

- [uv documentation](https://docs.astral.sh/uv/)
- [basedpyright documentation](https://docs.basedpyright.com/latest/)
- [Project repository](https://github.com/eddieland/tilemath)

## Questions?

If you have questions about contributing, please open an issue on GitHub or reach out to the maintainers.
