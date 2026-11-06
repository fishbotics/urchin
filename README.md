# Urchin

Urchin is a fork of the no-longer-updated `urdfpy`. The API is still mostly the
same. The main difference currently is that you can load urdfs without the
meshes (which is ~100x faster) by passing the flag `lazy_load_meshes=True` to the `URDF.load` function.

## Installation
You can install urchin directly from pip.
```bash
pip install urchin
```

## Development with uv

Install the project and development tools from the checked-in lockfile:

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked mypy urchin
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Use `uv sync --locked --python 3.10` to select a specific Python version.
The project supports Python 3.9 and later. The lockfile includes compatible
dependency versions for each supported Python version.

## Documentation

See the urdfpy documentation for the current docs. `urchin` will have its own
docs when it diverges enough to deserve it.

The urdfpy docs can be found [here](https://urdfpy.readthedocs.io/en/latest/examples/index.html) for
more information.
