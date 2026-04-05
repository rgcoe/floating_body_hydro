[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rgcoe/floating_body_hydro/HEAD?urlpath=%2Fdoc%2Ftree%2Fspheroid%2Fspheroid_6_1.ipynb)
[![CI](https://github.com/rgcoe/floating_body_hydro/actions/workflows/ci.yml/badge.svg)](https://github.com/rgcoe/floating_body_hydro/actions/workflows/ci.yml)

Example Python code for floating body hydrodynamics in AOE 3234/5334.

## Python

If you're new to Python, there are many guides to help you get started; see, e.g., in [Windows](https://learn.microsoft.com/en-us/windows/dev-environment/python?tabs=winget).

## Project structure

- `pdstripy.py`: Main Python module containing the `pdstripy` functionality
- `pyproject.toml`: Python packaging/build configuration and project metadata
- `README.md`: Project documentation and setup instructions
- `PDstrip.md`: Guide to installing and using `PDstrip`
- `hw07.md`: AOE 3234/5334 homework \#7
- `destroyer/`: Assets and script for generating a destroyer hull geometry
- `spheroid/`: Spheroid example case files, geometry, notebook, and generated outputs

## Set up

`pdstripy` is helps parse the results from `PDstrip`.
Use the following steps to set it up on your machine.

### 1. Create and activate a virtual environment

From the project root (`floating_body_hydro`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the package

Use one of these modes:

- Standard install (best for users)

```bash
pip install .
```

- Editable install (best for development; code changes are picked up immediately)

```bash
pip install -e .
```

### 3. Verify installation

```bash
python -c "import pdstripy; print(pdstripy.__name__)"
```

Expected output:

```text
pdstripy
```

### 4. Verify key dependencies (optional)

```bash
python -c "import pdstripy, capytaine, numpy, matplotlib, xarray, ipykernel; print('all good')"
```

### 5. Reinstall after dependency or metadata changes

If you modify `pyproject.toml` or `setup.py`, reinstall:

```bash
pip install --force-reinstall .
```

## Testing

### Install test dependencies

```bash
pip install -e .[dev]
```

### Fixture data for parser/integration tests

PDStrip output filenames are git-ignored in normal workflows, so committed test fixtures are vendored in:

- `tests/data/spheroid/pdstrip_run/`

The fixture files were copied from:

- `spheroid/pdstrip_run/`

If source outputs change intentionally, refresh the vendored fixture copy to keep tests aligned.

### Run tests

- Fast/default parser-focused suite:

```bash
pytest -m "not capytaine"
```

- Capytaine comparison suite (optional, slower):

```bash
pytest -m "capytaine"
```

- Full suite:

```bash
pytest
```

## CI

GitHub Actions runs CI on pushes to `main` and on pull requests with a matrix of:

- Operating systems: `ubuntu-latest`, `windows-latest`, `macos-latest`
- Python versions: `3.9`, `3.10`, `3.11`, `3.12`

The workflow has two required job groups:

- Fast tests:

```bash
pytest -m "not capytaine"
```

- Capytaine tests:

```bash
pytest -m "capytaine"
```

If capytaine jobs fail in CI, test reports are uploaded as workflow artifacts for debugging.
