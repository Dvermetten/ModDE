# Installation

```bash
pip install modde
```

modde needs Python 3.10 or newer. Binary wheels are provided for Windows
(x64), Linux (x86_64) and macOS (Apple Silicon, and Intel from macOS 10.15). On other platforms pip builds modde from source,
which needs a C++ compiler (see below).

The dependencies are installed automatically: NumPy, SciPy,
[ioh](https://iohprofiler.github.io/) (problems, logging) and
[ConfigSpace](https://automl.github.io/ConfigSpace/) (configuration space for
tuning).

Check the installation:

```python
import modde
print(modde.__version__)
```

## Building from source

Requirements:

- a C++17 compiler (tested with MSVC 2022 on Windows, GCC on Ubuntu and Apple Clang on macOS);
- CMake 3.18 or newer (installed automatically by the build if missing).

The C++ core uses [Eigen](https://eigen.tuxfamily.org) (a git submodule) and
[nanobind](https://github.com/wjakob/nanobind); the build uses
[scikit-build-core](https://scikit-build-core.readthedocs.io).

```bash
git clone --recursive https://github.com/Dvermetten/ModDE.git
cd ModDE
pip install .
```

If you cloned without `--recursive`, fetch Eigen with
`git submodule update --init`.

For development, install the test and documentation extras and run the tests:

```bash
pip install ".[test,docs]"
pytest
mkdocs serve        # documentation at http://127.0.0.1:8000
```

## The old version

The pure-Python 0.0.4 implementation is still included as `modde.legacy`. It is
deprecated and will be removed in a later release; see
[Migrating from 0.0.4](migration.md).
