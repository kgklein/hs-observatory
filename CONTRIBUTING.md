# Contributing to HS Observatory

Contributions to HS Observatory are welcome. This project is scientific
software, so changes should preserve both software correctness and the
scientific meaning of the calculations.

## Getting started

Clone the repository and install it in editable mode with the development
dependencies:

```bash
git clone https://github.com/kgklein/hs-observatory.git
cd hs-observatory
python3 -m pip install -e ".[dev]"
```
Optional dependencies are described in the
installation documentation.

## Before making a change

For substantial changes, especially changes to scientific algorithms,
configuration formats, or output products, please open an issue first so that
the intended approach can be discussed.

Small bug fixes, documentation improvements, and test additions may be
submitted directly as pull requests.

Development checks

Before submitting a pull request, please run:

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m pytest
```

Documentation changes should also pass:

```bash
mkdocs build --strict
```

The same checks are run automatically by GitHub Actions.

## Scientific changes

Changes to numerical or scientific calculations should include tests whenever
practical.

Useful tests include:

- recovery of known analytic solutions;
- comparison against an existing validated implementation;
- conservation or symmetry checks;
- regression tests for previously verified results;
- numerical agreement between alternative implementations.

If a change intentionally alters numerical results, describe:

- what changed;
- why the previous behavior was inadequate;
- how the new behavior was validated; and
- whether existing configurations or derived products are affected.

Please do not loosen numerical tolerances solely to make a failing test pass
without explaining the underlying numerical reason.

## Code organization

The primary package is located under:

`src/hs_observatory/`

Major components include:

- `fields/` — physical field models and field selection;
- `errors/` — coordinate and measurement-error models;
- `io/` — observation and simulation I/O;
- `analysis/` — scientific analysis methods;
- `spacecraft/` — spacecraft configurations and geometry.

General orchestration is handled by the generation and analysis entry points.

The architecture documentation provides a more detailed overview:

https://kgklein.github.io/hs-observatory/architecture/

## Adding a new field, error model, or analysis

New implementations should follow the existing base interfaces and configuration
patterns.

Please include:

- the implementation;
- configuration validation;
- at least one representative example configuration;
focused tests;
- documentation describing the method and assumptions.

Avoid putting general selection or dispatch logic inside an implementation-
specific module when a shared factory or registry is more appropriate.

## Documentation

User-facing features should be documented under docs/.

To preview the documentation locally:

```bash
python3 -m pip install -e ".[docs]"
mkdocs serve
```

Then open the local address printed by MkDocs, normally
http://127.0.0.1:8000/.

## Pull requests

A pull request should ideally:

- address one reasonably focused change;
- explain the motivation for the change;
- describe any scientific consequences;
- include or update tests where appropriate;
- update documentation when user-facing behavior changes; and
- pass the automated CI checks.

Large mechanical refactors should generally be separated from changes to
scientific behavior so that each can be reviewed independently.

## Reporting bugs

When reporting a bug, please include enough information to reproduce the
problem where possible:

- HS Observatory version or Git commit;
- Python version;
- relevant configuration file or reduced example;
- command used;
- error message or traceback; and
- expected versus observed behavior.

For numerical discrepancies, also describe the physical or mathematical result
you expected.