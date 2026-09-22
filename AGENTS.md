# Plotting-task guidance

For tasks in this directory:

- Restrict inspection to this directory and directly imported project modules.
- Do not perform a repo-wide codebase survey unless explicitly requested.
- Do not run full-repo pytest, ruff, mypy, or pre-commit.
- Validate only modified Python files with targeted ruff checks.
- Run only the plotting command needed to regenerate the requested figure.
- Do not refactor unrelated code.
- Prefer minimal patches.