# Plotting-task guidance

For tasks in this directory:

- Restrict inspection to this directory and directly imported project modules.
- Do not perform a repo-wide codebase survey unless explicitly requested.
- Do not run full-repo pytest, ruff, mypy, or pre-commit.
- Validate only modified Python files with targeted ruff checks.
- Run only the plotting command needed to regenerate the requested figure.
- Do not refactor unrelated code.
- Prefer minimal patches.

### Technical audits and development notes

Place internal technical audits, feasibility investigations, profiling reports, and Codex-generated implementation reviews in the repository-level `audits/` directory, **not** `docs/`.

Reserve `docs/` for public-facing MkDocs documentation. New Markdown files under `docs/` must be incorporated into the MkDocs navigation or explicitly handled by the documentation configuration.

Technical audit files should be version-controlled, but should not appear on the public documentation website unless explicitly requested.