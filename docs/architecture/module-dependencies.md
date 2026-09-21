# Module Dependencies

[Open the detailed SVG](module-dependencies.svg) for zooming, or download the
[DOT source](module-dependencies.dot).

![Static Python module dependencies](module-dependencies.svg)

This is a static source dependency graph extracted using Python's `ast` module.
An arrow points from the importing module to its dependency. **Solid** arrows
represent module-level imports; **dashed** arrows represent function/method-local
or lazy imports. When both occur, the arrow is solid. Imports guarded only by
`TYPE_CHECKING` are excluded.

The default view suppresses tests and standalone helpers, thin `cli_*.py`
wrappers, initializer/re-export relationships, analysis registration back-edges,
and version-only imports. Other dependencies and cycles remain visible.

This graph does not show runtime call order. In particular, important relationships
through saved HDF5 products may have no corresponding Python import. Start with
the [Architecture Overview](index.md) for a smaller conceptual view.

## Regeneration

From the repository root:

```bash
python3 scripts/generate_architecture_graph.py
```

No third-party Python graph package is required. Graphviz's `dot` executable is
optional for DOT generation and required for SVG rendering.

Use `--no-render` to write only DOT, `--full` to include initializer/wrapper and
registration noise in the detailed view, or `--hide-local-imports` to omit lazy
dependencies. `--dot PATH` and `--svg PATH` select detailed output paths; overview
files are written alongside them. `--detailed-only` skips overview generation.

Full extraction conventions remain in the repository's
[architecture README](https://github.com/kgklein/hs-observatory/blob/main/docs/architecture/README.md).
