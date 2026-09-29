# Design

The full design notes remain in
[`DESIGN.md`](https://github.com/kgklein/hs-observatory/blob/main/DESIGN.md)
at the repository root. This page summarizes the key architectural contracts;
it does not replace that document.

The design separates reading trajectories, defining fields, sampling fields,
applying errors, writing observations, visualization, and scientific analysis.
Field models receive time and positions and return truth values; error models
act on sampled values independently of the chosen physical field.

## Data conventions

- Spacecraft positions use `(spacecraft, time, 3)` arrays.
- Scalar truth uses `(time, spacecraft)`; vector truth adds a final component axis.
- Observed values retain a leading Monte Carlo axis.
- Each quantity can have its own native time grid.
- Field velocity is in the plasma frame; generation records the configured
  measurement frame and the information needed for downstream conversion.

HDF5 metadata records configuration and generation choices. Analysis methods
are selected by name from a registry rather than being hard-coded into the
generator. Shared Elsasser physics and pair operations live under
`analysis.derived` for reuse.

See the [Architecture Overview](../architecture/index.md) for responsibilities,
the [Module Dependencies](../architecture/module-dependencies.md) for actual
imports, and [Analysis Methods](../user-guide/analysis.md) for supported analyses.

The [data reference](../reference/data-products.md) owns current schema and frame
definitions; [developer guidance](index.md) owns extension recipes.
