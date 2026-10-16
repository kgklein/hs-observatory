# A Synthetic HelioSwarm Observatory

This code generates synthetic multipoint spacecraft measurements using representative HelioSwarm trajectories.
These synthetic data can then be analyzed with a variety of specified techniques.
The underlying fields used for the data generation include sine waves, static p-model turbulence, initialized linear eigenmodes, planar MHD shocks, and static direct numerical simulations.
The timeseries can include measurement errors, including timing and spatial knowledge uncertainties and instrumental noise, represented using a Monte Carlo ensemble.
**The addition of new input fields and analysis methods and encouraged.**

## Capabilities and status

*This is an active, evolving research codebase.*
The currently implemented methods intentionally have different scopes.
Some methods, e.g. timing calibration, are areas of active research.

The [intermittency recovery investigation](research-notes/intermittency-recovery.md)
tracks fit-range sensitivity, local logarithmic slopes, and direct 3D Malara
sampling under **Ongoing Research**. It distinguishes current findings from
smoke-test validation and the pilot calculations still to be run.

## Developer path

Start with the quick start and [public Python API](getting-started/first-analysis.md#use-python-instead-of-the-cli),
followed by [architecture](architecture/index.md) and
[data contracts](reference/data-products.md).
[Detailed import diagrams](architecture/module-dependencies.md)
are intended for general reference, and do not represent runtime sequence or HDF5 data flow.

## Project information

Use [CITATION.cff](https://github.com/kgklein/hs-observatory/blob/main/CITATION.cff)
and the method citations when publishing results. The repository
[README](https://github.com/kgklein/hs-observatory/blob/main/README.md) is the concise
landing page. The documentation is being prepared for public release; repository
links require access while private. See [local documentation development](developer/documentation.md).

## Researcher workflow

| Stage | Start here | Continue into reference |
| --- | --- | --- |
| Install | [Installation](getting-started/installation.md), [quick start](getting-started/quick-start.md) | [Configuration](reference/configuration.md) |
| Generate | [Generating observations](user-guide/generation.md) | [Fields](reference/fields.md), [errors](reference/errors.md), [data products](reference/data-products.md) |
| Analyze | [Choosing and running analyses](user-guide/analysis.md) | [Statistics](reference/statistical-methods.md), [gradients](reference/magnetic-gradients.md), [reconstruction](reference/reconstruction.md), [shock timing](reference/shock-timing.md), [timing research](reference/timing-calibration.md) |
| Visualize | [Figures and interpretation](user-guide/visualization.md) | Method-specific diagnostics and limitations |
| Movies | [Animated geometry](user-guide/movies.md) | Mission CDF snapshots, GIF/MP4 output, and frame selection |
