# HIVE 10xx Malara generation campaign

Derived from the attached 9xx generation YAMLs. Each file changes only `n_scales: 14` to `16`, `t_stop_s: 6000.0` to `3600.0`, and all case-specific paths and identifiers from 9xx to 10xx (`900→1000`, etc.). All physical parameters, seeds, enabled errors, and error models remain unchanged.

The HDF5 outputs are new files under `/xdisk/kgklein/hs-observatory/production/`, and plot outputs are placed in the matching case folders. At 0.125 s cadence, 3600 s inclusive gives 28,801 timestamps.

The 10xx files must be placed in the desired Puma config directory before generation.
