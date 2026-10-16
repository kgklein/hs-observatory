# Raw Malara field along nominal HelioSwarm trajectories

This analysis evaluates the existing deterministic Malara realization at the
saved nominal spacecraft-time positions. It does not generate trajectories,
read measurement-error MC members, or change field amplitudes. The two initial
recipes are `generation/malara-ts-500a-t1500.yaml` (p=0.9) and
`generation/malara-ts-502a-t1500.yaml` (p=0.5). HDF5 locations come from each
recipe's `output.path`. Paths are resolved relative to the project root.

## Commands

From the repository root, run a lightweight synthetic smoke test (no production
observations are read):

```bash
.venv/bin/python -m pytest -q tests/test_malara_raw_hs.py
.venv/bin/python -m ruff check src/hs_observatory/analysis/malara_raw_hs.py tests/test_malara_raw_hs.py
```

Validate both real inputs without field evaluation or pair selection:

```bash
bash scripts/malara_raw_hs_structure_function.sh all --dry-run
```

Run the full analysis on the analysis computer:

```bash
bash scripts/malara_raw_hs_structure_function.sh all
```

Select a case, or explicitly override budgets, seeds and evaluation resources:

```bash
bash scripts/malara_raw_hs_structure_function.sh 500a \
  --fit-pairs 500000 --outside-pairs 100000 \
  --seeds 12345 23456 34567 --backend parallel --workers 4 \
  --field-chunk-size 500 --chunk-size 100000
```

`HSOBS_PYTHON` can select an interpreter; otherwise the runner uses `.venv/bin/python`
or falls back to `python3`. There are no machine-specific worker requirements.
Field evaluation defaults to the backend and workers in the generation recipe.
Native positions/fields are held in memory; pair increments and evaluator caches
are bounded by their respective chunk sizes. Changing field backends or chunks
does not create a new realization.

Completed seeds are resumed by default after checking their scientific signature,
payload checksum, counts and populated moment shapes. `--force` recomputes them.
Each completed seed immediately updates the manifest. Outputs are separate:
`outputs/malara_raw_hs/500a/` and `outputs/malara_raw_hs/502a/`.

Regenerate figures and the report from completed seed files without reading the
original observations or evaluating the field:

```bash
bash scripts/malara_raw_hs_structure_function.sh all --plot-only
```

Overlay a native direct-3D HDF5 result (replace the path with your existing result):

```bash
bash scripts/malara_raw_hs_structure_function.sh 500a --plot-only \
  --direct3d outputs/path-to-existing-direct3d-result.h5
```

Multiple direct-3D files may follow `--direct3d`. Only cases with matching p are
used, and their complete effective evaluator configurations must match. These
curves are labeled **isotropic exact-separation estimates**. They are not matched
pairs and are not the same estimator as the finite-width spacecraft bins.

## Scientific contracts

Preflight compares the physical field, origin, realization, time, spacecraft and
measurement configurations to the embedded generation YAML. Backend settings may
differ without changing the physical field. It verifies the plasma-frame
`/truth/v` contract, position units, uniform native time grids, matching nominal
coordinates across n/v/B, constant density, and the shared v/B realization. The
initial implementation deliberately rejects differing native grids rather than
assuming the q identity survives interpolation. Unsupported non-affine geometry
is rejected by the existing sampler's geometry validator.

The adapter's own evaluator is used at `r - origin`, with its existing periodic
wrapping. Pair separation uses nominal **unwrapped** Euclidean positions,
including time-displaced pairs, exactly as in the duration-convergence estimator.
The 41 edges define 40 uniform logarithmic bins from 70 to 20,000 km. Geometric
bin centers determine budgets and fits: 500,000 pairs per primary fitting bin
(150–1200 km), 100,000 outside, capped at eligible populations. The sensitivity
window is 200–1000 km. Default selection seeds are 12345, 23456 and 34567.
Budgets can be explicitly reduced for experiments; defaults meet production targets.

Before sampling, every reconstructed q sample is checked against stored ideal
velocity and magnetic fields including configured backgrounds and amplitudes.
The existing Elsasser processor reads truth only. With constant proton density,

```
K = velocity_amplitude + magnetic_amplitude * alfven_conversion(density)
z_plus / K = q - mean_over_time(q, per_spacecraft)
```

The pointwise identity is checked numerically. One geometry-stratified pair-index
stream then feeds all three quantities. Every chunk also checks increment identity
and same-spacecraft mean cancellation. Native ideal-z+ moments are accumulated
first, then divided by `abs(K)**n`. Their equality with centered-q moments is
verified. Same-spacecraft and cross-spacecraft conditional moments are recorded
separately; their counts partition the full selected population. They are not
independently sampled populations. The shared index digest provides a compact
audit record without storing millions of pair indices.

Existing independently sampled truth SF caches are not reused as if they were
matched. Instead, ideal z+ is reprocessed from the existing stored truth and
accumulated on the exact same newly selected pairs as raw q. This supplies the
strict matched comparison even when older truth SF products are available.

## Products and interpretation

- `manifest.json`: recipes, embedded configuration and metadata, input SHA256,
  effective realization parameters, origin, git commit, relevant code hashes,
  NumPy version, bin edges, requested/eligible counts, seeds and seed-file references.
- `seed*.json`: native moment sums, rescaled moments, actual per-bin counts for
  all/same/cross-spacecraft contributions; both fit windows; local slopes;
  raw-minus-centered exponents; reconstruction/increment residuals and pair-index digest.
- `diagnostics.png` / `.pdf`: moments for orders 1–6, fitted exponents versus
  analytic p-model predictions, local slopes for n=3,4,6 over 300–2000 km,
  and raw-minus-centered exponent differences. Faint fit curves show individual seeds.
- `report.md`: seed pair counts and mean fitted exponents for every contribution,
  order and window, plus provenance and estimator qualifications.

All seeds probe one deterministic field realization. Their variation measures
pair-selection variability and is not independent-realization uncertainty. Fits
and three-point local slopes use the existing routines, preserving empty-bin
NaNs; `n_mc: 0` requires no observed members. A raw-q discrepancy from direct-3D
scaling indicates trajectory/bin-estimator effects; raw-versus-centered changes
isolate spacecraft-specific preprocessing under the verified assumptions.
