# Audit of existing Malara field-validation capabilities

Audit date: 2026-10-15. Scope: the current working tree in `hs-observatory`, based
on commit `c277a83def38b8706d3bbad3836d9b4df280b71d`, including existing uncommitted
research notes and the recent Study A diagnostics. This audit changes only this
report. No evaluator, field generation, observation generation, sampling job,
plotting command, or test suite was run. Inspection used source/configuration
reads, small saved HDF5 moment arrays, saved JSON summaries, and observation
metadata/shapes. No large observation arrays were loaded.

**Finding:** a reusable, trajectory-independent direct-3D structure-function
workflow already exists, with completed 2,000- and 5,000-pair pilots for
p=0.5, 0.7, and 0.9. Its high-order results do **not** establish converged
p-model scaling. In the 200–1,000 km window, the p=0.9 sixth-order fitted
exponent changes from **1.32849 to 1.64948** when the budget increases from
2,000 to 5,000 pairs/separation; theory is **1.28630**. This is existing saved
evidence, not a new field calculation (section 4).

Evidence labels throughout:

- **Code:** behavior verified by reading the implementation; this does not mean
  a new execution test was performed.
- **Saved result:** values inspected in existing local products, with dataset
  locations specified below.
- **Documentation:** a prior narrative/design claim, distinguished from a
  numerical validation.
- **Interpretation:** mathematical or scientific reasoning from that evidence.

## 1. Capability inventory

Source locations below are repository-relative; `:N–M` denotes source lines.
The table distinguishes actual field-only calculations from related machinery
that still operates on spacecraft samples.

| Capability | Implemented? | Source/configuration | Existing validation | Gap |
|---|---|---|---|---|
| Raw vector field at arbitrary 3D points | Yes | `src/hs_observatory/fields/malara.py:358–425`; adapter/base configuration at `:533–573` | Shape, deterministic realization, periodicity and finite-difference divergence tests in `tests/test_malara.py:34–115` | Tests do not establish spatial intermittency exponents or full rotational isotropy |
| Uniform positions and isotropic separation directions | Yes | `src/hs_observatory/analysis/spatial_structure_function.py:51–65` | Exact distances, position means and direction first/second moments, 30,000 inexpensive coordinate pairs: `tests/test_spatial_structure_function.py:67–98` | Sampler isotropy is not a test of field isotropy |
| Direct-3D raw-vector S1–S6 | Yes | `src/hs_observatory/analysis/spatial_structure_function.py:68–88,130–197`; `configs/example_spatial_structure_function.yaml`; `analysis/5000_spatial_structure_function.yaml` | Saved 2k/5k pilots at 25 exact separations for all three p values; synthetic-field arithmetic tests at `tests/test_spatial_structure_function.py:107–131` | p=0.9 high-order moments remain budget-sensitive; no saved direct amplitude distribution |
| Global exponents and fit diagnostics | Yes | `src/hs_observatory/analysis/structure_function.py:330–381`; `src/hs_observatory/analysis/spatial_structure_function.py:212–248` | Synthetic prescribed power laws recover their exponents, residuals and R²: `tests/test_spatial_structure_function.py:203–229` | No test of adequately sampled Malara increments against analytic exponents; R² is not a convergence test |
| Local slopes and analytic p-model reference | Yes | `src/hs_observatory/analysis/structure_function_local.py:27–83`; direct integration at `src/hs_observatory/analysis/spatial_structure_function.py:237–249` | Formula identities and synthetic power laws: `tests/test_structure_function_local.py:20–39`; saved pilot local slopes | No demonstrated stable high-order p=0.9 plateau |
| Spatial-budget checkpoints and batch variability | Yes | `src/hs_observatory/analysis/spatial_structure_function.py:130–197` | Quarter-prefix and batch sums checked against direct synthetic arithmetic: `tests/test_spatial_structure_function.py:107–131`; pilots retain both | Only one spatial sampling seed in saved pilots; batches are not field realizations; budget extension reevaluates prefixes |
| Cached refitting and PNG/PDF plots | Yes | `src/hs_observatory/analysis/spatial_structure_function.py:408–428,487–589`; CLI in `src/hs_observatory/cli_spatial_structure_function.py:93–108` | Test forbids all evaluator calls and checks source bytes are unchanged: `tests/test_spatial_structure_function.py:232–263` | Direct cache has no implementation/content fingerprint or sampling resume; comparison refresh depends on live HS inputs |
| Flatness of raw direct increments | Ingredients available | Shared `S4/S2²` helper: `src/hs_observatory/analysis/structure_function.py:317–327`; direct files retain both moments | Direct moments are available; no dedicated saved direct-flatness analysis was identified | Derivable without field evaluation, but not currently a direct plotting product; tail convergence still matters |
| Raw direct increment PDFs, moment contributions and concentration count | No integrated direct workflow | Reusable arithmetic exists in `src/hs_observatory/analysis/structure_function_moment_convergence.py:97–191` | Tested with synthetic amplitudes in `tests/test_structure_function_moment_convergence.py:21–80` | Direct sampler discards individual increments and S8/S12; cannot reconstruct C4/C6, top-event fractions or N_conc from old direct files |
| Elsasser PDFs, alignment and distribution distances | Yes, spacecraft-sampled rather than field-only | `src/hs_observatory/analysis/elsasser_increment_pdf_static.py:322–383,550–590,708–761,766–814`; `src/hs_observatory/analysis/increment_pdf_wasserstein.py:43–85`; `analysis/ELSASSER_INCREMENT_PDF_PUBLICATION.md` | Dedicated PDF and distance unit tests; existing publication workflow | Different quantity/projection/normalization and potentially truncated histogram tails; not independent uniform-volume field validation |
| High-order tails and temporal localization on 500a/502a | Yes, spacecraft truth only | `src/hs_observatory/analysis/structure_function_moment_convergence.py:32–41,68–191,216–257,403–479`; `scripts/structure_function_moment_convergence.sh` | Existing 500k/bin, three-seed products now present; section 6 distinguishes their scope | Does not sample the 3D volume; N_conc is not a correlation-corrected effective sample size |
| Evaluator performance/equivalence diagnostics | Yes | `scripts/benchmark_malara_evaluator.py:191–280`; `tests/test_malara_optimized_backends.py:21–109`; `tests/test_malara_evaluator_performance_helpers.py` | Direct/cached/parallel agreement, cache accounting, periodicity, chunk invariance | Accuracy relative to the reference evaluator is not accuracy relative to p-model statistics |

## 2. The existing direct-3D workflow

### Entry points and configurations

**Code:** the registered method is `SpatialStructureFunctionAnalysis`, method and
configuration section `spatial_structure_function`
(`src/hs_observatory/analysis/spatial_structure_function.py:279–285`). The package
exposes `hsobs-analyze-spatial-structure-function` and
`hsobs-plot-spatial-structure-function` (`pyproject.toml:30–31`), with a module
fallback `python -m hs_observatory.cli_spatial_structure_function sample|plot`
(`src/hs_observatory/cli_spatial_structure_function.py:111–114`). The normal
`hsobs-analyze CONFIG` dispatcher also supports the method; this is documented
at `docs/reference/statistical-methods.md:541–547` and exercised by
`tests/test_spatial_structure_function.py:232–235`.

Both existing recipes read **a generation YAML**,
`generation/malara-prod-401.yaml`, rather than observations. The example requests
2,000 pairs, while `analysis/5000_spatial_structure_function.yaml` requests 5,000.
Both request p=0.5,0.7,0.9, 25 log-spaced separations from 50–20,000 km, eight
batches, sampling seed 12345, chunks of 128 pairs, windows 200–1,000 and
100–1,300 km, and three fit points minimum (recipe lines 1–29). Their effective
saved settings use the parallel backend with four workers. Only p is replaced
in the adapter's base field configuration; the field seed, realization, cascade,
and raw normalization are preserved
(`src/hs_observatory/analysis/spatial_structure_function.py:284–345`).

### Exactly what is sampled

**Code:** for each separation λ, the sampler draws five independent uniform
variates per pair. Three set x uniformly within `[0,Lx) × [0,Ly) × [0,Lz)`.
The other two set azimuth uniformly on `[0,2π)` and cos θ uniformly on `[-1,1)`.
It evaluates `(x, x+λ rhat)`, so the separation direction is uniform on the
sphere and the unwrapped displacement has exactly magnitude λ. There is no
spacecraft geometry, mesh interpolation, field-dependent acceptance, or
increment-dependent stopping rule
(`src/hs_observatory/analysis/spatial_structure_function.py:51–65,142–171`).

**Interpretation:** both endpoints are marginally uniform in the periodic volume;
they are deliberately coupled by a prescribed separation. Isotropic sampling of
separation directions does not imply that the underlying lattice-based field is
exactly isotropic. Sampling with a fixed pseudo-random seed is reproducible,
not an independent ensemble of turbulent fields.

Coordinates are **domain-relative**. The raw evaluator wraps positions modulo L,
and each eddy uses minimum-image displacement relative to its lattice center
(`src/hs_observatory/fields/malara.py:257–264,358–396`). The observation adapter
subtracts its configured origin before raw evaluation (`:649–653`); direct
uniform-volume sampling therefore does not need to add a physical origin.
Default/log grids must stay below Lmin/2; explicit centers can override with a
warning (`src/hs_observatory/analysis/spatial_structure_function.py:293–300`).
All saved pilot separations are below 50,000 km, so their labels are unambiguous
minimum-image distances for the 100,000 km cube.

Each exact separation receives **n_pairs**, not n_pairs divided among scales or
batches. All six orders use the **same vector increments**. Different separations
have their own streams; all p values share the coordinates at a given separation.
The stream is keyed by the spatial seed and float64 bits of λ, independently
of p and chunk size. Increasing 2k to 5k reuses the first 2k coordinate prefix
but **recomputes those field values**; changing the numerical separation value
changes its stream. Code and tests:
`src/hs_observatory/analysis/spatial_structure_function.py:51–88,130–197`;
`tests/test_spatial_structure_function.py:77–98,107–131`.

### Quantities, fitting, and numerical behavior

The measured quantity is

\[
S_n^q(\lambda)=N^{-1}\sum_a
\left|\mathbf q(\mathbf x_a+\lambda\hat{\mathbf r}_a)-\mathbf q(\mathbf x_a)\right|^n,
\qquad n=1,\ldots,6.
\]

It uses the **full vector magnitude**, not a longitudinal or perpendicular
component. This is raw q with the adapter's base vector-potential normalization;
no velocity/magnetic output amplitudes, backgrounds, density conversion, Elsasser
variables, or trajectory means are applied. Code:
`src/hs_observatory/analysis/spatial_structure_function.py:68–88,284–292,349–356`.
Tests explicitly reject observation/Elsasser workflow calls and verify this
normalization (`tests/test_spatial_structure_function.py:168–198`).

The shared kernels sum `magnitude**n` and divide by actual count
(`src/hs_observatory/analysis/structure_function.py:385–393`). The direct workflow
uses compensated Kahan addition across chunk sums, but the within-chunk powers
and sums remain ordinary float64; nonfinite values/overflows raise errors
(`src/hs_observatory/analysis/spatial_structure_function.py:75–118`). It does not
use the scaled/log-moment arithmetic of the newer Study A module.

Global ξn is an **unweighted linear least-squares fit** of ln Sn against ln λ,
using actual positive finite sampled centers inside inclusive specified bounds.
Counts screen empty bins; they do not weight the fit. Fewer than the required
points produces NaN. ξn/ξ2 is also stored; the raw exponents should be compared
with the supplied ζn. The direct wrapper additionally stores intercept, log RMS
residual, and R² (`src/hs_observatory/analysis/structure_function.py:330–381`;
`src/hs_observatory/analysis/spatial_structure_function.py:212–249`).

Three-point local slopes regress three neighboring log values at actual centers,
with undefined endpoints; adjacent slopes use log differences at geometric
midpoints. Invalid neighborhoods are not bridged. Theory is calculated separately
as a reference, never used to adjust measured moments or slopes
(`src/hs_observatory/analysis/structure_function_local.py:27–83`).

### Budgets, cache, and what can be recovered

**Code:** each run saves contiguous, nonoverlapping sample batches and cumulative
25/50/75/100% moment checkpoints. For the pilots these are 250 or 625 pairs per
batch, and checkpoints `[500,1000,1500,2000]` or `[1250,2500,3750,5000]`.
Batch SD uses ddof=1; batch CV divides this SD by the full-sample mean. Local
slopes are also computed per batch. The displayed bands use batch-slope SD,
not standard error of the full estimate and not field-realization uncertainty
(`src/hs_observatory/analysis/spatial_structure_function.py:121–197,237–247,520–531`).

Tasks bound coordinate arrays and pending work; at most `2*n_workers` tasks are
queued. Serial caches clear after each task, and parallel tasks construct their
own evaluator/cache. Field caches retain deterministic eddy data, not reusable
sampled increments (`src/hs_observatory/analysis/spatial_structure_function.py:68–72,
91–107,155–174`; `src/hs_observatory/fields/malara.py:83–92,174–240`).

HDF5 retains source YAML, effective field configurations, RNG metadata,
full/batch/checkpoint moments and counts, local slopes, fits, and HS comparison
provenance (`src/hs_observatory/analysis/spatial_structure_function.py:343–356,
408–428`). **It does not retain sampled coordinates, amplitudes, or rank-ordered
contributions.** Thus existing files support refits, derivatives, flatness and
checkpoint/batch comparisons without field reevaluation, but not exact direct
PDFs, extreme-event contributions, S8/S12 or N_conc for n=4/6. Saved S4 does
permit an n=2 concentration count `N*S2²/S4`, subject to its sampling convergence;
it cannot supply the analogous higher-order quantities.

The dedicated plotting command regenerates separate HDF5 and PNG/PDF products
without constructing a field (`:565–589`; test at
`tests/test_spatial_structure_function.py:232–263`). This is statistical-product
reuse, **not sampling resume**. Direct sampling has one seed per invocation, no
multi-budget/seed campaign scheduler, no fingerprint-based completion cache,
and no automatic convergence acceptance criterion (sample CLI at
`src/hs_observatory/cli_spatial_structure_function.py:41–90`). Existing direct
products can be overwritten by another sampling run to the same output path;
use distinct output prefixes (`src/hs_observatory/analysis/spatial_structure_function.py:358–376`).

## 3. Field construction and what theory does and does not guarantee

### Cascade and normalization

**Code:** each cell has a reproducible seed hashed from
`base_seed|realization|scale|i|j|k`. Random draws select three distortions
γ in [-1,1], a sign ±1, a normalized Gaussian direction qhat, and exactly four
of eight child octants as high-weight children
(`src/hs_observatory/fields/malara.py:128–148`). p is absent from the identity
seed, so p comparisons preserve this geometry/random hierarchy and change
cascade weights.

Energy-transfer factor ε starts at 1 at the root. Every ancestor contributes
`2p` for a high child or `2(1-p)` for a low child; heritage is followed from the
root even when the field sum omits large scales below the injection index
(`src/hs_observatory/fields/malara.py:150–172`). **Interpretation from code:** the
eight-child average multiplier is exactly one, because four children receive
each factor. This is exact normalization of the cascade proxy. It is not an
assertion that overlapping curl eddies conserve field energy at every parent.
For p=0.5 all ε equal one; a unit test checks this at sampled cells
(`tests/test_malara.py:108–115`). There is no comparable full intermittent
cascade-moment identity test in the inspected Malara tests.

The signed coefficient at level m is

\[
A_m=s A_0\,\epsilon_m^{\alpha}2^{-mh},
\quad \alpha=\texttt{epsilon_exponent},\quad
h=\texttt{spectral_exponent_h}.
\]

Defaults α=h=1/3 are configurable, not enforced to equal each other
(`src/hs_observatory/fields/malara.py:35–63,208–217,287–295`). The adapter defaults
A0 to the cube root of the domain volume: 100,000 for the current cubic domain.
This compensates the gradient's 1/L dimension, rather than enforcing unit RMS
or normalizing sampled moments (`:559–573`).

**Interpretation:** over all cells at level m, the multiplicative construction
implies the exact cascade-proxy identity

\[
\langle\epsilon_m^a\rangle_{\rm cells}
=\left[\frac{(2p)^a+[2(1-p)]^a}{2}\right]^m.
\]

Consequently the cell-averaged **absolute coefficient** moments,
`<|A_m|^n>`, scale with level as
`2^(-m τn)`, with

\[
\tau_n=nh-\alpha n+1-\log_2[p^{\alpha n}+(1-p)^{\alpha n}].
\]

For α=h=1/3 this equals the requested ζn. This derivation concerns ε and
coefficients; it is not an identity for increments of the summed spatial field.
The plotting reference routine always uses the α=h=1/3 formula, even for custom
field configurations with other values. Such configurations therefore need a
reconsidered theoretical reference; no compatibility guard is present
(`src/hs_observatory/analysis/structure_function_local.py:27–32`;
`src/hs_observatory/analysis/spatial_structure_function.py:237–249,271–276`).

### Eddy geometry, vector field, and finite scales

**Code:** eddy centers are **fixed dyadic lattice cell centers**, not independently
uniform random positions. Compact supports extend one cell length each side,
so total support width is `2*L/2**m`. γ deforms each local coordinate by
`X + γ(X²-1/4)`, and qhat sets the vector-potential direction. The support box
remains aligned with Cartesian axes; there is no random SO(3) rotation of its
whole shape (`src/hs_observatory/fields/malara.py:257–295,334–355`). Random
orientations, signs and distortions are real implemented ingredients; “random
eddy positions” is not a literal description of these lattice centers.

The compact polynomial is `(1-4t²)^4` inside [-1/2,1/2], zero outside, with an
analytic derivative. Each eddy contributes
`A_m * grad(phi_m) cross qhat`, the curl of a scalar-shaped vector potential.
The gradient includes `2**(-m)/(2*cell_size)`; in a cube this prefactor is
independent of m. Summation includes injection through maximum scale, inclusive
(`src/hs_observatory/fields/malara.py:95–108,274–295,344–355`). This is the
mechanism behind the documented “solenoidal by construction” claim
(`docs/reference/fields.md:54–69`). Periodic position wrapping and periodic
support displacements are explicit, rather than post hoc tiling of samples.

Current 500a/502a settings include **13 levels, m=2,…,14**, with nominal cell
sizes 25,000 down to 6.1035 km and support widths 50,000 down to 12.207 km
(`generation/malara-ts-500a-t1500.yaml:19–36`;
`generation/malara-ts-502a-t1500.yaml:19–36`). Plot cascade marks are contextual
cell sizes, explicitly not proven inertial-range boundaries
(`src/hs_observatory/analysis/structure_function_local.py:21–24,107–138`).

**Interpretation:** the production configuration's compact curl construction
supports solenoidality, but exact rotational invariance, realization-level
homogeneity and the ζn scaling of spatial increments are statistical expectations
requiring measurements. The lattice and separable supports can leave directional
or scale-dependent signatures. Superposition across scales, spatial overlap,
vector cancellations, finite injection/cutoff scales and rare supports separate
field-increment statistics from cascade-coefficient identities. A finite smooth
field also approaches smooth-increment behavior at sufficiently small separation;
the p-model power law cannot hold at every λ.

There is **no mathematical enforcement or empirical unit-test guarantee** that
`S_n(λ) ∝ λ^ζn` for the actual vector-increment estimator over the chosen finite
fitting windows. This does not prove the field is wrong; it means validating
that expectation is an outstanding empirical task.

### Raw q versus ideal z-plus for these cases

The 500a/502a YAMLs specify constant n=5 cm⁻³, shared-realization velocity and
magnetic fluctuations of amplitudes 10 km/s and 0.5 nT, respectively, with no
realization offsets (`generation/malara-ts-{500a,502a}-t1500.yaml:38–51`). The
adapter shares an evaluator when effective realization indices match and applies
output amplitudes/backgrounds only afterward
(`src/hs_observatory/fields/malara.py:599–628,645–654`); this proportionality is
tested (`tests/test_malara_field.py:59–74`).

The Elsasser pipeline uses proton Alfvén conversion, aligns native time grids,
and subtracts each spacecraft's own full-record mean from velocity and magnetic
field (`src/hs_observatory/analysis/derived/elsasser.py:26–40,47–97`;
`src/hs_observatory/analysis/elsasser.py:286–296`). The truth velocity contract is
plasma-frame (`src/hs_observatory/analysis/elsasser.py:173–180`).

**Interpretation for these constant-density, shared-q, equal-cadence cases:** with
cA the constant nT-to-km/s conversion factor,

\[
\mathbf z_i^+(t)=K[\mathbf q(\mathbf r_i(t))-\bar{\mathbf q}_i],
\quad K=10+0.5c_A.
\]

For a pair from spacecraft i and j,

\[
\Delta\mathbf z^+=K[\mathbf q(\mathbf r_j(t_b))-
\mathbf q(\mathbf r_i(t_a))-\bar{\mathbf q}_j+\bar{\mathbf q}_i].
\]

For same-spacecraft pairs the means cancel exactly. For cross-spacecraft pairs,
the distinct finite-record means need not cancel. A single shared global mean
would cancel, and a constant K alone would change moment amplitudes, not
exponents. Therefore the processed/raw distinction here can be narrowed to a
specific finite-record mean effect, in addition to different spatial/directional
sampling and finite-width separation bins. This derivation should not be
extended to fluctuating density, independent v/B realizations, or mismatched
native time grids without revisiting those assumptions.

## 4. Existing saved direct-3D evidence

### Inventory and provenance

**Saved results inspected:**

- `outputs/direct-spatial-sf.h5`: 2,000 pairs × 25 separations × 3 p values;
  100,000 endpoint evaluations per p, 300,000 overall.
- `outputs/direct-spatial-sf-5000.h5`: 5,000 pairs × 25 separations × 3 p values;
  250,000 endpoint evaluations per p, 750,000 overall.
- Both have PNG/PDF companions named `_structure_functions`, `_local_slopes`,
  and `_comparison`. The documentation images
  `docs/assets/figures/intermittency-direct-spatial-{structure-functions,local-slopes,comparison}.png`
  accompany the 2k narrative at
  `docs/research-notes/intermittency-recovery.md:140–177`.

Both HDF5 files contain cases 0/1/2 for p=0.9/0.7/0.5, seed 12345, field seed
12345, realization 1, the 100,000 km cube, indices 2–14, h=α=1/3 and A0=100,000.
These were verified in `metadata_json`, `config_json` and `cases/*/field_config`.
The research notes document a 16-pair smoke calculation and cached refit
(`docs/research-notes/intermittency-recovery.md:118–132`), but those smoke HDF5
files were **not present** in the inspected output directory. The notes still
recommend increasing to 5k (`:179–188`); the saved 5k product shows that this has
already happened locally. Artifact existence is stronger evidence of execution
than that stale recommendation, but does not establish scientific adequacy.

For reproducibility of this audit snapshot, SHA-256 hashes of the two small
saved direct products are:

```text
direct-spatial-sf.h5
fbdd818535eae6d1cccf891a7ae2fc569e0e83ac63bf82b9b69e30131d57fd75
direct-spatial-sf-5000.h5
749e0b1ccc8a09ff700eb387681e054f4c311b4f8dfb12836d3e53c7254796bf
```

These hashes were calculated during read-only inspection. The original products
store source YAML and NumPy version but no evaluator implementation hash or
commit, so historical bitwise evaluator identity is not certified by their
metadata (`src/hs_observatory/analysis/spatial_structure_function.py:349–356`).

### What the pilots actually demonstrate

The following numbers are read from
`cases/*/q/scaling_exponents/raw[0,window,order]` and `cases/*/theory` in those
two files. The candidate window is 200–1,000 km and uses seven centers:
223.607, 287.015, 368.403, 472.871, 606.962, 779.078 and 1,000 km.

| p | ζ4 | ξ4, 2k | ξ4, 5k | ζ6 | ξ6, 2k | ξ6, 5k |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.33333 | 1.32357 | 1.32290 | 2.00000 | 1.99184 | 2.00597 |
| 0.7 | 1.28215 | 1.25696 | 1.30113 | 1.78588 | 1.76695 | 1.88191 |
| 0.9 | 1.12759 | 1.15175 | 1.31371 | 1.28630 | 1.32849 | 1.64948 |

The 2k p=0.9 ξ6 is only about 0.042 above theory; the 5k result is about 0.363
above. The latter is a roughly 24% increase in the measured fitted exponent.
The 100–1,300 km window gives p=0.9 ξ6=2.14708 at 2k and 1.78646 at 5k
(11 centers), showing substantial window dependence as well as budget dependence.
Candidate-window p=0.9 sixth-order R² rises from 0.69890 to 0.93958; that
improvement in fit appearance does not establish correct or converged scaling.

The saved moments and batches give more direct evidence of sampling sensitivity:

| p | Range of S6 change, 2k → 5k, across candidate centers | 5k S6 batch CV range | Largest absolute 75%-checkpoint difference from final S6, candidate centers |
|---|---:|---:|---:|
| 0.5 | −12.7% to +11.9% | 0.111–0.231 | 4.8% |
| 0.7 | −29.6% to +50.2% | 0.219–0.934 | 14.3% |
| 0.9 | −34.5% to +44.9% | 0.302–1.294 | 48.1% |

Dataset evidence: `cases/*/q/S_n[0,5,:]` in both files;
`cases/*/batch_cv[5,:]` and `cases/*/checkpoint_S_n[:,5,:]` in the 5k file.
The change is `S6_5000/S6_2000 - 1`; the checkpoint comparison is
`S6_3750/S6_5000 - 1`. These are reproducible, correlated prefix comparisons,
not independent replications or confidence intervals. Batch CV describes 625-pair
subsamples of one field. At approximately 472.871 km, p=0.9 S6 at 3,750 pairs
is 48.1% below its final 5k estimate. The full-sample local sixth-order slopes
at the seven candidate centers range from about 0.520 to 2.626
(`cases/0/local/slopes[0,5,:]`), rather than showing an established flat interval
at 1.286.

**Interpretation:** p=0.5 global fitted exponents are encouraging in this window,
but do not constitute a complete convergence study. p=0.7 sixth-order estimates
also retain material sensitivity. The prior near-theoretical p=0.9 global fit
was **not supported by adequately demonstrated convergence of high-order
moments**. Neither these data nor the diagnostic's existence establishes an
intrinsic field failure, a converged plateau, or spacecraft sampling as the
unique cause of discrepancies.

## 5. What the tests validate

Tests were read, not rerun for this report. Their assertions establish the
intended scope of coverage; no fresh pass claim is made here.

- `tests/test_malara.py:13–31`: compact basis, analytic derivative and boundary
  values. `:43–88`: deterministic parameters/realizations and periodic values.
  `:91–105`: finite-difference divergence on 27 diagnostic points with maximum
  scale 4 and injection 2, threshold 1e-4. This is useful local construction
  validation, not a global production-resolution divergence or isotropy audit.
- `tests/test_malara.py:108–115`: p=0.5 gives unit cascade transfer at selected
  cells. It does not assert measured S1–S6 exponents of the spatial field.
- `tests/test_malara_optimized_backends.py:21–109`: cached/parallel equivalence
  to direct values, cache lifecycle and periodicity on small fields. These protect
  numerical implementation equivalence rather than theoretical statistics.
- `tests/test_spatial_structure_function.py:67–131`: coordinate sampler isotropy,
  uniformity checks, reproducible prefixes, wrapping, arithmetic, batches and
  checkpoints. Most moment accounting uses a cheap synthetic sine evaluator.
  A small actual Malara case checks serial/task-parallel consistency (`:159–165`).
- The apparent “theoretical” regression check at
  `tests/test_spatial_structure_function.py:203–229` supplies **fabricated exact
  power laws** with exponents n/3 and p=0.5; it does not evaluate Malara fields.
  Likewise, `:266–288` fits fabricated HS power laws. These verify analysis
  machinery, not Malara intermittency.
- `tests/test_structure_function_local.py:35–39` checks ζn(0.5)=n/3 and ζ3(p)=1
  for the analytic formula. It does not compare that formula with measured
  direct-3D increments.

**No existing test was identified that measures adequately sampled actual Malara
3D structure functions and asserts agreement with analytic p-model exponents**
in the inspected Malara, direct-spatial, local-slope and structure-function tests.
No field-direction isotropy, intermittent cascade moment-spectrum, or converged
high-order spatial scaling acceptance test was found. The distinction matters:
verified estimators and deterministic generators do not automatically validate
the generator's finite-realization statistical scaling.

## 6. Applicability to 500a/502a and related existing diagnostics

### Same deterministic field parameters, different observation selection

**Saved metadata check:** after replacing p in the embedded source generation
YAML of the direct pilot, its complete `field` mapping equals the current
`generation/malara-ts-500a-t1500.yaml` and
`generation/malara-ts-502a-t1500.yaml` field mappings, respectively. Their existing
observation HDF5 `metadata/config_yaml` attributes also match their generation
YAMLs. Both observation products have truth velocity shape `(12001,9,3)`,
time endpoints 0 and 1,500 s, and n_mc=0. Only metadata, shape and the two time
endpoints were read for this check.

Thus the saved p=0.9 and p=0.5 direct results are **parameter-compatible with the
500a and 502a raw realizations**, under the current deterministic evaluator
contract. The direct pilot does not need to be discarded merely because its
source generation recipe is named prod-401 or has a different observation
trajectory/duration/error count. A field-only calculation does not use those
settings. This is parameter/provenance compatibility, subject to the missing
historical implementation fingerprint described above, not a new bytewise
field comparison.

### Exact centers and fitting windows can match; finite-width estimators cannot yet

`--field-config generation/malara-ts-500a-t1500.yaml --p 0.9 0.5` can select the
existing target field parameters without new field construction changes. The
CLI can import HS centers from a structure-function JSON or accept explicit
centers, and can use the same fit-window bounds and local-slope routine
(`src/hs_observatory/cli_spatial_structure_function.py:23–38,41–85`;
`src/hs_observatory/analysis/spatial_structure_function.py:252–268,437–469`).

However, **exact separation centers are not identical finite-width separation
bins**. Direct Sn(λcenter) does not average over the distribution of separations
or directions within an HS bin. `centers_from_hs` copies centers only; it does not
copy edges or construct a binned spatial estimator. Tests explicitly assert
that no fake direct bin edges are written
(`tests/test_spatial_structure_function.py:194–198`). This distinction persists
if all 40 geometric centers of the 70–20,000 km grid are supplied. Actual radial
and directional weights would also need to match for a strict estimator comparison.

The existing comparison YAML points to **prod-400/401/402**, not 500a/502a
(`analysis/malara-prod-combined_sf-publication.yaml:1–14`). Its older nine-center
HS grid has only two points in 200–1,000 km, so that fit is unavailable with the
three-point rule; four lie in 100–1,300 km. The saved direct files retain those
HS refits under `comparisons/*/fit`. This limitation is documented at
`docs/research-notes/intermittency-recovery.md:134–138`; it should not be mistaken
for a limitation of the newer 40-bin grid. New saved HS comparison inputs must
be explicitly selected rather than relabeling an old plot as a 500a/502a comparison.

There is also a cache limitation: cached direct postprocessing reloads the live
comparison YAML/products and replaces `comparisons`; missing comparison inputs
can drop a previously stored comparison, and changed inputs can change the
refit. Raw direct moments remain reusable, but comparison regeneration is not
fully self-contained (`src/hs_observatory/analysis/spatial_structure_function.py:
358–376,437–469,565–589`). No fix was made during this audit.

### Related machinery does not provide an independent field-only answer

The existing geometry-stratified duration/budget infrastructure operates on
stored spacecraft-time Elsasser samples, not uniformly sampled volume.
`structure_function_convergence.py` includes common-prefix multi-budget moments,
but its study driver specifically requires a 300 s, 2,401-time record and its
26-bin grid (`src/hs_observatory/analysis/structure_function_convergence.py:43–65,
127–164`). It is not a ready-to-run arbitrary-duration direct-spatial campaign.
The uniform-binning comparison has a 40-log-bin option, but likewise reads saved
HS truth and checks the original 300 s case
(`src/hs_observatory/analysis/structure_function_uniform_binning.py:37–49,114–128`).

The newer **Study A** diagnostics have already been run locally: the existing
`outputs/structure_function_moment_convergence/figures/summary.json` contains
15 rows per target case: five bins, seeds 12345/23456/34567, one requested budget
of 500,000. Across those rows, the largest 1% of increments contributes
**90.8–95.9% of S6 for 500a**, versus **27.5–29.4% for 502a**
(`rows[*].tail_contribution[2][1]`). These numbers were read from the saved
summary; this audit launched no Study A run. They establish concentration in
this sampled HS estimator, not uniform-volume concentration or independence of
events. Only one budget is present, so this snapshot does not demonstrate 1M/2M
budget convergence. Raw-amplitude curves, N_conc, midpoint blocks and lag-group
aggregation are implemented and tested separately
(`src/hs_observatory/analysis/structure_function_moment_convergence.py:97–191`;
`tests/test_structure_function_moment_convergence.py:21–123,155–186`). These
arithmetic routines are useful reuse candidates for future direct diagnostics,
but the current driver is explicitly coupled to HS truth processing.

The older PDF workflow supports full/perpendicular increments and separation
projections, but also requires local/global magnetic directions and uses
geometric-mean normalization (`src/hs_observatory/analysis/elsasser_increment_pdf_static.py:
136–150,322–383,562–590`). Its density normalizes over histogram counts inside
the saved finite range (`:708–728`); a percentile-selected upper edge can omit
rare tails. It cannot establish high-order convergence merely by looking smooth.
Although a low-level function accepts arrays of points, it is not a drop-in
uniform-volume raw-q PDF workflow with the same direct increment convention.
Flatness from S2/S4 is useful but does not resolve S6 convergence by itself.

Finally, `configs/malara_timeseries_fft.yaml:1–27` and the existing
`outputs/synthetic_observations_malara_observatory_fft.h5` concern a **time-series**
FFT of sampled observations/truth. Their existence is not evidence of an
isotropic 3D field spectrum. The evaluator benchmark measures performance and
backend agreement, not spatial spectra or high-order convergence
(`scripts/benchmark_malara_evaluator.py:191–280`).

## 7. Reuse, validation work, and missing functionality

### Existing functionality we can immediately reuse

1. Deterministic arbitrary-position evaluation, including the exact current
   500a/502a field parameters, periodicity, chunk processing and parallel tasks.
   The direct sampler already bypasses all trajectory/error generation.
2. Uniform-volume/isotropic exact-separation coordinates with shared p streams
   and reproducible budget prefixes; shared S1–S6, fits, local slopes, and theory.
3. Both existing direct HDF5 files for immediate moment/batch/checkpoint,
   fit-window and flatness analysis, with no evaluator calls. Refit/rendering
   interfaces and their source-protection tests already exist.
4. The recent Study A stable moment/tail/concentration arithmetic and saving
   conventions as reusable components, while recognizing that its sampling and
   temporal attribution driver is spacecraft-specific.

Sources and current coverage are itemized in sections 1, 2, 5 and 6.

### Existing functionality needing more validation or larger budgets

- Direct high-order moments, especially p=0.9: existing 2k→5k changes and 5k
  checkpoints are too large to claim convergence (section 4). More independent
  spatial sampling seeds are needed; a single nested prefix is insufficient.
- Apparent p=0.5 agreement: useful control, but validate both moment values and
  slope stability across windows/budgets, rather than accepting one ξ6≈2.
- Field isotropy/homogeneity and cascade-to-increment mapping: the coordinate
  sampler and local divergence tests do not validate these statistical claims
  (sections 3 and 5).
- Matched HS comparisons: specify the same realized field, quantity, windows
  and separation estimator, and quantify spacecraft mean offsets before
  attributing discrepancies solely to trajectory sampling (section 6).

### Missing functionality for convincing field-only intermittency validation

The missing items are integration/validation gaps, not requests to implement
another evaluator or duplicate the current direct sampler:

- A direct-spatial campaign collecting **increment-distribution/tail statistics**
  alongside the existing moments: Cn(x), top-event fractions, N_conc, full-tail
  PDFs, and spatial locations/eddy associations of dominant events. Old moment-only
  direct files cannot recover these. Temporal blocks are inappropriate for a
  static uniformly sampled field; spatial attribution is needed instead.
- A multi-seed, multi-budget direct workflow that retains checkpoints or resumes
  evaluated samples, preserves input/implementation fingerprints, and reports
  moment/tail stability before declaring a scaling interval. Existing separate
  invocations can perform the initial budget test without any new implementation.
- A direct finite-width-bin option with an explicit radial/directional sampling
  measure, if exact equality to the HS binned estimator is required. Matching
  centers alone is already supported but is a weaker comparison.
- Explicit validation of intermittent cascade coefficient moments separately from
  spatial increments, and directional/translated subvolume checks of the actual
  vector field. These would test intended statistical properties rather than
  infer them from a configuration label or curl construction.
- Reproducible statistical acceptance evidence for a resolved scaling interval:
  several interior scales, high-order moment stability, fit-window sensitivity,
  and multiple spatial seeds within a fixed realization. Only after that should
  independent field realizations test ensemble variability; new realizations are
  unnecessary for the first question about existing 500a/502a.

## 8. Minimal recommended sequence of future calculations

These are recommendations, **not commands executed by this audit**.

1. **Finish the cheap saved-product comparison first.** Use the existing 2k/5k
   moments and quarter checkpoints to tabulate S2/S4/S6, flatness and both local
   slope estimators on fixed windows. The tables above already expose the
   principal p=0.9 budget problem. Keep existing HS Study A concentration results
   alongside this table, with their different sampling measure clearly labeled.
2. **Single most useful next field calculation:** run a controlled **fixed-field,
   multi-budget, multi-spatial-seed direct-3D moment convergence experiment** for
   p=0.9 and p=0.5. Reuse the existing sampler and exact target field parameters;
   begin with 5k and 50k pairs/separation and at least three sampling seeds, then
   increase only if S4/S6 remain unstable. Restrict the first expensive step to
   the seven existing candidate-window separations listed in section 4, so the
   2k/5k pilots provide a directly comparable baseline. Each two-p, seven-scale,
   50k run evaluates 1.4 million endpoints per seed; it is substantial work and
   should be scheduled deliberately. Keep the field seed/realization fixed,
   output prefixes unique, and fit bounds fixed before seeing the result.
3. For that campaign, require empirical budget/seed stability of the individual
   moments before interpreting global exponents. Inspect quarter-prefix motion,
   batch variability and local-slope persistence; a visually plausible ξn or
   high R² is insufficient. Predeclare application-appropriate tolerances rather
   than choosing a window or stopping budget because it matches ζn. Existing
   sample and plot CLIs already support this stage, albeit with prefix replay.
4. If high-order sensitivity persists, integrate the existing stable tail kernels
   into the **existing** direct sampler to save distributions and compact spatial
   extremes. Establish whether rare volume locations dominate S4/S6 and whether
   additional pairs keep changing their contributions. This is a future extension,
   not something reconstructible from the old direct files.
5. Once direct estimates are adequately stable, extend the scale grid to match
   HS centers/windows and quantify the difference between exact-center and
   within-bin/directional weighting. Separately evaluate the finite-record
   spacecraft mean offsets in the raw-q→z-plus relation. If a stable raw-field
   discrepancy with ζn remains, prioritize coefficient-moment and directional
   field validation before generating new realizations or error ensembles.

For reference, the existing interfaces below suffice for steps 1–3. The plotting
example writes a separate product and evaluates no field; the sample example is
one future campaign member, not a complete convergence study:

```bash
# Saved-product refit; no field evaluation. Not run during this audit.
.venv/bin/hsobs-plot-spatial-structure-function outputs/direct-spatial-sf-5000.h5 \
  --fit-lambda-range 200 1000 --fit-window comparison 100 1300 \
  --output-prefix outputs/direct-spatial-sf-5000-audit-refit

# Future field-only calculation, explicitly expensive. Not run during this audit.
.venv/bin/hsobs-analyze-spatial-structure-function \
  configs/example_spatial_structure_function.yaml \
  --field-config generation/malara-ts-500a-t1500.yaml --p 0.9 0.5 \
  --lambda-centers-km 223.60679775 287.01471848 368.40314986 472.87080450 \
    606.96223100 779.07780805 1000 \
  --n-pairs 50000 --n-batches 8 --seed 23456 \
  --fit-lambda-range 200 1000 \
  --output-prefix outputs/direct-spatial-500a-502a-N50000-seed23456
```

The displayed centers are rounded for readability. For **identical** random
prefixes with old direct products, read their full-precision
`cases/*/lambda_bin_centers_km` values rather than retyping these decimals:
float64 separation bits key the per-scale RNG. Different sampling seeds should
be used for independent sampling repeats, not different field realizations.
The inherited comparison configuration still targets older prod-400/401/402
outputs; its automatic comparison panel is not a matched 500a/502a result.
