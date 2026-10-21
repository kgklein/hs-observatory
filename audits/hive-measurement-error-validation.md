# HIVE measurement-error validation: Stage 1 audit

Audit date: 2026-10-21. Source revision: `6b41b2b` (working tree also contains unrelated changes). Investigation only: read source/tests/configurations, opened HDF5 read-only, inspected metadata/shapes and bounded density/observation/offset slices. No tests, generators, production calculations, or Stage 2 diagnostics were run; no scientific code or products were changed. Findings describe implemented behavior, not certification against an external instrument specification.

## Findings and configuration scope

All seven `generation/malara-prod-{611,621,631,641,651,661,671}.yaml` files specify 0–6000 s, 0.125 s cadence for every quantity, seed 12345, 50 MC realizations, spacecraft-frame velocity, and `write_errors: true`. The time-axis implementation includes both endpoints: **48,001 samples**, not 48,000. Enabled components are respectively timing, position, density, speed, angle, DC+AC, and alignment. Expected spacecraft count is nine; diagnostics must read IDs rather than hard-code it.

The main interpretation issues are: radial position confidence differs from component sigma; density/speed requirements set **log-error** sigma rather than exact fractional-error sigma; velocity-angle sigma applies to **each** transverse coordinate; alignment uses Gaussian rotation-vector coordinates, not Euler angles; magnetic ASD is divided by both sigma level and sqrt(3). These distinctions must appear in numerical summaries.

| Error source | Existing coverage | Saved HDF5 data | Proposed statistic | Expected result | Missing capability |
| ------------ | ----------------- | --------------- | ------------------ | --------------- | ------------------ |
| Timing | Large-draw sigma test; deterministic coordinate application tests | `/errors/time_offset` | Mean, variance, Gaussian QQ; shifted-coordinate checks | Gaussian sigma 4.166667 ms, constant per MC/spacecraft | Shifted field truth/coordinates are not explicitly saved; full application check needs field values |
| Position | Large-draw 2σ/3σ radial quantile tests; deterministic coordinate tests | `/errors/position_offset` | Component covariance, Maxwell CDF, direction moments | Component sigma 0.882511 km; P(radius≤2.5 km)=0.9545 | No explicit sampled field truth; no existing isotropy campaign |
| Density | Pooled log-variance test at default 3σ; bias shape | `/errors/density_bias`, `/errors/density_log_error`, `/errors/n` | Bias/OU variance, innovations, ACF; observed/truth ratio | Log sigma 0.03; equal variance split; tau 300 s | No observatory-specific correlation/decomposition test or standalone report |
| Velocity speed | Pooled log-variance test at default 3σ; bias shape | `/errors/velocity_speed_bias`, `/errors/velocity_speed_log_error`, `/errors/v` | Log speed ratio, bias/OU variances, ACF, direction | Log sigma 0.015; equal split; tau 300 s | Same correlation/decomposition gap; frame-aware comparison needed |
| Velocity angle | Magnitude preservation, combined with alignment | `/errors/velocity_angle_alpha`, `/errors/velocity_angle_beta` | Component OU tests, Rayleigh angle CDF, tangent reconstruction, norm residual | Each component sigma 0.125°; total RMS 0.176777° | No distribution/OU validation of this angular implementation |
| Magnetic DC | Sigma conversion plus independent NumPy normal draws | `/errors/magnetic_dc_offset`, `/errors/B` | Per-axis Gaussian moments; residual time constancy | Sigma 0.05 nT per axis | Existing statistical test bypasses application routine |
| Magnetic AC | Averaged periodogram median ratio over 0.05–8 Hz | `/errors/magnetic_ac_noise` | Welch PSD/ASD, total/component normalization, integrated power | Component ASD = anchor/(3 sqrt(3)×1000) | No anchor-resolved Welch campaign; 10 Hz inaccessible at 8 Hz sampling |
| Alignment | Rotation/norm invariants, velocity norm test; serialization | B/V `/errors/*_alignment_rotvec`, `*_alignment_matrix` | Component moments, Maxwell rotation norm, Rodrigues/application residuals | Independent B/V rotvec coordinates N(0,(1°)²); constant matrices | No distribution, independence, or full B application statistical campaign |

## Source and existing-test map

Paths below are relative to the repository root.

- `src/hs_observatory/errors/observatory.py`: defaults/normalization at lines 19/118; scalar sigma conversion 174; radial conversion 189–232; exact stationary OU 235; Rodrigues rotation 279; local transverse basis/rotation 314–382; variance partition and lognormal application 394–429; ASD interpolation/conversion/synthesis 432–520; periodogram helper 523–547; measurement application 566 onward.
- `src/hs_observatory/errors/offsets.py`: Gaussian clock/position draws 35/57; factories 73/101 honor enabled flags and requirement conversion. A position `radial_probability` override supersedes sigma-level contour selection (absent in these YAMLs).
- `src/hs_observatory/generate.py`: inclusive grids 48; trajectory interpolation/extrapolation 86; velocity interpolation 152; frame truth 293–315; coordinate draws and shifted field evaluation 337–395; measurement application 397; provenance/frame metadata 479; writer call 539. Timing uses `t'=t+delta_t`, `r'=interpolate(r,t')+delta_r`; endpoint positions extrapolate with endpoint velocities or secants. Shifted spacecraft velocity interpolates with endpoint clamping. A configured reference spacecraft has zero clock offset; none is specified here.
- `src/hs_observatory/io/hdf5.py:21`: writer, gzip level 4 and automatic chunking. `read_observations:90` eagerly loads arrays: **do not use it** for the standalone validator.
- `src/hs_observatory/fields/malara.py:630`: quantity evaluation; density is constant, velocity/B sample the static spatial field. Clock errors affect this field through spacecraft motion, not intrinsic field evolution.
- `tests/test_observatory_errors.py:25,33,43`: genuine large-draw statistical timing sigma and position radial quantiles (20,000 and 120,000 draws). Lines 62/98 test conversions/aliases/rejections, not distributions. Line 126 tests pooled density/speed log variance to 15%, using default 3σ and tau 1 s, not these 2σ/300 s configurations; it does not test bias/OU split or correlation. Line 153 tests norm preservation, not angular/alignment distributions. Line 183 tests DC conversion with NumPy draws (not production application), and 48 averaged 4096-point AC periodograms at dt=0.03125 s; only a median ASD ratio within 20% is asserted. It does not resolve individual anchors, low-frequency plateau, component equality, or Nyquist behavior. Lines 213/220 check baseline algebra and small sine-field HDF5 serialization.
- `tests/test_errors.py`: seeded reproducibility, shapes and zero errors; the constant-offset test has no time dimension and does not independently verify pipeline constancy. `tests/test_coordinate_sampling_errors.py:7,72` verify shifted sine-field sampling, including motion in a static field: valuable deterministic application checks, not statistical validation. `tests/test_velocity_frames.py` verifies frame subtraction, metadata, and simple additive error bookkeeping. `tests/test_hdf5_roundtrip.py` checks schema/shapes/compression/round trip.
- `tests/test_ornstein_uhlenbeck.py` statistically tests stationarity, regular/irregular lag correlations and directional RMS for **different legacy models** in `errors/ornstein_uhlenbeck.py`. Those use a separate OU implementation; their vector model divides angular sigma by sqrt(2) and clamps relative magnitude at zero. They do not establish the observatory model's angular or multiplicative conventions.

## Mathematical targets and quantitative checks

### Timing and position

Clock draws are independent `delta_t[m,s] ~ N(0,(0.0125/3)²)` seconds. Check Gaussian mean/variance and QQ, cross-spacecraft covariance, and single-draw reuse across quantities. Time constancy is encoded structurally by the absence of a time dimension; verifying actual reuse requires application checks, not repeating the scalar as artificial samples. Pairwise clock differences have sigma sqrt(2) times the single-clock sigma, absent a fixed reference.

Position components are independent N(0,a²), where `a=2.5/q_chi3(0.9545)=0.8825114287 km`. Radius R/a is chi with 3 degrees of freedom (Maxwell):

`F_R(r)=erf(r/(sqrt(2)a))-sqrt(2/pi)(r/a) exp(-r²/(2a²))`.

Thus `E[R]=2a sqrt(2/pi)`, `E[R²]=3a²`, `Var(R)=(3-8/pi)a²`. Check all three component variances, off-diagonal covariance, radial CDF/coverage, and directions d: E[d]=0, E[dd^T]=I/3; a fixed-axis cosine is uniform on [-1,1]. Position is constant per series, not jitter. Baseline errors are differences of spacecraft offsets: component variance 2a² for distinct spacecraft, and baselines sharing spacecraft are correlated. The 3σ helper uses the rounded factor 3.762 for 99.73%; do not use 2.5/2 or 2.5/3 as component sigma.

### Density and velocity speed

For either component, `L=b+u`, `b~N(0,s_b²)`, stationary OU `u`, independent of b, and multiplier `M=exp(L)`. The normalized configured variance fractions set `s_b=s_total/sqrt(2)` and `s_u=s_total/sqrt(2)` here. Density has `s_total=0.06/2=0.03`; speed has `s_total=0.03/2=0.015`. Their component sigmas are 0.0212132 and 0.0106066 respectively. Fractions are normalized by their sum in code; they need not initially sum to one.

OU transitions are `u_i=phi_i u_(i-1)+s_u sqrt(1-phi_i²) z_i`, `phi_i=exp(-delta_t_i/300)`, with stationary Gaussian initialization (no burn-in). Covariance is `s_u² exp(-|h|/300)`; full log-error covariance is `s_b²+s_u² exp(-|h|/300)`, so full L autocorrelation has a 0.5 long-lag floor. At cadence 0.125 s, phi=0.9995834201.

Reconstruct u as saved L minus saved bias, not L minus its finite-series mean. Test bias variance across MC/spacecraft, OU variance/ACF at several lags including 30/300/600 s, independence of b and u, and standardized innovations (mean 0, variance 1, no serial correlation). Compare empirical L marginal variance with s_total². For fractional error f=M−1:

`E[f]=exp(s_total²/2)-1`, `Var(f)=exp(s_total²)(exp(s_total²)-1)`;
`q_p(f)=exp(s_total Phi^-1(p))-1`.

Expected fractional mean is about 0.000450101 for density and 0.000112506 for speed. There is no mean-centering correction `−s_total²/2`; positive mean bias is intended by this implementation. The nominal central 2σ interval is `[exp(-requirement_fraction)-1, exp(requirement_fraction)-1]`, not symmetric ±6%/±3%. Fractional covariance is `exp(s_total²)[exp(C_L(h))-1]`.

For 631 compare `log(observed/n / truth/n)` directly to L (truth is 5 cm^-3). For 641 compare `log(norm(observed/v)/norm(truth/v_spacecraft_frame))` to L and check direction preservation. Mask zero denominators explicitly. Rotations preserve speed even in combined products, but coordinate offsets change the relevant sampled velocity there.

### Velocity angle

The observatory model draws independent stationary OU alpha and beta, each with `s=deg2rad(0.25/2)` and tau=300 s. For unit input direction u and code's local basis e1,e2, `w=alpha e1+beta e2`, `theta=norm(w)`, and

`v_out=norm(v_in)[cos(theta)u + sin(theta)w/theta]`.

The latent theta is Rayleigh(scale s), **not Gaussian**: E[theta²]=2s², E[theta]=s sqrt(pi/2), `P(theta≤r)=1−exp(-r²/(2s²))`. RMS is 0.1767767°, 95.45% quantile 0.3107472°; only `1−exp(-2)=86.4665%` lies within 0.25°. If the requirement meant a 95.45% total-angle radius, the implementation does not encode that interpretation. Observed principal angle is exactly `acos(cos(theta))`; Rayleigh applies directly at these tiny angles with negligible wrapping probability.

Test alpha/beta variances, zero cross-correlation, stationary start and OU innovations; angle CDF; speed preservation; and full vector reconstruction in 651. Use atan2(norm(cross),dot) for measured angles. Longitudinal residual is `norm(v)(cos(theta)−1)` and transverse norm `norm(v) abs(sin(theta))`. Recover signed tangent coordinates from the direction change and compare with saved alpha/beta. The deterministic helper-axis switch at |u_x|=0.9 means Cartesian perturbation ACF need not itself be an OU ACF as the input direction moves; test the saved local coordinates.

### Magnetic DC and AC

DC is one independent N(0,0.05² nT²) offset per axis/MC/spacecraft, reused over time. In 661 verify `B_obs=B_truth+dc+ac`; its time-average residual should recover dc because the synthesized AC series has exactly zero mean up to roundoff. Check per-axis Gaussian distributions and off-diagonal covariance. Separate the stored AC before testing DC constancy.

Let A(f) be total-vector requirement ASD in pT/sqrt(Hz). Implemented A is 1500 below 0.001 Hz (a **flat low-frequency plateau**), `1500(f/0.001)^(-2/3)` between 0.001 and 1 Hz, and `15 f^(-1)` between 1 and 10 Hz, continuing that slope above 10. Per-axis one-sigma ASD is `a(f)=A(f)/(3 sqrt(3)×1000)` nT/sqrt(Hz). Component anchor values are 0.2886751, 0.002886751, 0.0002886751 nT/sqrt(Hz); quadrature total one-sigma ASD is A/3000. Components are independent Gaussian series with equal PSD; “uniform” denotes equal component allocation, not uniform random amplitudes. `component_distribution` and most measurement `model` labels are descriptive configuration fields rather than alternate algorithm dispatch.

For interior rFFT bins, synthesis sets real and imaginary coefficient variances to `N a(f_k)²/(4dt)`; consequently the usual one-sided periodogram has expectation a². DC coefficient is forced to zero. **For even N**, the real Nyquist coefficient variance is `N a(f_Nyq)²/(2dt)`: with the undoubled endpoint periodogram this gives **a²/2**, not a². The docstring's normalization claim therefore has an endpoint exception worth testing. These seven configurations have odd N=48,001 and no exact Nyquist bin; the exception does not affect them. Existing AC test uses even N but excludes Nyquist. No error model change is proposed in Stage 1.

At dt=0.125, fs=8 Hz and Nyquist=4 Hz: the 10 Hz anchor and >10 Hz extrapolation cannot be validated from production data. Fundamental frequency is 8/48001≈0.000166663 Hz, allowing limited resolution around the 0.001 Hz knee. Use full-record/long-window ensemble spectra near that knee and shorter Welch windows at high frequencies. Compare **averaged PSDs**, then square-root for ASD; averaging ASDs introduces bias. For Welch compare to the target PSD convolved with the actual window response (including detrending), or use suitable band integrals away from knees/edges. Check three components separately, cross-spectra, quadrature total, zero mean, and Parseval/integrated power. Finite-record expected variance is `sum(interior a_k²)/(N dt)` plus `a_Nyq²/(2Ndt)` for even N; DC is absent. Spectral synthesis is periodic and band limited, not an aliased observation of a continuous process above Nyquist. Small error-only samples at higher cadence are needed to exercise the 10 Hz anchor and both odd/even endpoint cases.

### Alignment

For each MC/spacecraft, velocity and B have independent Gaussian rotvecs w with coordinate sigma `deg2rad(3/3)=1°`. R=Rodrigues(w) is constant over all times; `norm(w)` is Maxwell(scale 1°), RMS sqrt(3)°. These are rotation-vector coordinates, not independently composed Euler rotations. Test coordinate variances/covariances, radial CDF, B/V independence, `R^T R=I`, det(R)=1 and saved-R/Rodrigues agreement.

For a fixed vector direction u and axis k=w/norm(w), the exact deflection obeys `cos(delta)=cos(theta)+(1−cos(theta))(k dot u)²`. At small angles delta is approximately Rayleigh(scale 1°), RMS sqrt(2)°; use the exact mapping for acceptance calibration. Constant rotation does **not** imply constant observed deflection when truth direction changes. Check `v_obs=R_v v_truth_spacecraft` and `B_obs=R_B B_truth` throughout selected chunks in 671, plus norm preservation. In combined products B is rotated **after** adding dc/ac; velocity is rotated after speed and direction errors. A single constant-direction vector cannot determine rotation about itself; saved matrices remove that identifiability problem.

## Saved products and exact schema

Final inspection after the user confirmed 601's download: **`outputs/production/malara-prod-601.h5` opens successfully**, size 3,352,336,155 bytes. It has all eight errors enabled, all 18 error datasets listed below, M=50, S=9, T=48,001 and time endpoints 0/6000 s. Its normalized requirements match the seven isolated configurations. Bounded observation/offset reads succeeded; this is schema inspection, not a full integrity scan or statistical validation. An initial truncated-file error was resolved by download completion.

At the final target-file inspection, `outputs/production/malara-prod-611.h5` had appeared but was **still truncated** (106,954,752 bytes available versus HDF5 stored EOF 1,000,838,039). It was not usable. No 621–671 products were found under `outputs/`, including `outputs/production/`; no isolated targets were at the YAMLs' declared top-level output paths. Download status is transient: recheck read-only before Stage 2; do not classify incomplete inputs as failed scientific tests.

Other inspected products:

| Products | Actual content and usefulness |
| --- | --- |
| `outputs/malara-prod-{100,101,102}.h5` | All eight enabled, M=50/S=9/T=2401, 0–300 s, 0.125 s cadence; position uses 3σ radial contour, other levels match the campaign. Useful for latent distributions/application bookkeeping, not long-lag or 0.001 Hz resolution. |
| `outputs/malara-prod-{400,401,402}.h5` | Same dimensions/duration, all eight enabled; position uses 2σ, so requirements match the campaign. Full schema/metadata inspected in 402. |
| `outputs/malara/synthetic_observations_malara.h5` | Legacy scalar/vector OU, M=10; only aggregate n/v/B and coordinate errors. Does not validate observatory lognormal/angular definitions. |
| `outputs/example/synthetic_observations.h5` | Legacy Gaussian models, M=10; useful schema comparison only. |
| `outputs/malara-ts-500c-t600.h5` | Truth-only, M=0, T=4801, no realized error datasets; cannot test error distributions. |

All six older production products and 601 record seed 12345. Do not pool these as independent realizations: common seed and matching draw order can reuse latent draws. They provide different field contexts, not automatically more independent errors.

The following schema is directly observed in 601 and 402 and is the current writer's expected schema for the isolated files. Let M be MC count, S spacecraft count, T_q time count. Scalar/vector data are float64; IDs are HDF5 strings. Except IDs, datasets are gzip-compressed/chunked. Read slices/chunks with `h5py.File(...,'r')`.

| Exact dataset paths | Shape | Physical units; actual unit attribute |
| --- | --- | --- |
| `/errors/time_offset` | (M,S) | s; `units=s` |
| `/errors/position_offset` | (M,S,3) | km; `units=km` |
| `/errors/density_bias` | (M,S) | Dimensionless log bias; no attribute |
| `/errors/density_log_error` | (M,T_n,S) | Dimensionless log error; no attribute |
| `/errors/velocity_speed_bias` | (M,S) | Dimensionless log bias; no attribute |
| `/errors/velocity_speed_log_error` | (M,T_v,S) | Dimensionless log error; no attribute |
| `/errors/velocity_angle_alpha`, `/errors/velocity_angle_beta` | (M,T_v,S) each | rad; `units=rad` |
| `/errors/magnetic_dc_offset` | (M,S,3) | nT; `units=nT` |
| `/errors/magnetic_ac_noise` | (M,T_B,S,3) | nT; `units=nT` |
| `/errors/velocity_alignment_rotvec`, `/errors/magnetic_alignment_rotvec` | (M,S,3) each | rad; `units=rad` |
| `/errors/velocity_alignment_matrix`, `/errors/magnetic_alignment_matrix` | (M,S,3,3) each | Dimensionless; no attribute |
| `/errors/n`, `/errors/v`, `/errors/B` | (M,T_n,S), (M,T_v,S,3), (M,T_B,S,3) | cm^-3, km/s, nT; no attributes |
| `/observed/n`, `/observed/v`, `/observed/B` | Same shapes as aggregate errors | cm^-3, km/s, nT; no attributes |
| `/truth/n`, `/truth/v`, `/truth/B` | (T_n,S), (T_v,S,3), (T_B,S,3) | cm^-3, plasma-frame km/s, nT; no attributes |
| `/truth/v_spacecraft_frame` | (T_v,S,3) | Spacecraft-frame km/s; no attribute |
| `/time/n`, `/time/v`, `/time/B` | (T_q,) each | s; no attributes |
| `/spacecraft/r/n`, `/spacecraft/r/v`, `/spacecraft/r/B` | (S,T_q,3) each | Nominal positions in km; no attributes |
| `/spacecraft/id` | (S,) | Identifiers |

Disabled measurement components still have latent datasets filled with zero (alignment matrices identity). The presence of a dataset does not imply its component is enabled. No separate OU dataset is stored: subtract bias from log_error. No per-MC shifted time, position, or sampled-field-truth dataset is saved. Spacecraft velocity is not stored separately, but nominal velocity can be inferred from plasma minus spacecraft-frame velocity truth.

Relevant **attributes of `/metadata`**, not datasets: `config_yaml`, `config_path`, `observatory_error_config` (normalized JSON), `observatory_error_model`, `error_models` (JSON), `random_seed`, `n_mc`, `schema_version=2`, `created_utc`, `field_model`, `field_metadata`, `spacecraft_source` (JSON), `velocity_truth_frame`, `velocity_observed_frame`, `velocity_error_applied_frame`, `timing_sampling_policy`; 601 also has `mc_generation_enabled`. Read provenance from each product rather than assuming the current YAML generated it. 601/402 have `field_metadata={}` and no source commit provenance. Their spacecraft metadata records a CDF nearest-record snapshot selected at 2030-08-25T11:50:35.801 for requested 12:00; this is useful trajectory provenance and not a measurement timing perturbation.

## Independent verification boundaries

1. **Entirely saved latent data:** every distribution, OU decomposition, rotation, and in-band AC statistic above is available in 601 and older observatory products. This verifies generated perturbations without Malara evaluation. For 611/621, offset distributions are likewise sufficient once complete, but their arrays alone cannot prove application at every time.
2. **Independent truth/observation application checks:** 631/641/651/661/671 disable coordinate errors, so nominal truth (with the correct velocity frame) is the independent input. Reconstruct all intermediate operations from saved latent errors and compare against observations. In 601/older combined products, constant density still allows a truly independent density application check against 5 cm^-3. A bounded eight-sample 402 density bookkeeping check had zero floating residual; no population statistics were evaluated.
3. **Bookkeeping reconstruction:** `sampled_q=observed_q−errors_q` recovers hidden coordinate-shifted input because aggregate errors are defined relative to that input, **not nominal truth**. Reapply exp(L), local angular rotation, DC/AC, and final alignment to this recovered input. This tests consistency among saved fields, but is partly circular and cannot independently establish correct coordinate sampling. Comparing `observed−truth` to `/errors/q` is generally wrong when coordinate errors are enabled.
4. **Small synthetic error-only samples:** needed to extend AC tests through 10 Hz/above, exercise even/odd FFT endpoints, and obtain enough independent draws for precise rare-tail checks. Also recommended for observatory OU innovations, constant/different-direction angular inputs and exact alignment mapping. They need no Malara or trajectory data; use cheap array inputs and component helpers only. Deterministic sine-field coordinate tests already establish basic shifted-sampling semantics; additional small synthetic boundary tests could cover trajectory endpoint handling. These are Stage 2 proposals, not work performed here.
5. **Field reevaluation:** independently establishing actual B/v coordinate application in 611/621 or combined 601 requires evaluating the Malara field at selected shifted positions and comparing to the recovered sampled input. Nominal truth sampled along a trajectory is insufficient to determine an arbitrary displaced 3D field exactly; interpolation along time is only approximate. Full reevaluation is expensive and unnecessary for uncertainty distributions. Keep it outside the default campaign; a future explicit sparse check could limit cost. Constant density cannot detect coordinate errors. No regeneration of observations is needed or authorized by this audit.

## Acceptance criteria with finite ensembles

Use exact-model statistical intervals, report effect sizes and uncertainty, and distinguish deterministic identity failures from sampling excursions. For nine spacecraft there are 450 independent scalar constant draws per component, not 50×48,001×9. Per-spacecraft checks have only 50. For Gaussian mean use SE=sigma/sqrt(K); for centered variance use `(K−1)s_hat²/sigma² ~ chi-square(K−1)`. At K=450 relative SE of standard deviation is about 3.34%; at K=50 about 10.1%. Check Gaussianity/QQ in addition to moments. Use binomial intervals for radial/tail coverage and calibrated CDF tests; 450 draws yield only about 20.5 expected exceedances at 95.45% coverage and 1.2 at 99.73%. Precise 3σ quantiles require supplementary independent synthetic draws, not a tight production-tail tolerance.

OU time samples are correlated. With T≈6000 s and tau=300 s, one path has about T/(2tau)=10 effective samples for a mean, versus 48,001 timestamps. Squared zero-mean OU has correlation exp(−2h/tau), giving roughly T/tau=20 effective samples for its second moment. Full L includes a permanent bias and must be handled hierarchically. In the old 300 s files, a path spans only one tau: temporal mean subtraction severely distorts variance/ACF. Prefer saved bias subtraction and exact transition innovations. Innovations are independent under the model, but estimating parameters introduces fitting uncertainty; do not convert a tiny lag-one correlation discrepancy into an overconfident tau estimate. Use multiple physical lags, covariance-based likelihood or parametric calibration with the actual cadence/duration, and resample independent MC/spacecraft paths (MC clusters if checking cross-spacecraft relationships). Do not flatten path boundaries as adjacent time samples. For fixed-time cross-sectional comparisons, the 450 paths remain independent.

For AC interior-bin averaged periodograms, PSD/target approximately follows chi-square(nu)/nu with nu=2K for K independent series at one bin. Welch window overlap/frequency averaging changes effective degrees of freedom; account for those correlations or calibrate with cheap error-only simulations passed through the identical estimator. Include detrending/window leakage near the low-frequency knee and endpoint differences. Use simultaneous bands or predeclared band-power tests; do not require every noisy spectral bin to be within a fixed percentage. Bootstrap independent series, not individual frequency/time samples indiscriminately.

Proposed gate: predeclare primary statistics with familywise alpha 0.01 (Holm adjustment or calibrated joint bounds), and mark exploratory QQ/ACF/spectral excursions separately. Use precision- and scale-aware numerical tolerances for application/orthogonality/zero-mean identities; report max and RMS residuals. The CLI should distinguish pass, fail, insufficient data/bandwidth, incomplete input and skipped checks, with K/effective sample information. Independent simulations for calibration should not merely call the same conversion helper as the scientific expectation oracle: implement the formulas above separately, then compare to production helpers.

## Recommended Stage 2 order and proposed interface

1. Read-only input inventory, normalized provenance, enabled components, dimensions/units/frames and download-completeness checks; streamed JSON/CSV summary. Start with completed 601; inspect isolated files as they arrive.
2. Constant timing/position/DC/alignment distributions and matrix invariants; independent application identities on isolated products.
3. Density/speed bias–OU decomposition, innovations/physical-lag ACF and exact lognormal fractional targets; angular component OU/Rayleigh and vector reconstruction.
4. AC component/total Welch and full-record low-frequency spectra, Parseval and confidence bands; separate inexpensive high-cadence/endpoints samples.
5. Optional future sparse Malara coordinate-application investigation, clearly separated from default error-distribution validation.

Proposed, **not implemented**, standalone script interface:

```bash
python scripts/validate_measurement_errors.py saved \
  --inputs outputs/production/malara-prod-601.h5 \
  --configs generation/malara-prod-601.yaml \
  --components all --mc all --spacecraft all \
  --chunk-times 4096 --confidence 0.99 --seed 20261021 \
  --output-dir outputs/error_validation/601

python scripts/validate_measurement_errors.py saved \
  --inputs 'outputs/production/malara-prod-6[1-7]1.h5' \
  --config-dir generation --allow-missing \
  --output-dir outputs/error_validation/isolated

python scripts/validate_measurement_errors.py synthetic \
  --config generation/malara-prod-661.yaml --components magnetic_ac \
  --duration-s 6000 --cadence-s 0.03125 --n-realizations 32 \
  --seed 20261021 --output-dir outputs/error_validation/synthetic_ac
```

The saved mode should resolve globs itself, check both declared/top-level and production paths, pair by product provenance, report mismatches, read one MC/spacecraft series or bounded chunks at a time, and never import/instantiate field evaluators. Synthetic mode must be explicit and generate error-only inputs, with bounded batching and documented sample counts. Add configurable Welch segment/overlap/window parameters and even/odd sample cases. Output `summary.json`, a flat statistics CSV, provenance/estimator settings, and PNG/PDF QQ/CDF, decomposition/ACF, rotation residual, and ASD plots under `outputs/error_validation/`; save confidence intervals and skipped reasons alongside numerical targets. This campaign should remain independent of standard pytest and must not write to source HDF5 files.
