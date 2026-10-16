# Research & Development Roadmap

This roadmap connects scientific questions and software additions and improvements with people
interested in pursuing them. It is a place to discuss scope and record interest.
It is **not** a commitment to delivery dates or inclusion in a particular release.

The initial projects are proposed candidates grounded in existing capabilities and documented limitations.
The [Research Notes](research-notes/index.md) can be used to describe ongoing investigations and historical experiments.

Consider using these statuses:

- **Proposed:** an idea awaiting agreement on scope and whether to pursue it.
- **Seeking contributors:** a project agreed for pursuit that is inviting participation.
- **Active:** contributors have confirmed that work on the stated scope is underway.
- **Completed:** the stated scope has been delivered and its outcomes and validation linked.

Priority and contributers are **TBD** until maintainers agree on it.
Record names only with the person's consent.

## Project summary

| Project | Status | Priority | Scope | Interested contributors | Active contributors |
| --- | --- | --- | --- | --- | --- |
| [High-order intermittency convergence](#high-order-intermittency-convergence) | Proposed | TBD | Fixed-field sampling and spatial-tail diagnostics | TBD | TBD |
| [Timing-calibration validity and noise assumptions](#timing-calibration-validity-and-noise-assumptions) | Proposed | TBD | Controlled validation and instrument-model review | TBD | TBD |

## High-order intermittency convergence

**Motivation.** The intermittency investigation reports sensitivity of high-order
moments to rare increments. Agreement of a fitted exponent with p-model theory
does not establish moment convergence or a resolved scaling interval.

**Proposed work and outcomes.** Review saved checkpoints first, then assess the
documented 500k-pair extension and additional sampling seeds where warranted.
Report moment stability, fit-window sensitivity, and local slopes separately.
If tails remain unstable, investigate spatial concentration diagnostics for the
direct sampler, reusing Study A arithmetic where appropriate. Expected outcomes
are a reproducible convergence assessment and explicit limits on interpretation,
including negative or inconclusive findings. Independent field realizations are
a later scope decision; spatial sampling seeds within one field are not an
ensemble of field realizations.

**Existing capabilities and documentation.** The direct spatial sampler and
campaign runner already support fixed-field sampling, saved moments, checkpoints,
and reporting. See [Intermittency recovery](research-notes/intermittency-recovery.md),
the [direct-3D campaign](research-notes/sf/malara_direct3d_convergence.md), and
[Study A](research-notes/sf/structure_function_moment_convergence.md).
The notes describe the 500k extension as unfinished; confirm its current status
before scheduling work.

- **Status:** Proposed. **Priority:** TBD. **Scope:** fixed-field validation first;
  tail-diagnostic implementation and ensemble extensions require separate agreement.
- **Interested contributors:** TBD. **Active contributors:** TBD. **Issue:** TBD.


## Timing-calibration validity and noise assumptions

**Motivation.** ...

**Proposed work and outcomes.**  ...

**Existing capabilities and documentation.** Polyhedron, affine, and quadratic
calibration methods, geometry-dependent instrument-noise weights, and independent
resampled diagnostics already exist. See the
[timing reference](reference/timing-calibration.md) and
[timing investigations](research-notes/timing-investigations.md).

- **Status:** Proposed. **Priority:** TBD. **Scope:** validation and assumption review;
  algorithm changes are outside the initial scope.
- **Interested contributors:** TBD. **Active contributors:** TBD. **Issue:** TBD.



## Getting involved

Open or comment on a relevant [GitHub Issue](https://github.com/kgklein/hs-observatory/issues)
with the project name, your interests, relevant experience, and a proposed first
step. If no issue exists, propose a bounded scope and its scientific acceptance
criteria. Follow the repository's
[contribution guidance](https://github.com/kgklein/hs-observatory/blob/main/CONTRIBUTING.md),
especially before changing scientific algorithms or configuration formats.

With your consent, add your name or GitHub handle to **Interested contributors**
in both the summary table and project section. Interest is not an assignment.
Move to **Active contributors** and update the status only after confirming that
work is underway and agreeing on scope. Replace the issue's TBD field with its
actual link when one is established; do not invent issue numbers. On completion,
link the delivered work and validation, and update both table and section together.
