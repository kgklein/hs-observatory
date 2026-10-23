"""Machine-readable reports and bounded aggregate figures."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INTERPRETATIONS = [
    "PASS means consistency at the chosen familywise significance level, not proof of correctness.",
    "Primary statistical p-values use Holm correction within this input. Listed confidence intervals "
    "are marginal; they are not simultaneous confidence bands.",
    "Gaussian fixed-time tests use independent MC/spacecraft draws. OU path-summary Student "
    "intervals are approximate and require at least 30 paths; saved biases are subtracted, "
    "never empirical time means.",
    "Position requirements are radial confidence contours; 3-sigma conversion uses the implemented "
    "rounded Maxwell factor 3.762. Alignment coordinates are Gaussian rotation vectors, not Euler angles.",
    "Magnetic anchors are total-vector sigma-level ASD requirements, divided by sigma_level*sqrt(3) "
    "and converted pT to nT. PSDs are averaged before square roots; theoretical curves include "
    "finite-record zero DC, taper leakage and the even-N half-power Nyquist exception. "
    "Frequencies at/above Nyquist cannot establish higher-frequency requirements.",
    "No truth/observation application checks, field evaluation or high-cadence synthetic validation "
    "are performed. Separate products with the same seed must not be pooled as independent draws.",
]


def model_interpretations(config):
    notes = list(INTERPRETATIONS)
    for name in ("density", "velocity_speed"):
        s = config.get(name, {})
        if not s.get("enabled", False):
            continue
        if s.get("model") == "gaussian_fractional_bias_plus_ou":
            notes.append(
                f"{name}: Gaussian fractional bias plus OU; multiplier is 1+epsilon, "
                "with zero expected fractional mean. Nonpositive factors are rejected, "
                "not clipped/resampled; this is not a truncated-Gaussian model. "
                "Aborting appreciably probable invalid realizations can select the saved ensemble."
            )
        else:
            notes.append(
                f"{name}: Gaussian log bias plus OU; multiplier is exp(L), "
                "implying a positive expected fractional mean."
            )
    s = config.get("velocity_angle", {})
    if s.get("enabled", False):
        meaning = (
            "a total-angle Rayleigh confidence radius at erf(k/sqrt(2)) coverage"
            if s.get("model") == "transverse_ou_rotation_total_angle"
            else "a per-transverse-coordinate Gaussian sigma-level bound"
        )
        notes.append(
            f"Velocity-angle requirement is {meaning}. The latent total angle is Rayleigh; "
            "principal vector deflection wraps above pi. See the theoretical wrapping "
            "probability in the numerical results; no small-angle reinterpretation is applied."
        )
    return notes


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def plot_figures(records, output):
    """At most six distribution panels per component page; no per-spacecraft figures."""
    paths = []

    def save(fig, stem):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            name = stem + "." + extension
            fig.savefig(output / name, dpi=150)
            paths.append(name)
        plt.close(fig)

    for component in sorted({r["component"] for r in records}):
        distributions = [
            r for r in records if r["component"] == component and r["kind"] == "distribution"
        ]
        if component == "alignment":
            pages = [
                (
                    index,
                    f"{component}: {field}",
                    [r for r in distributions if r["title"].startswith(field)],
                )
                for index, field in enumerate(("velocity", "magnetic"), start=1)
            ]
        else:
            pages = [
                (start // 6 + 1, component, distributions[start : start + 6])
                for start in range(0, len(distributions), 6)
            ]
        for page, title, group in pages:
            if not group:
                continue
            fig, axes = plt.subplots(
                len(group), 2, figsize=(11, max(3, 2.6 * len(group))), squeeze=False
            )
            for row, record in enumerate(group):
                x = np.sort(record["values"])
                dist = record["distribution"]
                if not len(x):
                    continue
                left, right = axes[row]
                if np.std(x) > 0 and np.isfinite(dist.ppf(0.99)):
                    grid = np.linspace(
                        min(x.min(), dist.ppf(0.005)), max(x.max(), dist.ppf(0.995)), 200
                    )
                    left.hist(
                        x, bins=min(40, max(5, int(np.sqrt(len(x))))), density=True, alpha=0.5
                    )
                    left.plot(grid, dist.pdf(grid), label="configured density")
                    probs = (np.arange(len(x)) + 0.5) / len(x)
                    target = dist.ppf(probs)
                    right.plot(target, x, ".", markersize=2)
                    limits = [min(x.min(), target.min()), max(x.max(), target.max())]
                    right.plot(limits, limits, "k--", linewidth=1)
                else:
                    left.axvline(x[0])
                    right.plot(x, (np.arange(len(x)) + 1) / len(x))
                left.set_title(record["title"], fontsize=9)
                left.set_xlabel(record["units"])
                right.set_xlabel("Configured quantile")
                right.set_ylabel("Saved quantile")
            fig.suptitle(title + ": distributions and QQ comparisons", y=1.005)
            save(fig, f"{component}_distributions_{page}")
        acfs = [r for r in records if r["component"] == component and r["kind"] == "acf"]
        if acfs:
            fig, axes = plt.subplots(1, len(acfs), figsize=(6 * len(acfs), 4), squeeze=False)
            for ax, record in zip(axes[0], acfs):
                values = record["values"]
                if values:
                    x, measured, target, lo, hi = np.array(values, dtype=float).T
                    ax.plot(x, target, "k--", label="configured OU")
                    ax.plot(x, measured, "o", label="saved covariance / configured variance")
                    ax.vlines(x, lo, hi)
                ax.set(xlabel="Lag (s)", ylabel="ACF (known mean/variance)", title=record["title"])
                ax.legend(fontsize=7)
            save(fig, component + "_autocorrelation")
    for index, record in enumerate(r for r in records if r["kind"] == "spectrum"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        f = record["frequency"]
        mask = f > 0
        for axis, label in enumerate("xyz"):
            axes[0].loglog(f[mask], np.sqrt(record["psd"][axis, mask]), label=label, alpha=0.8)
            axes[0].fill_between(
                f[mask],
                np.sqrt(record["lower"][axis, mask]),
                np.sqrt(record["upper"][axis, mask]),
                alpha=0.1,
            )
        axes[0].loglog(
            f[mask], np.sqrt(record["target"][mask]), "k--", label="window-adjusted component"
        )
        axes[0].loglog(
            f[mask], np.sqrt(record["raw_target"][mask]), "k:", label="unwindowed requirement"
        )
        axes[1].loglog(f[mask], np.sqrt(record["psd"][:, mask].sum(axis=0)), label="saved total")
        axes[1].loglog(
            f[mask], np.sqrt(3 * record["target"][mask]), "k--", label="window-adjusted total"
        )
        for ax in axes:
            ax.set(xlabel="Frequency (Hz)", ylabel="ASD (nT / sqrt(Hz))")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2, which="both")
        fig.suptitle(record["title"] + f"; {record['n_paths']} paths; shaded marginal intervals")
        save(fig, f"magnetic_ac_asd_{index + 1}")
    return paths


def write_reports(output, payload, records):
    output.mkdir(parents=True, exist_ok=True)
    figures = plot_figures(records, output)
    payload["figures"] = figures
    interpretations = model_interpretations(
        payload.get("provenance", {}).get("embedded_configuration", {})
    )
    payload["interpretations"] = interpretations
    payload = clean(payload)
    (output / "validation_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    rows = payload["results"]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "validation_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# HIVE measurement-error validation — Phase A",
        "",
        f"Overall status: **{payload['status']}**; exit code {payload['exit_code']}.",
        "",
        f"Input: `{payload['provenance']['input']}`",
        "",
        "PASS indicates statistical consistency, not proof of correctness.",
        "",
        "## Component summary",
        "",
        "| Component | PASS | FAIL | INSUFFICIENT_DATA | SKIPPED | INCOMPLETE_INPUT |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for component in sorted({r["component"] for r in rows}):
        counts = [
            sum(r["component"] == component and r["status"] == status for r in rows)
            for status in ("PASS", "FAIL", "INSUFFICIENT_DATA", "SKIPPED", "INCOMPLETE_INPUT")
        ]
        lines.append("| " + component + " | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "## Configuration cross-check", ""]
    discrepancies = payload.get("discrepancies", [])
    if discrepancies:
        lines += [
            "Embedded parameters were used. Configuration discrepancies make this run unsuccessful.",
            "",
        ]
        lines += [
            f"- `{d['parameter']}`: supplied `{d['supplied']}`, embedded `{d['embedded']}`."
            for d in discrepancies
        ]
    else:
        lines.append(
            "No discrepancies detected"
            if payload.get("configuration_checked")
            else "Configuration cross-check unavailable."
        )
    lines += ["", "## Findings requiring attention", ""]
    attention = [r for r in rows if r["status"] not in ("PASS",) and r["primary"]]
    lines += [
        f"- **{r['status']}** `{r['component']}.{r['statistic']}`: {r['note']}" for r in attention
    ]
    if not attention:
        lines.append("No primary statistical failures or coverage gaps.")
    lines += ["", "## Interpretation and limits", ""] + ["- " + s for s in interpretations]
    lines += [
        "",
        "Plot distributions retain at most 4,096 values; correlated OU plot samples use up to "
        "256 points from each of the first eight selected paths. The recorded seed makes selection reproducible. "
        "Statistical tests use the independent samples/path summaries described in CSV/JSON, not plotting samples.",
        "",
        "Detailed numerical results, effect sizes, counts and marginal intervals: "
        "[CSV](validation_results.csv) · [JSON](validation_results.json).",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- [{name}]({name})" for name in figures if name.endswith(".png")]
    (output / "validation_report.md").write_text("\n".join(lines) + "\n")
