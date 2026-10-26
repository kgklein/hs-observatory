"""Compact configured-versus-saved Phase A overview; no fitted distributions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy import special, stats

from .model import angular_sigma, position_sigma, split_sigma


OBSERVED_COLOR = "tab:blue"
MODEL_COLOR = "tab:orange"
COMPONENT_COLORS = ("tab:red", "tab:green", "tab:purple")
TOTAL_ASD_COLOR = "0.15"
HIST_ALPHA = 0.4
MODEL_WIDTH = 1.5
SIGMA_STYLES = {2: ("0.55", "--"), 3: ("0.3", "-.")}


PANELS = (
    ("timing", "Timing knowledge", "Timing error (ms)", "ms⁻¹"),
    ("position", "Spacecraft-position knowledge", "Radial position error (km)", "km⁻¹"),
    ("density", "Proton density errors", "Total fractional density error δn/n", "fraction⁻¹"),
    ("velocity_speed", "Proton speed errors", "Total fractional speed error δV/V", "fraction⁻¹"),
    (
        "velocity_angle",
        "Proton velocity angular precision",
        "Total transverse angle (deg)",
        "deg⁻¹",
    ),
    ("magnetic_dc", "DC magnetic-field offsets", "Component offset (nT)", "nT⁻¹"),
    ("magnetic_ac", "AC magnetic-field noise", "Frequency (Hz)", ""),
    (
        "alignment",
        "FC–MAG relative alignment (FC reference)",
        "Rotation-vector component (deg)",
        "deg⁻¹",
    ),
)


def sigma_bounds(component, section, sigma, distribution):
    """Statistical reference bounds in display units, with no model refitting."""
    bounds = {}
    for level in (2, 3):
        if component in ("position", "velocity_angle"):
            bounds[level] = [distribution.ppf(special.erf(level / np.sqrt(2))) if sigma else 0.0]
        else:
            values = np.array([-level * sigma, level * sigma])
            if (
                component in ("density", "velocity_speed")
                and section["model"] != "gaussian_fractional_bias_plus_ou"
            ):
                values = np.expm1(values)
            bounds[level] = values
    return bounds


def distribution_panel(ax, samples, distribution, bounds, units, *, zero=False):
    """Render one or three histograms with a known PDF and shared physical bins."""
    all_values = np.concatenate(samples)
    limits = [all_values.min(), all_values.max(), *bounds[2], *bounds[3]]
    if not zero:
        limits.extend(distribution.ppf([0.001, 0.999]))
    lo, hi = min(limits), max(limits)
    padding = (hi - lo) * 0.04 if hi > lo else max(abs(lo) * 0.04, 0.01)
    ax.set_xlim(lo - padding, hi + padding)
    bins = np.linspace(
        lo - padding, hi + padding, min(40, max(5, int(np.sqrt(len(samples[0]))))) + 1
    )
    colors = (OBSERVED_COLOR,) if len(samples) == 1 else COMPONENT_COLORS
    for index, (values, color) in enumerate(zip(samples, colors)):
        label = "Saved" if len(samples) == 1 else "xyz"[index]
        ax.hist(values, bins=bins, density=True, color=color, alpha=HIST_ALPHA, label=label)
    if zero:
        ax.axvline(0, color=MODEL_COLOR, linewidth=MODEL_WIDTH, label="Configured point mass")
    else:
        grid = np.linspace(lo - padding, hi + padding, 600)
        ax.plot(
            grid,
            distribution.pdf(grid),
            color=MODEL_COLOR,
            linewidth=MODEL_WIDTH,
            label="Configured PDF",
        )
    for level, (color, linestyle) in SIGMA_STYLES.items():
        for bound in dict.fromkeys(bounds[level]):
            ax.axvline(bound, color=color, linestyle=linestyle, linewidth=0.9, label=f"{level}σ")
    ax.set_ylabel(f"Probability density ({units})")


def build_overview(records, config):
    """Build eight fixed panels from bounded validator records and embedded settings."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.065, 1, 0.87))
    #title = fig.suptitle("HIVE measurement-error validation summary", fontsize=15, y=0.995)
    #title.set_in_layout(False)
    handles = [
        Patch(facecolor=OBSERVED_COLOR, alpha=HIST_ALPHA, label="Saved realizations"),
        Line2D([], [], color=MODEL_COLOR, linewidth=MODEL_WIDTH, label="Configured model"),
        *[
            Line2D([], [], color=color, linestyle=style, linewidth=0.9, label=f"{level}σ")
            for level, (color, style) in SIGMA_STYLES.items()
        ],
        Line2D([], [], color=TOTAL_ASD_COLOR, linewidth=MODEL_WIDTH, label="Saved total ASD"),
        *[
            Patch(facecolor=color, alpha=HIST_ALPHA, label=axis)
            for axis, color in zip("xyz", COMPONENT_COLORS)
        ],
    ]
    legend = fig.legend(
        handles=handles[:4],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=4,
        fontsize=14,
        frameon=False,
        columnspacing=1.3,
    )
    legend.set_in_layout(False)
    axes.flat[5].legend(
        handles=handles[5:],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.85),
        fontsize=8,
        frameon=False,
    )
    axes.flat[6].legend(
        handles=handles[4:5],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.85),
        fontsize=8,
        frameon=False,
    )
    for label, ax, (component, title, xlabel, units) in zip("abcdefgh", axes.flat, PANELS):
        ax.text(
            0.97,
            0.96,
            f"({label})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=5,
        )
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=0.18)
        section = config[component]
        message = None
        if not section["enabled"]:
            message = "Disabled in configuration"
        elif component == "alignment" and section["model"] != "relative_constant_rotation_B_to_V":
            message = "Relative alignment unavailable (legacy independent model)"
        if message:
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", wrap=True, fontsize=9)
            continue
        available = [r for r in records if r["component"] == component]
        if component == "magnetic_ac":
            spectra = [r for r in available if r["kind"] == "spectrum"]
            if spectra:
                record = max(spectra, key=lambda r: len(r["frequency"]))
                f = record["frequency"]
                observed = np.sqrt(record["psd"].sum(axis=0))
                target = np.sqrt(3 * record["target"])
                mask = (f > 0) & (observed > 0) & (target > 0)
                if np.any(mask):
                    ax.loglog(
                        f[mask],
                        observed[mask],
                        color=TOTAL_ASD_COLOR,
                        linewidth=MODEL_WIDTH,
                        label="Saved total ASD",
                    )
                    ax.loglog(
                        f[mask],
                        target[mask],
                        color=MODEL_COLOR,
                        linewidth=MODEL_WIDTH,
                        label="Configured total ASD",
                    )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Zero ASD; log scale unavailable",
                        transform=ax.transAxes,
                        ha="center",
                    )
                ax.set_ylabel("Total ASD (nT / √Hz)")
            else:
                ax.text(0.5, 0.5, "Saved spectrum unavailable", transform=ax.transAxes, ha="center")
        else:
            scale = 1.0
            if component == "timing":
                ids = ["offset.0"]
                scale = 1000.0
                requirement = section["requirement_ms"]
                sigma = requirement / section["sigma_level"]
                dist = stats.norm(scale=sigma)
                if section.get("reference_spacecraft") is not None:
                    ax.text(
                        0.02,
                        0.96,
                        "Fixed reference clock excluded",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
            elif component == "position":
                ids = ["radius"]
                sigma, _ = position_sigma(section)
                dist = stats.maxwell(scale=sigma)
                if "radial_probability" in section:
                    ax.text(
                        0.02,
                        0.96,
                        "Requirement uses configured radial coverage",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
            elif component in ("density", "velocity_speed"):
                ids = ["initial_full"]
                sigma, _, _ = split_sigma(section)
                dist = (
                    stats.norm(scale=sigma)
                    if section["model"] == "gaussian_fractional_bias_plus_ou"
                    else stats.lognorm(s=sigma, loc=-1, scale=1)
                )
            elif component == "velocity_angle":
                ids = ["total_angle"]
                scale = 180 / np.pi
                sigma = angular_sigma(section) * scale
                dist = stats.rayleigh(scale=sigma)
                if section["model"] != "transverse_ou_rotation_total_angle":
                    ax.text(
                        0.02,
                        0.96,
                        "Requirement is legacy per-coordinate; lines show total-angle coverage",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
            else:
                ids = [
                    f"{'relative' if component == 'alignment' else 'offset'}.{i}" for i in range(3)
                ]
                scale = 180 / np.pi if component == "alignment" else 1.0
                requirement = section[
                    "requirement_deg_per_axis"
                    if component == "alignment"
                    else "requirement_nT_per_axis"
                ]
                sigma = requirement / section["sigma_level"]
                dist = stats.norm(scale=sigma)
            selected = [
                next((r for r in available if r.get("record_id") == key), None) for key in ids
            ]
            if all(r is not None and len(r["values"]) for r in selected):
                samples = [np.asarray(r["values"]) * scale for r in selected]
                if (
                    component in ("density", "velocity_speed")
                    and section["model"] != "gaussian_fractional_bias_plus_ou"
                ):
                    samples = [np.expm1(values) for values in samples]
                    ax.text(
                        0.02,
                        0.96,
                        "Legacy log model: fractional bounds are asymmetric",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=8,
                    )
                bounds = sigma_bounds(component, section, sigma, dist)
                distribution_panel(ax, samples, dist, bounds, units, zero=sigma == 0)
            else:
                ax.text(0.5, 0.5, "Saved samples unavailable", transform=ax.transAxes, ha="center")
    #fig.text(
    #    0.5,
    #    0.015,
    #    "Density, speed and angle: first timestamp across selected MC/spacecraft paths.\n"
    #    "2σ / 3σ denote model confidence levels; the other level is not an additional requirement.\n"
    #    "Total ASD: configured sigma-level normalization; finite-record and Hann-window effects.",
    #    ha="center",
    #    va="bottom",
    #    fontsize=9,
    #)
    return fig


def plot_overview(records, config, output):
    """Save the overview in both formats; return report-relative filenames."""
    fig = build_overview(records, config)
    names = []
    try:
        for extension in ("png", "pdf"):
            name = f"measurement_error_summary_overview.{extension}"
            fig.savefig(output / name, dpi=150)
            names.append(name)
    finally:
        plt.close(fig)
    return names
