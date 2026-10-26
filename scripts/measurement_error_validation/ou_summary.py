"""Path-level, configured-variance OU covariance summary for a validation run."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .overview import MODEL_COLOR, MODEL_WIDTH, OBSERVED_COLOR


PANELS = (
    ("density", "Proton density OU validation"),
    ("velocity_speed", "Proton speed OU validation"),
    ("velocity_angle", "Proton velocity-angle OU validation"),
)


def build_ou_summary(records, config):
    """Plot existing path summaries; never load HDF5 or fit model parameters."""
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.88))
    #title = fig.suptitle("HIVE measurement-error OU validation summary", y=0.995, fontsize=15)
    #title.set_in_layout(False)
    handles = [
        Line2D([], [], color=OBSERVED_COLOR, marker="o", label="Empirical (95% CI)"),
        Line2D([], [], color=MODEL_COLOR, linewidth=MODEL_WIDTH, label="Configured OU model"),
        Line2D([], [], color=OBSERVED_COLOR, marker="o", label="α (95% CI)"),
        Line2D([], [], color="tab:purple", marker="s", linestyle="--", label="β (95% CI)"),
    ]
    for ax, entries in ((axes[0], handles[:2]), (axes[2], handles[2:])):
        ax.legend(
            handles=entries,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.85),
            fontsize=9,
            frameon=False,
        )
    for label, ax, (component, panel_title) in zip("abc", axes, PANELS):
        ax.set(title=panel_title, ylabel=r"$C(\tau) / \sigma_{\mathrm{config}}^2$", xlim=(-15, 915))
        ax.grid(alpha=0.18)
        ax.text(
            0.97,
            0.95,
            f"({label})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        section = config[component]
        selected = [r for r in records if r["kind"] == "ou_summary" and r["component"] == component]
        if not section["enabled"] or not selected:
            message = (
                "Disabled in configuration"
                if not section["enabled"]
                else "Saved OU samples unavailable"
            )
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center")
            continue
        lag_grid = np.linspace(0, 900, 500)
        ax.plot(
            lag_grid,
            np.exp(-lag_grid / section["ou_correlation_time_s"]),
            color=MODEL_COLOR,
            linewidth=MODEL_WIDTH,
        )
        notes = []
        if component != "velocity_angle" and section["model"] != "gaussian_fractional_bias_plus_ou":
            notes.append("Legacy log-error OU coordinates")
        if component == "velocity_angle" and len(selected) != 2:
            notes.append("One transverse coordinate unavailable")
        for record in selected:
            if record["zero_variance"]:
                notes.append("Configured OU variance is zero; normalization undefined")
                continue
            beta = record["record_id"] == "velocity_angle_beta"
            ax.errorbar(
                record["lag"],
                record["measured"],
                yerr=np.array(
                    [record["measured"] - record["lower"], record["upper"] - record["measured"]]
                ),
                color="tab:purple" if beta else OBSERVED_COLOR,
                marker="s" if beta else "o",
                linestyle="--" if beta else "-",
                linewidth=1,
                markersize=4,
                capsize=3,
            )
            if record["n_paths"] < 30:
                notes.append("95% CI unavailable: fewer than 30 independent paths")
        if notes:
            ax.text(
                0.02,
                0.03,
                "\n".join(dict.fromkeys(notes)),
                transform=ax.transAxes,
                va="bottom",
                fontsize=8,
            )
    axes[-1].set_xticks(np.arange(0, 901, 100))
    fig.supxlabel("Lag (s)", y=0.065)
    #fig.text(
    #    0.5,
    #    0.015,
    #    "Saved constant biases removed; covariance normalized by configured stationary OU variance.\n"
    #    "Marginal 95% Student-t intervals across MC/spacecraft paths (≥30 paths).\n"
    #    "Only lags resolved on the saved time grid are shown; timestamps are not independent draws.",
    #    ha="center",
    #    va="bottom",
    #    fontsize=8,
    #)
    return fig


def plot_ou_summary(records, config, output):
    """Save the companion figure alongside the overview and detailed diagnostics."""
    fig = build_ou_summary(records, config)
    names = []
    try:
        for extension in ("png", "pdf"):
            name = f"measurement_error_validation_ou_summary.{extension}"
            fig.savefig(output / name, dpi=150)
            names.append(name)
    finally:
        plt.close(fig)
    return names
