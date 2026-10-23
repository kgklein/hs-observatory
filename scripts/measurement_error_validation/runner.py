"""Bounded, read-only HDF5 diagnostics. No plasma-quantity application checks."""

from itertools import combinations
import json
import math

import numpy as np
from scipy import signal, stats

from .model import (
    DEFAULTS,
    angular_sigma,
    component_asd,
    differences,
    normalize,
    position_sigma,
    rodrigues,
    spectral_target,
    split_sigma,
)
from .statistics import Results


class InvalidInput(ValueError):
    pass


def selection(text, available, *, numeric=False):
    if text == "all":
        return list(range(len(available)))
    tokens = text.split(",")
    try:
        result = [int(t) if numeric else available.index(t) for t in tokens]
    except (ValueError, TypeError) as exc:
        raise InvalidInput(f"invalid selection {text!r}; available={available}") from exc
    if (
        not result
        or len(set(result)) != len(result)
        or any(i < 0 or i >= len(available) for i in result)
    ):
        raise InvalidInput(f"empty, duplicate or out-of-range selection: {text}")
    return result


def read_series(dataset, mc, spacecraft, chunk, axis=None):
    """Never materialize a dataset's MC/spacecraft ensemble."""
    output = np.empty(dataset.shape[1], dtype=float)
    for start in range(0, len(output), chunk):
        stop = min(start + chunk, len(output))
        key = (mc, slice(start, stop), spacecraft)
        if axis is not None:
            key += (axis,)
        output[start:stop] = dataset[key]
    if not np.all(np.isfinite(output)):
        raise InvalidInput(f"nonfinite saved errors in {dataset.name}")
    return output


def time_axis(handle, quantity, chunk):
    path = f"time/{quantity}"
    if path not in handle or handle[path].ndim != 1:
        raise InvalidInput(f"missing/invalid {path}")
    dataset = handle[path]
    t = np.empty(dataset.shape[0])
    for start in range(0, len(t), chunk):
        t[start : start + chunk] = dataset[start : start + chunk]
    if len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise InvalidInput(f"{path} needs >=2 finite increasing samples")
    return t


class Validator:
    def __init__(self, handle, supplied, args):
        self.h = handle
        self.args = args
        self.results = Results(args.confidence)
        self.plots = []
        self.rng = np.random.default_rng(args.seed)
        if "metadata" not in handle or "spacecraft/id" not in handle:
            raise InvalidInput("missing metadata or spacecraft/id")
        self.meta = dict(handle["metadata"].attrs)
        try:
            self.cfg = normalize(json.loads(self.meta["observatory_error_config"]))
            self.mc_count = int(self.meta["n_mc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidInput(f"invalid embedded observatory configuration: {exc}") from exc
        if self.mc_count < 1:
            raise InvalidInput("no saved MC errors")
        self.ids = list(handle["spacecraft/id"].asstr()[:])
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise InvalidInput("spacecraft IDs must be nonempty and unique")
        self.mcs = selection(args.mc, list(range(self.mc_count)), numeric=True)
        self.scs = selection(args.spacecraft, self.ids)
        self.paths = [(m, s) for m in self.mcs for s in self.scs]
        self.k = len(self.paths)
        self.times = {q: time_axis(handle, q, args.chunk_times) for q in ("n", "v", "B")}
        self.discrepancies = differences(normalize(supplied.get("errors", {})), self.cfg, "errors")
        for key in ("n_mc", "random_seed"):
            configured = supplied.get("errors", {}).get(key, 1 if key == "n_mc" else None)
            embedded = self.meta.get(key)
            if embedded == "null":
                embedded = None
            self.discrepancies += differences(configured, embedded, f"errors.{key}")
        for q, t in self.times.items():
            spec = supplied.get("time", {})
            cadence = spec.get("cadence_s")
            cadence = cadence.get(q) if isinstance(cadence, dict) else cadence
            if cadence is None or not np.isfinite(cadence) or cadence <= 0:
                raise InvalidInput(f"supplied YAML requires positive cadence for {q}")
            actual = dict(start=float(t[0]), stop=float(t[-1]), cadence=float(t[1] - t[0]))
            desired = dict(
                start=spec.get("t_start_s", 0.0), stop=spec.get("t_stop_s"), cadence=cadence
            )
            for name, value in desired.items():
                if value is None or not np.isclose(actual[name], value, rtol=1e-10, atol=1e-10):
                    self.discrepancies.append(
                        dict(parameter=f"time.{q}.{name}", supplied=value, embedded=actual[name])
                    )
            if not np.allclose(np.diff(t), actual["cadence"], rtol=1e-10, atol=1e-10):
                # Irregular OU transitions are supported, but cadence provenance must say so.
                self.discrepancies.append(
                    dict(parameter=f"time.{q}.uniform", supplied=True, embedded=False)
                )
        if self.cfg["timing"].get("reference_spacecraft") not in (None, *self.ids):
            raise InvalidInput("embedded reference_spacecraft not found in IDs")

    def dataset(self, component, name, shape):
        path = "errors/" + name
        if path not in self.h:
            self.results.unavailable(component, name, f"missing /{path}", "SKIPPED")
            return None
        ds = self.h[path]
        if ds.shape != shape:
            raise InvalidInput(f"/{path} shape {ds.shape}; expected {shape}")
        units = {
            "time_offset": "s",
            "position_offset": "km",
            "magnetic_dc_offset": "nT",
            "magnetic_ac_noise": "nT",
            "velocity_angle_alpha": "rad",
            "velocity_angle_beta": "rad",
            "velocity_alignment_rotvec": "rad",
            "magnetic_alignment_rotvec": "rad",
        }
        if name in units:
            actual = ds.attrs.get("units")
            if actual != units[name]:
                self.results.add(
                    component,
                    name + ".units",
                    status="FAIL",
                    note=f"expected {units[name]!r}, found {actual!r}",
                )
        if component in ("density", "velocity_speed", "velocity_angle"):
            section = self.cfg[component]
            if "model" in ds.attrs and ds.attrs["model"] != section["model"]:
                self.results.add(
                    component,
                    name + ".model",
                    status="FAIL",
                    note="dataset model contradicts embedded configuration",
                )
            if component == "velocity_angle":
                key, meaning = "requirement_interpretation", section["requirement_interpretation"]
            else:
                key = "interpretation"
                meaning = section["error_interpretation"] + (
                    "_bias" if name.endswith("_bias") else ""
                )
                if (
                    section["error_interpretation"] == "fractional_error"
                    and ds.attrs.get("units") != "1"
                ):
                    self.results.add(
                        component,
                        name + ".units",
                        status="FAIL",
                        note="fractional errors require units='1'",
                    )
            if key in ds.attrs and ds.attrs[key] != meaning:
                self.results.add(
                    component,
                    name + ".interpretation",
                    status="FAIL",
                    note="dataset interpretation contradicts embedded configuration",
                )
        return ds

    def constant(self, component, name, vector=False, matrix=False):
        tail = (3, 3) if matrix else (3,) if vector else ()
        ds = self.dataset(component, name, (self.mc_count, len(self.ids), *tail))
        if ds is None:
            return None
        values = np.array([ds[m, s] for m, s in self.paths])
        if not np.all(np.isfinite(values)):
            raise InvalidInput(f"nonfinite saved errors in {ds.name}")
        return values

    def draw_plot(self, component, title, values, distribution, units):
        values = np.asarray(values).ravel()
        if len(values) > 4096:
            values = self.rng.choice(values, size=4096, replace=False)
        self.plots.append(
            dict(
                kind="distribution",
                component=component,
                title=title,
                values=values,
                distribution=distribution,
                units=units,
            )
        )

    def gaussian_vector(self, component, name, values, sigma, units, spacecraft=True):
        if values is None:
            return
        x = values.reshape(self.k, -1)
        for axis in range(x.shape[1]):
            self.results.gaussian(component, f"{name}.{axis}", x[:, axis], sigma, units)
            self.draw_plot(
                component, f"{name} axis {axis}", x[:, axis], stats.norm(scale=sigma), units
            )
        if sigma == 0:
            return
        for a, b in combinations(range(x.shape[1]), 2):
            self.results.correlation(
                component, f"{name}.axis_correlation_{a}_{b}", x[:, a], x[:, b]
            )
            self.results.path_mean(
                component,
                f"{name}.covariance_{a}_{b}",
                x[:, a] * x[:, b],
                0.0,
                units + "^2",
                note="known-zero-mean component covariance",
            )
        if spacecraft:
            y = x.reshape(len(self.mcs), len(self.scs), -1)
            for a, b in combinations(range(len(self.scs)), 2):
                for axis in range(x.shape[1]):
                    self.results.correlation(
                        component,
                        f"{name}.spacecraft_{self.ids[self.scs[a]]}_{self.ids[self.scs[b]]}.{axis}",
                        y[:, a, axis],
                        y[:, b, axis],
                    )

    def radial(self, component, name, vectors, sigma):
        if vectors is None or sigma == 0:
            return
        r = np.linalg.norm(vectors, axis=-1)
        self.results.distribution(component, name, r, stats.maxwell(scale=sigma))
        self.results.path_mean(component, name + ".squared_radius", r * r, 3 * sigma * sigma)
        self.draw_plot(
            component,
            name,
            r,
            stats.maxwell(scale=sigma),
            "km" if component == "position" else "rad",
        )

    def constants(self):
        if self.cfg["timing"]["enabled"]:
            values = self.constant("timing", "time_offset")
            if values is not None:
                sigma = (
                    self.cfg["timing"]["requirement_ms"] / self.cfg["timing"]["sigma_level"] * 1e-3
                )
                reference = self.cfg["timing"].get("reference_spacecraft")
                if reference is None:
                    self.gaussian_vector("timing", "offset", values, sigma, "s")
                else:
                    mask = np.array([self.ids[s] != reference for _, s in self.paths])
                    self.results.identity(
                        "timing",
                        "reference_zero",
                        np.max(np.abs(values[~mask]), initial=0),
                        1e-14,
                        int((~mask).sum()),
                    )
                    self.results.gaussian("timing", "offset_nonreference", values[mask], sigma, "s")
                    y = values.reshape(len(self.mcs), len(self.scs))
                    for a, b in combinations(range(len(self.scs)), 2):
                        if reference not in (self.ids[self.scs[a]], self.ids[self.scs[b]]):
                            self.results.correlation(
                                "timing", f"spacecraft_{a}_{b}", y[:, a], y[:, b]
                            )
        if self.cfg["position"]["enabled"]:
            x = self.constant("position", "position_offset", vector=True)
            sigma, probability = position_sigma(self.cfg["position"])
            self.gaussian_vector("position", "offset", x, sigma, "km")
            self.radial("position", "radius", x, sigma)
            if x is not None and sigma > 0:
                r = np.linalg.norm(x, axis=-1)
                self.results.coverage(
                    "position",
                    "radial_coverage",
                    r,
                    self.cfg["position"]["requirement_magnitude_km"],
                    probability,
                )
                d = np.divide(x, r[:, None], out=np.zeros_like(x), where=r[:, None] > 0)
                for axis in range(3):
                    self.results.path_mean("position", f"direction_mean.{axis}", d[:, axis], 0)
                    self.results.distribution(
                        "position", f"direction_cosine.{axis}", d[:, axis], stats.uniform(-1, 2)
                    )
                    for b in range(axis, 3):
                        self.results.path_mean(
                            "position",
                            f"direction_second_moment.{axis}_{b}",
                            d[:, axis] * d[:, b],
                            1 / 3 if axis == b else 0,
                        )
        if self.cfg["magnetic_dc"]["enabled"]:
            x = self.constant("magnetic_dc", "magnetic_dc_offset", vector=True)
            s = self.cfg["magnetic_dc"]
            self.gaussian_vector(
                "magnetic_dc", "offset", x, s["requirement_nT_per_axis"] / s["sigma_level"], "nT"
            )
        if self.cfg["alignment"]["enabled"]:
            self.alignment()

    def alignment(self):
        section = self.cfg["alignment"]
        sigma = np.deg2rad(section["requirement_deg_per_axis"] / section["sigma_level"])
        rotations = []
        for field in ("velocity", "magnetic"):
            vec = self.constant("alignment", field + "_alignment_rotvec", vector=True)
            mat = self.constant("alignment", field + "_alignment_matrix", matrix=True)
            rotations.append(vec)
            self.gaussian_vector("alignment", field, vec, sigma, "rad")
            self.radial("alignment", field + ".magnitude", vec, sigma)
            if mat is not None:
                eye_res = np.max(np.abs(np.swapaxes(mat, -1, -2) @ mat - np.eye(3)))
                self.results.identity("alignment", field + ".orthogonality", eye_res, 1e-12, self.k)
                self.results.identity(
                    "alignment",
                    field + ".determinant",
                    np.max(np.abs(np.linalg.det(mat) - 1)),
                    1e-12,
                    self.k,
                )
            if mat is not None and vec is not None:
                expected = np.array([rodrigues(v) for v in vec])
                self.results.identity(
                    "alignment", field + ".rodrigues", np.max(np.abs(mat - expected)), 1e-12, self.k
                )
        if all(v is not None for v in rotations) and sigma > 0:
            for a in range(3):
                for b in range(3):
                    self.results.correlation(
                        "alignment",
                        f"B_V_independence.{a}_{b}",
                        rotations[0][:, a],
                        rotations[1][:, b],
                    )

    def ou(
        self,
        component,
        dataset_name,
        time,
        sigma,
        tau,
        *,
        biases=None,
        total_sigma=None,
        fractional=False,
    ):
        ds = self.dataset(component, dataset_name, (self.mc_count, len(time), len(self.ids)))
        if ds is None:
            return None
        n = len(time)
        phi = np.exp(-np.diff(time) / tau)
        innovation_scale = sigma * np.sqrt(-np.expm1(-2 * np.diff(time) / tau))
        lags = {lag: int(np.argmin(np.abs(time - time[0] - lag))) for lag in (30.0, 300.0, 600.0)}
        lags = {
            lag: step
            for lag, step in lags.items()
            if step > 0
            and np.isclose(time[step] - time[0], lag, rtol=0, atol=1e-7)
            and np.allclose(time[step:] - time[:-step], lag, rtol=0, atol=1e-7)
        }
        summaries = {"mean": [], "second_moment": [], "fractional_mean": [], "full_second": []}
        covariance = {lag: [] for lag in lags}
        full_covariance = {lag: [] for lag in lags}
        innovation_correlations = {lag: [] for lag in (1, 2, 10)}
        first, first_full, samples = [], [], []
        innovation_sum = innovation_square = 0.0
        max_zero = 0.0
        invalid_factors = 0
        for index, (mc, sc) in enumerate(self.paths):
            full = read_series(ds, mc, sc, self.args.chunk_times)
            bias = 0.0 if biases is None else biases[index]
            u = full - bias
            first.append(u[0])
            first_full.append(full[0])
            summaries["mean"].append(float(u.mean()))
            summaries["second_moment"].append(float(np.mean(u * u)))
            summaries["full_second"].append(float(np.mean(full * full)))
            summaries["fractional_mean"].append(
                float(np.mean(full if fractional else np.expm1(full)))
            )
            if fractional:
                invalid_factors += int(np.count_nonzero(1.0 + full <= 0))
            max_zero = max(max_zero, float(np.max(np.abs(u))))
            if index < 8:
                take = np.sort(self.rng.choice(n, size=min(n, 256), replace=False))
                samples.extend(u[take])
            for lag, step in lags.items():
                covariance[lag].append(float(np.mean(u[:-step] * u[step:])))
                full_covariance[lag].append(float(np.mean(full[:-step] * full[step:])))
            if sigma > 0:
                z = (u[1:] - phi * u[:-1]) / innovation_scale
                innovation_sum += float(z.sum())
                innovation_square += float(np.dot(z, z))
                for lag in innovation_correlations:
                    if len(z) > lag:
                        innovation_correlations[lag].append(float(np.mean(z[:-lag] * z[lag:])))
        label = dataset_name
        units = (
            "rad" if component == "velocity_angle" else "fraction" if fractional else "log fraction"
        )
        self.results.gaussian(component, label + ".initial_ou", first, sigma, units)
        if biases is not None:
            self.results.gaussian(
                component,
                label + (".initial_fractional" if fractional else ".initial_log"),
                first_full,
                total_sigma,
                units,
            )
            if fractional:
                self.results.identity(
                    component, label + ".nonpositive_factor_count", invalid_factors, 0, self.k, n
                )
                self.results.coverage(
                    component,
                    label + ".symmetric_requirement_coverage",
                    np.abs(first_full),
                    self.cfg[component]["requirement_fraction"],
                    math.erf(self.cfg[component]["sigma_level"] / math.sqrt(2))
                    if total_sigma > 0
                    else 1.0,
                )
            self.draw_plot(
                component,
                label + " (initial independent paths)",
                first_full,
                stats.norm(scale=total_sigma),
                units,
            )
            self.results.path_mean(
                component,
                label + (".fractional_second_moment" if fractional else ".log_second_moment"),
                summaries["full_second"],
                total_sigma**2,
                units + "^2",
                n,
            )
            if total_sigma > 0:
                self.results.path_mean(
                    component,
                    label + ".bias_variance_fraction",
                    biases**2 / total_sigma**2,
                    (total_sigma**2 - sigma**2) / total_sigma**2,
                    n_time=1,
                    note="known-zero-mean bias second moment / configured total variance",
                )
                self.results.path_mean(
                    component,
                    label + ".ou_variance_fraction",
                    np.asarray(summaries["second_moment"]) / total_sigma**2,
                    sigma**2 / total_sigma**2,
                    n_time=n,
                    note="known-zero-mean OU second moment / configured total variance",
                )
            self.results.path_mean(
                component,
                label + ".fractional_mean",
                summaries["fractional_mean"],
                0.0 if fractional else np.expm1(total_sigma**2 / 2),
                "fraction",
                n,
            )
            if np.std(biases) > 0 and sigma > 0:
                self.results.correlation(component, label + ".bias_ou_independence", biases, first)
        if sigma == 0:
            self.results.identity(component, label + ".zero_ou", max_zero, 1e-14, self.k, n, units)
            return np.asarray(first_full)
        self.results.path_mean(component, label + ".ou_mean", summaries["mean"], 0, units, n)
        self.results.path_mean(
            component,
            label + ".ou_variance",
            summaries["second_moment"],
            sigma**2,
            units + "^2",
            n,
            "zero-known-mean second moment; no empirical time demeaning",
        )
        lag_plot = []
        for lag in (30.0, 300.0, 600.0):
            if lag not in covariance:
                self.results.unavailable(
                    component, label + f".acf_{lag:g}s", "lag not resolvable on saved time grid"
                )
                continue
            target = np.exp(-lag / tau)
            row = self.results.path_mean(
                component,
                label + f".acf_{lag:g}s",
                np.array(covariance[lag]) / sigma**2,
                target,
                n_time=n - lags[lag],
                note="known-mean covariance normalized by configured OU variance",
            )
            lag_plot.append((lag, row["measured"], target, row["ci_low"], row["ci_high"]))
            if biases is not None:
                self.results.path_mean(
                    component,
                    label
                    + (
                        f".full_fractional_covariance_{lag:g}s"
                        if fractional
                        else f".full_log_covariance_{lag:g}s"
                    ),
                    full_covariance[lag],
                    total_sigma**2 - sigma**2 + sigma**2 * target,
                    units + "^2",
                    n - lags[lag],
                    "constant bias creates a nonzero long-lag covariance floor",
                )
        count = self.k * (n - 1)
        if count < 2:
            self.results.unavailable(component, label + ".innovations", "requires >=2 innovations")
            return np.asarray(first_full)
        mean = innovation_sum / count
        variance = max(0, (innovation_square - count * mean**2) / (count - 1))
        alpha = 1 - self.args.confidence
        zcrit = stats.norm.ppf(1 - alpha / 2)
        r = self.results.add(
            component,
            label + ".innovations.mean",
            expected=0.0,
            measured=mean,
            ci=(mean - zcrit / np.sqrt(count), mean + zcrit / np.sqrt(count)),
            p=float(2 * stats.norm.sf(abs(mean) * np.sqrt(count))),
            n_paths=self.k,
            n_time=n - 1,
            method="exact known-parameter Gaussian innovation mean",
            note=f"independent innovations={count}; parameters not fitted",
        )
        r["n_independent_draws"] = count
        q = (count - 1) * variance
        r = self.results.add(
            component,
            label + ".innovations.variance",
            expected=1.0,
            measured=variance,
            ci=(
                q / stats.chi2.ppf(1 - alpha / 2, count - 1),
                q / stats.chi2.ppf(alpha / 2, count - 1),
            ),
            p=float(min(1, 2 * min(stats.chi2.cdf(q, count - 1), stats.chi2.sf(q, count - 1)))),
            n_paths=self.k,
            n_time=n - 1,
            method="exact known-parameter chi-square innovation variance",
        )
        r["n_independent_draws"] = count
        for lag, values in innovation_correlations.items():
            self.results.path_mean(
                component,
                label + f".innovations.lag_{lag}",
                values,
                0,
                n_time=max(n - 1 - lag, 0),
                note="within-path uncentered covariance of unit innovations",
            )
        self.draw_plot(
            component,
            label + " OU (plot samples are correlated)",
            samples,
            stats.norm(scale=sigma),
            units,
        )
        self.plots.append(
            dict(kind="acf", component=component, title=label + " OU ACF", values=lag_plot)
        )
        return np.asarray(first_full)

    def processes(self):
        for component, quantity, stem in (
            ("density", "n", "density"),
            ("velocity_speed", "v", "velocity_speed"),
        ):
            s = self.cfg[component]
            if not s["enabled"]:
                continue
            total, bias_sigma, ou_sigma = split_sigma(s)
            fractional = s["model"] == "gaussian_fractional_bias_plus_ou"
            bias = self.constant(component, stem + "_bias")
            if bias is not None:
                self.gaussian_vector(
                    component,
                    "bias",
                    bias,
                    bias_sigma,
                    "fraction" if fractional else "log fraction",
                )
                self.ou(
                    component,
                    stem + ("_fractional_error" if fractional else "_log_error"),
                    self.times[quantity],
                    ou_sigma,
                    s["ou_correlation_time_s"],
                    biases=bias,
                    total_sigma=total,
                    fractional=fractional,
                )
            else:
                self.results.unavailable(
                    component,
                    "OU decomposition",
                    "saved bias required; empirical demeaning forbidden",
                    "SKIPPED",
                )
        s = self.cfg["velocity_angle"]
        if s["enabled"]:
            sigma = angular_sigma(s)
            values = [
                self.ou(
                    "velocity_angle",
                    "velocity_angle_" + name,
                    self.times["v"],
                    sigma,
                    s["ou_correlation_time_s"],
                )
                for name in ("alpha", "beta")
            ]
            if all(v is not None for v in values) and sigma > 0:
                self.results.correlation(
                    "velocity_angle", "alpha_beta.initial_independence", *values
                )
                angle = np.hypot(*values)
                self.results.distribution(
                    "velocity_angle",
                    "total_angle.initial_cdf",
                    angle,
                    stats.rayleigh(scale=sigma),
                    "rad",
                )
                self.draw_plot(
                    "velocity_angle",
                    "Latent total angle (initial independent paths)",
                    angle,
                    stats.rayleigh(scale=sigma),
                    "rad",
                )
                self.results.path_mean(
                    "velocity_angle", "total_angle.second_moment", angle**2, 2 * sigma**2, "rad^2"
                )
                radius = np.deg2rad(s["requirement_deg"])
                self.results.coverage(
                    "velocity_angle",
                    "total_angle.requirement_coverage",
                    angle,
                    radius,
                    float(stats.rayleigh.cdf(radius, scale=sigma)),
                )
            self.results.add(
                "velocity_angle",
                "total_angle.wrapping_probability",
                expected=float(np.exp(-0.5 * (np.pi / sigma) ** 2)) if sigma > 0 else 0.0,
                status="SKIPPED",
                primary=False,
                note="theoretical P(latent angle > pi); principal direction deflection wraps",
            )

    def run(self):
        for component in DEFAULTS:
            if not self.cfg[component]["enabled"]:
                self.results.add(
                    component,
                    "component",
                    note="disabled in embedded configuration",
                    status="SKIPPED",
                    primary=False,
                )
        self.constants()
        self.processes()
        if self.cfg["magnetic_ac"]["enabled"]:
            self.spectra()
        return self.results.finalize()

    def spectra(self):
        component = "magnetic_ac"
        t = self.times["B"]
        n, dt = len(t), float(t[1] - t[0])
        if not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-10):
            self.results.unavailable(component, "spectrum", "uniform cadence required")
            return
        ds = self.dataset(component, "magnetic_ac_noise", (self.mc_count, n, len(self.ids), 3))
        if ds is None:
            return
        section = self.cfg[component]
        lengths = sorted(set((min(8192, n), n)))
        targets = {length: spectral_target(n, dt, section, length) for length in lengths}
        summed = {length: np.zeros((3, length // 2 + 1)) for length in lengths}
        squared = {length: np.zeros((3, length // 2 + 1)) for length in lengths}
        power = [[] for _ in range(3)]
        total_power = []
        band_values = {}
        max_mean = max_parseval = 0.0
        power_target = targets[n][2]
        for mc, sc in self.paths:
            this_total = 0.0
            for axis in range(3):
                series = read_series(ds, mc, sc, self.args.chunk_times, axis)
                rms2 = float(np.mean(series**2))
                power[axis].append(rms2)
                this_total += rms2
                # Ratio gives a scale-aware zero-mean identity tolerance.
                max_mean = max(max_mean, abs(float(series.mean())) / (np.sqrt(rms2) + 1e-30))
                fft = np.fft.rfft(series)
                norm = np.abs(fft) ** 2
                norm[1 : -1 if n % 2 == 0 else None] *= 2
                reconstructed = float(norm.sum() / n**2)
                max_parseval = max(max_parseval, abs(reconstructed - rms2) / (rms2 + 1e-30))
                for length in lengths:
                    f, psd = signal.welch(
                        series,
                        fs=1 / dt,
                        window="hann",
                        nperseg=length,
                        noverlap=length // 2,
                        detrend=False,
                        scaling="density",
                    )
                    target = targets[length][1]
                    summed[length][axis] += psd
                    squared[length][axis] += psd**2
                    masks = {}
                    if length == n:
                        for anchor in section["anchors"]:
                            freq = anchor["frequency_Hz"]
                            df = f[1] - f[0]
                            if freq < 1 / (2 * dt) and freq >= 2 * df:
                                width = max(freq * 0.25, 2 * df)
                                mask = (f >= freq - width) & (f <= freq + width) & (f > 0)
                                if mask.sum() >= 3:
                                    masks[f"anchor_{freq:g}Hz"] = mask
                    if length == min(8192, n):
                        anchors = section["anchors"]
                        for left, right in zip(anchors[:-1], anchors[1:]):
                            lo, hi = left["frequency_Hz"], right["frequency_Hz"]
                            mask = (f >= max(lo, 3 / (length * dt))) & (
                                f <= min(hi, 0.95 / (2 * dt))
                            )
                            if mask.sum() >= 8:
                                masks[f"band_{lo:g}_{hi:g}Hz"] = mask
                                x = np.log(f[mask])
                                weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
                                baseline = float(np.dot(weights, np.log(target[mask])) / 2)
                                slope = baseline + float(
                                    np.dot(weights, psd[mask] / target[mask] - 1) / 2
                                )
                                key = (
                                    axis,
                                    f"slope_{lo:g}_{hi:g}Hz",
                                    baseline,
                                    "linearized ASD slope",
                                )
                                band_values.setdefault(key, []).append(slope)
                    for name, mask in masks.items():
                        ratio = float(psd[mask].sum() / target[mask].sum())
                        key = (axis, name, 1.0, "band PSD / window-adjusted expected band PSD")
                        band_values.setdefault(key, []).append(ratio)
            total_power.append(this_total)
        self.results.identity(component, "zero_mean_relative_to_rms", max_mean, 1e-11, self.k, n)
        self.results.identity(
            component, "parseval_relative_residual", max_parseval, 1e-12, self.k, n
        )
        for axis in range(3):
            self.results.path_mean(
                component,
                f"axis_{axis}.integrated_variance",
                power[axis],
                power_target,
                "nT^2",
                n,
                "finite-record zero-DC expectation including even-N endpoint convention",
            )
        self.results.path_mean(
            component, "total.integrated_variance", total_power, 3 * power_target, "nT^2", n
        )
        for (axis, name, target, method), values in band_values.items():
            self.results.path_mean(
                component,
                f"axis_{axis}.{name}",
                values,
                target,
                n_time=n,
                note=method
                + "; one summary per independent series; overlapping windows/bins not counted as draws",
            )
        for anchor in section["anchors"]:
            freq = anchor["frequency_Hz"]
            if not any(f"anchor_{freq:g}Hz" == key[1] for key in band_values):
                self.results.unavailable(
                    component,
                    f"anchor_{freq:g}Hz",
                    f"anchor requires >=3 resolved bins below Nyquist={1 / (2 * dt):g} Hz",
                )
        self.results.add(
            component,
            "above_10Hz_extrapolation",
            status="SKIPPED",
            primary=False,
            note="high-cadence synthetic validation belongs to Phase C",
        )
        for left, right in zip(section["anchors"][:-1], section["anchors"][1:]):
            name = f"slope_{left['frequency_Hz']:g}_{right['frequency_Hz']:g}Hz"
            if not any(name == key[1] for key in band_values):
                self.results.unavailable(component, name, "requires >=8 resolved in-band bins")
        for length in lengths:
            f, target, _ = targets[length]
            mean = summed[length] / self.k
            se = np.sqrt(
                np.maximum(0, (squared[length] - self.k * mean**2) / max(self.k - 1, 1)) / self.k
            )
            critical = stats.t.ppf((1 + self.args.confidence) / 2, max(self.k - 1, 1))
            self.plots.append(
                dict(
                    kind="spectrum",
                    component=component,
                    title=f"Magnetic AC: Hann L={length}, overlap={length // 2}, no detrend",
                    frequency=f,
                    psd=mean,
                    target=target,
                    raw_target=component_asd(f, section) ** 2,
                    lower=np.maximum(0, mean - critical * se),
                    upper=mean + critical * se,
                    n_paths=self.k,
                    n_segments=1 + (n - length) // (length - length // 2),
                )
            )
        self.results.add(
            component,
            "spectral_interpretation",
            status="SKIPPED",
            primary=False,
            note="ASD=sqrt(mean PSD); plotted intervals are marginal, not simultaneous. "
            "Zero-DC finite periodic model; flat below first anchor; total ASD uses quadrature. "
            "Slopes use a linearized log-PSD contrast with exact zero expected residual.",
        )
