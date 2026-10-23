"""Statistical records, exact marginal tests and independent-path inference."""

import numpy as np
from scipy import stats


class Results:
    def __init__(self, confidence=0.99):
        self.confidence = confidence
        self.rows = []

    def add(
        self,
        component,
        statistic,
        *,
        expected=None,
        measured=None,
        units="1",
        ci=None,
        p=None,
        status=None,
        n_paths=0,
        n_time=1,
        method="",
        note="",
        primary=True,
    ):
        if p is not None and not np.isfinite(p):
            p, status = None, "FAIL"
            note += "; nonfinite derived p-value"
        row = dict(
            component=component,
            statistic=statistic,
            units=units,
            expected=expected,
            measured=measured,
            ci_low=None if ci is None else ci[0],
            ci_high=None if ci is None else ci[1],
            confidence=self.confidence,
            effect_size=None if measured is None or expected is None else measured - expected,
            relative_effect=None
            if expected in (None, 0) or measured is None
            else (measured - expected) / abs(expected),
            raw_p=p,
            adjusted_p=None,
            status=status,
            n_paths=n_paths,
            n_time=n_time,
            method=method,
            note=note,
            primary=primary,
        )
        self.rows.append(row)
        return row

    def unavailable(self, component, statistic, reason, status="INSUFFICIENT_DATA"):
        return self.add(component, statistic, status=status, note=reason)

    def identity(self, component, name, residual, tolerance, n_paths, n_time=1, units="1"):
        return self.add(
            component,
            name,
            expected=0.0,
            measured=float(residual),
            units=units,
            status="PASS" if residual <= tolerance else "FAIL",
            n_paths=n_paths,
            n_time=n_time,
            method="deterministic tolerance",
            note=f"tolerance={tolerance:g}",
        )

    def path_mean(self, component, name, values, target, units="1", n_time=1, note=""):
        """Student interval over independent path summaries, not individual timestamps."""
        x = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(x)):
            return self.add(component, name, status="FAIL", note="nonfinite path statistic")
        n = len(x)
        estimate = float(np.mean(x)) if n else None
        if n < 30:
            return self.add(
                component,
                name,
                expected=target,
                measured=estimate,
                units=units,
                status="INSUFFICIENT_DATA",
                n_paths=n,
                n_time=n_time,
                note="requires >=30 independent paths; " + note,
            )
        se = float(np.std(x, ddof=1) / np.sqrt(n))
        critical = stats.t.ppf((1 + self.confidence) / 2, n - 1)
        p = (
            1.0
            if se == 0 and estimate == target
            else (0.0 if se == 0 else float(2 * stats.t.sf(abs(estimate - target) / se, n - 1)))
        )
        return self.add(
            component,
            name,
            expected=float(target),
            measured=estimate,
            units=units,
            ci=(estimate - critical * se, estimate + critical * se),
            p=p,
            n_paths=n,
            n_time=n_time,
            method="approximate Student independent-path mean",
            note=note,
        )

    def gaussian(self, component, prefix, values, sigma, units="1", n_time=1):
        x = np.asarray(values).ravel()
        n = len(x)
        if n < 2:
            self.unavailable(component, prefix, "requires >=2 independent draws")
            return
        if sigma == 0:
            self.identity(component, prefix + ".zero", np.max(np.abs(x)), 1e-14, n, n_time, units)
            return
        alpha = 1 - self.confidence
        mean, sd = float(x.mean()), float(x.std(ddof=1))
        crit = stats.norm.ppf(1 - alpha / 2)
        self.add(
            component,
            prefix + ".mean",
            expected=0.0,
            measured=mean,
            units=units,
            ci=(mean - crit * sigma / np.sqrt(n), mean + crit * sigma / np.sqrt(n)),
            p=float(2 * stats.norm.sf(abs(mean) * np.sqrt(n) / sigma)),
            n_paths=n,
            n_time=n_time,
            method="exact normal known-sigma mean",
        )
        q = (n - 1) * sd**2 / sigma**2
        self.add(
            component,
            prefix + ".sigma",
            expected=float(sigma),
            measured=sd,
            units=units,
            ci=(
                sd * np.sqrt((n - 1) / stats.chi2.ppf(1 - alpha / 2, n - 1)),
                sd * np.sqrt((n - 1) / stats.chi2.ppf(alpha / 2, n - 1)),
            ),
            p=float(min(1, 2 * min(stats.chi2.cdf(q, n - 1), stats.chi2.sf(q, n - 1)))),
            n_paths=n,
            n_time=n_time,
            method="exact chi-square variance; interval for sigma",
        )
        self.distribution(component, prefix + ".cdf", x, stats.norm(scale=sigma), units)
        ordered = np.sort(x)
        for probability in (0.025, 0.5, 0.975):
            target = float(stats.norm.ppf(probability, scale=sigma))
            lo = int(stats.binom.ppf(alpha / 2, n, probability))
            hi = int(stats.binom.ppf(1 - alpha / 2, n, probability)) + 1
            interval = (
                None if lo == 0 else float(ordered[lo - 1]),
                None if hi > n else float(ordered[hi - 1]),
            )
            self.add(
                component,
                f"{prefix}.quantile_{probability:g}",
                expected=target,
                measured=float(np.quantile(x, probability)),
                units=units,
                ci=interval,
                p=float(stats.binomtest(int(np.count_nonzero(x <= target)), n, probability).pvalue),
                n_paths=n,
                method="binomial inversion/order-statistic quantile interval",
                note="null interval endpoint denotes an unbounded tail",
            )

    def distribution(self, component, name, values, distribution, units="1"):
        x = np.asarray(values).ravel()
        if len(x) < 8:
            self.unavailable(component, name, "requires >=8 independent draws for CDF diagnostic")
            return
        result = stats.kstest(x, distribution.cdf, method="exact")
        self.add(
            component,
            name,
            expected=0.0,
            measured=float(result.statistic),
            units="CDF distance",
            p=float(result.pvalue),
            n_paths=len(x),
            method="fixed-parameter one-sample KS",
            note=f"distribution units={units}; no parameters fitted to these draws",
        )

    def correlation(self, component, name, x, y):
        n = len(x)
        if n < 4 or np.std(x) == 0 or np.std(y) == 0:
            self.unavailable(component, name, "correlation requires >=4 draws and nonzero variance")
            return
        r, p = stats.pearsonr(x, y)
        z = np.arctanh(np.clip(r, -1 + 1e-15, 1 - 1e-15))
        delta = stats.norm.ppf((1 + self.confidence) / 2) / np.sqrt(n - 3)
        self.add(
            component,
            name,
            expected=0.0,
            measured=float(r),
            ci=np.tanh([z - delta, z + delta]),
            p=float(p),
            n_paths=n,
            method="exact normal-pair Pearson test; approximate Fisher CI",
        )

    def coverage(self, component, name, values, radius, probability):
        x = np.asarray(values)
        test = stats.binomtest(int(np.count_nonzero(x <= radius)), len(x), probability)
        ci = test.proportion_ci(self.confidence, method="exact")
        self.add(
            component,
            name,
            expected=float(probability),
            measured=float(test.statistic),
            ci=(ci.low, ci.high),
            p=float(test.pvalue),
            n_paths=len(x),
            method="exact binomial coverage",
            note=f"radius={radius:g}",
        )

    def finalize(self):
        """Holm family includes all valid primary p-values, regardless of dependence."""
        eligible = [r for r in self.rows if r["primary"] and r["raw_p"] is not None]
        eligible.sort(key=lambda r: r["raw_p"])
        running = 0.0
        for index, row in enumerate(eligible):
            running = max(running, min(1.0, (len(eligible) - index) * row["raw_p"]))
            row["adjusted_p"] = running
            row["status"] = "FAIL" if running < 1 - self.confidence else "PASS"
        for row in self.rows:
            if row["status"] is None:
                row["status"] = "SKIPPED"
        return self.rows
