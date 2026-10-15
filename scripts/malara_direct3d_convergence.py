"""Small CLI campaign wrapper and independent saved-product report; no sampling kernels."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

import h5py
import numpy as np
import yaml

from hs_observatory.analysis.spatial_structure_function import read_spatial_result
from hs_observatory.analysis.structure_function import _fit_scaling_exponents, _to_serializable
from hs_observatory.analysis.structure_function_local import p_model_exponents
from hs_observatory.fields.malara import MalaraTurbulenceField

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGETS = (5000, 50000)
BUDGETS = (*DEFAULT_BUDGETS, 500000)
SEEDS = (12345, 23456, 34567)
P_VALUES = (.9, .5)
WINDOW = {"candidate": [200., 1000.]}
ORDERS = np.arange(1, 7)
NOTE = ("Seed spread measures spatial pair sampling within one fixed Malara realization, "
        "not variability between independent field realizations or measurement-error MC. "
        "Budgets share coordinate prefixes; paired budget changes are correlated.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(_to_serializable(value), sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prefix(budget, seed, p_values=P_VALUES):
    label = "-".join(f"p{round(p*10):02d}" for p in sorted(p_values, reverse=True))
    return f"direct3d_{label}_N{budget}_seed{seed}"


def job_p_values(job):
    # Legacy jobs omitted p_values and always contain both cases.
    return tuple(job.get("p_values", P_VALUES))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def field_configs(field):
    # Read the adapter normalization without evaluating any field point.
    base = MalaraTurbulenceField(field).base_config
    return {str(p): json.loads(json.dumps(asdict(replace(base, intermittency_p=p))))
            for p in P_VALUES}


def validate_result(result, fields, *, centers=None, budget=None, seed=None, p_values=P_VALUES):
    """Validate numerical completion and scientific identity, not merely HDF5 readability."""
    meta, config, cases = result["metadata"], result["config"], result["cases"]
    require(meta["field_model"] == "malara_turbulence" and meta["quantity"] == "q"
            and meta["separation_mode"] == "exact" and not meta.get("smoke"),
            "expected non-smoke exact-separation raw Malara product")
    require(meta["sampling_rng"] == "PCG64" and meta["sample_stream_version"] == 1,
            "unsupported spatial RNG stream")
    require(config["n_batches"] == 8, "expected eight batches")
    if budget is not None:
        require(config["n_pairs"] == budget, "pair budget mismatch")
    if budget == 500000:
        require(config["backend"] == "parallel" and config["chunk_size"] > 0,
                "500k requires the bounded parallel evaluator")
    if seed is not None:
        require(config["seed"] == seed == meta["sampling_seed"], "sampling seed mismatch")
    require(len(cases) == len({float(c["p_value"]) for c in cases}), "duplicate p cases")
    require(set(p_values).issubset({float(c["p_value"]) for c in cases}), "missing requested p cases")
    if centers is not None:
        require({float(c["p_value"]) for c in cases} == set(p_values)
                and config["scale_ranges_km"] == WINDOW
                and config["fit_range"] == "candidate" and config["min_fit_bins"] == 3,
                "campaign cases or fitting configuration mismatch")
    for case in cases:
        p = float(case["p_value"])
        if p not in P_VALUES:
            continue  # Original pilot also contains p=0.7.
        require(signature(case["field_config"]) == signature(fields[str(p)]),
                f"p={p}: physical field configuration mismatch")
        x = case["lambda_bin_centers_km"]
        require(x.dtype == np.dtype("float64") and x.ndim == 1 and np.isfinite(x).all()
                and (x > 0).all() and (np.diff(x) > 0).all(), "invalid separation centers")
        if centers is not None:
            require(np.array_equal(x.view("u8"), np.asarray(centers).view("u8")),
                    "exact separation bits differ from pilot")
        require(np.array_equal(case["orders"], ORDERS), "expected orders 1..6")
        n, size = config["n_pairs"], len(x)
        counts = case["counts"]
        values, sums = case["q"]["S_n"], case["moment_sums"]
        require(counts.shape == (1, size) and np.all(counts == n), "incomplete pair counts")
        require(values.shape == sums.shape == (1, 6, size) and np.isfinite(values).all()
                and np.all(values > 0) and np.isfinite(sums).all(), "invalid moments")
        require(np.allclose(values * n, sums, rtol=1e-12, atol=0), "moment sums/counts disagree")
        batch_counts = np.broadcast_to(np.diff(np.linspace(0, n, 9, dtype=int))[:, None],
                                       (8, size))
        cp_counts = np.broadcast_to(np.ceil(n*np.array([.25, .5, .75, 1])).astype(int)[:, None],
                                    (4, size))
        require(np.array_equal(case["batch_counts"], batch_counts)
                and np.array_equal(case["checkpoint_counts"], cp_counts),
                "incomplete batch/checkpoint counts")
        batch, batch_sums = case["batch_S_n"], case["batch_moment_sums"]
        cp = case["checkpoint_S_n"]
        require(batch.shape == batch_sums.shape == (8, 6, size)
                and cp.shape == (4, 6, size) and np.isfinite(batch).all()
                and np.isfinite(cp).all() and (batch > 0).all() and (cp > 0).all(),
                "invalid batch/checkpoint moments")
        require(np.allclose(batch * batch_counts[:, None], batch_sums, rtol=1e-12, atol=0)
                and np.allclose(batch_sums.sum(axis=0), sums[0], rtol=1e-12, atol=0)
                and np.allclose(cp[-1], values[0], rtol=1e-12, atol=0), "batch/checkpoint accounting mismatch")
        require(np.allclose(case["theory"], p_model_exponents(ORDERS, p), rtol=1e-12, atol=0),
                "analytic reference mismatch")
        fit = case["q"]["scaling_exponents"]
        expected = _fit_scaling_exponents(values, counts, x, ORDERS.tolist(),
                                         config["scale_ranges_km"], config["min_fit_bins"])
        require(fit["range_names"] == expected["range_names"]
                and np.allclose(fit["raw"], expected["raw"], rtol=1e-12, equal_nan=True)
                and np.array_equal(fit["bins_used"], expected["bins_used"]), "stored fit mismatch")
        if centers is not None:
            require(np.isfinite(fit["raw"]).all(), "incomplete fitted exponents")
            local = case["local"]
            require(local["slopes"].shape == (1, 6, size)
                    and np.array_equal(local["coordinates_km"], x)
                    and np.isfinite(local["slopes"][..., 1:-1]).all(),
                    "incomplete three-point local slopes")
    return result


def discover(root=ROOT):
    """Read seven full-precision pilot centers and validate fixed physical parameters."""
    source = root / "generation/malara-ts-500a-t1500.yaml"
    pilot = root / "outputs/direct-spatial-sf-5000.h5"
    recipe = root / "configs/example_spatial_structure_function.yaml"
    executable = root / ".venv/bin/hsobs-analyze-spatial-structure-function"
    require(executable.is_file() and os.access(executable, os.X_OK),
            f"missing executable: {executable}")
    generation = yaml.safe_load(source.read_text())
    fields = field_configs(generation["field"])
    result = read_spatial_result(pilot)
    validate_result(result, fields, budget=5000, seed=12345)
    saved_field = yaml.safe_load(result["metadata"]["source_field_yaml"])["field"]
    for p in P_VALUES:
        current = json.loads(json.dumps(generation["field"]))
        original = json.loads(json.dumps(saved_field))
        current["turbulence"]["intermittency_p"] = p
        original["turbulence"]["intermittency_p"] = p
        require(current == original, "pilot and generation field mappings differ beyond p")
    grids = [c["lambda_bin_centers_km"] for c in result["cases"]]
    require(all(np.array_equal(grids[0], grid) for grid in grids), "pilot p grids differ")
    centers = grids[0][(grids[0] >= 200) & (grids[0] <= 1000)]
    require(len(centers) == 7, "pilot must contain exactly seven centers in 200–1000 km")
    require(centers[-1] < min(fields["0.9"]["domain_size"])/2,
            "campaign centers must stay below half the periodic box")
    config = yaml.safe_load(recipe.read_text())
    require(config["analysis"]["method"] == "spatial_structure_function", "wrong CLI recipe method")
    options = config["spatial_structure_function"]
    options.pop("comparison_config", None)
    options.update(plot={"enabled": False}, scale_ranges_km=WINDOW, fit_range="candidate",
                   min_fit_bins=3, local_slope_method="three-point")
    # Existing schema; runtime recipe disables plots and unrelated HS comparisons.
    config["input"]["path"] = str(source)
    inputs = dict(field_yaml_sha256=sha256(source), pilot_sha256=sha256(pilot),
                  recipe_sha256=sha256(recipe), fields=fields,
                  centers_hex=[float(x).hex() for x in centers], config=config,
                  numpy_version=np.__version__, wrapper_sha256=sha256(__file__),
                  implementation_sha256={
                      name: sha256(root / name) for name in (
                          "src/hs_observatory/analysis/spatial_structure_function.py",
                          "src/hs_observatory/analysis/structure_function.py",
                          "src/hs_observatory/analysis/structure_function_local.py",
                          "src/hs_observatory/cli_spatial_structure_function.py",
                          "src/hs_observatory/fields/malara.py")})
    campaign = dict(version=1, inputs=inputs, fingerprint=signature(inputs), jobs=[
        dict(budget=b, seed=s, prefix=prefix(b, s)) for b in DEFAULT_BUDGETS for s in SEEDS])
    return campaign


def job_signature(campaign, job):
    return signature(dict(campaign=campaign["fingerprint"], job=job))


def validated_job(output, campaign, job):
    """Only a matching receipt plus hash and a complete numerical product is successful."""
    path = output / (job["prefix"] + ".h5")
    receipt_path = output / (job["prefix"] + ".complete.json")
    receipt = json.loads(receipt_path.read_text())
    require(receipt["fingerprint"] == job_signature(campaign, job), "stale completion fingerprint")
    require(receipt["status"] == "complete" and np.isfinite(receipt["elapsed_s"])
            and receipt["elapsed_s"] >= 0 and receipt["product_sha256"] == sha256(path),
            "invalid completion receipt or product hash")
    centers = np.array([float.fromhex(x) for x in campaign["inputs"]["centers_hex"]])
    result = validate_result(read_spatial_result(path), campaign["inputs"]["fields"],
                             centers=centers, budget=job["budget"], seed=job["seed"],
                             p_values=job_p_values(job))
    require(sha256_text(result["metadata"]["source_field_yaml"])
            == campaign["inputs"]["field_yaml_sha256"], "source generation YAML mismatch")
    return result, receipt


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def complete(output, campaign, job):
    try:
        validated_job(output, campaign, job)
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def command_for(root, recipe, target, campaign, job):
    centers = [repr(float.fromhex(x)) for x in campaign["inputs"]["centers_hex"]]
    command = [str(root / ".venv/bin/hsobs-analyze-spatial-structure-function"), str(recipe),
               "--field-config", str(root / "generation/malara-ts-500a-t1500.yaml"),
               "--p", *map(str, job_p_values(job)), "--lambda-centers-km", *centers,
               "--n-pairs", str(job["budget"]), "--n-batches", "8", "--seed", str(job["seed"]),
               "--fit-lambda-range", "200", "1000", "--output-prefix", str(target)]
    if job["budget"] == 500000:
        chunk_size = campaign["inputs"]["config"]["spatial_structure_function"].get("chunk_size", 128)
        command += ["--backend", "parallel", "--chunk-size", str(chunk_size)]
    return command


def scientific_inputs(inputs):
    # A wrapper/report update must not invalidate fixed, unchanged scientific inputs.
    return {key: value for key, value in inputs.items() if key != "wrapper_sha256"}


def extend_campaign(saved, current, requested):
    require(saved["version"] == 1 and signature(saved["inputs"]) == saved["fingerprint"],
            "invalid saved campaign manifest")
    require(scientific_inputs(saved["inputs"]) == scientific_inputs(current["inputs"]),
            "campaign scientific inputs changed; use a separate output directory")
    # Preserve the original inputs/fingerprint and exact legacy job dictionaries:
    # their existing receipts remain valid. New jobs are append-only.
    result = dict(saved, jobs=list(saved["jobs"]))
    for job in requested:
        matches = [j for j in result["jobs"] if j["prefix"] == job["prefix"]]
        require(not matches or matches == [job], "conflicting job identity")
        if not matches:
            result["jobs"].append(job)
    return result


def run_campaign(root, output, budgets=DEFAULT_BUDGETS, seeds=SEEDS, *, dry_run=False,
                 resume=False, p_values=P_VALUES):
    p_values = tuple(sorted(set(p_values), reverse=True))
    require(p_values and set(p_values).issubset(P_VALUES), "unsupported p values")
    require(500000 not in budgets or p_values == (.9,), "500k requires --p 0.9")
    current = discover(root)
    jobs = []
    for budget in dict.fromkeys(budgets):
        for seed in dict.fromkeys(seeds):
            job = dict(budget=budget, seed=seed, prefix=prefix(budget, seed, p_values))
            if p_values != P_VALUES:
                job["p_values"] = list(p_values)
            jobs.append(job)
    manifest = output / "campaign.json"
    saved = json.loads(manifest.read_text()) if manifest.exists() else current
    # A completed combined-p job already supplies a requested single-p small-budget run.
    jobs = [next((old for old in saved["jobs"] if old["budget"] == job["budget"]
                  and old["seed"] == job["seed"]
                  and set(job_p_values(job)).issubset(job_p_values(old))
                  and complete(output, saved, old)), job) for job in jobs]
    campaign = extend_campaign(saved, current, jobs)
    pending = []
    for job in jobs:
        exists = any((output / (job["prefix"]+suffix)).exists()
                     for suffix in (".h5", ".complete.json"))
        done = complete(output, campaign, job)
        require(resume or not exists, f"existing job {job['prefix']}; use --resume")
        print(f"{'SKIP complete' if done and resume else 'PLAN'} {job['prefix']}", flush=True)
        if not (resume and done):
            pending.append(job)
    evaluations = sum(2 * len(job_p_values(j)) * 7 * j["budget"] for j in pending)
    print(f"Selected {len(jobs)} jobs; pending {len(pending)}; "
          f"planned field endpoint evaluations: {evaluations:,}", flush=True)
    print("Exact centers [km]: " + " ".join(repr(float.fromhex(x)) for x in
                                           campaign["inputs"]["centers_hex"]), flush=True)
    for job in pending:
        print(shlex.join(command_for(root, output/"<attempt>/analysis.yaml",
                                    output/"<attempt>"/job["prefix"], campaign, job)), flush=True)
    if dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".campaign.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another campaign runner holds the output lock") from exc
        # Recheck after acquiring the lock, before modifying any campaign artifacts.
        if manifest.exists():
            campaign = extend_campaign(json.loads(manifest.read_text()), current, jobs)
        atomic_json(manifest, campaign)
        for job in pending:
            if resume and complete(output, campaign, job):
                continue
            require(resume or not any((output/(job["prefix"]+suffix)).exists()
                                     for suffix in (".h5", ".complete.json")),
                    f"job appeared during preflight: {job['prefix']}; use --resume")
            work = output / ".incomplete"
            work.mkdir(exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix=job["prefix"]+"-", dir=work))
            recipe = attempt / "analysis.yaml"
            recipe.write_text(yaml.safe_dump(campaign["inputs"]["config"]))
            command = command_for(root, recipe, attempt/job["prefix"], campaign, job)
            print(f"RUN {job['prefix']}; log: {attempt/'run.log'}", flush=True)
            started = time.monotonic()
            status = dict(status="running", command=command,
                          started_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(attempt/"status.json", status)
            try:
                with (attempt/"run.log").open("w") as log:
                    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
                path = attempt / (job["prefix"]+".h5")
                centers = np.array([float.fromhex(x) for x in campaign["inputs"]["centers_hex"]])
                result = validate_result(read_spatial_result(path), campaign["inputs"]["fields"],
                                         centers=centers, budget=job["budget"], seed=job["seed"],
                                         p_values=job_p_values(job))
                require(sha256_text(result["metadata"]["source_field_yaml"])
                        == campaign["inputs"]["field_yaml_sha256"], "generation input changed")
                require(scientific_inputs(discover(root)["inputs"]) ==
                        scientific_inputs(campaign["inputs"]), "campaign inputs changed during sampling")
                elapsed = time.monotonic()-started
                result["metadata"].update(sampling_product=str(output/path.name),
                                          campaign_attempt_product=str(path))
                with h5py.File(path, "r+") as handle:
                    handle.attrs["metadata_json"] = json.dumps(result["metadata"])
                # Keep any incomplete prior outputs until the replacement has validated.
                for suffix in (".h5", ".complete.json"):
                    previous = output / (job["prefix"]+suffix)
                    if previous.exists():
                        previous.replace(attempt / ("previous"+suffix))
                path.replace(output/path.name)
                status.update(status="complete", elapsed_s=elapsed,
                              fingerprint=job_signature(campaign, job),
                              product_sha256=sha256(output/path.name),
                              executed_wrapper_sha256=current["inputs"]["wrapper_sha256"])
                atomic_json(output/(job["prefix"]+".complete.json"), status)
                atomic_json(attempt/"status.json", status)
                print(f"DONE {job['prefix']}: {elapsed:.3f} s", flush=True)
            except (Exception, KeyboardInterrupt) as exc:
                status.update(status="failed", elapsed_s=time.monotonic()-started, error=str(exc))
                atomic_json(attempt/"status.json", status)
                raise RuntimeError(f"{job['prefix']} failed; preserved diagnostics: {attempt}") from exc


def aggregate(records, p, budget, key):
    values = [r[key] for r in records if r["p"] == p and r["budget"] == budget]
    return np.stack(values) if values else None


def stats(values):
    if values is None or not len(values):
        return "missing"
    if not np.isfinite(values).all():
        return "undefined (zero reference exponent)"
    if len(values) == 1:
        return f"{values[0]:.6g} (single seed; SD unavailable)"
    return (f"{np.mean(values):.6g} ± {np.std(values, ddof=1):.3g} "
            f"[{np.min(values):.6g}, {np.max(values):.6g}]")


def budget_changes(records, p, key, low_budget=5000, high_budget=50000):
    low = {r["seed"]: r[key] for r in records if r["p"] == p and r["budget"] == low_budget}
    high = {r["seed"]: r[key] for r in records if r["p"] == p and r["budget"] == high_budget}
    seeds = sorted(low.keys() & high.keys())
    return np.stack([np.divide(high[s]-low[s], low[s], out=np.full_like(low[s], np.nan),
                               where=low[s] != 0) for s in seeds]) if seeds else None


def report_budgets(campaign, p):
    return sorted(set(DEFAULT_BUDGETS) | {j["budget"] for j in campaign["jobs"]
                                        if p in job_p_values(j)})


def budget_pairs(budgets):
    adjacent = list(zip(budgets[:-1], budgets[1:]))
    return adjacent + ([(budgets[0], budgets[-1])] if len(budgets) > 2 else [])


def budget_label(budget):
    return f"{budget//1000}k"


def report(output, threshold=.1):
    """Read receipts and small numerical products only; never invoke the sampling CLI."""
    campaign = json.loads((output/"campaign.json").read_text())
    require(campaign["version"] == 1 and signature(campaign["inputs"]) == campaign["fingerprint"],
            "invalid campaign manifest")
    records, lines = [], ["# Direct-3D Malara convergence report", "", NOTE, "",
        "All moments use raw vector-increment magnitudes, one fixed field seed/realization, "
        "seven exact pilot centers and the 200–1000 km fit. SD uses ddof=1; one seed has "
        "undefined SD. Ranges are min–max across available seeds. Missing jobs are explicit; "
        "a quiet diagnostic is not proof of convergence.", "",
        "| Job | Status | Wall time [s] |", "|---|---|---:|"]
    for job in campaign["jobs"]:
        try:
            result, receipt = validated_job(output, campaign, job)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            lines.append(f"| {job['prefix']} | unavailable/incomplete: {exc} | — |")
            continue
        lines.append(f"| {job['prefix']} | complete | {receipt['elapsed_s']:.3f} |")
        for case in result["cases"]:
            record = dict(p=float(case["p_value"]), budget=job["budget"], seed=job["seed"],
                          moments=case["q"]["S_n"][0],
                          xi=case["q"]["scaling_exponents"]["raw"][0, 0],
                          checkpoint=np.max(abs(case["checkpoint_S_n"][:-1] /
                                                case["q"]["S_n"][0] - 1), axis=0))
            duplicates = [r for r in records if all(r[k] == record[k] for k in ("p", "budget", "seed"))]
            if duplicates:
                require(all(np.allclose(duplicates[0][k], record[k], rtol=1e-12, atol=1e-12)
                            for k in ("moments", "xi", "checkpoint")),
                        "conflicting products for the same p/budget/seed")
            else:
                records.append(record)
    centers = np.array([float.fromhex(x) for x in campaign["inputs"]["centers_hex"]])
    lines += ["", f"Screening threshold: {100*threshold:g}%. Flags below mean a moment's maximum "
              "nonfinal checkpoint deviation, available between-seed CV, or maximum absolute "
              "matched-seed budget change exceeds this heuristic threshold. They are "
              "diagnostic warnings, not statistical significance tests. Checkpoint variation is "
              "max(|S_n(N/4,N/2,3N/4)/S_n(N) − 1|), not independent replicate uncertainty. "
              "One seed supplies a value and a paired budget change, not a seed-spread uncertainty "
              "estimate. No flag with missing budgets/seeds means insufficient evidence, not convergence.", ""]
    for p in P_VALUES:
        budgets = report_budgets(campaign, p)
        pairs = budget_pairs(budgets)
        labels = [budget_label(b) for b in budgets]
        pair_labels = [f"{budget_label(lo)}→{budget_label(hi)}" for lo, hi in pairs]
        lines += [f"## p={p}", "", "### Fitted exponents", "",
                  "Values with multiple seeds are mean ± sample SD [min, max]. Single-seed "
                  "values have no uncertainty estimate.", "",
                  "| n | ζ_n | " + " | ".join(f"ξ_n, {label}" for label in labels) + " |",
                  "|---:|---:|" + "---|"*len(budgets)]
        for i, n in enumerate(ORDERS):
            columns = []
            for b in budgets:
                values = aggregate(records, p, b, "xi")
                columns.append(stats(None if values is None else values[:, i]))
            lines.append(f"| {n} | {p_model_exponents([n], p)[0]:.6g} | " + " | ".join(columns) + " |")
        xi_changes = [budget_changes(records, p, "xi", lo, hi) for lo, hi in pairs]
        lines += ["", "Fitted-exponent changes: paired (ξ_high/ξ_low − 1) × 100, using only "
                  "matched seeds. A single matched seed is a difference, not an uncertainty estimate.", "",
                  "| n | " + " | ".join(f"Δ {label} [%]" for label in pair_labels) + " |",
                  "|---:|" + "---|"*len(pairs)]
        for i, n in enumerate(ORDERS):
            columns = [stats(None if change is None else 100*change[:, i]) for change in xi_changes]
            lines.append(f"| {n} | " + " | ".join(columns) + " |")
        lines += ["", "### Moment convergence by separation", "",
                  "Moments: seed mean ± sample SD [min, max] when multiple seeds are available, "
                  "in raw field units^n. CP lists maximum checkpoint deviation across available "
                  "seeds. CP and seed counts follow budget order: " + " / ".join(labels) + ".", "",
                  "| λ [km] | n | " + " | ".join(f"S_n, {label}" for label in labels)
                  + " | Max CP [%] | Seeds | Screen |",
                  "|---:|---:|" + "---|"*(len(budgets)+3)]
        changes = [budget_changes(records, p, "moments", lo, hi) for lo, hi in pairs]
        for si, separation in enumerate(centers):
            for n in (2, 4, 6):
                i = n-1
                columns, checkpoints, counts, reasons = [], [], [], []
                for b in budgets:
                    values = aggregate(records, p, b, "moments")
                    cp = aggregate(records, p, b, "checkpoint")
                    counts.append(str(0 if values is None else len(values)))
                    columns.append(stats(None if values is None else values[:, i, si]))
                    checkpoints.append("missing" if cp is None else f"{100*cp[:, i, si].max():.3g}")
                    if cp is not None and cp[:, i, si].max() > threshold:
                        reasons.append(f"CP{b}")
                    if values is not None and len(values) > 1:
                        cv = values[:, i, si].std(ddof=1)/values[:, i, si].mean()
                        if cv > threshold:
                            reasons.append(f"seedCV{b}")
                for label, change in zip(pair_labels, changes):
                    if change is not None and abs(change[:, i, si]).max() > threshold:
                        reasons.append(f"budget {label}")
                adequate = all(int(c) >= 2 for c in counts)
                screen = ", ".join(reasons) or ("below screen; not proof" if adequate else "insufficient jobs")
                if not adequate and reasons:
                    screen += "; insufficient seed repeats"
                lines.append(f"| {separation:.12g} | {n} | " + " | ".join(columns)
                             + " | " + " / ".join(checkpoints) + " | " + " / ".join(counts)
                             + f" | {screen} |")
        lines += ["", "### Paired moment changes by separation", "",
                  "Δ = (S_high/S_low − 1) × 100, matching seeds before aggregation. This table "
                  "compares individual moments; agreement of fitted exponents is not sufficient.", "",
                  "| λ [km] | n | " + " | ".join(f"Δ {label} [%]" for label in pair_labels) + " |",
                  "|---:|---:|" + "---|"*len(pairs)]
        for si, separation in enumerate(centers):
            for n in (2, 4, 6):
                columns = [stats(None if change is None else 100*change[:, n-1, si]) for change in changes]
                lines.append(f"| {separation:.12g} | {n} | " + " | ".join(columns) + " |")
        lines += ["", "### Individual saved moments and high-order checkpoints", "",
                  "| Budget | Seed | λ [km] | S2 | S4 | S6 | Max CP4 [%] | Max CP6 [%] |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in records:
            if r["p"] == p:
                for si, x in enumerate(centers):
                    numbers = [r["moments"][n-1, si] for n in (2, 4, 6)]
                    numbers += [100*r["checkpoint"][n-1, si] for n in (4, 6)]
                    lines.append(f"| {r['budget']} | {r['seed']} | {x:.12g} | "
                                 + " | ".join(f"{v:.7g}" for v in numbers) + " |")
    lines += ["", "Interpret the screening flags at individual scales before comparing fitted "
              "exponents with theory. Increasing the budget replays existing prefixes; it does "
              "not create new field realizations. Theory agreement alone is not convergence.", ""]
    target = output/"convergence_report.md"
    if any(j["budget"] == 500000 for j in campaign["jobs"]):
        archive = output/"reports_before_500k"
        for name in ("convergence_report.md", "convergence_exponents.png", "convergence_exponents.pdf"):
            previous = output/name
            if previous.exists() and not (archive/name).exists():
                archive.mkdir(exist_ok=True)
                shutil.copy2(previous, archive/name)
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(target)
    plot_report(records, output, {p: report_budgets(campaign, p) for p in P_VALUES})
    print(target)
    return records


def plot_report(records, output, budgets_by_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, p in zip(axes, P_VALUES):
        for i, n in enumerate(ORDERS):
            color = plt.get_cmap("tab10")(i)
            xs, means = [], []
            for b in budgets_by_p[p]:
                values = aggregate(records, p, b, "xi")
                if values is None:
                    continue
                xs.append(b)
                means.append(values[:, i].mean())
                if len(values) > 1:
                    ax.errorbar(b, means[-1], yerr=values[:, i].std(ddof=1),
                                color=color, capsize=3)
                ax.scatter(np.full(len(values), b), values[:, i], color=color, alpha=.3, s=12)
            ax.plot(xs, means, marker="o", color=color, label=f"n={n}")
            ax.axhline(p_model_exponents([n], p)[0], color=color, linestyle="--", alpha=.7, lw=.8)
        ax.set(xscale="log", xlabel="Pairs per exact separation", title=f"p={p}",
               xticks=budgets_by_p[p], xticklabels=[budget_label(b) for b in budgets_by_p[p]])
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r"Fitted $\xi_n$; dashed: $\zeta_n(p)$")
    axes[1].legend(ncol=2, fontsize=8)
    fig.suptitle("Raw direct-3D; 200–1000 km; one fixed field", fontsize=10)
    fig.text(.5, .01, "SD bars only with ≥2 seeds; single-seed points have no uncertainty estimate.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output/f"convergence_exponents.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="invoke existing sampling CLI; expensive unless --dry-run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--budget", action="append", type=int, choices=BUDGETS)
    run.add_argument("--seed", action="append", type=int, choices=SEEDS)
    run.add_argument("--p", nargs="+", type=float, choices=P_VALUES,
                     help="p cases; 500k is restricted to --p 0.9")
    saved = sub.add_parser("report", help="saved results only; never samples fields")
    saved.add_argument("--threshold", type=float, default=.1)
    for command in (run, saved):
        command.add_argument("--output", type=Path, default=ROOT/"outputs/malara_direct3d_convergence")
    args = parser.parse_args(argv)
    try:
        if args.mode == "run":
            budgets = args.budget or DEFAULT_BUDGETS
            p_values = args.p if args.p is not None else ((.9,) if 500000 in budgets else P_VALUES)
            seeds = args.seed or ((12345,) if 500000 in budgets else SEEDS)
            run_campaign(ROOT, args.output.resolve(), budgets, seeds,
                         dry_run=args.dry_run, resume=args.resume, p_values=p_values)
        else:
            require(np.isfinite(args.threshold) and args.threshold > 0, "threshold must be positive")
            report(args.output.resolve(), args.threshold)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
