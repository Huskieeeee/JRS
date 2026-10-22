#!/usr/bin/env python3
"""Source-backed paired exploratory analysis; no summary-only significance tests."""
from __future__ import annotations
import argparse
import csv
import json
import pathlib
import sys
import numpy as np
from scipy.stats import binomtest

from .paths import ROOT
from . import runner


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    highest = 0.0
    for rank, index in enumerate(order):
        highest = max(highest, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, highest)
    return adjusted.tolist()


def bootstrap_ci(values, strata, seed=44512, repeats=10000):
    """Pairs stay together; independent physical worlds resampled within layout."""
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    if not len(values):
        raise ValueError("No pairs")
    rng = np.random.default_rng(seed)
    samples = np.zeros(repeats)
    for label in sorted(set(strata)):
        indices = np.flatnonzero(strata == label)
        drawn = rng.choice(indices, size=(repeats, len(indices)), replace=True)
        samples += values[drawn].sum(axis=1) / len(values)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def signflip_p(values, seed=88013, repeats=20000):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > 1e-14]
    if len(values) == 0:
        return 1.0
    observed = abs(values.sum())
    if len(values) <= 18:
        count = 0
        for mask in range(1 << len(values)):
            total = sum(value if mask & (1 << i) else -value for i, value in enumerate(values))
            count += abs(total) >= observed - 1e-12
        return count / (1 << len(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, repeats, 1000):
        signs = rng.choice([-1., 1.], size=(min(1000, repeats-start), len(values)))
        exceed += int(np.count_nonzero(np.abs(signs @ values) >= observed - 1e-12))
    return (1 + exceed) / (repeats + 1)


def row_from(record):
    job, metrics = record["job"], record["metrics"]
    for required in ("quality_success", "cost", "false_releases", "inspections", "repairs", "travel", "waiting_time"):
        if required not in metrics:
            raise ValueError("Missing metric: " + required)
    return {"world_id": job["world_id"], "condition": job["condition"], "layout": job["layout"],
            "method": job["method"], "planner_replicate": job["planner_replicate"],
            "qualified_completion": float(metrics["quality_success"]), "J": metrics["cost"],
            **{k: metrics[k] for k in ("false_releases", "inspections", "repairs", "travel", "waiting_time",
                                       "successor_revocations", "deadline_misses", "stale_authorizations",
                                       "quality_false_releases", "charged_time", "time", "qualified_tasks")},
            "decision_count": record["decision_count"], "planner_wall_seconds": record["planner"].get("total_wall_s"),
            "episode_wall_seconds": record["wall_seconds"], "planner_cpu_seconds": record["planner_cpu_seconds"], "audit_pass": record["audit"]["pass"],
            "base_world_sha256": record["pairing"]["base_world_sha256"]}


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir, out):
    config = runner.load_json(run_dir / "protocol.json")
    freeze = runner.load_json(run_dir / "FREEZE.json")
    if runner.digest(runner.canonical(config)) != freeze["config_sha256"]:
        raise ValueError("Protocol changed after freeze")
    jobs = list(runner.jobs_for(config))
    if runner.digest(runner.canonical(jobs)) != freeze["jobs_sha256"]:
        raise ValueError("Job manifest changed")
    missing, rows, audit_events = [], [], 0
    for job in jobs:
        folder = run_dir / "jobs" / job["job_id"]
        if not (folder / "result.json").is_file():
            missing.append(job["job_id"])
            continue
        record = runner.audit_saved(folder, job)
        if not record["audit"]["pass"]:
            raise ValueError("Saved audit did not pass")
        rows.append(row_from(record))
        audit_events += record["audit"]["events"]
    if missing:
        raise RuntimeError(f"Analysis requires the frozen complete campaign; {len(missing)} jobs absent")
    worlds = {}
    for row in rows:
        worlds.setdefault(row["world_id"], set()).add(row["base_world_sha256"])
    if any(len(hashes) != 1 for hashes in worlds.values()):
        raise ValueError("Paired geometry/initial state identity differs across treatment or method")
    grouped = {}
    for row in rows:
        grouped.setdefault((row["world_id"], row["condition"], row["method"]), []).append(row)
    aggregated = {}
    metrics = ("qualified_completion", "J", "false_releases", "inspections", "repairs", "travel", "waiting_time", "decision_count", "episode_wall_seconds", "planner_cpu_seconds",
               "successor_revocations", "deadline_misses", "stale_authorizations", "quality_false_releases", "charged_time", "time", "qualified_tasks")
    for key, group in grouped.items():
        aggregated[key] = {m: float(np.mean([r[m] for r in group])) for m in metrics}
        aggregated[key]["layout"] = group[0]["layout"]
    summary_rows, paired_rows, comparisons = [], [], []
    conditions = [c["id"] for c in config["conditions"]]
    for condition in conditions:
        for method in config["methods"]:
            group = [r for (w, c, m), r in aggregated.items() if c == condition and m == method]
            summary_rows.append({"condition": condition, "method": method, "worlds": len(group),
                                 **{m: float(np.mean([r[m] for r in group])) for m in metrics},
                                 "qualified_completion_count": float(sum(r["qualified_completion"] for r in group))})
        reference = config.get("primary_method", "certificate_window")
        if reference not in config["methods"]:
            continue
        for control in [m for m in config["methods"] if m != reference]:
            identifiers = sorted({w for w, c, m in aggregated if c == condition and m == reference})
            a = [aggregated[w, condition, reference] for w in identifiers]
            b = [aggregated[w, condition, control] for w in identifiers]
            completion = np.array([x["qualified_completion"]-y["qualified_completion"] for x,y in zip(a,b)])
            cost = np.array([x["J"]-y["J"] for x,y in zip(a,b)])
            strata = [x["layout"] for x in a]
            wins, losses = int(np.count_nonzero(completion > 0)), int(np.count_nonzero(completion < 0))
            binary = all(x["qualified_completion"] in (0., 1.) for x in a+b)
            p_comp = float(binomtest(wins, wins+losses, .5).pvalue) if binary and wins+losses else 1.0
            if not binary:
                p_comp = signflip_p(completion)
            comparisons.append({"condition": condition, "method": reference, "control": control, "worlds": len(a),
                                "completion_difference": float(np.mean(completion)), "completion_ci95": bootstrap_ci(completion,strata),
                                "completion_gains": wins, "completion_losses": losses, "completion_p_exploratory": p_comp,
                                "J_difference": float(np.mean(cost)), "J_ci95": bootstrap_ci(cost,strata),
                                "J_p_exploratory": signflip_p(cost),
                                "completion_test": "two-sided exact McNemar" if binary else "paired world sign flip"})
            for w,x,y in zip(identifiers,a,b):
                paired_rows.append({"world_id": w, "condition": condition, "layout": x["layout"], "method": reference, "control": control,
                                    **{m+"_difference": x[m]-y[m] for m in metrics}})
    for field in ("completion", "J"):
        adjusted = holm([c[field+"_p_exploratory"] for c in comparisons])
        for row,value in zip(comparisons,adjusted):
            row[field+"_holm_p_exploratory"] = value
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out / "episodes.csv", rows)
    csv_write(out / "method_condition.csv", summary_rows)
    if paired_rows:
        csv_write(out / "paired_differences.csv", paired_rows)
    report = {"schema":"repair-round-analysis-1", "status":"complete_exploratory_development",
              "primary_method":reference,
              "source_freeze_sha256": runner.digest((run_dir / "FREEZE.json").read_bytes()),
              "episodes":len(rows), "unique_worlds":len(worlds), "structural_events_audited":audit_events,
              "summary":summary_rows, "comparisons":comparisons,
              "statistical_notes":["Each method comparison uses identical physical worlds and treatment.",
                   "One physical world is the independent unit; planner repeats average within world.",
                   "95% percentile CIs resample paired worlds inside layout strata (10,000 draws); not simultaneous CIs.",
                   "Separate Holm families span every method-by-condition completion test and every cost test.",
                   "These are development diagnostics, not confirmatory superiority evidence.",
                   "No pooling of treatment episodes as independent samples.",
                   "Cost J is secondary to qualified completion; zero unsafe releases alone is not success."],
              "pairing_audit":"one invariant base-world SHA per world across all methods and conditions"}
    runner.write_json(out / "analysis.json", report)
    print(json.dumps({"episodes":len(rows),"worlds":len(worlds),"out":str(out)}))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=pathlib.Path)
    parser.add_argument("--out",required=True,type=pathlib.Path)
    args=parser.parse_args(argv)
    analyze(args.run,args.out)


if __name__ == "__main__":
    main()
