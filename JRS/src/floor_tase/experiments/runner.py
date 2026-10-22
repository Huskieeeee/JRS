#!/usr/bin/env python3
"""Paired repair-round development runner with exact resume identities.

This is a development experiment. Completeness and structural audit success do
not establish superiority, physical validity or author-baseline reproduction.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import multiprocessing
import pathlib
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

from .paths import ROOT


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe(value):
    if dataclasses.is_dataclass(value):
        return safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if hasattr(value, "item"):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(pathlib.Path(path).read_text())


def stable_seed(namespace, *parts):
    return int.from_bytes(hashlib.blake2b(canonical([namespace, *parts]), digest_size=8).digest(), "big") % (2**31 - 1)


def jobs_for(config):
    ids = [c["id"] for c in config["conditions"]]
    if len(ids) != len(set(ids)) or len(config["methods"]) != len(set(config["methods"])):
        raise ValueError("Conditions and method identifiers must be unique.")
    width=int(config['worlds_per_layout'])
    shard=config.get('world_shard',{'index':0,'count':1})
    if not 0<=shard['index']<shard['count']:raise ValueError('Invalid shard')
    for layout_index,layout in enumerate(config['layouts']):
        for replicate in range(shard['index'],width,shard['count']):
            world_id=int(config['world_start'])+layout_index*width+replicate
            for condition in config["conditions"]:
                for method in config["methods"]:
                    for planner_rep in range(int(config.get("planner_replicates", 1))):
                        yield {
                            "job_id": f"w{world_id}__{condition['id']}__{method}__p{planner_rep}",
                            "world_id": world_id, "layout": layout, "world_replicate": replicate,
                            "condition": condition["id"], "intervention": condition.get("changes", {}),
                            "deadline_multiplier": condition.get("deadline_multiplier", 1.0),
                            "method": method, "planner_replicate": planner_rep,
                            "scenario_seed": stable_seed("physical", config["split"], world_id),
                            "measurement_seed": stable_seed("measurement", config["split"], world_id),
                            "planner_seed": stable_seed("planner", config["split"], world_id, planner_rep),
                            "action_mode": "deterministic_given_seed",
                            "budget": config["planning_budget"],
                        }


def source_identity():
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    hashes = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in files}
    return {"files": hashes, "sha256": digest(canonical(hashes))}


def model_identity(config):
    budget = config['planning_budget']
    model_key = 'release_model_path' if budget.get('release_model_path') else 'surrogate_path'
    hash_key = 'release_model_sha256' if model_key == 'release_model_path' else 'surrogate_sha256'
    if not budget.get(model_key):
        return None
    import numpy as np
    path = ROOT / budget[model_key]
    actual = digest(path.read_bytes())
    if actual != budget.get(hash_key):
        raise RuntimeError('Frozen surrogate hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
    evaluation = {f"{config['split']}:{j['world_id']}" for j in jobs_for(config)}
    used = set(meta['training_worlds']) | set(meta['validation_worlds'])
    if evaluation & used:
        raise RuntimeError('Evaluation overlaps surrogate training/validation worlds')
    return {'sha256':actual,'metadata':meta}


def receipt_for(config):
    jobs = list(jobs_for(config))
    if not jobs:
        raise ValueError("Empty experiment")
    return {
        "schema": "floor-repair-paired-1", "created_utc": utc(),
        "config_sha256": digest(canonical(config)), "source": source_identity(),
        "jobs_sha256": digest(canonical(jobs)), "n_jobs": len(jobs),
        "n_worlds": len({j["world_id"] for j in jobs}), "split": config["split"],
        "python": platform.python_version(), "platform": platform.platform(),
        "status": "local_development_freeze_not_third_party_preregistration",
        "training_jobs": 0, "external_model":model_identity(config),
    }


def assert_resume(old, new):
    for key in ("schema", "config_sha256", "jobs_sha256", "n_jobs", "n_worlds", "split", "external_model"):
        if old.get(key) != new.get(key):
            raise RuntimeError(f"Resume identity changed: {key}. Use a new output directory.")
    if old["source"]["sha256"] != new["source"]["sha256"]:
        raise RuntimeError("Runtime source changed after freeze. Use a new output directory.")


def check_formal(config):
    """Hard block: this release provides development entrypoints, not a final claim."""
    if config.get("split") == "confirmation" or config.get("formal", False):
        gate = load_json(ROOT / "protocol" / "CONFIRMATION_GATE.json")
        required = ("algorithm_validated", "native_author_adapter_validated", "physical_calibration_validated", "protocol_reviewed")
        missing = [k for k in required if gate.get(k) is not True]
        # Native/physical status must be backed by concrete readable evidence,
        # not merely a boolean entered in a configuration.
        for key in ("author_artifact", "measurement_artifact", "frozen_protocol_artifact"):
            value = gate.get(key) or {}
            path = ROOT / value.get("path", "__missing__")
            if not path.is_file() or digest(path.read_bytes()) != value.get("sha256"):
                missing.append(key)
        if missing:
            raise RuntimeError("FORMAL EXECUTION BLOCKED: " + ", ".join(missing))


def scenario_for(job, config):
    from floor_tase.model import ScenarioConfig, make_scenario
    # Treatments are applied to ONE generated base scenario. In particular a
    # budget treatment must NOT regenerate horizon-scaled task deadlines.
    settings = dict(config["base_scenario"], layout=job["layout"])
    base = make_scenario(job["scenario_seed"], ScenarioConfig(**settings), measurement_seed=job["measurement_seed"])
    allowed = {"collateral_probability", "repair_failure_prob", "assumed_collateral_probability", "assumed_repair_failure_prob", "shared_passage", "horizon", "freshness"}
    if set(job["intervention"]) - allowed:
        raise ValueError("Intervention modifies non-isolated base geometry/task fields")
    cfg = dataclasses.replace(base.config, **job["intervention"])
    if job['method'] in ('whole_reset','jrs_rg_no_reuse'):
        cfg=dataclasses.replace(cfg,evidence_policy='whole_reset')
    multiplier = float(job["deadline_multiplier"])
    tasks = tuple(dataclasses.replace(t, deadline=t.deadline * multiplier) for t in base.tasks)
    scenario = dataclasses.replace(base, config=cfg, tasks=tasks, id=f"paired-w{job['world_id']}-{job['condition']}")
    invariant = {"seed": base.seed, "measurement_seed": base.measurement_seed,
                 "cells": base.cells, "robots": base.robots, "jobs": base.jobs,
                 "views": base.views, "initial_quality": base.initial_quality,
                 "base_tasks": base.tasks}
    return scenario, {
        "base_world_sha256": digest(canonical(invariant)),
        "scenario_sha256": digest(canonical(scenario)),
        "actual_deadlines_sha256": digest(canonical([(t.id, t.deadline) for t in scenario.tasks])),
        "shared_latent_noise": "physical/sensor keyed seeds contain neither method nor treatment",
        "deadline_changed_intentionally": multiplier != 1.0,
    }


def planner_for(job):
    budget = job["budget"]
    if job['method'].startswith('jrs_rg'):
        from floor_tase.rg_variants import make_rg
        return make_rg(job['method'],job['planner_seed'],budget)
    if job['method'] in ('qa_sequential','robust_local','whole_reset','overlap_batch','independent'):
        from floor_tase.planners import make_planner
        return make_planner(job['method'],job['planner_seed'])
    if job['method'].startswith('jrs_release'):

        from floor_tase.release_planner import ReleasePlanner,ReleaseOptions
        variants={'jrs_release':{},'jrs_release_no_chain':{'chain_guard':False},
                  'jrs_release_no_commit':{'commit_events':0},'jrs_release_no_gate':{'gain_gate':False}}
        if job['method'] not in variants:raise ValueError('Unknown release method')
        o=ReleaseOptions(model_path=str(ROOT/budget['release_model_path']),
            model_sha256=budget['release_model_sha256'],device=budget.get('release_device','cuda'),
            candidates=budget.get('release_candidates',8),provider_candidates=budget.get('max_candidates',32),
            provider_cover_nodes=budget.get('cover_nodes',2000),commit_events=budget.get('release_commit_events',2),
            completion_margin=budget.get('release_completion_margin',.05),cost_margin=budget.get('release_cost_margin',.10),
            completion_tolerance=budget.get('release_completion_tolerance',.02),ensemble_penalty=budget.get('release_ensemble_penalty',1.))
        return ReleasePlanner(job['planner_seed'],dataclasses.replace(o,**variants[job['method']]))
    if job["method"] in ("arch_staged", "arch_dispatch"):
        from floor_tase.joint_recovery import ArchitecturePlanner
        return ArchitecturePlanner(job["method"])
    if job['method'].startswith('jrs_evidence'):
        from floor_tase.evidence_recovery import EvidenceRecoveryPlanner, EvidenceOptions
        variants = {'jrs_evidence':{}, 'jrs_evidence_no_guard':{'protect_evidence':False},
            'jrs_evidence_single':{'continuations':('evidence',)},
            'jrs_evidence_handoff':{'continuations':('handoff',)},
            'jrs_evidence_screened':{}}
        if job['method'] not in variants: raise ValueError('Unknown evidence method')
        screened = job['method'] == 'jrs_evidence_screened'
        if screened and not budget.get('surrogate_path'):
            raise ValueError('Screened method requires a trained model')
        o = EvidenceOptions(candidates=budget.get('evidence_candidates',8),
            samples=budget.get('evidence_samples',4), max_events=budget.get('rollout_events',2000),
            provider_candidates=budget.get('max_candidates',32),provider_cover_nodes=budget.get('cover_nodes',2000),
            max_wait=budget.get('evidence_max_wait',2.),slack_fraction=budget.get('evidence_slack_fraction',.25),
            surrogate_path=str(ROOT / budget['surrogate_path']) if screened else '',
            surrogate_sha256=budget.get('surrogate_sha256','') if screened else '',
            device=budget.get('surrogate_device','cpu'),shortlist=budget.get('surrogate_shortlist',4))
        return EvidenceRecoveryPlanner(job['planner_seed'],dataclasses.replace(o,**variants[job['method']]))
    if job["method"].startswith("jrs_joint"):
        from floor_tase.joint_recovery import JointRecoveryPlanner, JointOptions
        options = JointOptions(candidates=budget.get("joint_candidates", 8),
            samples=budget.get("joint_samples", 4), max_events=budget.get("rollout_events", 2000),
            provider_candidates=budget.get("max_candidates", 32),
            provider_cover_nodes=budget.get("cover_nodes", 2000),
            tail_weight=budget.get("joint_tail_weight", .25),
            collateral_margin=budget.get("joint_collateral_margin", .10),
            failure_margin=budget.get("joint_failure_margin", .05))
        variants = {"jrs_joint": {}, "jrs_joint_no_provider": {"retain_provider": False},
            "jrs_joint_local_scope": {"adaptive_scope": False},
            "jrs_joint_no_diagnostic": {"diagnostic_checks": False},
            "jrs_joint_global_wait": {"local_dependencies": False},
            "jrs_joint_central": {"robust_models": False},
            "jrs_joint_central_no_tail": {"robust_models":False,"tail_weight":0.},
            "jrs_joint_no_tail": {"tail_weight": 0.},
            "jrs_joint_dispatch": {"provider": "arch_dispatch"},
            "jrs_joint_dispatch_continuation": {"continuation": "dispatch"}}
        if job["method"] not in variants:
            raise ValueError("Unknown joint method: " + job["method"])
        return JointRecoveryPlanner(job["planner_seed"], dataclasses.replace(options, **variants[job["method"]]))
    if job["method"].startswith("handoff_adaptive"):
        from floor_tase.adaptive_handoff import AdaptiveOptions, AdaptiveHandoffPlanner
        options = AdaptiveOptions(max_candidates=budget.get("adaptive_candidates",6),
                                  finalists=budget.get("adaptive_finalists",2),
                                  min_samples=budget.get("adaptive_min_samples",2),
                                  max_samples=budget.get("adaptive_max_samples",4),
                                  max_rollout_events=budget.get("rollout_events",2000),
                                  nominal_weight=budget.get("adaptive_nominal_weight",.5))
        variants = {"handoff_adaptive": {},
                    "handoff_adaptive_always_diagnostic": {"diagnostic_policy":"always"},
                    "handoff_adaptive_no_fast": {"fast_dispatch":False},
                    "handoff_adaptive_no_protection": {"protect_handoffs":False},
                    "handoff_adaptive_full_risk": {"nominal_weight":0.},
                    "handoff_adaptive_local_scope": {"adaptive_scope":False}}
        if job["method"] not in variants:raise ValueError("Unknown adaptive variant: "+job["method"])
        return AdaptiveHandoffPlanner(seed=job["planner_seed"],options=dataclasses.replace(options,**variants[job["method"]]))
    if job["method"].startswith("handoff"):
        from floor_tase.handoff import HandoffOptions, HandoffPlanner
        from dataclasses import replace
        options = HandoffOptions(max_candidates=budget.get("handoff_candidates", 8),
                                 samples=budget.get("handoff_samples", 3),
                                 max_rollout_events=budget.get("rollout_events", 2000))
        variants = {"handoff": {}, "handoff_rule": {},
                    "handoff_no_diagnostic": {"diagnostic_checks": False},
                    "handoff_global_wait": {"local_dependencies": False},
                    "handoff_local_scope": {"adaptive_scope": False},
                    "handoff_optimistic": {"prediction_mode": "optimistic"},
                    "handoff_cascade": {"prediction_mode": "cascade"}}
        if job["method"] not in variants:
            raise ValueError(f"Unknown handoff method: {job['method']}")
        return HandoffPlanner(seed=job["planner_seed"], options=replace(options, **variants[job["method"]]),
                              rollout=job["method"] != "handoff_rule")
    if job["method"] in ("fixed_jrs", "fixed_random"):
        from floor_tase.deterministic_legacy import FixedBudgetPlanner
        return FixedBudgetPlanner(name="jrs" if job["method"] == "fixed_jrs" else "random_rollout",
                                  seed=job["planner_seed"], max_candidates=budget["max_candidates"],
                                  cover_nodes=budget["cover_nodes"])
    if job["method"] in ("certificate_window", "certificate_window_no_cuts", "certificate_window_staged"):
        from floor_tase.window_planner import WindowPlanner
        return WindowPlanner(seed=job["planner_seed"], max_candidates=budget["max_candidates"],
                             max_nodes=budget["window_nodes"], use_cuts=job["method"] != "certificate_window_no_cuts",
                             coordinate_windows=job["method"] != "certificate_window_staged")
    raise ValueError(f"Unknown method: {job['method']}")


def run_job(job, config, folder):
    from floor_tase.environment import RecoveryEnv, audit_trace
    scenario, pairing = scenario_for(job, config)
    env = RecoveryEnv(scenario)
    wall_start = time.perf_counter()
    planner = planner_for(job)
    planner_init_wall = time.perf_counter()-wall_start
    records, snapshots = [], []
    snapshot_indices = set(config.get("snapshot_decisions", [0, 5, 15]))
    while not env.done:
        if len(records) >= config.get("max_decisions", 10000):
            raise RuntimeError("Decision guard reached; not a physical infeasibility result")
        before, allowed = env.observe(), env.allowed_actions()
        index = len(records)
        if index in snapshot_indices:
            snapshots.append({"decision_index": index, "observation": safe(before), "allowed_actions": safe(allowed)})
        start = time.perf_counter()
        cpu_start = time.process_time()
        actions = tuple(planner.select(before, allowed))
        records.append({"index": index, "time": before.time, "actions": safe(actions),
                        "wall_seconds": time.perf_counter() - start,
                        "cpu_seconds": time.process_time() - cpu_start,
                        "diagnostics": safe(getattr(planner, "last_diagnostics", {})),
                        "idle_free_robots": sum(r.busy_until <= before.time + 1e-9 and
                                                r.id not in {a.robot_id for a in actions} for r in before.robots),
                        "legally_available_actions": len(allowed)})
        if actions:
            env.step_joint(actions)
        else:
            env.advance()
    metrics, trace = safe(env.result()), safe(env.get_trace())
    audit = audit_trace(trace)
    if not audit["pass"]:
        raise RuntimeError("Structural trajectory audit failed: " + str(audit["errors"]))
    write_json(folder / "scenario.json", scenario)
    write_json(folder / "decisions.json", records)
    write_json(folder / "public_snapshots.json", snapshots)
    with (folder / "trajectory.jsonl").open("w") as stream:
        stream.write(json.dumps({"record_type": "metadata", "job": job, "pairing": pairing}, sort_keys=True) + "\n")
        for event in trace:
            stream.write(json.dumps({"record_type": "event", **event}, sort_keys=True) + "\n")
    artifacts = {name: digest((folder / name).read_bytes()) for name in
                 ("scenario.json", "decisions.json", "public_snapshots.json", "trajectory.jsonl")}
    return {"job": job, "status": "complete", "finished_utc": utc(), "metrics": metrics,
            "audit": audit, "pairing": pairing, "planner": safe(planner.diagnostics()),
            "planner_initialization_wall_seconds":planner_init_wall,
            "planner_wall_seconds":sum(r["wall_seconds"] for r in records),
            "wall_seconds": time.perf_counter() - wall_start, "planner_cpu_seconds": sum(r["cpu_seconds"] for r in records),
            "decision_count": len(records), "artifacts": artifacts}


def audit_saved(folder, job):
    result = load_json(folder / "result.json")
    if result.get("job") != job or result.get("status") != "complete":
        raise RuntimeError(f"Saved job is mismatched/incomplete: {job['job_id']}. Use a new output or investigate; errors are not skipped.")
    for name, expected in result["artifacts"].items():
        if digest((folder / name).read_bytes()) != expected:
            raise RuntimeError(f"Saved artifact changed: {job['job_id']}/{name}")
    return result


def execute_and_save(job, config, folder):
    folder.mkdir(parents=True, exist_ok=True)
    try:
        record = run_job(job, config, folder)
    except Exception as exc:
        record = {"job": job, "status": "execution_error", "error": str(exc), "traceback": traceback.format_exc(), "finished_utc": utc()}
    write_json(folder / "result.json", record)
    return {"job":job["job_id"], "status":record["status"]}


def run(config_path, out, freeze_only=False, max_jobs=None, workers=1):
    config = load_json(config_path)
    check_formal(config)
    if not freeze_only and any(str(config['planning_budget'].get(k,'cpu')).startswith('cuda') for k in ('release_device','rg_device')):
        import torch
        if not torch.cuda.is_available():raise RuntimeError('This campaign requires CUDA; no CPU fallback')
    expected = receipt_for(config)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "FREEZE.json"
    if receipt_path.exists():
        assert_resume(load_json(receipt_path), expected)
    else:
        if any(out.iterdir()):
            raise RuntimeError("Non-empty output has no freeze receipt")
        write_json(receipt_path, expected)
        write_json(out / "protocol.json", config)
        if expected['external_model']:
            budget=config['planning_budget']
            model = ROOT / (budget.get('release_model_path') or budget['surrogate_path'])
            (out/'surrogate_snapshot.npz').write_bytes(model.read_bytes())
        for relative in expected["source"]["files"]:
            target = out / "runtime_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / relative).read_bytes())
    if freeze_only:
        print(json.dumps({"status": "frozen_not_run", "jobs": expected["n_jobs"], "out": str(out)}), flush=True)
        return 0
    jobs = list(jobs_for(config))
    done, executed, failures, pending = 0, 0, 0, []
    for job in jobs:
        folder = out / "jobs" / job["job_id"]
        if (folder / "result.json").exists():
            audit_saved(folder, job)
            done += 1
        else:
            pending.append(job)
    if max_jobs is not None:
        pending = pending[:max_jobs]
    workers = max(1, int(workers))
    # Bounded batches avoid an unbounded submitted queue on a detected error.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for start in range(0, len(pending), workers):
            batch = pending[start:start+workers]
            futures = [pool.submit(execute_and_save, job, config, out / "jobs" / job["job_id"]) for job in batch]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                executed += 1
                if record["status"] == "complete":
                    done += 1
                else:
                    failures += 1
                print(json.dumps({**record, "completed":done, "planned":len(jobs), "errors":failures}), flush=True)
            if failures:
                break
    complete = done == len(jobs) and not failures
    write_json(out / "COMPLETION.json", {"updated_utc": utc(), "planned": len(jobs), "complete": done,
               "errors": failures, "data_complete": complete, "executed_this_invocation": executed,
               "workers":workers, "performance_superiority": "not determined by completeness", "physical_validation": False})
    return 0 if complete else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        raise SystemExit(run(args.config, args.out, args.freeze_only, args.max_jobs, args.workers))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
