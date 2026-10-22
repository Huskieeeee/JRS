"""Repository-aware command line interface for research workflows."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


COMMANDS = {
    "prepare": ("campaigns", "Generate a v1.6 smoke/probe/main protocol"),
    "run": ("runner", "Run or resume a frozen paired campaign"),
    "analyze": ("analysis", "Audit and summarize a complete paired campaign"),
    "report": ("reporting", "Generate v1.6 comparisons and factorial reports"),
    "plot": ("plotting", "Export v1.6 PDF/PNG figures"),
    "benchmark": ("benchmark", "Benchmark graph evaluation and CPU/CUDA parity"),
    "merge": ("merge", "Merge completed GPU shards"),
    "budget": ("budget", "Run the candidate budget sweep"),
    "gpu": ("gpu", "Inspect Torch/CUDA availability"),
}


def doctor(require_cuda=False):
    """Check actual imports and model hashes; CPU work does not need Torch."""
    from .experiments.paths import ROOT
    from .experiments.campaigns import prepare
    from .experiments.runner import jobs_for, model_identity

    versions = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        importlib.import_module(name)
        versions[name] = importlib.metadata.version(name)
    config = prepare("smoke", "cpu")
    identity = model_identity(config)
    state = {
        "python": platform.python_version(),
        "repository": str(ROOT),
        "dependencies": versions,
        "model_sha256": identity["sha256"],
        "v16_smoke_jobs": len(list(jobs_for(config))),
        "cuda_available": "not_checked_cpu_workflow",
    }
    if require_cuda:
        import torch
        state["cuda_available"] = torch.cuda.is_available()
        if not state["cuda_available"]:
            raise RuntimeError("CUDA requested but unavailable")
        state["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(state, indent=2))
    return 0


def smoke(out, workers=1):
    """Six CPU jobs for developer feedback, using separate development seeds."""
    from .experiments.campaigns import prepare
    from .experiments.runner import load_json, run, write_json

    config = prepare("smoke", "cpu")
    config.update(
        split="dev_ci_v16", world_start=16900000, layouts=["hall"],
        methods=["jrs_rg", "fixed_jrs"],
        claims="Developer integration check only; not a performance experiment.",
    )
    config["planning_budget"].update(rg_candidates=4, rg_samples=8)
    out = Path(out).resolve()
    config_path = out.parent / (out.name + ".config.json")
    if config_path.exists() and load_json(config_path) != config:
        raise RuntimeError("Development protocol changed; choose a new --out path")
    write_json(config_path, config)
    status = run(config_path, out, workers=workers)
    if status == 0:
        from .experiments.analysis import analyze
        analyze(out, out / "analysis")
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, (_, help_text) in COMMANDS.items():
        sub.add_parser(command, help=help_text, add_help=False)
    check = sub.add_parser("doctor", help="Check CPU dependencies and bundled assets")
    check.add_argument("--require-cuda", action="store_true")
    quick = sub.add_parser("smoke", help="Run six small CPU integration jobs")
    quick.add_argument("--out", type=Path, required=True)
    quick.add_argument("--workers", type=int, default=1)
    args, remaining = parser.parse_known_args(argv)
    if args.command in COMMANDS:
        module = importlib.import_module(".experiments." + COMMANDS[args.command][0], "floor_tase")
        return module.main(remaining)
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.command == "doctor":
        return doctor(args.require_cuda)
    if args.workers < 1:
        parser.error("--workers must be positive")
    return smoke(args.out, args.workers)


if __name__ == "__main__":
    sys.exit(main())
