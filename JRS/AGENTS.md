# JRS / FLOOR-MAS v1.6 主体代码

This is a Python research simulator for rework, recertification and successor handoff. Communicate in Chinese. This compact source distribution contains 73 delivered files; generated environments and experiment outputs are excluded.

Read README.md and docs/ARCHITECTURE_ZH.md. The method implementation is src/floor_tase/recovery_graph.py and rg_variants.py; shared physical dynamics and original J are in model.py and environment.py. Experiment configuration, execution, statistics and plotting live in src/floor_tase/experiments/. Retain the provided v1.6 compatibility scripts.

For Codex Cloud, use Python 3.12 and run `bash scripts/dev/cloud_setup.sh` from the repository checkout. The script creates a project .venv, installs CPU dependencies and runs checks. Later task shells should enter the project directory and use `.venv/bin/python -m floor_tase`. No API key or Torch is required for CPU work. Environment setup does not need to leave an activated shell behind.

Validation: `bash scripts/dev/check.sh` runs compilation, Shell syntax, 18 tests, asset checks and six paired developer jobs. For targeted work, run relevant tests and a new developer smoke output directory. Standard full smoke has 198 jobs; main has 66,000. Routine development uses the six-job check. Run larger campaigns when the task requires them.

Keep paired physical and observation seeds, original cost J, quality and deadline definitions consistent unless the requested research change requires otherwise. Preserve model checksums, trace audits, freeze and resume identity. New runtime source means a new output directory. Keep failed episodes. Explicit CUDA requests must fail clearly when CUDA is unavailable. Preserve the existing validated server Torch installation for GPU work.

Keep .venv, caches, generated configs, results, logs and plots out of Git. Add behavior-focused regression tests when changing scientific logic or the runner. Report changed behavior and actual checks; distinguish local CPU validation from GPU or published-cloud execution. Add new source files when the task requires them; the 73-file count describes this delivery.

## Code Review Rules

Flag changed job pairing, hidden world state reaching planners, altered metric definitions, silently substituted CPU execution, dropped failures, and broken multiprocessing-spawn imports. Ensure the single aggregate ensemble model continues to support all v1.6 methods.
