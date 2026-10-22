#!/usr/bin/env python3
"""Run a frozen, paired handoff campaign using the shared recovery environment."""
import argparse
from pathlib import Path
from run_repair_round import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/handoff_smoke.json'))
    parser.add_argument('--out', type=Path, default=Path('results/handoff_smoke'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    args = parser.parse_args()
    raise SystemExit(run(args.config, args.out, args.freeze_only, args.max_jobs, args.workers))
