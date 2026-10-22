"""Regression checks for protocol identity and the packaged experiment workflow."""
import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from floor_tase.experiments.campaigns import prepare
from floor_tase.experiments.runner import assert_resume, receipt_for, source_identity


class ExperimentPackagingTests(unittest.TestCase):
    def test_v16_protocols_match_original_release(self):
        manifest = json.loads((ROOT / "tests/v16_protocol_hashes.json").read_text())
        for stage, device in (("smoke", "cpu"), ("smoke", "cuda"), ("probe", "cuda"), ("main", "cuda")):
            with self.subTest(stage=stage, device=device):
                payload = (json.dumps(prepare(stage, device), indent=2) + "\n").encode()
                key = f"configs/generated/v16_{stage}_{device}_0of1.json"
                self.assertEqual(hashlib.sha256(payload).hexdigest(), manifest[key])

    def test_compatibility_exports_share_implementation(self):
        for old, new, function in (
            ("prepare_v16", "campaigns", "prepare"),
            ("run_repair_round", "runner", "run"),
            ("analyze_repair_round", "analysis", "analyze"),
            ("report_v16", "reporting", "report"),
            ("merge_v16", "merge", "merge"),
        ):
            with self.subTest(script=old):
                self.assertIs(getattr(importlib.import_module(old), function),
                              getattr(importlib.import_module("floor_tase.experiments." + new), function))

    def test_freeze_covers_relocated_implementations(self):
        files = source_identity()["files"]
        for name in ("campaigns", "runner", "analysis", "reporting", "paths"):
            self.assertIn(f"src/floor_tase/experiments/{name}.py", files)

    def test_changed_source_cannot_resume(self):
        receipt = receipt_for(prepare("smoke", "cpu"))
        changed = copy.deepcopy(receipt)
        changed["source"]["sha256"] = "modified-source"
        with self.assertRaisesRegex(RuntimeError, "Runtime source changed"):
            assert_resume(receipt, changed)

    def test_cli_works_outside_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "protocol.json"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            result = subprocess.run(
                [sys.executable, "-m", "floor_tase", "prepare", "smoke", "--device", "cpu", "--out", str(target)],
                cwd=temp, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(target.read_text()), prepare("smoke", "cpu"))

    def test_cli_freeze_only_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            from floor_tase.experiments.runner import write_json
            config = Path(temp) / "protocol.json"
            write_json(config, prepare("smoke", "cpu"))
            result = subprocess.run(
                [sys.executable, "-m", "floor_tase", "run", "--config", str(config),
                 "--out", str(Path(temp) / "run"), "--freeze-only"],
                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((Path(temp) / "run/FREEZE.json").read_text())
            self.assertEqual(record["n_jobs"], 198)
            self.assertFalse((Path(temp) / "run/COMPLETION.json").exists())

    def test_cli_propagates_incomplete_campaign_exit_status(self):
        with tempfile.TemporaryDirectory() as temp:
            from floor_tase.experiments.runner import write_json
            config = Path(temp) / "protocol.json"
            write_json(config, prepare("smoke", "cpu"))
            result = subprocess.run(
                [sys.executable, "-m", "floor_tase", "run", "--config", str(config),
                 "--out", str(Path(temp) / "run"), "--max-jobs", "0"],
                env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            record = json.loads((Path(temp) / "run/COMPLETION.json").read_text())
            self.assertFalse(record["data_complete"])
            self.assertEqual(record["complete"], 0)


if __name__ == "__main__":
    unittest.main()
