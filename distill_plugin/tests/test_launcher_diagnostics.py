"""Regression checks for actionable failures and a strictly read-only inspector."""
import argparse
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def launcher():
    path = Path(__file__).resolve().parents[1] / "start_2026-11-17.py"
    spec = importlib.util.spec_from_file_location("launcher_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LauncherDiagnosticsTests(unittest.TestCase):
    def test_gpu_rejection_is_visible_and_preserved(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(run_dir=Path(folder), mode="baseline", seed=0, student_batch=128)
            stderr = io.StringIO()
            with mock.patch.object(module, "reserve_gpu", side_effect=RuntimeError("GPU 0 occupied (test)")), \
                    mock.patch.object(module.subprocess, "Popen") as spawn, \
                    contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.execute(args), 1)
            spawn.assert_not_called()
            self.assertIn("GPU 0 occupied (test)", stderr.getvalue())
            state = json.loads((args.run_dir / "status.json").read_text())
            summary = json.loads((args.run_dir / "summary.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(summary["error"], state["error"])
            self.assertEqual(state["jobs"], {})

    def test_inspector_bypasses_training_and_missing_weight_checks(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "status.json").write_text('{"status":"failed","error":"example failure"}')
            (path / "baseline.log").write_text("example traceback\nRuntimeError: example failure\n")
            stdout = io.StringIO()
            with mock.patch.object(module.sys, "argv", ["launcher", "--inspect-run", folder]), \
                    mock.patch.object(module, "reserve_gpu") as reserve, \
                    mock.patch.object(module.subprocess, "Popen") as spawn, \
                    mock.patch.object(module.subprocess, "run") as query, contextlib.redirect_stdout(stdout):
                query.return_value = argparse.Namespace(returncode=0, stdout="mock GPU status\n")
                self.assertEqual(module.main(), 0)
            reserve.assert_not_called()
            spawn.assert_not_called()
            query.assert_called_once()
            self.assertEqual(query.call_args[0][0], ["nvidia-smi"])
            self.assertIn("example traceback", stdout.getvalue())
            self.assertIn("mock GPU status", stdout.getvalue())

    def test_training_crash_shows_the_child_traceback(self):
        module = launcher()
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(run_dir=Path(folder), mode="baseline", seed=0, student_batch=128)
            stderr = io.StringIO()
            def crash(*args, **kwargs):
                kwargs["stdout"].write("RuntimeError: injected training failure\n")
                kwargs["stdout"].flush()
                return argparse.Namespace(pid=123456, poll=lambda: 2)
            with mock.patch.object(module, "reserve_gpu", return_value=mock.Mock()), \
                    mock.patch.object(module, "command", return_value=["python", "train.py"]), \
                    mock.patch.object(module.subprocess, "Popen", side_effect=crash), \
                    contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.execute(args), 1)
            self.assertIn("RuntimeError: injected training failure", stderr.getvalue())
            state = json.loads((args.run_dir / "status.json").read_text())
            self.assertEqual(state["jobs"]["baseline"]["returncode"], 2)
            self.assertEqual(state["jobs"]["baseline"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
