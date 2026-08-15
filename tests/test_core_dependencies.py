from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from moe_cache_lab.granite_dependencies import (
    GRANITE_DEPENDENCY_MESSAGE,
    GraniteDependencyError,
    load_granite_modules,
)
from moe_cache_lab.switch_dependencies import (
    SwitchDependencyError,
    load_switch_modules,
)


ROOT = Path(__file__).resolve().parents[1]


class CoreDependencyBoundaryTests(unittest.TestCase):
    def test_offline_import_graph_does_not_import_torch_or_transformers(self) -> None:
        script = r'''
import builtins
import contextlib
import io
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'transformers'}:
        raise AssertionError('offline import attempted heavy dependency: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import moe_cache_lab
import moe_cache_lab.cli as cli
import moe_cache_lab.preflight
import moe_cache_lab.workflow
from moe_cache_lab.trace import load_trace_schema
assert load_trace_schema()['$id'] == 'urn:moe-cache-lab:schema:routing-trace:1'
sys_argv = __import__('sys').argv
__import__('sys').argv = ['moe-cache-lab', '--help']
try:
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            cli.main()
        except SystemExit as error:
            assert error.code == 0
finally:
    __import__('sys').argv = sys_argv
print(moe_cache_lab.__version__)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "0.7.0")

    def test_missing_dependency_is_translated_to_bounded_granite_error(self) -> None:
        missing = ModuleNotFoundError("No module named 'torch'", name="torch")
        with patch("moe_cache_lab.granite_dependencies.import_module", side_effect=missing):
            with self.assertRaisesRegex(GraniteDependencyError, r"moe-cache-lab\[granite\]"):
                load_granite_modules()

    def test_missing_dependency_is_translated_to_bounded_switch_error(self) -> None:
        missing = ModuleNotFoundError("No module named 'torch'", name="torch")
        with patch("moe_cache_lab.switch_dependencies.import_module", side_effect=missing):
            with self.assertRaisesRegex(SwitchDependencyError, r"moe-cache-lab\[switch\]"):
                load_switch_modules()

    def test_collector_commands_report_extra_without_traceback(self) -> None:
        commands = (
            ["moe-cache-lab", "inspect"],
            ["moe-cache-lab", "collect", "--prompt", "x", "--output", "trace.jsonl"],
            ["moe-cache-lab", "collect-corpus", "--output-dir", "corpus"],
            ["moe-cache-lab", "collect-stage1", "--output-dir", "stage1"],
            ["moe-cache-lab", "collect-v04", "--output-root", "v04"],
        )
        for arguments in commands:
            stdout = StringIO()
            stderr = StringIO()
            with self.subTest(command=arguments[1]), patch.object(
                sys, "argv", arguments
            ), patch(
                "moe_cache_lab.cli.load_granite_modules",
                side_effect=GraniteDependencyError(GRANITE_DEPENDENCY_MESSAGE),
            ), patch(
                "moe_cache_lab.workflow.load_granite_modules",
                side_effect=GraniteDependencyError(GRANITE_DEPENDENCY_MESSAGE),
            ), redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as raised:
                from moe_cache_lab.cli import main

                main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("moe-cache-lab[granite]", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")

    def test_dependency_metadata_declares_only_model_collector_extras(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("dependencies = []", pyproject)
        self.assertIn("[project.optional-dependencies]", pyproject)
        self.assertIn('granite = [', pyproject)
        self.assertIn('switch = [', pyproject)
        self.assertIn('"torch==2.12.0"', pyproject)
        self.assertIn('"transformers==5.12.0"', pyproject)


if __name__ == "__main__":
    unittest.main()
