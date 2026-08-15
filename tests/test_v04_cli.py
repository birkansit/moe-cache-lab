from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
from unittest import mock

from moe_cache_lab import cli


class V04CliTests(unittest.TestCase):
    def test_retry_dispatch_preserves_corpus_output_and_lineage_arguments(self):
        corpus = Path("benchmarks/custom-corpus.json")
        output_root = Path("results/custom-root")
        retry_of = Path(
            "results/v0.4-cpu-profiler-attempt-"
            "d6f7cac2-d67a-46e8-8267-d65c861ea009/v04-set.json"
        )
        terminal = Path("results/new-attempt/v04-set.json")
        argv = [
            "moe-cache-lab", "collect-v04",
            "--corpus", str(corpus),
            "--output-root", str(output_root),
            "--retry-of", str(retry_of),
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch(
            "moe_cache_lab.v04.collect_v04", return_value=terminal,
        ) as collect, redirect_stdout(StringIO()) as stdout:
            cli.main()

        collect.assert_called_once_with(
            corpus, output_root, rerun_of=None, retry_of=retry_of,
        )
        self.assertEqual(stdout.getvalue().strip(), str(terminal))


if __name__ == "__main__":
    unittest.main()
