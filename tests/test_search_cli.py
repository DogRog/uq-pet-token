"""Verify the shared search command's execution boundaries without training."""

import subprocess
import sys
from pathlib import Path

import pytest

SEARCH_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bert_token_uq_search.py"


@pytest.mark.parametrize(
    ("args", "returncode", "expected"),
    [
        (["--help"], 0, "--config-json"),
        ([], 2, "--trials"),
        (["--trials", "0"], 2, "--trials must be positive"),
        (["--trials", "1", "--config-json", '{"sampler":"invalid"}'], 2, "sampler"),
    ],
)
def test_search_execution_boundaries(args, returncode, expected, tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SEARCH_PATH), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == returncode, completed.stderr
    assert expected in completed.stdout + completed.stderr
    assert list(tmp_path.iterdir()) == []
