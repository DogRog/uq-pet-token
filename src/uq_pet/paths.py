"""Where the project keeps its data, results, and `.env`, resolved from this checkout."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
