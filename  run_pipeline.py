from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    PROJECT_ROOT / "src" / "bronze.py",
    PROJECT_ROOT / "src" / "silver.py",
    PROJECT_ROOT / "src" / "gold.py",
]


def main() -> None:
    for script in PIPELINE_STEPS:
        print(f"\n{'=' * 60}")
        print(f"Running {script.name}")
        print(f"{'=' * 60}")

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:
            raise SystemExit(
                f"Pipeline stopped because {script.name} failed."
            )

    print("\nClinical Lakehouse pipeline completed successfully.")


if __name__ == "__main__":
    main()