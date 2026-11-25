
import subprocess
import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stage 1 experiments in order
experiments = [
    "baseline.py",
    "preprocessing.py",
    "logistic_regression.py",
    "threshold_analysis.py",
    "balanced_logistic.py"
]

print("=" * 60)
print("BANK ACCOUNT FRAUD DETECTION - STAGE 1")
print("=" * 60)

for experiment in experiments:

    script_path = PROJECT_ROOT / "stage1" / experiment

    print(f"\nRunning {experiment}...", flush=True)
    print("-" * 60, flush=True)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True
    )

    print(
        f"\n{experiment} completed successfully!",
        flush=True
    )

print("\n" + "=" * 60)
print("ALL STAGE 1 EXPERIMENTS COMPLETED!")
print("=" * 60)

print("\nGenerated figures are in:")
print(PROJECT_ROOT / "reports" / "stage1")
