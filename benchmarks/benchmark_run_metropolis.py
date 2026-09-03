"""Reproducible benchmark for :func:`ising_mc.run_metropolis`.

Run from the repository root with::

    uv run python benchmarks/benchmark_run_metropolis.py --label my-change

By default, each invocation appends one row to ``benchmarks/results.csv``.
Use ``--no-save`` for an unrecorded exploratory run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ising_mc import run_metropolis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "src" / "ising_mc" / "functions" / "function_defs.py"
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark run_metropolis and optionally append the result to CSV."
    )
    parser.add_argument("--label", default="manual", help="Description of this run.")
    parser.add_argument("--size", type=int, default=16, help="Linear lattice size.")
    parser.add_argument("--temperature", type=float, default=2.269)
    parser.add_argument("--coupling", type=float, default=-1.0)
    parser.add_argument("--field", type=float, default=1e-2)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument(
        "--burn-in-steps",
        type=int,
        default=None,
        help="Single-spin burn-in steps; defaults to 10 * size**2.",
    )
    parser.add_argument(
        "--steps-between-samples",
        type=int,
        default=None,
        help="Single-spin steps per sample; defaults to size**2 // 4.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_902)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the benchmark without appending it to the output CSV.",
    )
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")

    if args.burn_in_steps is None:
        args.burn_in_steps = 10 * args.size**2
    if args.steps_between_samples is None:
        args.steps_between_samples = args.size**2 // 4
    if args.burn_in_steps < 0:
        parser.error("--burn-in-steps cannot be negative")
    if args.steps_between_samples < 0:
        parser.error("--steps-between-samples cannot be negative")

    return args


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def source_digest() -> str:
    return hashlib.sha256(SOURCE_FILE.read_bytes()).hexdigest()[:12]


def benchmark(args: argparse.Namespace) -> tuple[list[float], tuple[float, ...]]:
    parameters = {
        "L": args.size,
        "beta": 1 / args.temperature,
        "J": args.coupling,
        "h": args.field,
        "num_samples": args.samples,
        "burn_in_steps": args.burn_in_steps,
        "steps_between_samples": args.steps_between_samples,
    }

    for _ in range(args.warmups):
        np.random.seed(args.seed)
        run_metropolis(**parameters)

    durations = []
    reference_result = None
    for _ in range(args.repeats):
        np.random.seed(args.seed)
        start = time.perf_counter()
        result = run_metropolis(**parameters)
        durations.append(time.perf_counter() - start)

        if reference_result is None:
            reference_result = result
        else:
            np.testing.assert_allclose(result, reference_result, rtol=0, atol=1e-12)

    return durations, tuple(float(value) for value in reference_result)


def make_result_row(
    args: argparse.Namespace,
    durations: list[float],
    observables: tuple[float, ...],
) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "git_revision": git_revision(),
        "git_dirty": git_is_dirty(),
        "source_sha256": source_digest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "size": args.size,
        "temperature": args.temperature,
        "coupling": args.coupling,
        "field": args.field,
        "samples": args.samples,
        "burn_in_steps": args.burn_in_steps,
        "steps_between_samples": args.steps_between_samples,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "seed": args.seed,
        "median_seconds": statistics.median(durations),
        "mean_seconds": statistics.fmean(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "stdev_seconds": statistics.pstdev(durations),
        "raw_seconds": ";".join(f"{duration:.9f}" for duration in durations),
        "energy_per_site": observables[1],
        "energy_variance": observables[2],
        "magnetization_per_site": observables[3],
        "magnetization_variance": observables[4],
        "absolute_magnetization": observables[5],
    }


def append_result(path: Path, row: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_result(
    args: argparse.Namespace,
    durations: list[float],
    observables: tuple[float, ...],
) -> None:
    print(
        f"L={args.size}, T={args.temperature}, samples={args.samples}, "
        f"burn-in steps={args.burn_in_steps}, "
        f"steps between samples={args.steps_between_samples}"
    )
    print("runs (s):", ", ".join(f"{duration:.6f}" for duration in durations))
    print(f"median (s): {statistics.median(durations):.6f}")
    print(f"mean (s): {statistics.fmean(durations):.6f}")
    print("observables:", ", ".join(f"{value:.12g}" for value in observables))


def main() -> None:
    args = parse_args()
    durations, observables = benchmark(args)
    print_result(args, durations, observables)

    if not args.no_save:
        row = make_result_row(args, durations, observables)
        append_result(args.output, row)
        print(f"appended result to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
