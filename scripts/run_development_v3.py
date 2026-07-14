#!/usr/bin/env python
"""Run resumable V3 synthetic development experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

from pacbayes_tsk.experiments.development_v3 import (
    DevelopmentSettings,
    aggregate_development,
    run_series,
)


def _worker(payload):
    process, seed, settings, output_dir, protocol_paths = payload
    # Avoid nested BLAS oversubscription in multi-process runs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return run_series(
        process=process,
        seed=seed,
        settings=settings,
        output_dir=output_dir,
        protocol_paths=protocol_paths,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/v3/protocol_v3.yaml")
    parser.add_argument("--synthetic", default="configs/v3/synthetic_v3.yaml")
    parser.add_argument("--development", default="configs/v3/development_v3.yaml")
    parser.add_argument("--output", default="results/development/full_v3")
    parser.add_argument("--processes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = DevelopmentSettings.from_files(
        protocol_path=args.protocol,
        synthetic_path=args.synthetic,
        development_path=args.development,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_paths = tuple(str(Path(x).resolve()) for x in (args.protocol, args.synthetic, args.development))

    if args.aggregate_only:
        print(json.dumps(aggregate_development(output), indent=2))
        return 0

    processes = tuple(args.processes) if args.processes else settings.processes
    seeds = tuple(args.seeds) if args.seeds else settings.seeds
    invalid_processes = set(processes) - set(settings.processes)
    invalid_seeds = set(seeds) - set(settings.seeds)
    if invalid_processes or invalid_seeds:
        raise SystemExit(
            f"Outside frozen development support: processes={sorted(invalid_processes)}, seeds={sorted(invalid_seeds)}"
        )

    tasks = []
    skipped = []
    for process in processes:
        for seed in seeds:
            stem = f"{process}_seed{seed}"
            selected_path = output / "series" / f"{stem}_selected.csv"
            audit_path = output / "series" / f"{stem}_audit.json"
            if not args.no_resume and selected_path.exists() and audit_path.exists():
                skipped.append((process, seed))
                continue
            tasks.append((process, seed, settings, str(output), protocol_paths))

    print(f"Development tasks: {len(tasks)}; resumed/skipped: {len(skipped)}")
    results = []
    if args.workers <= 1:
        for task in tasks:
            result = _worker(task)
            results.append(result)
            print(
                f"completed {result['process']} seed={result['seed']} "
                f"candidates={result['candidate_count']} runtime={result['runtime_seconds']:.1f}s"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_worker, task): task[:2] for task in tasks}
            for future in as_completed(futures):
                process, seed = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    print(f"FAILED {process} seed={seed}: {type(error).__name__}: {error}", file=sys.stderr)
                    raise
                results.append(result)
                print(
                    f"completed {result['process']} seed={result['seed']} "
                    f"candidates={result['candidate_count']} runtime={result['runtime_seconds']:.1f}s"
                )

    if tasks or skipped:
        paths = aggregate_development(output)
        print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
