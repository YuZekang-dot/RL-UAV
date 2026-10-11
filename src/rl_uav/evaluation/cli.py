"""Command-line entry point for evidence aggregation and synthetic smoke data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from rl_uav.evaluation.evidence import (
    EpisodeRecord,
    aggregate_records,
    analyze_sample_efficiency,
    read_jsonl,
    verify_real_evidence,
    write_aggregate_csv,
    write_jsonl,
)


def _synthetic_records() -> list[EpisodeRecord]:
    rng = np.random.default_rng(2026)
    records: list[EpisodeRecord] = []
    speeds = {"multitask_ppo": 2.4, "flat_ppo": 5.0, "separate_ppo": 3.8}
    for algorithm, speed in speeds.items():
        for seed in range(5):
            for step_million in range(1, 11):
                progress = 1.0 - np.exp(-step_million / speed)
                for episode in range(50):
                    score = float(np.clip(96.0 * progress + rng.normal(0.0, 3.0), 0.0, 100.0))
                    success_probability = float(np.clip(0.10 + 0.90 * progress, 0.0, 1.0))
                    success = bool(rng.random() < success_probability)
                    collision = int(rng.random() < max(0.0, 0.20 * (1.0 - progress)))
                    records.append(
                        EpisodeRecord(
                            algorithm=algorithm,
                            seed=seed,
                            env_steps=step_million * 1_000_000,
                            checkpoint=f"{algorithm}-s{seed}-{step_million}m",
                            eval_profile="official",
                            episode_seed=20260300 + episode,
                            total_score=score if success and not collision else 0.0,
                            success=success and not collision,
                            nav_score=float(np.clip(progress + rng.normal(0, 0.03), 0, 1)),
                            hover_score=float(np.clip(progress + rng.normal(0, 0.03), 0, 1)),
                            waypoint_score=float(np.clip(progress + rng.normal(0, 0.04), 0, 1)),
                            collisions=collision,
                            nav_steps=max(100, int(1400 - 1100 * progress + rng.normal(0, 30))),
                            hover_error_m=float(max(0.01, 0.35 * (1.0 - progress) + rng.normal(0, 0.01))),
                            synthetic=True,
                        )
                    )
    return records


def _write_report(records: list[EpisodeRecord], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_records(records)
    write_aggregate_csv(output / "aggregates.csv", aggregates)
    analysis = analyze_sample_efficiency(records, bootstrap_samples=500)
    try:
        evidence = verify_real_evidence(records)
    except ValueError as error:
        evidence = {"verified": False, "reason": str(error)}
    summary = {"sample_efficiency": analysis, "evidence": evidence}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser("synthetic", help="generate clearly marked smoke-test data")
    synthetic.add_argument("--output", type=Path, default=Path("outputs/synthetic"))
    aggregate = subparsers.add_parser("aggregate", help="aggregate an existing JSONL evidence file")
    aggregate.add_argument("input", type=Path)
    aggregate.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args(argv)

    if args.command == "synthetic":
        records = _synthetic_records()
        write_jsonl(args.output / "synthetic_records.jsonl", records)
        _write_report(records, args.output)
        print(f"Synthetic smoke report written to {args.output}; it is not real evidence.")
        return 0
    records = read_jsonl(args.input)
    _write_report(records, args.output)
    print(json.dumps(asdict(aggregate_records(records)[0]), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

