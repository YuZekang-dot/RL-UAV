"""Strict JSONL evidence schema and reproducible aggregation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    algorithm: str
    seed: int
    env_steps: int
    checkpoint: str
    eval_profile: str
    episode_seed: int
    total_score: float
    success: bool
    nav_score: float
    hover_score: float
    waypoint_score: float
    collisions: int
    nav_steps: int
    hover_error_m: float
    synthetic: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EpisodeRecord":
        unknown = set(payload) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown evidence fields: {sorted(unknown)}")
        record = cls(**payload)  # type: ignore[arg-type]
        if record.env_steps < 0 or record.nav_steps < 0 or record.collisions < 0:
            raise ValueError("step and collision counters must be non-negative")
        if not 0.0 <= record.total_score <= 100.0:
            raise ValueError("total_score must be within [0, 100]")
        return record


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    algorithm: str
    seed: int | None
    env_steps: int
    eval_profile: str
    episodes: int
    total_score: float
    success_rate: float
    nav_score: float
    hover_score: float
    waypoint_score: float
    collision_rate: float
    hover_error_m: float
    synthetic: bool


@dataclass(frozen=True, slots=True)
class ParityCriteria:
    total_score_ratio_min: float = 0.95
    success_rate_delta_min: float = -0.02
    waypoint_score_delta_min: float = -0.02
    collision_rate_delta_max: float = 0.01
    hover_error_ratio_max: float = 1.05
    consecutive_evaluations: int = 3


def read_jsonl(path: str | Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(EpisodeRecord.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid evidence at line {line_number}: {error}") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[EpisodeRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
    return destination


def aggregate_records(
    records: Iterable[EpisodeRecord], *, across_seeds: bool = False
) -> list[AggregateMetrics]:
    grouped: dict[tuple[object, ...], list[EpisodeRecord]] = {}
    for record in records:
        seed_key = None if across_seeds else record.seed
        key = (record.algorithm, seed_key, record.env_steps, record.eval_profile)
        grouped.setdefault(key, []).append(record)
    aggregates = []
    for (algorithm, seed, env_steps, profile), group in sorted(grouped.items()):
        aggregates.append(
            AggregateMetrics(
                algorithm=str(algorithm),
                seed=None if seed is None else int(seed),
                env_steps=int(env_steps),
                eval_profile=str(profile),
                episodes=len(group),
                total_score=float(np.mean([item.total_score for item in group])),
                success_rate=float(np.mean([item.success for item in group])),
                nav_score=float(np.mean([item.nav_score for item in group])),
                hover_score=float(np.mean([item.hover_score for item in group])),
                waypoint_score=float(np.mean([item.waypoint_score for item in group])),
                collision_rate=float(np.mean([item.collisions > 0 for item in group])),
                hover_error_m=float(np.mean([item.hover_error_m for item in group])),
                synthetic=any(item.synthetic for item in group),
            )
        )
    return aggregates


def _baseline_target(
    aggregates: list[AggregateMetrics], baseline: str, profile: str
) -> AggregateMetrics:
    candidates = [
        item for item in aggregates if item.algorithm == baseline and item.eval_profile == profile
    ]
    if not candidates:
        raise ValueError(f"no baseline data for {baseline}/{profile}")
    final_steps = sorted({item.env_steps for item in candidates})[-3:]
    selected = [item for item in candidates if item.env_steps in final_steps]
    return AggregateMetrics(
        algorithm=baseline,
        seed=None,
        env_steps=max(final_steps),
        eval_profile=profile,
        episodes=sum(item.episodes for item in selected),
        total_score=float(np.mean([item.total_score for item in selected])),
        success_rate=float(np.mean([item.success_rate for item in selected])),
        nav_score=float(np.mean([item.nav_score for item in selected])),
        hover_score=float(np.mean([item.hover_score for item in selected])),
        waypoint_score=float(np.mean([item.waypoint_score for item in selected])),
        collision_rate=float(np.mean([item.collision_rate for item in selected])),
        hover_error_m=float(np.mean([item.hover_error_m for item in selected])),
        synthetic=any(item.synthetic for item in selected),
    )


def _passes_parity(
    candidate: AggregateMetrics,
    baseline: AggregateMetrics,
    criteria: ParityCriteria,
) -> bool:
    hover_limit = baseline.hover_error_m * criteria.hover_error_ratio_max
    return bool(
        candidate.total_score >= baseline.total_score * criteria.total_score_ratio_min
        and candidate.success_rate >= baseline.success_rate + criteria.success_rate_delta_min
        and candidate.waypoint_score >= baseline.waypoint_score + criteria.waypoint_score_delta_min
        and candidate.collision_rate <= baseline.collision_rate + criteria.collision_rate_delta_max
        and candidate.hover_error_m <= hover_limit
    )


def _first_parity_step(
    aggregates: list[AggregateMetrics],
    algorithm: str,
    profile: str,
    baseline_target: AggregateMetrics,
    criteria: ParityCriteria,
) -> int | None:
    curve = sorted(
        (
            item
            for item in aggregates
            if item.algorithm == algorithm and item.eval_profile == profile
        ),
        key=lambda item: item.env_steps,
    )
    consecutive = 0
    first_step: int | None = None
    for item in curve:
        if _passes_parity(item, baseline_target, criteria):
            consecutive += 1
            first_step = item.env_steps if consecutive == 1 else first_step
            if consecutive >= criteria.consecutive_evaluations:
                return first_step
        else:
            consecutive = 0
            first_step = None
    return None


def analyze_sample_efficiency(
    records: list[EpisodeRecord],
    *,
    candidate: str = "multitask_ppo",
    baseline: str = "flat_ppo",
    profile: str = "official",
    criteria: ParityCriteria | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
) -> dict[str, object]:
    criteria = criteria or ParityCriteria()
    pooled = aggregate_records(records, across_seeds=True)
    target = _baseline_target(pooled, baseline, profile)
    candidate_step = _first_parity_step(pooled, candidate, profile, target, criteria)
    baseline_step = _first_parity_step(pooled, baseline, profile, target, criteria)
    ratio = candidate_step / baseline_step if candidate_step is not None and baseline_step else None

    available_seeds = sorted({item.seed for item in records})
    rng = np.random.default_rng(bootstrap_seed)
    ratios: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(available_seeds, size=len(available_seeds), replace=True)
        sample_records: list[EpisodeRecord] = []
        for output_seed, source_seed in enumerate(sampled):
            for item in records:
                if item.seed == int(source_seed):
                    payload = asdict(item)
                    payload["seed"] = output_seed
                    sample_records.append(EpisodeRecord(**payload))
        sample_aggregates = aggregate_records(sample_records, across_seeds=True)
        sample_target = _baseline_target(sample_aggregates, baseline, profile)
        sample_candidate = _first_parity_step(
            sample_aggregates, candidate, profile, sample_target, criteria
        )
        sample_baseline = _first_parity_step(
            sample_aggregates, baseline, profile, sample_target, criteria
        )
        if sample_candidate is not None and sample_baseline:
            ratios.append(sample_candidate / sample_baseline)
    confidence_interval = (
        [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
        if ratios
        else None
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "profile": profile,
        "candidate_first_parity_steps": candidate_step,
        "baseline_first_parity_steps": baseline_step,
        "interaction_ratio": ratio,
        "bootstrap_95_ci": confidence_interval,
        "bootstrap_valid_samples": len(ratios),
        "criteria": asdict(criteria),
        "contains_synthetic_data": any(item.synthetic for item in records),
    }


def verify_real_evidence(
    records: list[EpisodeRecord],
    *,
    expected_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    episodes_per_evaluation: int = 50,
    required_algorithms: tuple[str, ...] = ("multitask_ppo", "flat_ppo", "separate_ppo"),
) -> dict[str, object]:
    if not records:
        raise ValueError("evidence set is empty")
    if any(record.synthetic for record in records):
        raise ValueError("synthetic records can never verify a real result")
    algorithms = {record.algorithm for record in records}
    missing_algorithms = set(required_algorithms) - algorithms
    if missing_algorithms:
        raise ValueError(f"missing required algorithms: {sorted(missing_algorithms)}")
    seeds = {record.seed for record in records}
    if seeds != set(expected_seeds):
        raise ValueError(f"expected seeds {expected_seeds}, got {sorted(seeds)}")
    groups: dict[tuple[str, int, int, str], list[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault(
            (record.algorithm, record.seed, record.env_steps, record.eval_profile), []
        ).append(record)
    invalid = [
        key
        for key, group in groups.items()
        if len(group) != episodes_per_evaluation
        or len({item.episode_seed for item in group}) != episodes_per_evaluation
    ]
    if invalid:
        raise ValueError(f"incomplete or duplicate 50-episode groups: {invalid[:5]}")
    return {
        "verified": True,
        "records": len(records),
        "evaluation_groups": len(groups),
        "algorithms": sorted(algorithms),
        "seeds": sorted(seeds),
    }


def write_aggregate_csv(path: str | Path, aggregates: Iterable[AggregateMetrics]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in aggregates]
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(AggregateMetrics.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    return destination

