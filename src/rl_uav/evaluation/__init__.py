"""Evidence ingestion, aggregation, and sample-efficiency analysis."""

from .evidence import (
    AggregateMetrics,
    EpisodeRecord,
    ParityCriteria,
    aggregate_records,
    analyze_sample_efficiency,
    read_jsonl,
    verify_real_evidence,
)

__all__ = [
    "AggregateMetrics",
    "EpisodeRecord",
    "ParityCriteria",
    "aggregate_records",
    "analyze_sample_efficiency",
    "read_jsonl",
    "verify_real_evidence",
]

