# Reproduction protocol

1. Use Python 3.10/3.11 and install `.[dev]` for CPU checks.
2. Run `pytest -q` before connecting any private platform package.
3. Install the authorized KaiwuDRL, Isaac Sim, OmniDrones, task configuration,
   and protocol packages separately.
4. Train seeds 0–4 with `configs/curriculum.toml`. Persist the curriculum state
   in every v2 checkpoint.
5. Every 1M transitions, evaluate the same 50 episode seeds under both profiles
   from `configs/benchmark.toml`.
6. Export one JSONL row per episode using the exact schema in
   `docs/result-schema.md`; do not pre-average raw evidence.
7. Run `rl-uav-benchmark aggregate records.jsonl`. A result is publishable only
   when `summary.json` reports `evidence.verified=true`.

Synthetic output from `rl-uav-benchmark synthetic` exists only to exercise the
pipeline and always carries `synthetic=true`.

