# Result evidence schema

Each JSONL line is one immutable evaluation episode:

```json
{
  "algorithm": "multitask_ppo",
  "seed": 0,
  "env_steps": 1000000,
  "checkpoint": "model-s0-1m",
  "eval_profile": "official",
  "episode_seed": 20260300,
  "total_score": 0.0,
  "success": false,
  "nav_score": 0.0,
  "hover_score": 0.0,
  "waypoint_score": 0.0,
  "collisions": 0,
  "nav_steps": 1500,
  "hover_error_m": 0.5,
  "synthetic": false
}
```

Publication verification requires algorithms `multitask_ppo`, `flat_ppo`, and
`separate_ppo`; seeds 0–4; and 50 unique episode seeds for every
algorithm/seed/checkpoint/profile group. Any row with `synthetic=true` makes the
dataset ineligible as real evidence.

