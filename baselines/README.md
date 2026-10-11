# Baselines

The baseline models live in `rl_uav.models.baselines` and intentionally reuse
the same observation builder, target manager, reward functions, curriculum,
and evaluation profiles as the multi-task model.

- **FlatPPO** concatenates shared state, padded task state, and phase one-hot,
  then trains one actor and one critic. It is the primary sample-efficiency
  comparison.
- **SeparatePPO** owns independent actor/critic networks for navigation and
  hover and switches them by `TaskId`. It is a specialization upper bound,
  not the deployment architecture.

Do not compare against the archived competition template: it used different
features, reward timing, imports, and action-distribution semantics.

