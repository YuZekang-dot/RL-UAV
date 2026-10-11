# Third-party dependencies and research attribution

This repository does not redistribute the competition SDK, Isaac Sim,
OmniDrones, generated protocol files, or the referenced papers.

## Research used as design input

- Jiaxu Xing, Ismail Geles, Yunlong Song, Elie Aljalbout, and Davide
  Scaramuzza, **Multi-Task Reinforcement Learning for Quadrotors**, IEEE
  Robotics and Automation Letters, 2024, arXiv:2412.12442. The shared
  dynamics encoder, task-specific encoders, shared actor, and multi-critic
  separation are adapted to the navigation/hover phases in this project.
- Elia Kaufmann et al., **Champion-level drone racing using deep
  reinforcement learning**, Nature 620, 2023,
  https://doi.org/10.1038/s41586-023-06419-4. This work provides background
  on learned low-level quadrotor control and sim-to-real practice; this
  repository does not claim to reproduce Swift.

## External runtime dependencies

- NVIDIA Isaac Sim and OmniDrones
- Tencent KaiwuDRL competition SDK and the ObstacleHover task package
- PyTorch, NumPy, Matplotlib, Pillow, and pytest under their respective
  upstream licenses

Platform-specific code is loaded only through the optional `agent_ppo`
adapter. Consult the competition rules and vendor licenses before combining
this public source with any private SDK files.

