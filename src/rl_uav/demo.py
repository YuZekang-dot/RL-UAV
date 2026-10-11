"""CPU-only planner animation plus an optional untrained-model smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_uav.planning import TargetManager
from rl_uav.types import ObservationBatch, TargetKind, TaskId


def simulate_planner(steps: int = 180) -> dict[str, object]:
    """Follow planner targets with a point-mass visualization.

    This is deliberately not a quadrotor simulator and is never used for
    performance claims.
    """
    manager = TargetManager()
    position = np.array([0.5, 2.5, 0.15], dtype=np.float32)
    goal = np.array([4.5, 2.5, 1.5], dtype=np.float32)
    waypoints = np.array(
        [[1.20, 4.10, 0.9], [2.70, 1.00, 1.2], [3.80, 4.00, 1.4]], dtype=np.float32
    )
    pending = np.ones(len(waypoints), dtype=bool)
    obstacles = np.array(
        [[1.45, 2.85, 1.5], [2.55, 2.20, 1.5], [3.55, 2.90, 1.5]], dtype=np.float32
    )
    radii = np.array([0.38, 0.42, 0.38], dtype=np.float32)
    positions = [position.copy()]
    targets = []
    kinds = []
    route_versions = []
    for step in range(steps):
        for index, waypoint in enumerate(waypoints):
            if pending[index] and np.linalg.norm(position - waypoint) < 0.20:
                pending[index] = False
        context = manager.update(
            position,
            waypoints,
            pending,
            goal,
            obstacles,
            radii,
            remaining_time=max(2.0, 30.0 - step * 0.10),
        )
        delta = context.control_target - position
        distance = float(np.linalg.norm(delta))
        if distance > 1.0e-6:
            position = position + min(0.07, distance) * delta / distance
        positions.append(position.copy())
        targets.append(context.control_target.copy())
        kinds.append(context.kind.value)
        route_versions.append(context.route_version)
        if np.linalg.norm(position - goal) < 0.12 and not pending.any():
            break
    return {
        "positions": np.asarray(positions),
        "targets": np.asarray(targets),
        "kinds": kinds,
        "route_versions": route_versions,
        "waypoints": waypoints,
        "obstacles": obstacles,
        "radii": radii,
        "goal": goal,
        "visited_waypoints": int((~pending).sum()),
        "reached_goal": bool(np.linalg.norm(position - goal) < 0.12),
    }


def model_smoke_test() -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError:
        return {"status": "skipped", "reason": "PyTorch is not installed"}
    from rl_uav.models import MultiTaskActorCritic

    torch.manual_seed(7)
    shared = torch.zeros((2, 19), dtype=torch.float32)
    task = torch.zeros((2, 31), dtype=torch.float32)
    mask = torch.zeros_like(task)
    mask[0, :31] = 1.0
    mask[1, :5] = 1.0
    observations = ObservationBatch(
        shared_obs=shared,
        task_obs=task,
        task_id=torch.tensor([TaskId.NAVIGATION, TaskId.HOVER]),
        task_obs_mask=mask,
    )
    model = MultiTaskActorCritic().eval()
    with torch.no_grad():
        actions, values, log_probs = model.act(observations, deterministic=True)
    return {
        "status": "passed",
        "task_ids": observations.task_id.tolist(),
        "actions": actions.tolist(),
        "values": values.tolist(),
        "log_probs": log_probs.tolist(),
        "actions_in_range": bool(torch.all((actions >= -1.0) & (actions <= 1.0))),
        "critic_count": len(model.critics),
    }


def render_demo(simulation: dict[str, object], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Circle
    except ModuleNotFoundError as error:
        raise RuntimeError("install the demo extra: python -m pip install -e '.[demo]'") from error

    positions = np.asarray(simulation["positions"])
    targets = np.asarray(simulation["targets"])
    kinds = list(simulation["kinds"])
    waypoints = np.asarray(simulation["waypoints"])
    obstacles = np.asarray(simulation["obstacles"])
    radii = np.asarray(simulation["radii"])
    goal = np.asarray(simulation["goal"])
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.set(xlim=(0, 5), ylim=(0, 5), aspect="equal", xlabel="x [m]", ylabel="y [m]")
    axis.set_title("Planner-only demo — not quadrotor dynamics")
    axis.grid(alpha=0.2)
    for center, radius in zip(obstacles, radii, strict=True):
        axis.add_patch(Circle(center[:2], radius, color="#df6c5b", alpha=0.65))
        axis.add_patch(Circle(center[:2], radius + 0.13, fill=False, color="#a63d2f", linestyle="--"))
    axis.scatter(waypoints[:, 0], waypoints[:, 1], marker="D", color="#2864dc", label="Waypoints")
    axis.scatter([goal[0]], [goal[1]], marker="*", s=180, color="#159447", label="Goal")
    trail, = axis.plot([], [], color="#202b45", linewidth=2, label="Point-mass path")
    drone, = axis.plot([], [], marker="o", color="#111827", markersize=7)
    target_marker, = axis.plot([], [], marker="x", color="#f59e0b", markersize=10, markeredgewidth=2)
    status = axis.text(0.02, 0.98, "", transform=axis.transAxes, va="top")
    axis.legend(loc="lower right")

    def update(frame: int):
        trail.set_data(positions[: frame + 1, 0], positions[: frame + 1, 1])
        drone.set_data([positions[frame, 0]], [positions[frame, 1]])
        target_index = min(frame, len(targets) - 1)
        target_marker.set_data([targets[target_index, 0]], [targets[target_index, 1]])
        kind = kinds[target_index]
        target_marker.set_color("#f59e0b" if kind == TargetKind.SAFETY_GUIDE.value else "#159447")
        status.set_text(f"current target: {kind}")
        return trail, drone, target_marker, status

    if output.suffix.lower() == ".gif":
        animation = FuncAnimation(figure, update, frames=len(positions), interval=60, blit=True)
        animation.save(output, writer=PillowWriter(fps=16))
    else:
        update(len(positions) - 1)
        figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo.gif"))
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)
    simulation = simulate_planner(args.steps)
    smoke = model_smoke_test()
    if not args.no_render:
        render_demo(simulation, args.output)
    summary = {
        "artifact": None if args.no_render else str(args.output),
        "disclaimer": "planner/point-mass visualization; not learned flight or quadrotor dynamics",
        "visited_waypoints": simulation["visited_waypoints"],
        "reached_goal": simulation["reached_goal"],
        "target_kinds": sorted(set(simulation["kinds"])),
        "model_smoke": smoke,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

