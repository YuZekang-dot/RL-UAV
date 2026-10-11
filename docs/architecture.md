# Architecture notes

## Observation contract

The public decoder accepts the documented 95-dimensional ObstacleHover
observation. It derives a 19-dimensional dynamics state shared by both tasks:
position (3), the first two rotation-matrix columns (6), linear velocity (3),
angular velocity (3), and previous CTBR action (4).

Navigation receives 31 task dimensions: current and next target features (8),
target-kind one-hot (4), top-four obstacle danger features (12), six boundary
risks, and remaining-time fraction. Hover receives goal distance/direction (4)
and time fraction. Padding plus a validity mask supports mixed-task batches.

## Target ownership

`TargetManager` is the single owner of the active target. Its immutable
`TargetContext` is stored with every transition and is consumed by both the
policy observation builder and navigation reward. Policy and reward code must
never run independent planners.

## Critic boundary

The navigation-to-hover transition terminates navigation GAE. Hover begins a
new value segment, so returns never bootstrap from a different task's critic.
The shared actor still receives gradients from both task objectives.

