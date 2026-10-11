"""Authorized-platform hook without redistributing the platform rollout code."""

from __future__ import annotations

from typing import Any


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Delegate rollout ownership to an injected authorized platform driver.

    The private SDK should pass ``rollout_driver`` with a ``run`` method. This
    keeps the public repository importable while preserving the configured
    workflow entry point and avoiding a fake reimplementation of KaiwuDRL's
    distributed actor/learner lifecycle.
    """
    driver = kwargs.get("rollout_driver")
    if driver is None:
        raise RuntimeError(
            "KaiwuDRL rollout driver is external. Pass rollout_driver from the authorized "
            "platform integration; CPU tests do not require this workflow."
        )
    if not envs or not agents:
        raise ValueError("workflow requires at least one environment and one agent")
    return driver.run(env=envs[0], agent=agents[0], logger=logger, monitor=monitor)

