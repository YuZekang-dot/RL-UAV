"""Local smoke test by default; optional delegation to an installed KaiwuDRL SDK."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="store_true")
    args = parser.parse_args()
    if args.platform:
        try:
            from kaiwudrl.common.utils.train_test_utils import run_train_test
        except ModuleNotFoundError as error:
            raise RuntimeError("KaiwuDRL is not installed in this public environment") from error
        run_train_test(algorithm_name="ppo", algorithm_name_list=["ppo"])
        return 0

    from rl_uav.demo import model_smoke_test, simulate_planner

    simulation = simulate_planner(120)
    result = {
        "planner_reached_goal": simulation["reached_goal"],
        "planner_visited_waypoints": simulation["visited_waypoints"],
        "model": model_smoke_test(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

