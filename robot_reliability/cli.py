from __future__ import annotations

import argparse
import json

from .experiment import run_experiment, save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot Reliability OS simulation benchmark")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    baseline_results, baseline = run_experiment(args.episodes, args.seed, supervised=False)
    supervised_results, supervised = run_experiment(args.episodes, args.seed, supervised=True)

    save_results(f"{args.output_dir}/baseline.json", baseline_results, baseline)
    save_results(f"{args.output_dir}/supervised.json", supervised_results, supervised)

    report = {
        "baseline": baseline.to_dict(),
        "supervised": supervised.to_dict(),
        "delta": {
            "success_rate": supervised.success_rate - baseline.success_rate,
            "collision_rate": supervised.collision_rate - baseline.collision_rate,
            "drop_rate": supervised.drop_rate - baseline.drop_rate,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
