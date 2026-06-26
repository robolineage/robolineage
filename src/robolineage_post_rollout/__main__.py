from __future__ import annotations

import argparse
from pathlib import Path

from robolineage_shared_agents.visual_snapshot.vlm_runner import make_vlm_runner_from_env

from .formal_review import DEFAULT_MAX_REVIEW_IMAGES, PostRolloutReviewAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-rollout review for one rollout directory.")
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--no-vlm", action="store_true", help="write deterministic artifacts without VLM polish")
    parser.add_argument("--max-review-images", type=int, default=DEFAULT_MAX_REVIEW_IMAGES)
    args = parser.parse_args()

    runner = None
    if not args.no_vlm:
        runner = make_vlm_runner_from_env(
            "POST_REVIEW_VLM",
            timeout_default=60.0,
            max_output_tokens_default=4096,
            min_timeout_s=60.0,
            min_output_tokens=4096,
        )
    agent = PostRolloutReviewAgent(
        vlm_runner=runner,
        use_vlm=not args.no_vlm,
        max_review_images=args.max_review_images,
    )
    result = agent.run(args.rollout_dir)
    print(f"status={result.status} rollout_id={result.rollout_id} used_vlm={result.used_vlm}")
    for name, path in sorted(result.artifacts.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
