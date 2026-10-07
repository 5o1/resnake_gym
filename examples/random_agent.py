"""Run a reproducible random-policy baseline without real-time throttling."""

from __future__ import annotations

import argparse

import gymnasium as gym

import resnake_gym  # noqa: F401 - importing the package registers the environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--logic-fps", type=float, default=10.0)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument(
        "--action-mode",
        choices=("absolute", "relative"),
        default="absolute",
    )
    parser.add_argument(
        "--ansi",
        action="store_true",
        help="print every board state (considerably slower than headless sampling)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_mode = "ansi" if args.ansi else None
    env = gym.make(
        "resnake_gym/ReSnake-v0",
        action_mode=args.action_mode,
        logic_fps=args.logic_fps,
        frame_skip=args.frame_skip,
        render_mode=render_mode,
    )
    env.action_space.seed(args.seed)

    try:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            _, info = env.reset(seed=episode_seed)
            total_reward = 0.0
            terminated = truncated = False

            if args.ansi:
                print(env.render())

            while not (terminated or truncated):
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if args.ansi:
                    print(env.render())

            outcome = info.get(
                "termination_reason", "time_limit" if truncated else "unknown"
            )
            print(
                f"episode={episode + 1} seed={episode_seed} "
                f"score={info['score']} length={info['length']} "
                f"reward={total_reward:.1f} ticks={info['logic_steps']} "
                f"outcome={outcome}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
