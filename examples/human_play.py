"""Play ReSnake with a keyboard through the Gymnasium environment."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

import resnake_gym  # noqa: F401 - importing the package registers the environment
from resnake_gym import gamepad

try:
    import pygame
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    raise SystemExit(
        'pygame is required; install it with: python -m pip install -e ".[render]"'
    ) from exc


KEY_TO_DIRECTION = {
    pygame.K_UP: 0,
    pygame.K_w: 0,
    pygame.K_RIGHT: 1,
    pygame.K_d: 1,
    pygame.K_DOWN: 2,
    pygame.K_s: 2,
    pygame.K_LEFT: 3,
    pygame.K_a: 3,
}

DPAD_INDEX_BY_DIRECTION = {
    0: 0,  # up
    1: 3,  # right
    2: 1,  # down
    3: 2,  # left
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--logic-fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--height", type=int, default=20)
    return parser.parse_args()


def facing_direction(observation: np.ndarray) -> int:
    """Read the current absolute direction from observation channels 5..8."""

    direction_channels = observation[5:9].reshape(4, -1)
    return int(np.argmax(direction_channels.max(axis=1)))


def dpad_report(direction: int) -> np.ndarray:
    """Encode an absolute direction as a normalized gamepad report."""

    report = gamepad.neutral()
    report[DPAD_INDEX_BY_DIRECTION[direction]] = 1.0
    return report


def main() -> None:
    args = parse_args()
    env = gym.make(
        resnake_gym.ENV_ID,
        width=args.width,
        height=args.height,
        logic_fps=args.logic_fps,
        frame_skip=1,
        render_mode="human",
    )

    observation, _ = env.reset(seed=args.seed)
    action = dpad_report(facing_direction(observation))
    game_over = False
    running = True

    print("Arrow keys / WASD: turn | R: restart | Esc: quit")

    try:
        while running:
            restart = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        restart = True
                    elif event.key in KEY_TO_DIRECTION:
                        action = dpad_report(KEY_TO_DIRECTION[event.key])

            if not running:
                break

            if restart:
                observation, _ = env.reset(seed=args.seed)
                action = dpad_report(facing_direction(observation))
                game_over = False
                continue

            if game_over:
                # Keep processing window events without burning a CPU core.
                pygame.time.wait(10)
                continue

            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                reason = info.get(
                    "termination_reason", "time_limit" if truncated else "unknown"
                )
                print(
                    f"game over: score={info['score']} length={info['length']} "
                    f"reason={reason}; press R to restart"
                )
                game_over = True
    finally:
        env.close()


if __name__ == "__main__":
    main()
