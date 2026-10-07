"""Pretrain an SB3 PPO policy from the Hamiltonian oracle.

Samples are generated online, so no demonstration dataset is stored on disk.
This is the initialization stage for later PPO fine-tuning, not a pure-RL
result; evaluations must label it as behavior cloning.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from resnake_gym.baselines import (
    HamiltonianOracle,
    absolute_action,
    is_cycle_aligned,
    relative_action,
)
from resnake_gym.envs import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=31)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--updates", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--balanced-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="oversample the rare turn cells instead of mostly sampling straight",
    )
    return parser.parse_args()


class OracleBatchGenerator:
    """Generate exact full-scene observations without storing a dataset."""

    def __init__(self, width: int, height: int, seed: int) -> None:
        self.width = width
        self.height = height
        self.oracle = HamiltonianOracle(width, height)
        self.cycle = self.oracle.cycle
        self.capacity = len(self.cycle)
        self.rng = np.random.default_rng(seed)
        self._cycle_xy = np.asarray(self.cycle, dtype=np.int64)
        self._directions = np.asarray(
            [
                absolute_action(self.cycle[index - 1], cell)
                for index, cell in enumerate(self.cycle)
            ],
            dtype=np.int64,
        )
        self.labels = np.asarray(
            [self._label_for_head(index) for index in range(self.capacity)],
            dtype=np.int64,
        )
        self.indices_by_action = tuple(
            np.flatnonzero(self.labels == action) for action in range(3)
        )
        if any(len(indices) == 0 for indices in self.indices_by_action):
            raise ValueError(
                "the selected cycle does not contain every relative action"
            )
        self._action_counts = np.asarray(
            [len(indices) for indices in self.indices_by_action],
            dtype=np.int64,
        )
        self._action_starts = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(self._action_counts[:-1]))
        )
        self._balanced_head_indices = np.concatenate(self.indices_by_action)

    def _label_for_head(self, head_index: int) -> int:
        predecessor = self.cycle[(head_index - 1) % self.capacity]
        head = self.cycle[head_index]
        successor = self.cycle[(head_index + 1) % self.capacity]
        direction = absolute_action(predecessor, head)
        target = absolute_action(head, successor)
        return relative_action(direction, target)

    def batch(
        self,
        batch_size: int,
        *,
        balanced_actions: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if balanced_actions:
            sampled_actions = self.rng.integers(0, 3, size=batch_size)
            offsets_within_action = self.rng.integers(
                0,
                self._action_counts[sampled_actions],
                size=batch_size,
            )
            head_indices = self._balanced_head_indices[
                self._action_starts[sampled_actions] + offsets_within_action
            ]
        else:
            head_indices = self.rng.integers(0, self.capacity, size=batch_size)

        lengths = self.rng.integers(3, self.capacity, size=batch_size)
        food_offsets = self.rng.integers(
            1,
            self.capacity - lengths + 1,
            size=batch_size,
        )
        observations = np.empty(
            (batch_size, 9, self.height, self.width),
            dtype=np.float32,
        )
        observations.fill(0.0)
        actions = self.labels[head_indices].copy()

        # Flatten all variable-length bodies into one set of advanced-index writes.
        # Compared with a (batch, capacity) mask, this only materializes occupied
        # cells and avoids both Python loops from the reference implementation.
        samples = np.arange(batch_size, dtype=np.int64)
        body_samples = np.repeat(samples, lengths)
        body_starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        body_offsets = np.arange(len(body_samples), dtype=np.int64) - body_starts
        body_indices = (head_indices[body_samples] - body_offsets) % self.capacity
        body_xy = self._cycle_xy[body_indices]
        body_lengths = lengths[body_samples]
        observations[body_samples, 0, body_xy[:, 1], body_xy[:, 0]] = 1.0
        observations[body_samples, 1, body_xy[:, 1], body_xy[:, 0]] = (
            body_lengths - body_offsets
        ) / body_lengths

        head_xy = self._cycle_xy[head_indices]
        tail_indices = (head_indices - lengths + 1) % self.capacity
        tail_xy = self._cycle_xy[tail_indices]
        food_indices = (head_indices + food_offsets) % self.capacity
        food_xy = self._cycle_xy[food_indices]
        observations[samples, 2, head_xy[:, 1], head_xy[:, 0]] = 1.0
        observations[samples, 3, tail_xy[:, 1], tail_xy[:, 0]] = 1.0
        observations[samples, 4, food_xy[:, 1], food_xy[:, 0]] = 1.0
        observations[
            samples,
            5 + self._directions[head_indices],
            head_xy[:, 1],
            head_xy[:, 0],
        ] = 1.0
        return observations, actions

    def observation(
        self,
        head_index: int,
        length: int,
        food_offset: int,
    ) -> np.ndarray:
        observation = np.zeros((9, self.height, self.width), dtype=np.float32)
        for offset in range(length):
            x, y = self.cycle[(head_index - offset) % self.capacity]
            observation[0, y, x] = 1.0
            observation[1, y, x] = (length - offset) / length
        head_x, head_y = self.cycle[head_index]
        tail_x, tail_y = self.cycle[(head_index - length + 1) % self.capacity]
        food_x, food_y = self.cycle[(head_index + food_offset) % self.capacity]
        predecessor = self.cycle[(head_index - 1) % self.capacity]
        direction = absolute_action(predecessor, (head_x, head_y))
        observation[2, head_y, head_x] = 1.0
        observation[3, tail_y, tail_x] = 1.0
        observation[4, food_y, food_x] = 1.0
        observation[5 + direction, head_y, head_x] = 1.0
        return observation

    def exhaustive_evaluation_set(self) -> tuple[np.ndarray, np.ndarray]:
        lengths = sorted({3, self.capacity // 4, self.capacity // 2, self.capacity - 1})
        observations: list[np.ndarray] = []
        actions: list[int] = []
        for length in lengths:
            if length < 3 or length >= self.capacity:
                continue
            free_count = self.capacity - length
            for head_index in range(self.capacity):
                food_offset = 1 + (head_index % free_count)
                observations.append(self.observation(head_index, length, food_offset))
                actions.append(int(self.labels[head_index]))
        return np.stack(observations), np.asarray(actions, dtype=np.int64)


def _accuracy(model: object, observations: np.ndarray, actions: np.ndarray) -> float:
    predicted, _ = model.predict(observations, deterministic=True)
    return float(np.mean(np.asarray(predicted).reshape(-1) == actions))


def main() -> None:
    args = parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.eval_every < 1:
        raise ValueError("updates, batch-size and eval-every must be positive")

    try:
        import stable_baselines3
        import torch
        import torch.nn.functional as functional
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'RL dependencies are missing; install with `pip install -e ".[rl]"`'
        ) from exc

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = args.output_dir / "tensorboard"

    generator = OracleBatchGenerator(args.width, args.height, args.seed)
    probe_env = SnakeEnv(
        width=args.width,
        height=args.height,
        action_mode="relative",
    )
    try:
        probe_env.reset(seed=args.seed)
        if not is_cycle_aligned(probe_env.snake, generator.cycle):
            raise ValueError(
                "the default initial snake is not aligned with this board's cycle; "
                "choose a board such as 8x8 or 31x20"
            )
    finally:
        probe_env.close()

    vec_env = DummyVecEnv(
        [
            lambda: SnakeEnv(
                width=args.width,
                height=args.height,
                action_mode="relative",
                max_logic_steps=(args.width * args.height)
                * (args.width * args.height + 1)
                // 2,
            )
        ]
    )
    policy_kwargs = {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.SiLU,
    }
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=0,
    )
    optimizer = model.policy.optimizer
    evaluation_observations, evaluation_actions = generator.exhaustive_evaluation_set()

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir.resolve())
    config["python"] = platform.python_version()
    config["torch"] = torch.__version__
    config["stable_baselines3"] = stable_baselines3.__version__
    config["training_method"] = "hamiltonian_behavior_cloning"
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    writer = SummaryWriter(log_dir=tensorboard_dir)
    started = time.perf_counter()
    last_loss = float("nan")
    last_accuracy = 0.0
    try:
        model.policy.set_training_mode(True)
        for update in range(1, args.updates + 1):
            observations, actions = generator.batch(
                args.batch_size,
                balanced_actions=args.balanced_actions,
            )
            observations_tensor = torch.as_tensor(
                observations,
                device=model.device,
            )
            actions_tensor = torch.as_tensor(actions, device=model.device)
            distribution = model.policy.get_distribution(observations_tensor)
            logits = distribution.distribution.logits
            loss = functional.cross_entropy(logits, actions_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            writer.add_scalar("bc/cross_entropy", last_loss, update)

            if update == 1 or update % args.eval_every == 0:
                model.policy.set_training_mode(False)
                last_accuracy = _accuracy(
                    model,
                    evaluation_observations,
                    evaluation_actions,
                )
                writer.add_scalar("bc/exhaustive_accuracy", last_accuracy, update)
                print(
                    f"update={update} loss={last_loss:.6f} "
                    f"exhaustive_accuracy={last_accuracy:.6f}"
                )
                model.policy.set_training_mode(True)
        if args.updates != 1 and args.updates % args.eval_every != 0:
            model.policy.set_training_mode(False)
            last_accuracy = _accuracy(
                model,
                evaluation_observations,
                evaluation_actions,
            )
            writer.add_scalar("bc/exhaustive_accuracy", last_accuracy, args.updates)
            print(
                f"update={args.updates} loss={last_loss:.6f} "
                f"exhaustive_accuracy={last_accuracy:.6f}"
            )
        model.policy.set_training_mode(False)
        model.save(args.output_dir / "bc_model")
    finally:
        writer.close()
        vec_env.close()

    summary = {
        "updates": args.updates,
        "samples": args.updates * args.batch_size,
        "final_cross_entropy": last_loss,
        "exhaustive_accuracy": last_accuracy,
        "wall_seconds": time.perf_counter() - started,
        "model_path": str((args.output_dir / "bc_model.zip").resolve()),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "bc_summary.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
