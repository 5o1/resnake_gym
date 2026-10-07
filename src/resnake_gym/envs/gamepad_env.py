"""Current gamepad-only interface, with the old direction kernel kept internal."""

import gymnasium as gym

import resnake_gym.gamepad as gamepad
from resnake_gym.envs.snake_env import SnakeEnv, SnakeState


class GamepadSnakeEnv(gym.Env):
    """Gamepad adapter around a single explicitly selected Snake kernel."""

    metadata = SnakeEnv.metadata
    kernel_type = SnakeEnv

    def __init__(self, *, stick_deadzone=0.2, **kwargs):
        if "action_mode" in kwargs:
            raise ValueError("v1 accepts gamepad reports only; no action_mode")
        if not 0 <= stick_deadzone < 1:
            raise ValueError("stick_deadzone must be in [0, 1)")
        self._kernel = self.kernel_type(action_mode="absolute", **kwargs)
        self.stick_deadzone = stick_deadzone
        self.action_space = gamepad.action_space()
        self.observation_space = self._kernel.observation_space
        self.render_mode = self._kernel.render_mode
        self.metadata = dict(self._kernel.metadata)
        self.last_report = gamepad.neutral()

    @property
    def kernel(self) -> SnakeEnv:
        """Kernel extension boundary for task-specific environment adapters."""

        return self._kernel

    @property
    def state(self) -> SnakeState:
        """Current immutable kernel snapshot."""

        return self._kernel.state

    @property
    def logic_fps(self) -> float:
        """Number of game ticks represented by one simulated second."""

        return self._kernel.logic_fps

    @property
    def frame_skip(self) -> int:
        """Number of game ticks advanced by one environment step."""

        return self._kernel.frame_skip

    def apply_kernel_update(self, kernel: SnakeEnv) -> None:
        """Adopt a validated counterfactual kernel without changing the adapter."""

        if not isinstance(kernel, self.kernel_type):
            raise TypeError(
                f"expected {self.kernel_type.__name__}, got {type(kernel).__name__}"
            )
        self._kernel = kernel

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_report = gamepad.neutral()
        observation, info = self._kernel.reset(seed=seed, options=options)
        info["gamepad_report"] = self.last_report.copy()
        info["gamepad_layout"] = gamepad.LAYOUT_VERSION
        return observation, info

    def step(self, action):
        # The only nondifferentiable conversion belongs at the environment edge.
        self.last_report = gamepad.digital_report(action)
        request = gamepad.direction_request(self.last_report, self.stick_deadzone)
        if request is None:
            request = self._kernel.direction
        observation, reward, terminated, truncated, info = self._kernel.step(request)
        info["gamepad_report"] = self.last_report.copy()
        info["gamepad_layout"] = gamepad.LAYOUT_VERSION
        return observation, reward, terminated, truncated, info

    def render(self):
        return self._kernel.render()

    def close(self):
        self._kernel.close()
