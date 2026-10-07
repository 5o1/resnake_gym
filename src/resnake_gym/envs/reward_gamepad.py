"""Bounded food-efficiency reward and optional potential-based shaping.

PBRS: Ng, Harada & Russell (ICML 1999). Coefficients are project hypotheses,
not literature-optimal values. Reward state is private environment bookkeeping.
"""

from resnake_gym.envs.perturbed_gamepad import PerturbedGamepadEnv


class RewardGamepadEnv(PerturbedGamepadEnv):
    def __init__(
        self, *, reward_gamma=0.99, shaping_scale=0.25, death_cost=3.0, **kwargs
    ):
        super().__init__(**kwargs)
        if not 0 < reward_gamma <= 1 or not 0 <= shaping_scale <= 1 or death_cost <= 0:
            raise ValueError("invalid reward parameters")
        self.reward_gamma = reward_gamma
        self.shaping_scale = shaping_scale
        self.death_cost = death_cost

    def _start_food(self):
        state = self.state
        n = state.width * state.height - len(self.kernel.obstacle_cells)
        self.food_free = n - len(state.snake)
        self.food_cells = n
        self.food_distance = self._distance()
        self.food_start_tick = state.logic_steps

    def _distance(self):
        state = self.state
        if state.food is None:
            return 0
        return sum(abs(a - b) for a, b in zip(state.snake[0], state.food, strict=True))

    def _potential(self):
        state = self.state
        if state.food is None:
            return 0.0
        return self.shaping_scale * (
            1 - self._distance() / (state.width + state.height - 2)
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._start_food()
        return obs, info

    def step(self, action):
        previous_phi = self._potential()
        previous_score = self.state.score
        obs, _, terminated, truncated, info = super().step(action)
        state = self.state
        ate = state.score > previous_score
        base, food_event = 0.0, None
        if ate:
            elapsed = state.logic_steps - self.food_start_tick
            efficiency = self.food_distance / max(self.food_distance, elapsed, 1)
            density = 1 - self.food_free / self.food_cells
            base = 1 + 0.5 * density + 0.5 * efficiency
            food_event = dict(
                tick=state.logic_steps,
                ticks=elapsed,
                distance=self.food_distance,
                free=self.food_free,
                cells=self.food_cells,
                efficiency=efficiency,
                reward=base,
            )
            self._start_food()
        if terminated:
            base += 3.0 if info["won"] else -self.death_cost
        # A training cutoff is not a terminal state: retain potential/bootstrap.
        next_phi = 0.0 if terminated else self._potential()
        shaping = self.reward_gamma * next_phi - previous_phi
        info.update(
            reward_version="food-efficiency-pbrs-v1",
            reward_base=base,
            reward_shaping=shaping,
            food_event=food_event,
            ticks_since_food=self.state.logic_steps - self.food_start_tick,
        )
        return obs, base + shaping, terminated, truncated, info
