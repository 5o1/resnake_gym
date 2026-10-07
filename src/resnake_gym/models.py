"""Head-local spatial encoding and differentiable, semantic gamepad conditioning.

No pretrained weights or demonstrations. Input is the delivered observation,
never the hidden current game state. Batches group equal board sizes.
"""

import torch
from torch import nn
from torch.nn import functional as F

import resnake_gym.gamepad as gamepad


class _MarginalControlDistribution:
    """Sample every control, but score only a causal prefix of them.

    The controller report must stay complete, so sampling still delegates to the
    full distribution.  ``log_prob`` and ``entropy`` instead describe its
    marginal over the controls that can affect Snake.  The component
    distributions used here are independent across controls (including
    ``ChunkNormal``, whose correlation is only across chunk time), so dropping
    the other per-control terms is the exact marginal rather than a heuristic
    rescaling of the joint score.
    """

    def __init__(self, distribution, active_controls):
        self.distribution = distribution
        self.active_controls = active_controls

    def _active_terms(self, terms):
        if terms.shape[-1] < self.active_controls:
            raise ValueError("distribution has fewer controls than its causal prefix")
        mask = torch.arange(terms.shape[-1], device=terms.device)
        return terms * (mask < self.active_controls).to(terms.dtype)

    def log_prob(self, value):
        return self._active_terms(self.distribution.log_prob(value))

    def entropy(self):
        return self._active_terms(self.distribution.entropy())

    def __getattr__(self, name):
        # Preserve the normal distribution API (sample, logits, loc, scale,
        # batch_shape, ...), including full-size samples for the 20-D report.
        return getattr(self.distribution, name)


DPAD5_LABELS = gamepad.DPAD5_LABELS


class HeldDpad5Categorical(torch.distributions.Categorical):
    """One categorical decision expanded to a constant controller chunk.

    ``chunk_length`` is execution metadata, not an additional random-variable
    dimension.  Keeping this type distinct prevents generic chunk code from
    accidentally multiplying the same action probability once per report slot.
    """

    def __init__(self, *, logits, chunk_length):
        super().__init__(logits=logits)
        if chunk_length < 1:
            raise ValueError("chunk_length must be positive")
        self.chunk_length = int(chunk_length)


def _dpad5_templates(*, device, dtype):
    """Return the five legal full-report realizations of the dpad5 action set."""
    return torch.tensor(gamepad.dpad5_reports(), device=device, dtype=dtype)


def _dpad5_controls_with_gradient(logits):
    """Deterministic legal report in forward, soft categorical gradient backward."""
    probabilities = logits.softmax(-1)
    categories = probabilities.argmax(-1)
    hard = F.one_hot(categories, num_classes=5).to(probabilities.dtype)
    one_hot = hard - probabilities.detach() + probabilities
    return one_hot @ _dpad5_templates(device=logits.device, dtype=logits.dtype)


def _held_dpad5_controls_with_gradient(logits, chunk_length):
    """Apply ST once, then hold that single report across the whole chunk."""
    report = _dpad5_controls_with_gradient(logits)
    return report.unsqueeze(-2).expand(*report.shape[:-1], chunk_length, 20)


def normalized_controls(logits):
    """Preserve autograd: soft buttons, four stick axes, two trigger axes."""
    return torch.cat(
        (
            logits[..., :14].sigmoid(),
            logits[..., 14:18].tanh(),
            logits[..., 18:20].sigmoid(),
        ),
        dim=-1,
    )


def hard_controls_with_gradient(logits, *, temperature=1.0, stochastic=True):
    """ST Gumbel-Softmax (Jang et al., ICLR 2017, section 2.2).

    Forward buttons are binary, backward uses a biased continuous surrogate.
    Do not mistake this for differentiating physical contacts or game rewards.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    button_logits = logits[..., :14]
    if stochastic:
        pair = torch.stack((torch.zeros_like(button_logits), button_logits), -1)
        buttons = F.gumbel_softmax(pair, tau=temperature, hard=True, dim=-1)[..., 1]
    else:
        soft = (button_logits / temperature).sigmoid()
        hard = (button_logits >= 0).to(soft.dtype)
        buttons = hard - soft.detach() + soft
    return torch.cat(
        (buttons, logits[..., 14:18].tanh(), logits[..., 18:20].sigmoid()), -1
    )


class HeadLocalEncoder(nn.Module):
    def __init__(self, channels=9, dim=64, radius=3, spatial_pool=False):
        super().__init__()
        self.radius = radius
        self.channels = channels
        self.local = nn.Sequential(
            nn.Conv2d(channels + 1, dim, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(dim * (2 * radius + 1) ** 2, dim),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(channels + 1, dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.position = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.entities = (
            nn.Sequential(
                nn.Linear(channels + 3, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU()
            )
            if spatial_pool
            else None
        )
        self.output = nn.Linear((4 if spatial_pool else 3) * dim, dim)

    def forward(self, board):
        batch, _, height, width = board.shape
        # Explicit on-board channel: padding is not empty playable space.
        if board.shape[1] != self.channels:
            raise ValueError("scene channels do not match this model checkpoint")
        mask = (
            board[:, 10:11] > 0.5
            if self.channels == 14
            else torch.ones_like(board[:, :1], dtype=torch.bool)
        )
        if not mask.flatten(1).any(1).all():
            raise ValueError("scene has no playable cells")
        board = torch.where(mask, board, torch.zeros_like(board))
        board = torch.cat((board, mask.to(board.dtype)), dim=1)
        head = board[:, 2].flatten(1).argmax(1)
        hx, hy = head % width, head // width
        yy, xx = torch.meshgrid(
            torch.arange(height, device=board.device),
            torch.arange(width, device=board.device),
            indexing="ij",
        )
        dx = xx.flatten()[None] - hx[:, None]
        dy = yy.flatten()[None] - hy[:, None]
        offsets = torch.stack((dx, dy), -1).to(board.dtype)
        relative = offsets.sign() * offsets.abs().log1p()
        r = self.radius
        padded = F.pad(board, (r, r, r, r))
        steps = torch.arange(2 * r + 1, device=board.device)
        patch = padded[
            torch.arange(batch, device=board.device)[:, None, None],
            :,
            hy[:, None, None] + steps[None, :, None],
            hx[:, None, None] + steps[None, None, :],
        ].permute(0, 3, 1, 2)
        local = self.local(patch)
        # Mask between convolutions as well: bias in padded cells must not leak
        # back into valid boundary cells in the next convolution.
        spatial = self.spatial[1](self.spatial[0](board)) * mask
        tokens = (self.spatial[2](spatial) * mask).flatten(2).transpose(1, 2)
        scores = (self.key(tokens) * self.query(local)[:, None]).sum(-1)
        scores = scores / tokens.shape[-1] ** 0.5 + self.position(relative).squeeze(-1)
        scores = scores.masked_fill(~mask.flatten(1), -torch.inf)
        values = self.value(tokens)
        # One local-biased readout and one unrestricted readout.
        near = scores - offsets.abs().sum(-1).log1p()
        near = (near.softmax(-1)[..., None] * values).sum(1)
        global_read = (scores.softmax(-1)[..., None] * values).sum(1)
        reads = [local, near, global_read]
        if self.entities is not None:
            # Shared cell encoder with relative coordinates, then symmetric pooling.
            # No path, action labels, or privileged food coordinates are supplied.
            entities = self.entities(
                torch.cat((board.flatten(2).transpose(1, 2), relative / 4), -1)
            )
            entities = entities.masked_fill(~mask.flatten(1)[..., None], -torch.inf)
            reads.append(entities.max(1).values)
        return self.output(torch.cat(reads, -1))


class GamepadPolicy(nn.Module):
    def __init__(
        self,
        dim=64,
        chunk_length=8,
        *,
        channels=9,
        time_features=False,
        decoder="parallel",
        spatial_pool=False,
        chunk_rho=None,
        action_head="raw",
    ):
        super().__init__()
        if action_head not in ("raw", "dpad5", "held_dpad5"):
            raise ValueError("action_head must be raw, dpad5 or held_dpad5")
        if action_head in ("dpad5", "held_dpad5") and chunk_rho is not None:
            raise ValueError("chunk_rho is not supported by a dpad5 action head")
        self.chunk_length = chunk_length
        self.chunk_rho = chunk_rho
        self.action_head = action_head
        if chunk_rho is not None:
            self.button_log_std = nn.Parameter(torch.zeros(chunk_length, 14))
        self.encoder = HeadLocalEncoder(
            channels=channels, dim=dim, spatial_pool=spatial_pool
        )
        self.time_features = time_features
        if decoder not in ("parallel", "gru"):
            raise ValueError("decoder must be parallel or gru")
        self.decoder = decoder
        snake_control_mask = torch.zeros(20)
        snake_control_mask[:4] = 1
        snake_control_mask[14:16] = 1
        # Keep the public controller tensor complete while making the Snake
        # policy state independent of controls the game cannot observe.  This
        # is required before their score terms can be marginalized without
        # bias: otherwise sampled shoulder/face buttons could feed back through
        # report/history/previous_chunk and change a later effective action.
        self.register_buffer("snake_control_mask", snake_control_mask, persistent=False)
        self.time_encoder = nn.Linear(5 + chunk_length, dim) if time_features else None
        # Ordered controller feedback is encoded event-by-event by the caller.
        # Timing: observation age seconds, elapsed event seconds, new-frame flag.
        self.memory = nn.GRUCell(dim + 20 + 3, dim)
        self.feedback_memory = nn.GRUCell(21 if time_features else 20, dim)
        self.context = nn.Linear(dim + chunk_length * 20 + 1, dim)
        self.chunk_memory = nn.GRUCell(dim + 1, dim) if decoder == "gru" else None
        action_width = 20 if action_head == "raw" else 5
        actor_width = (
            5
            if action_head == "held_dpad5"
            else action_width
            if decoder == "gru"
            else chunk_length * action_width
        )
        self.actor = nn.Linear(dim, actor_width)
        self.critic = nn.Linear(dim, 1)
        if action_head == "raw":
            self.log_std = nn.Parameter(torch.full((chunk_length, 6), -1.0))

    def _state(
        self,
        board,
        actual_report,
        timing,
        hidden=None,
        *,
        history=None,
        history_mask=None,
        previous_chunk=None,
        previous_age=None,
        time_context=None,
        target_dt=None,
        history_age=None,
    ):
        actual_report = actual_report * self.snake_control_mask
        features = self.encoder(board)
        if self.time_features:
            if time_context is None or target_dt is None or history_age is None:
                raise ValueError("time-conditioned policy requires timing-v2 fields")
            features = features + self.time_encoder(
                torch.cat((time_context, target_dt), -1)
            )
        if history is not None:
            if history_mask is None or previous_chunk is None or previous_age is None:
                raise ValueError("history requires mask and previous command context")
            feedback = torch.zeros_like(features)
            for j in range(history.shape[1]):
                event = history[:, j] * self.snake_control_mask
                if self.time_features:
                    event = torch.cat((event, history_age[:, j, None]), -1)
                candidate = self.feedback_memory(event, feedback)
                feedback = torch.where(
                    history_mask[:, j, None].bool(), candidate, feedback
                )
            features = features + self.context(
                torch.cat(
                    (
                        feedback,
                        (previous_chunk * self.snake_control_mask).flatten(1),
                        previous_age,
                    ),
                    -1,
                )
            )
        hidden = self.memory(torch.cat((features, actual_report, timing), -1), hidden)
        return hidden

    def _logits(self, hidden, target_dt):
        action_width = 20 if self.action_head == "raw" else 5
        if self.action_head == "held_dpad5":
            return self.actor(hidden)
        if self.decoder == "parallel":
            return self.actor(hidden).reshape(-1, self.chunk_length, action_width)
        if target_dt is None:
            raise ValueError("chunk decoder requires target time offsets")
        state, outputs = hidden, []
        for j in range(self.chunk_length):
            state = self.chunk_memory(
                torch.cat((hidden, target_dt[:, j, None]), -1), state
            )
            outputs.append(self.actor(state))
        return torch.stack(outputs, 1)

    def forward(self, board, actual_report, timing, hidden=None, **context):
        hidden = self._state(board, actual_report, timing, hidden, **context)
        logits = self._logits(hidden, context.get("target_dt"))
        if self.action_head == "held_dpad5":
            report = _held_dpad5_controls_with_gradient(logits, self.chunk_length)
        elif self.action_head == "dpad5":
            report = _dpad5_controls_with_gradient(logits)
        else:
            report = hard_controls_with_gradient(logits, stochastic=False)
        return (
            report,
            self.critic(hidden),
            hidden,
        )

    def distribution(self, board, actual_report, timing, hidden=None, **context):
        hidden = self._state(board, actual_report, timing, hidden, **context)
        mean = self._logits(hidden, context.get("target_dt"))
        if self.action_head == "held_dpad5":
            # One score-function variable controls every report slot.  This is
            # a diagnostic action-repeat head, not L identical samples.
            return (
                HeldDpad5Categorical(logits=mean, chunk_length=self.chunk_length),
                self.critic(hidden),
                hidden,
            )
        if self.action_head == "dpad5":
            # Each position in the command chunk is an independent choice from
            # a fixed legal controller-report set.  This mapping never reads
            # game state and performs no collision or reverse-action repair.
            return (
                torch.distributions.Categorical(logits=mean),
                self.critic(hidden),
                hidden,
            )
        buttons = torch.distributions.Bernoulli(logits=mean[..., :14])
        axes = torch.distributions.Normal(mean[..., 14:], self.log_std.exp())
        if self.chunk_rho is not None:
            from resnake_gym.chunk_normal import ChunkNormal

            buttons = ChunkNormal(
                mean[..., :14], self.button_log_std.exp(), self.chunk_rho
            )
            axes = ChunkNormal(mean[..., 14:], self.log_std.exp(), self.chunk_rho)
        # All 20 controls are sampled and remain available to downstream users,
        # but Snake can only observe D-pad buttons and the two left-stick axes.
        # PPO/SIL therefore score the exact marginal over those six controls;
        # unrelated controller noise must not enter their ratios or entropy.
        buttons = _MarginalControlDistribution(buttons, active_controls=4)
        axes = _MarginalControlDistribution(axes, active_controls=2)
        # Score-function RL uses the causal marginal and its axis latents.
        # Reparameterized downstream training uses hard_controls_with_gradient.
        # All six Gaussian latents must be transformed before device execution.
        return (buttons, axes), self.critic(hidden), hidden


def reports_from_samples(buttons, axis_latents, *, continuous_buttons=False):
    """Fixed transforms; PPO ratios can be computed in latent space.

    The parameter-independent transform Jacobian cancels between old/new
    probabilities. Entropy of these latents is not physical-report entropy.
    """
    return torch.cat(
        (
            buttons.sigmoid() if continuous_buttons else buttons,
            axis_latents[..., :4].tanh(),
            axis_latents[..., 4:].sigmoid(),
        ),
        -1,
    )


def sample_log_prob(distributions, buttons, axis_latents):
    button_dist, axis_dist = distributions
    return button_dist.log_prob(buttons).sum((-1, -2)) + axis_dist.log_prob(
        axis_latents
    ).sum((-1, -2))


def sample_policy_action(distribution, stochastic=True):
    """Sample either policy head, retaining its exact score-function latent."""
    if isinstance(distribution, torch.distributions.Categorical):
        return distribution.sample() if stochastic else distribution.logits.argmax(-1)
    button_dist, axis_dist = distribution
    if stochastic:
        return button_dist.sample(), axis_dist.sample()
    buttons = (
        (button_dist.probs >= 0.5).to(button_dist.probs.dtype)
        if hasattr(button_dist, "probs")
        else button_dist.loc
    )
    return buttons, axis_dist.loc


def policy_action_log_prob(distribution, action):
    """Return one joint chunk log probability per batch element."""
    if isinstance(distribution, HeldDpad5Categorical):
        return distribution.log_prob(action)
    if isinstance(distribution, torch.distributions.Categorical):
        return distribution.log_prob(action).sum(-1)
    buttons, axis_latents = action
    return sample_log_prob(distribution, buttons, axis_latents)


def policy_action_entropy(distribution):
    """Return one summed latent-action entropy per batch element."""
    if isinstance(distribution, HeldDpad5Categorical):
        return distribution.entropy()
    if isinstance(distribution, torch.distributions.Categorical):
        return distribution.entropy().sum(-1)
    return sum(component.entropy().sum((-1, -2)) for component in distribution)


def reports_from_policy_action(distribution, action, *, continuous_buttons=False):
    """Map a score-function latent to the normalized 20-control report API."""
    if isinstance(distribution, torch.distributions.Categorical):
        if action.dtype.is_floating_point or action.dtype == torch.bool:
            raise TypeError("dpad5 actions must be integer category indices")
        if torch.any((action < 0) | (action >= len(DPAD5_LABELS))):
            raise ValueError("dpad5 category is outside [0, 5)")
        templates = _dpad5_templates(
            device=action.device, dtype=distribution.logits.dtype
        )
        reports = templates[action]
        if isinstance(distribution, HeldDpad5Categorical):
            return reports.unsqueeze(-2).expand(
                *reports.shape[:-1], distribution.chunk_length, 20
            )
        return reports
    buttons, axis_latents = action
    return reports_from_samples(
        buttons, axis_latents, continuous_buttons=continuous_buttons
    )


def policy_inputs(observation, device):
    """Convert delivered environment observations, never diagnostics or truth."""
    return {
        key: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0)
        for key, value in observation.items()
    }


def distribution_from_observation(model, observation, hidden=None):
    timed = {
        key: observation[key]
        for key in ("time_context", "target_dt", "history_age")
        if key in observation
    }
    return model.distribution(
        observation["board"],
        observation["report"],
        observation["timing"],
        hidden,
        history=observation["history"],
        history_mask=observation["history_mask"],
        previous_chunk=observation["previous_chunk"],
        previous_age=observation["previous_age"],
        **timed,
    )
