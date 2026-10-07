"""Head-local spatial encoding and differentiable, semantic gamepad conditioning.

No pretrained weights or demonstrations. Input is the delivered observation,
never the hidden current game state. Batches group equal board sizes.
"""

import torch
from torch import nn
from torch.nn import functional as F

import resnake_gym.gamepad as gamepad

DPAD5_LABELS = gamepad.DPAD5_LABELS


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
    ):
        super().__init__()
        self.chunk_length = chunk_length
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
        actor_width = 5 if decoder == "gru" else chunk_length * 5
        self.actor = nn.Linear(dim, actor_width)
        self.critic = nn.Linear(dim, 1)

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
        if self.decoder == "parallel":
            return self.actor(hidden).reshape(-1, self.chunk_length, 5)
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
        report = _dpad5_controls_with_gradient(logits)
        return (
            report,
            self.critic(hidden),
            hidden,
        )

    def distribution(self, board, actual_report, timing, hidden=None, **context):
        hidden = self._state(board, actual_report, timing, hidden, **context)
        mean = self._logits(hidden, context.get("target_dt"))
        # Every command slot is an independent categorical D-pad choice.  It is
        # converted to the public full [L, 20] controller-report tensor only at
        # the environment boundary.
        return (
            torch.distributions.Categorical(logits=mean),
            self.critic(hidden),
            hidden,
        )


def sample_policy_action(distribution, stochastic=True):
    """Sample the categorical D-pad action for every command slot."""
    return distribution.sample() if stochastic else distribution.logits.argmax(-1)


def policy_action_log_prob(distribution, action):
    """Return one joint chunk log probability per batch element."""
    return distribution.log_prob(action).sum(-1)


def policy_action_entropy(distribution):
    """Return one summed categorical-action entropy per batch element."""
    return distribution.entropy().sum(-1)


def reports_from_policy_action(distribution, action):
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
        return reports
    raise TypeError("policy distribution must be categorical")


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
