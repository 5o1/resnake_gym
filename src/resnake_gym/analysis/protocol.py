"""Frozen protocol constants for formal V-trace v3 experiments."""

REPORT_FORMAT = "resnake-vtrace-v3-offline-analysis-v1"
EXPECTED_FORMAT = "gamepad-vtrace-v3"
EXPECTED_COLLECTION = "fifo-stitched-credit-trace-v3"
EXPECTED_RECURRENT = "stored-state-burnin-window-tbptt-v5"
EXPECTED_METRICS = "received-vs-trained-observability-v1"
SCORE_THRESHOLDS = (1, 2, 5, 10, 25, 50, 100, 250)
DEATH_REASONS = frozenset(("wall_collision", "self_collision", "obstacle_collision"))

FORMAL_ARMS = {
    "U0": {
        "action_head": "dpad5",
        "entropy_coefficient": 0.001,
        "batch_min_transitions": 2048,
        "credit_trace_max_transitions": 128,
    },
    "J0": {
        "action_head": "dpad5",
        "entropy_coefficient": 0.001,
        "batch_min_transitions": 512,
        "credit_trace_max_transitions": 128,
    },
    "H0": {
        "action_head": "held_dpad5",
        "entropy_coefficient": 0.001,
        "batch_min_transitions": 512,
        "credit_trace_max_transitions": 128,
    },
    "H8": {
        "action_head": "held_dpad5",
        "entropy_coefficient": 0.008,
        "batch_min_transitions": 512,
        "credit_trace_max_transitions": 128,
    },
    "JL": {
        "action_head": "dpad5",
        "entropy_coefficient": 0.001,
        "batch_min_transitions": 512,
        "credit_trace_max_transitions": 2048,
    },
    "HL": {
        "action_head": "held_dpad5",
        "entropy_coefficient": 0.008,
        "batch_min_transitions": 512,
        "credit_trace_max_transitions": 2048,
    },
}

FORMAL_COMMON_CONFIG = {
    "width": 31,
    "height": 20,
    "gamma": 0.9999,
    "learning_rate": 0.0001,
    "actor_processes": 8,
    "envs_per_actor": 2,
    "actor_sync_steps": 128,
    "unroll_length": 128,
    "recurrent_burn_in": 64,
    "bptt_window": 128,
    "batch_max_transitions": 8192,
    "batch_food_target": 4,
}

FORMAL_CONTRASTS = (
    ("J0-U0", "J0", "U0"),
    ("H0-J0", "H0", "J0"),
    ("H8-H0", "H8", "H0"),
    ("H8-J0", "H8", "J0"),
    ("JL-J0", "JL", "J0"),
    ("HL-H8", "HL", "H8"),
)
