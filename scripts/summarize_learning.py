"""Read training JSONL without loading weights or transferring replay archives."""

import argparse
import json
from pathlib import Path


def summarize(path, window):
    rows = []
    for line in (path / "metrics.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # An active writer may not have completed its last line.
    if not rows:
        return {"run": str(path), "updates": 0}
    selected = rows[-window:]
    episodes = [episode for row in selected for episode in row["episodes"]]
    count = len(episodes)
    return {
        "run": str(path),
        "update": rows[-1]["update"],
        "policy_version": rows[-1]["sample_policy_version"],
        "logic_ticks": rows[-1]["logic_ticks"],
        "window_updates": len(selected),
        "completed_games": count,
        "wins": sum(e["won"] for e in episodes),
        "mean_score": sum(e["score"] for e in episodes) / max(count, 1),
        "max_score": max((e["score"] for e in episodes), default=0),
        "mean_lifetime_ticks": sum(e["ticks"] for e in episodes) / max(count, 1),
        "food_events": sum(row["food_count"] for row in selected),
        "sil_rows": sum(row.get("sil_rows", 0) for row in selected),
        "sil_valid": sum(row["sil_valid"] for row in selected),
        "replay_episodes": rows[-1]["replay_episodes"],
        "scope": "training, not held-out evaluation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    for path in args.runs:
        print(json.dumps(summarize(path, args.window)))
