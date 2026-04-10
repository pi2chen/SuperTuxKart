"""Evaluation / live-play entry-point for the SuperTuxKart deep learning agent.

Usage
-----
::

    python src/evaluate.py \\
        --checkpoint checkpoints/rl_agent.pt \\
        --track zengarden \\
        --laps 1 \\
        --render

The script loads a trained checkpoint (IL or RL), initialises a pystk race, and
drives the kart autonomously while printing per-step telemetry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from agent import ILAgent, PPOAgent
from utils import compute_reward, pack_action


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint: str,
    track: str,
    laps: int,
    render: bool,
    agent_type: str,
    device: str,
    image_size: int,
) -> None:
    """Load a checkpoint and run the agent in a pystk race.

    Args:
        checkpoint: Path to a ``.pt`` file saved by ``ILAgent.save`` or ``PPOAgent.save``.
        track: pystk track name.
        laps: Number of laps to race.
        render: If ``True``, print frame-by-frame telemetry (verbose).
        agent_type: ``"il"`` or ``"rl"``.
        device: Torch device string.
        image_size: Square image size expected by the model.
    """
    try:
        import pystk
    except ImportError:
        print("ERROR: pystk is not installed.  Run: pip install pystk")
        sys.exit(1)

    # Load agent
    if agent_type == "il":
        agent = ILAgent.load(checkpoint, device=device)
        print(f"Loaded IL agent from {checkpoint}")
    else:
        agent = PPOAgent.load(checkpoint, device=device)
        print(f"Loaded PPO (RL) agent from {checkpoint}")

    # Set up pystk
    gfx = pystk.GraphicsConfig.hd()
    gfx.screen_width = image_size
    gfx.screen_height = image_size
    pystk.init(gfx)

    race_config = pystk.RaceConfig()
    race_config.track = track
    race_config.laps = laps
    race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

    race = pystk.Race(race_config)
    race.start()
    race.step()

    state = pystk.WorldState()
    state.update()

    rewards: List[float] = []
    prev_dist = state.players[0].kart.overall_distance
    step = 0

    try:
        while True:
            image = race.render_data[0].image

            if agent_type == "il":
                steer, accel, brake = agent.predict(image)
            else:
                steer, accel, brake, _, _ = agent.select_action(image)

            action = pystk.Action()
            action.steer = steer
            action.acceleration = accel
            action.brake = brake
            race.step(action)

            state.update()
            kart = state.players[0].kart
            curr_dist = kart.overall_distance
            speed = kart.velocity_lc[1]

            off_track = bool(curr_dist - prev_dist < 0.01 and speed < 1.0)
            wrong_way = bool(speed < -0.5)
            reward = compute_reward(prev_dist, curr_dist, speed, off_track, wrong_way)
            rewards.append(reward)

            if render:
                print(
                    f"step={step:5d}  steer={steer:+.3f}  accel={accel:.3f}"
                    f"  brake={int(brake)}  speed={speed:6.2f}  reward={reward:7.3f}"
                    f"  dist={curr_dist:.2f}"
                )

            prev_dist = curr_dist
            step += 1

            if kart.lap >= laps:
                print(f"\nFinished {laps} lap(s) in {step} steps.")
                break

    finally:
        race.stop()
        pystk.clean()

    total_reward = sum(rewards)
    mean_reward = total_reward / max(len(rewards), 1)
    print(f"Total reward : {total_reward:.2f}")
    print(f"Mean reward  : {mean_reward:.4f}")
    print(f"Steps        : {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SuperTuxKart Deep Learning Agent – Evaluation"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--track", default="zengarden", help="pystk track name")
    parser.add_argument("--laps", type=int, default=1)
    parser.add_argument(
        "--render", action="store_true", help="Print per-step telemetry"
    )
    parser.add_argument(
        "--agent-type",
        choices=["il", "rl"],
        default="il",
        help="'il' for imitation learning, 'rl' for PPO",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--image-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        track=args.track,
        laps=args.laps,
        render=args.render,
        agent_type=args.agent_type,
        device=args.device,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
