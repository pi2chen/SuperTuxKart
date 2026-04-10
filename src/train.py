"""Training entry-point for the SuperTuxKart deep learning agent.

Commands
--------
collect
    Record human demonstrations to a pickle dataset.
il
    Train the agent using imitation learning (behaviour cloning).
rl
    Fine-tune or train from scratch using PPO.

Usage examples
--------------
::

    python src/train.py collect --track zengarden --laps 3 --out data/demo.pkl
    python src/train.py il --data data/demo.pkl --epochs 20 --save checkpoints/il.pt
    python src/train.py rl --checkpoint checkpoints/il.pt --timesteps 200000 \\
                           --save checkpoints/rl.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, random_split

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from agent import ILAgent, PPOAgent
from utils import (
    DemonstrationDataset,
    Transition,
    compute_reward,
    pack_action,
    save_transitions,
)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_demonstrations(track: str, laps: int, out: str) -> None:
    """Collect human (or scripted AI) demonstrations using pystk.

    The kart is controlled by the built-in AI at ``Difficulty.MEDIUM`` to avoid
    requiring a physical keyboard/gamepad in a headless environment.  Replace the
    action generation logic with human input if running interactively.

    Args:
        track: pystk track name (e.g. ``"zengarden"``).
        laps: Number of laps to record.
        out: Output path for the pickle dataset.
    """
    try:
        import pystk
    except ImportError:
        print("ERROR: pystk is not installed.  Run: pip install pystk")
        sys.exit(1)

    config = pystk.GraphicsConfig.hd()
    config.screen_width = 128
    config.screen_height = 128
    pystk.init(config)

    race_config = pystk.RaceConfig()
    race_config.track = track
    race_config.laps = laps
    race_config.players[0].controller = pystk.PlayerConfig.Controller.AI_CONTROL

    race = pystk.Race(race_config)
    race.start()
    race.step()

    transitions: List[Transition] = []

    try:
        while True:
            state = pystk.WorldState()
            state.update()
            kart = state.players[0].kart

            # Retrieve the rendered image (H×W×3 uint8)
            image = race.render_data[0].image

            # Read back the AI action that was applied this step
            action = race.render_data[0].action  # pystk.Action

            transitions.append(
                Transition(
                    image=image.copy(),
                    steer=float(action.steer),
                    acceleration=float(action.acceleration),
                    brake=bool(action.brake),
                )
            )

            done = race.step() is None or kart.lap >= laps
            if done:
                break
    finally:
        race.stop()
        pystk.clean()

    save_transitions(transitions, out)
    print(f"Saved {len(transitions)} transitions → {out}")


# ---------------------------------------------------------------------------
# Imitation learning
# ---------------------------------------------------------------------------


def train_il(
    data: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    save: str,
    device: str,
    image_size: int,
) -> None:
    """Train the policy with behaviour cloning.

    Args:
        data: Path to the demonstration pickle file.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        lr: Learning rate for AdamW.
        val_split: Fraction of data used for validation.
        save: Checkpoint save path.
        device: Torch device string.
        image_size: Square image size for the model.
    """
    dataset = DemonstrationDataset(data, image_size=image_size, train=True)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    agent = ILAgent(image_size=image_size, lr=lr, device=device)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        # ----- training pass -----
        train_losses: List[float] = []
        agent.model.train()
        for batch in train_loader:
            metrics = agent.train_step(
                batch["image"],
                batch["steer"],
                batch["acceleration"],
                batch["brake"],
            )
            train_losses.append(metrics["loss"])
        agent.step_scheduler()

        # ----- validation pass -----
        val_losses: List[float] = []
        agent.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(agent.device)
                steer_t = batch["steer"].to(agent.device)
                accel_t = batch["acceleration"].to(agent.device)
                brake_t = batch["brake"].to(agent.device)
                pred_s, pred_a, pred_b = agent.model(images)
                loss = (
                    torch.nn.functional.mse_loss(pred_s, steer_t)
                    + torch.nn.functional.mse_loss(pred_a, accel_t)
                    + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(pred_b, brake_t)
                )
                val_losses.append(loss.item())

        mean_train = sum(train_losses) / len(train_losses)
        mean_val = sum(val_losses) / len(val_losses)
        print(f"[Epoch {epoch:3d}/{epochs}]  train={mean_train:.4f}  val={mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            agent.save(save)
            print(f"  ✓ Saved checkpoint → {save}  (val={best_val_loss:.4f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Reinforcement learning (PPO)
# ---------------------------------------------------------------------------


def train_rl(
    track: str,
    timesteps: int,
    save: str,
    checkpoint: str | None,
    device: str,
    image_size: int,
) -> None:
    """Fine-tune (or train from scratch) with PPO.

    Args:
        track: pystk track name.
        timesteps: Total environment steps to collect.
        save: Checkpoint save path.
        checkpoint: Optional path to an IL checkpoint to initialise the encoder.
        device: Torch device string.
        image_size: Square image size.
    """
    try:
        import pystk
    except ImportError:
        print("ERROR: pystk is not installed.  Run: pip install pystk")
        sys.exit(1)

    # Build agent
    if checkpoint is not None:
        print(f"Loading encoder weights from IL checkpoint: {checkpoint}")
        il_ckpt = torch.load(checkpoint, map_location=device)
        agent = PPOAgent(image_size=image_size, device=device)
        # Transfer encoder weights only
        enc_state = {
            k.removeprefix("encoder."): v
            for k, v in il_ckpt["model_state"].items()
            if k.startswith("encoder.")
        }
        agent.model.encoder.load_state_dict(enc_state)
    else:
        agent = PPOAgent(image_size=image_size, device=device)

    # Initialise pystk
    gfx = pystk.GraphicsConfig.hd()
    gfx.screen_width = image_size
    gfx.screen_height = image_size
    pystk.init(gfx)

    race_config = pystk.RaceConfig()
    race_config.track = track
    race_config.laps = 1
    race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

    step = 0
    episode = 0

    while step < timesteps:
        race = pystk.Race(race_config)
        race.start()
        race.step()

        state = pystk.WorldState()
        state.update()
        prev_dist = state.players[0].kart.overall_distance
        ep_reward = 0.0
        ep_steps = 0

        while True:
            image = race.render_data[0].image
            steer, accel, brake, log_prob, value = agent.select_action(image)

            action = pystk.Action()
            action.steer = steer
            action.acceleration = accel
            action.brake = brake
            race.step(action)

            state.update()
            kart = state.players[0].kart
            curr_dist = kart.overall_distance
            speed = kart.velocity_lc[1]

            # Detect off-track heuristically: very low speed + no progress
            off_track = bool(curr_dist - prev_dist < 0.01 and speed < 1.0)
            wrong_way = bool(speed < -0.5)

            reward = compute_reward(prev_dist, curr_dist, speed, off_track, wrong_way)
            done = kart.lap >= 1 or ep_steps > 3000

            agent.buffer.push(image, steer, accel, float(brake), reward, value, log_prob, done)
            prev_dist = curr_dist
            ep_reward += reward
            ep_steps += 1
            step += 1

            if agent.buffer.is_full or done:
                # Bootstrap value for truncated rollout
                if not done:
                    _, _, _, _, last_val = agent.select_action(image)
                else:
                    last_val = 0.0

                metrics = agent.update(last_val)
                print(
                    f"[Step {step:7d}/{timesteps}] ep={episode:4d}"
                    f"  reward={ep_reward:8.2f}"
                    f"  π_loss={metrics['policy_loss']:.4f}"
                    f"  v_loss={metrics['value_loss']:.4f}"
                    f"  entropy={metrics['entropy']:.4f}"
                )

            if done:
                break

        race.stop()
        episode += 1

    pystk.clean()
    agent.save(save)
    print(f"RL training complete.  Saved → {save}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SuperTuxKart Deep Learning Agent – Training"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # collect sub-command
    p_collect = sub.add_parser("collect", help="Record AI demonstrations")
    p_collect.add_argument("--track", default="zengarden", help="pystk track name")
    p_collect.add_argument("--laps", type=int, default=3, help="Number of laps to record")
    p_collect.add_argument("--out", default="data/demo.pkl", help="Output pickle path")

    # il sub-command
    p_il = sub.add_parser("il", help="Train with imitation learning")
    p_il.add_argument("--data", required=True, help="Path to demonstration pickle")
    p_il.add_argument("--epochs", type=int, default=30)
    p_il.add_argument("--batch-size", type=int, default=64)
    p_il.add_argument("--lr", type=float, default=1e-3)
    p_il.add_argument("--val-split", type=float, default=0.1)
    p_il.add_argument("--save", default="checkpoints/il_agent.pt")
    p_il.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_il.add_argument("--image-size", type=int, default=128)

    # rl sub-command
    p_rl = sub.add_parser("rl", help="Train / fine-tune with PPO")
    p_rl.add_argument("--track", default="zengarden")
    p_rl.add_argument("--timesteps", type=int, default=500_000)
    p_rl.add_argument("--save", default="checkpoints/rl_agent.pt")
    p_rl.add_argument("--checkpoint", default=None, help="IL checkpoint to warm-start encoder")
    p_rl.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_rl.add_argument("--image-size", type=int, default=128)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        collect_demonstrations(args.track, args.laps, args.out)

    elif args.command == "il":
        train_il(
            data=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            save=args.save,
            device=args.device,
            image_size=args.image_size,
        )

    elif args.command == "rl":
        train_rl(
            track=args.track,
            timesteps=args.timesteps,
            save=args.save,
            checkpoint=args.checkpoint,
            device=args.device,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
