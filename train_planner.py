print("Time to train")

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import RoadDataset, load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    """Generic training loop for planner models predicting waypoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", batch_size=batch_size, shuffle=True, num_workers=2)
    val_data = load_data("drive_data/val", batch_size=batch_size, shuffle=False, num_workers=2)

    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # This code was written by GPT-5 (train_planner.py:51-141)
    for epoch in range(num_epoch):
        model.train()
        train_losses = []

        for batch in train_data:
            image = batch["image"]
            track_left = batch["track_left"]
            track_right = batch["track_right"]
            waypoints = batch["waypoints"]
            waypoints_mask = batch["waypoints_mask"]

            image = image.to(device)
            track_left = track_left.to(device)
            track_right = track_right.to(device)
            waypoints = waypoints.to(device)
            waypoints_mask = waypoints_mask.to(device)

            # Forward pass
            if "vit" in model_name.lower():
                preds = model(image)
            else:
                preds = model(track_left, track_right)

            # Mask out invalid waypoints before computing loss
            mask = waypoints_mask.unsqueeze(-1).expand_as(waypoints)
            masked_loss = loss_func(preds[mask], waypoints[mask])

            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()

            train_losses.append(masked_loss.item())
            logger.add_scalar("Loss/train_iter", masked_loss.item(), global_step)
            global_step += 1

        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses, lat_errs, long_errs = [], [], []

        with torch.no_grad():
            for batch in val_data:
                image = batch["image"]
                track_left = batch["track_left"]
                track_right = batch["track_right"]
                waypoints = batch["waypoints"]
                waypoints_mask = batch["waypoints_mask"]

                image = image.to(device)
                track_left = track_left.to(device)
                track_right = track_right.to(device)
                waypoints = waypoints.to(device)
                waypoints_mask = waypoints_mask.to(device)

                if "vit" in model_name.lower():
                    preds = model(image)
                else:
                    preds = model(track_left, track_right)

                mask = waypoints_mask.unsqueeze(-1).expand_as(waypoints)
                val_loss = loss_func(preds[mask], waypoints[mask])
                val_losses.append(val_loss.item())

                diff = torch.abs(preds - waypoints)
                lateral_error = diff[..., 0].mean().item()
                longitudinal_error = diff[..., 1].mean().item()
                lat_errs.append(lateral_error)
                long_errs.append(longitudinal_error)

        avg_val_loss = np.mean(val_losses)
        avg_lat_err = np.mean(lat_errs)
        avg_long_err = np.mean(long_errs)

        logger.add_scalar("Loss/train", avg_train_loss, epoch)
        logger.add_scalar("Loss/val", avg_val_loss, epoch)
        logger.add_scalar("Error/lateral", avg_lat_err, epoch)
        logger.add_scalar("Error/longitudinal", avg_long_err, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:02d}/{num_epoch:02d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Lateral Err: {avg_lat_err:.3f} | "
                f"Longitudinal Err: {avg_long_err:.3f}"
            )

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train planner models (MLP, Transformer, etc.)")

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the planner model to train (mlp, transformer, vit, etc.)")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    train(**vars(args))
