"""Agent wrappers for imitation learning and PPO-based reinforcement learning.

Classes
-------
ILAgent
    Behaviour-cloning agent that wraps ``KartPolicy``.
PPOAgent
    Proximal Policy Optimisation agent that wraps ``KartActorCritic``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from PIL import Image as PILImage

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Normal

from model import KartActorCritic, KartPolicy
from utils import RolloutBuffer, image_to_tensor, get_train_transform


# ---------------------------------------------------------------------------
# Imitation-learning agent
# ---------------------------------------------------------------------------


class ILAgent:
    """Behaviour-cloning agent backed by ``KartPolicy``.

    Args:
        image_size: Square input image dimension.
        feat_dim: Encoder output dimension.
        base_channels: CNN width multiplier.
        hidden_dim: Policy head hidden width.
        lr: AdamW learning rate.
        device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self,
        image_size: int = 128,
        feat_dim: int = 512,
        base_channels: int = 32,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size

        self.model = KartPolicy(image_size, feat_dim, base_channels, hidden_dim).to(
            self.device
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-5
        )

        self._steer_loss = nn.MSELoss()
        self._accel_loss = nn.MSELoss()
        self._brake_loss = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        images: torch.Tensor,
        target_steer: torch.Tensor,
        target_accel: torch.Tensor,
        target_brake: torch.Tensor,
    ) -> Dict[str, float]:
        """Run one gradient-descent step.

        Args:
            images: ``(B, 3, H, W)`` float tensor on the agent device.
            target_steer: ``(B, 1)`` float tensor.
            target_accel: ``(B, 1)`` float tensor.
            target_brake: ``(B, 1)`` float tensor (0 or 1).

        Returns:
            Dictionary with scalar loss values.
        """
        self.model.train()
        images = images.to(self.device)
        target_steer = target_steer.to(self.device)
        target_accel = target_accel.to(self.device)
        target_brake = target_brake.to(self.device)

        pred_steer, pred_accel, pred_brake_logit = self.model(images)

        loss_steer = self._steer_loss(pred_steer, target_steer)
        loss_accel = self._accel_loss(pred_accel, target_accel)
        loss_brake = self._brake_loss(pred_brake_logit, target_brake)

        loss = loss_steer + loss_accel + 0.5 * loss_brake

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "loss_steer": loss_steer.item(),
            "loss_accel": loss_accel.item(),
            "loss_brake": loss_brake.item(),
        }

    def step_scheduler(self) -> None:
        self.scheduler.step()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[float, float, bool]:
        """Predict actions for a single raw RGB frame.

        Args:
            image: ``(H, W, 3)`` uint8 numpy array.

        Returns:
            ``(steer, acceleration, brake)``
        """
        self.model.eval()
        tensor = image_to_tensor(image, image_size=self.image_size).to(self.device)
        steer, accel, brake_logit = self.model(tensor)
        steer_val = steer.item()
        accel_val = accel.item()
        brake_val = torch.sigmoid(brake_logit).item() > 0.5
        return steer_val, accel_val, brake_val

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": {
                    "image_size": self.image_size,
                    "feat_dim": self.model.encoder.feat_dim,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ILAgent":
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]
        agent = cls(image_size=cfg["image_size"], feat_dim=cfg["feat_dim"], device=device)
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return agent


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------


class PPOAgent:
    """Proximal Policy Optimisation agent backed by ``KartActorCritic``.

    Uses Gaussian distributions for the continuous actions (steer, acceleration)
    and a Bernoulli distribution for the discrete brake action.

    Args:
        image_size: Square input image dimension.
        feat_dim: Encoder output dimension.
        base_channels: CNN width multiplier.
        hidden_dim: Actor/critic head hidden width.
        lr: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: GAE coefficient.
        clip_eps: PPO clipping epsilon.
        vf_coef: Value function loss coefficient.
        ent_coef: Entropy bonus coefficient.
        max_grad_norm: Gradient clipping norm.
        n_epochs: Number of PPO mini-batch epochs per update.
        batch_size: Mini-batch size.
        buffer_size: Rollout buffer capacity.
        device: Torch device string.
    """

    # Learnable log-std for continuous actions
    LOG_STD_MIN = -4.0
    LOG_STD_MAX = 0.5

    def __init__(
        self,
        image_size: int = 128,
        feat_dim: int = 512,
        base_channels: int = 32,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 256,
        buffer_size: int = 2048,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.image_size = image_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model = KartActorCritic(
            image_size, feat_dim, base_channels, hidden_dim
        ).to(self.device)

        # Learnable log-std parameters (separate from the policy network)
        self.log_std_steer = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_std_accel = nn.Parameter(torch.zeros(1, device=self.device))

        self.optimizer = optim.Adam(
            list(self.model.parameters()) + [self.log_std_steer, self.log_std_accel],
            lr=lr,
        )

        self.buffer = RolloutBuffer(max_size=buffer_size)

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def select_action(
        self, image: np.ndarray
    ) -> Tuple[float, float, bool, float, float]:
        """Sample actions from the current policy.

        Args:
            image: ``(H, W, 3)`` uint8 numpy array.

        Returns:
            ``(steer, acceleration, brake, log_prob, value)``
        """
        self.model.eval()
        with torch.no_grad():
            tensor = image_to_tensor(image, image_size=self.image_size).to(self.device)
            (mu_steer, mu_accel, brake_logit), value = self.model(tensor)

            log_std_s = self.log_std_steer.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
            log_std_a = self.log_std_accel.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

            dist_steer = Normal(mu_steer, log_std_s.exp())
            dist_accel = Normal(mu_accel, log_std_a.exp())
            dist_brake = Bernoulli(logits=brake_logit)

            s = dist_steer.sample()
            a = dist_accel.sample()
            b = dist_brake.sample()

            log_prob = (
                dist_steer.log_prob(s)
                + dist_accel.log_prob(a)
                + dist_brake.log_prob(b)
            ).sum().item()

        steer = float(s.clamp(-1.0, 1.0).item())
        accel = float(a.clamp(0.0, 1.0).item())
        brake = bool(b.item() > 0.5)
        return steer, accel, brake, log_prob, value.item()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, last_value: float) -> Dict[str, float]:
        """Run a PPO update on the current rollout buffer.

        Args:
            last_value: Bootstrap value for the final state.

        Returns:
            Dictionary with mean losses over all mini-batch updates.
        """
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Build tensors from the buffer
        transform = get_train_transform(self.image_size)
        images = torch.stack(
            [
                transform(PILImage.fromarray(img.astype(np.uint8)))
                for img in self.buffer.images
            ]
        ).to(self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32, device=self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        target_steers = torch.tensor(self.buffer.steers, dtype=torch.float32, device=self.device).unsqueeze(1)
        target_accels = torch.tensor(self.buffer.accelerations, dtype=torch.float32, device=self.device).unsqueeze(1)
        target_brakes = torch.tensor(self.buffer.brakes, dtype=torch.float32, device=self.device).unsqueeze(1)

        T = len(self.buffer.rewards)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = torch.randperm(T)
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                self.model.train()
                (mu_steer, mu_accel, brake_logit), values = self.model(images[idx])

                log_std_s = self.log_std_steer.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
                log_std_a = self.log_std_accel.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

                dist_steer = Normal(mu_steer, log_std_s.exp())
                dist_accel = Normal(mu_accel, log_std_a.exp())
                dist_brake = Bernoulli(logits=brake_logit)

                new_log_probs = (
                    dist_steer.log_prob(target_steers[idx])
                    + dist_accel.log_prob(target_accels[idx])
                    + dist_brake.log_prob(target_brakes[idx])
                ).sum(dim=1)

                entropy = (
                    dist_steer.entropy() + dist_accel.entropy() + dist_brake.entropy()
                ).mean()

                ratio = (new_log_probs - old_log_probs[idx]).exp()
                adv = advantages[idx]
                policy_loss = -torch.min(
                    ratio * adv,
                    ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv,
                ).mean()

                value_pred = values.squeeze(1)
                value_loss = nn.functional.mse_loss(value_pred, returns[idx])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + [self.log_std_steer, self.log_std_accel],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "log_std_steer": self.log_std_steer.data,
                "log_std_accel": self.log_std_accel.data,
                "config": {
                    "image_size": self.image_size,
                    "feat_dim": self.model.encoder.feat_dim,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PPOAgent":
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]
        agent = cls(image_size=cfg["image_size"], feat_dim=cfg["feat_dim"], device=device)
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.log_std_steer.data = checkpoint["log_std_steer"].to(agent.device)
        agent.log_std_accel.data = checkpoint["log_std_accel"].to(agent.device)
        return agent
