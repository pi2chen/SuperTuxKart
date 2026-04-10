"""Neural network models for the SuperTuxKart deep learning agent.

Provides:
  - ``KartEncoder``: A lightweight ResNet-style CNN that maps raw RGB frames
    to a compact feature vector.
  - ``PolicyHead``: Fully-connected head that predicts continuous steering /
    acceleration and a discrete brake action.
  - ``ValueHead``: Critic head used by PPO.
  - ``KartPolicy``: Combines encoder + policy head (actor-only, for IL).
  - ``KartActorCritic``: Combines encoder + policy head + value head (for PPO).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """A single residual block: two 3×3 convolutions with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class KartEncoder(nn.Module):
    """Lightweight CNN encoder.

    Converts a batch of RGB images (B, 3, H, W) → feature vectors (B, feat_dim).

    Architecture
    ------------
    Stem → 3 stages of (conv stride-2 + ResBlock) → global average pool → Linear

    Args:
        image_size: Spatial size of the square input image (default 128).
        feat_dim: Output feature dimensionality (default 512).
        base_channels: Width of the first stage (doubles each stage, default 32).
    """

    def __init__(
        self,
        image_size: int = 128,
        feat_dim: int = 512,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            ResBlock(c * 2),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            ResBlock(c * 4),
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
            ResBlock(c * 8),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c * 8, feat_dim)

        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(B, 3, H, W)`` with values in ``[0, 1]``.

        Returns:
            Feature vector of shape ``(B, feat_dim)``.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class PolicyHead(nn.Module):
    """Maps encoder features → driving actions.

    Output
    ------
    steer        : ``(B, 1)`` in ``[-1, 1]`` (tanh)
    acceleration : ``(B, 1)`` in ``[0, 1]``  (sigmoid)
    brake_logit  : ``(B, 1)`` raw logit for brake (BCEWithLogitsLoss / Bernoulli)
    """

    def __init__(self, feat_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.steer_head = nn.Linear(hidden_dim, 1)
        self.accel_head = nn.Linear(hidden_dim, 1)
        self.brake_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.mlp(features)
        steer = torch.tanh(self.steer_head(h))
        acceleration = torch.sigmoid(self.accel_head(h))
        brake_logit = self.brake_head(h)
        return steer, acceleration, brake_logit


class ValueHead(nn.Module):
    """Maps encoder features → scalar state value (for PPO critic)."""

    def __init__(self, feat_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


# ---------------------------------------------------------------------------
# Full models
# ---------------------------------------------------------------------------


class KartPolicy(nn.Module):
    """Actor-only model: encoder + policy head.

    Suitable for imitation learning (behaviour cloning).

    Args:
        image_size: Spatial size of the square input image.
        feat_dim: Latent feature dimension.
        base_channels: CNN width multiplier.
        hidden_dim: Width of the policy MLP.
    """

    def __init__(
        self,
        image_size: int = 128,
        feat_dim: int = 512,
        base_channels: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = KartEncoder(image_size, feat_dim, base_channels)
        self.policy = PolicyHead(feat_dim, hidden_dim)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(steer, acceleration, brake_logit)`` for a batch of images."""
        features = self.encoder(images)
        return self.policy(features)


class KartActorCritic(nn.Module):
    """Actor-Critic model: shared encoder + policy head + value head.

    Suitable for on-policy RL (e.g. PPO).

    Args:
        image_size: Spatial size of the square input image.
        feat_dim: Latent feature dimension.
        base_channels: CNN width multiplier.
        hidden_dim: Width of the policy / value MLPs.
    """

    def __init__(
        self,
        image_size: int = 128,
        feat_dim: int = 512,
        base_channels: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = KartEncoder(image_size, feat_dim, base_channels)
        self.policy = PolicyHead(feat_dim, hidden_dim)
        self.value = ValueHead(feat_dim, hidden_dim)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return ``((steer, acceleration, brake_logit), value)``."""
        features = self.encoder(images)
        actions = self.policy(features)
        value = self.value(features)
        return actions, value

    def act(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method: return only the action tuple (no gradient)."""
        with torch.no_grad():
            features = self.encoder(images)
            return self.policy(features)
