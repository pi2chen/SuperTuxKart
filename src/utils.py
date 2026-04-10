"""Utility helpers for the SuperTuxKart deep learning project.

Includes:
  - Image pre-processing transforms
  - ``DemonstrationDataset`` for imitation learning
  - ``ReplayBuffer`` for on-policy rollout storage (PPO)
  - Miscellaneous helpers (action packing/unpacking, reward shaping)
"""

from __future__ import annotations

import pickle
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def get_train_transform(image_size: int = 128) -> transforms.Compose:
    """Return the augmentation pipeline used during training."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomHorizontalFlip(p=0.0),  # disabled: flipping breaks steer
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def get_eval_transform(image_size: int = 128) -> transforms.Compose:
    """Return the deterministic transform used during evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


def image_to_tensor(
    image: np.ndarray, transform: Optional[transforms.Compose] = None, image_size: int = 128
) -> torch.Tensor:
    """Convert a ``H×W×3`` uint8 numpy array to a normalised ``(1,3,H,W)`` tensor.

    Args:
        image: Raw RGB frame from pystk (dtype uint8, values 0–255).
        transform: Optional torchvision transform pipeline.  Defaults to the
            evaluation transform.
        image_size: Target square size used when ``transform`` is ``None``.

    Returns:
        Float tensor of shape ``(1, 3, image_size, image_size)``.
    """
    if transform is None:
        transform = get_eval_transform(image_size)
    pil_img = Image.fromarray(image.astype(np.uint8))
    return transform(pil_img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Demonstration dataset
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    """A single ``(image, steer, acceleration, brake)`` demonstration step."""

    image: np.ndarray          # uint8, shape (H, W, 3)
    steer: float               # in [-1, 1]
    acceleration: float        # in [0, 1]
    brake: bool


class DemonstrationDataset(Dataset):
    """PyTorch dataset backed by a pickle file of ``Transition`` objects.

    Args:
        path: Path to the ``.pkl`` file produced by the ``collect`` command.
        image_size: Target image size for the transform.
        train: If ``True`` apply training augmentations, else evaluation transform.
    """

    def __init__(self, path: str | Path, image_size: int = 128, train: bool = True) -> None:
        self.transitions: List[Transition] = self._load(Path(path))
        self.transform = (
            get_train_transform(image_size) if train else get_eval_transform(image_size)
        )

    @staticmethod
    def _load(path: Path) -> List[Transition]:
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of Transition objects, got {type(data)}")
        return data

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.transitions[idx]
        pil = Image.fromarray(t.image.astype(np.uint8))
        image = self.transform(pil)
        return {
            "image": image,
            "steer": torch.tensor([t.steer], dtype=torch.float32),
            "acceleration": torch.tensor([t.acceleration], dtype=torch.float32),
            "brake": torch.tensor([float(t.brake)], dtype=torch.float32),
        }


def save_transitions(transitions: List[Transition], path: str | Path) -> None:
    """Serialise a list of transitions to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(transitions, f)


# ---------------------------------------------------------------------------
# Rollout buffer (PPO)
# ---------------------------------------------------------------------------


@dataclass
class RolloutBuffer:
    """Stores a single on-policy rollout for PPO updates.

    Attributes:
        max_size: Maximum number of steps to store before ``is_full`` is True.
    """

    max_size: int = 2048

    images: List[np.ndarray] = field(default_factory=list)
    steers: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    brakes: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def push(
        self,
        image: np.ndarray,
        steer: float,
        acceleration: float,
        brake: float,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Append one step to the buffer."""
        self.images.append(image)
        self.steers.append(steer)
        self.accelerations.append(acceleration)
        self.brakes.append(brake)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    @property
    def is_full(self) -> bool:
        return len(self.rewards) >= self.max_size

    def clear(self) -> None:
        """Reset the buffer after a PPO update."""
        self.images.clear()
        self.steers.clear()
        self.accelerations.clear()
        self.brakes.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns.

        Args:
            last_value: Bootstrap value for the last state.
            gamma: Discount factor.
            gae_lambda: GAE smoothing coefficient.

        Returns:
            ``(returns, advantages)`` both of shape ``(T,)``.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        values = np.array(self.values + [last_value], dtype=np.float32)

        for t in reversed(range(T)):
            mask = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return torch.from_numpy(returns), torch.from_numpy(advantages)


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def pack_action(
    steer: float, acceleration: float, brake: bool
) -> Dict[str, float | bool]:
    """Package scalar action values into a dict compatible with ``pystk.Action``."""
    return {
        "steer": float(np.clip(steer, -1.0, 1.0)),
        "acceleration": float(np.clip(acceleration, 0.0, 1.0)),
        "brake": bool(brake),
        "fire": False,
        "drift": False,
        "nitro": False,
        "rescue": False,
    }


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


def compute_reward(
    prev_distance: float,
    curr_distance: float,
    speed: float,
    off_track: bool,
    wrong_way: bool,
) -> float:
    """Heuristic reward function for on-track progress.

    Args:
        prev_distance: Track fraction at the previous step.
        curr_distance: Track fraction at the current step.
        speed: Current kart speed (m/s).
        off_track: Whether the kart is off the racing surface.
        wrong_way: Whether the kart is heading the wrong direction.

    Returns:
        Scalar reward.
    """
    progress_reward = (curr_distance - prev_distance) * 100.0
    speed_bonus = speed * 0.01
    penalty = -1.0 if off_track else 0.0
    penalty += -2.0 if wrong_way else 0.0
    return progress_reward + speed_bonus + penalty
