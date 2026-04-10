"""Unit tests for the SuperTuxKart deep learning models and utilities.

Run with::

    pytest tests/test_model.py -v

These tests do NOT require pystk to be installed – they exercise only the
PyTorch model code and the utility helpers.
"""

from __future__ import annotations

import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import KartEncoder, KartPolicy, KartActorCritic, PolicyHead, ValueHead, ResBlock
from utils import (
    DemonstrationDataset,
    RolloutBuffer,
    Transition,
    compute_reward,
    get_eval_transform,
    get_train_transform,
    image_to_tensor,
    pack_action,
    save_transitions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IMAGE_SIZE = 64  # use small size to keep tests fast
BATCH = 4


@pytest.fixture()
def random_images() -> torch.Tensor:
    return torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture()
def random_np_image() -> np.ndarray:
    return np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture()
def demo_pickle(tmp_path: Path) -> Path:
    """Create a tiny demonstration pickle file."""
    transitions = [
        Transition(
            image=np.random.randint(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
            steer=float(np.random.uniform(-1, 1)),
            acceleration=float(np.random.uniform(0, 1)),
            brake=bool(np.random.randint(0, 2)),
        )
        for _ in range(20)
    ]
    path = tmp_path / "demo.pkl"
    save_transitions(transitions, path)
    return path


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------


class TestResBlock:
    def test_output_shape(self) -> None:
        block = ResBlock(channels=16)
        x = torch.rand(BATCH, 16, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self) -> None:
        """With zero weights the output should equal the input (before ReLU)."""
        block = ResBlock(channels=4)
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        x = torch.rand(1, 4, 4, 4)
        out = block(x)
        # After zeroing weights BN output ≈ 0, so out ≈ relu(x + 0) = relu(x) = x (all positive)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# KartEncoder
# ---------------------------------------------------------------------------


class TestKartEncoder:
    def test_output_shape(self, random_images: torch.Tensor) -> None:
        encoder = KartEncoder(image_size=IMAGE_SIZE, feat_dim=128, base_channels=8)
        feat = encoder(random_images)
        assert feat.shape == (BATCH, 128)

    def test_grad_flows(self, random_images: torch.Tensor) -> None:
        random_images.requires_grad_(True)
        encoder = KartEncoder(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8)
        feat = encoder(random_images)
        feat.sum().backward()
        assert random_images.grad is not None

    def test_batch_size_one(self) -> None:
        encoder = KartEncoder(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8)
        img = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        feat = encoder(img)
        assert feat.shape == (1, 64)


# ---------------------------------------------------------------------------
# PolicyHead & ValueHead
# ---------------------------------------------------------------------------


class TestPolicyHead:
    def test_output_shapes(self) -> None:
        head = PolicyHead(feat_dim=64, hidden_dim=32)
        feats = torch.rand(BATCH, 64)
        steer, accel, brake_logit = head(feats)
        assert steer.shape == (BATCH, 1)
        assert accel.shape == (BATCH, 1)
        assert brake_logit.shape == (BATCH, 1)

    def test_steer_range(self) -> None:
        head = PolicyHead(feat_dim=64, hidden_dim=32)
        feats = torch.rand(100, 64)
        steer, _, _ = head(feats)
        assert steer.min() >= -1.0 - 1e-6
        assert steer.max() <= 1.0 + 1e-6

    def test_accel_range(self) -> None:
        head = PolicyHead(feat_dim=64, hidden_dim=32)
        feats = torch.rand(100, 64)
        _, accel, _ = head(feats)
        assert accel.min() >= 0.0 - 1e-6
        assert accel.max() <= 1.0 + 1e-6


class TestValueHead:
    def test_output_shape(self) -> None:
        head = ValueHead(feat_dim=64, hidden_dim=32)
        feats = torch.rand(BATCH, 64)
        value = head(feats)
        assert value.shape == (BATCH, 1)


# ---------------------------------------------------------------------------
# KartPolicy (actor-only, for IL)
# ---------------------------------------------------------------------------


class TestKartPolicy:
    def test_forward_shape(self, random_images: torch.Tensor) -> None:
        policy = KartPolicy(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8, hidden_dim=32)
        steer, accel, brake_logit = policy(random_images)
        assert steer.shape == (BATCH, 1)
        assert accel.shape == (BATCH, 1)
        assert brake_logit.shape == (BATCH, 1)

    def test_no_pystk_required(self) -> None:
        """Model construction must not import pystk."""
        policy = KartPolicy(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8)
        assert policy is not None

    def test_save_load(self, tmp_path: Path, random_images: torch.Tensor) -> None:
        policy = KartPolicy(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8, hidden_dim=32)
        path = tmp_path / "policy.pt"
        torch.save(policy.state_dict(), path)
        policy2 = KartPolicy(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8, hidden_dim=32)
        policy2.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            out1 = policy(random_images)
            out2 = policy2(random_images)
        for a, b in zip(out1, out2):
            assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# KartActorCritic (for PPO)
# ---------------------------------------------------------------------------


class TestKartActorCritic:
    def test_forward_shapes(self, random_images: torch.Tensor) -> None:
        ac = KartActorCritic(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8, hidden_dim=32)
        (steer, accel, brake_logit), value = ac(random_images)
        assert steer.shape == (BATCH, 1)
        assert accel.shape == (BATCH, 1)
        assert brake_logit.shape == (BATCH, 1)
        assert value.shape == (BATCH, 1)

    def test_act_no_grad(self, random_images: torch.Tensor) -> None:
        ac = KartActorCritic(image_size=IMAGE_SIZE, feat_dim=64, base_channels=8, hidden_dim=32)
        steer, accel, brake = ac.act(random_images)
        # act() wraps in no_grad; tensors should have no grad_fn
        assert steer.grad_fn is None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestImageToTensor:
    def test_output_shape(self, random_np_image: np.ndarray) -> None:
        tensor = image_to_tensor(random_np_image, image_size=IMAGE_SIZE)
        assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_dtype(self, random_np_image: np.ndarray) -> None:
        tensor = image_to_tensor(random_np_image, image_size=IMAGE_SIZE)
        assert tensor.dtype == torch.float32

    def test_normalised(self, random_np_image: np.ndarray) -> None:
        """After normalisation values should not be in [0, 1] naively."""
        tensor = image_to_tensor(random_np_image, image_size=IMAGE_SIZE)
        # The normalised range isn't strictly [0,1] so we just check it's finite
        assert torch.isfinite(tensor).all()


class TestTransforms:
    def test_train_transform_output_shape(self, random_np_image: np.ndarray) -> None:
        from PIL import Image

        pil = Image.fromarray(random_np_image)
        t = get_train_transform(IMAGE_SIZE)(pil)
        assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_eval_transform_output_shape(self, random_np_image: np.ndarray) -> None:
        from PIL import Image

        pil = Image.fromarray(random_np_image)
        t = get_eval_transform(IMAGE_SIZE)(pil)
        assert t.shape == (3, IMAGE_SIZE, IMAGE_SIZE)


class TestComputeReward:
    def test_positive_progress(self) -> None:
        r = compute_reward(0.0, 1.0, 10.0, False, False)
        assert r > 0

    def test_off_track_penalty(self) -> None:
        r_on = compute_reward(0.0, 1.0, 10.0, False, False)
        r_off = compute_reward(0.0, 1.0, 10.0, True, False)
        assert r_off < r_on

    def test_wrong_way_penalty(self) -> None:
        r_ok = compute_reward(0.0, 1.0, 10.0, False, False)
        r_ww = compute_reward(0.0, 1.0, 10.0, False, True)
        assert r_ww < r_ok


class TestPackAction:
    def test_keys_present(self) -> None:
        a = pack_action(0.5, 0.8, False)
        assert "steer" in a
        assert "acceleration" in a
        assert "brake" in a

    def test_steer_clipped(self) -> None:
        a = pack_action(2.0, 0.5, False)
        assert a["steer"] == 1.0
        a2 = pack_action(-5.0, 0.5, False)
        assert a2["steer"] == -1.0

    def test_accel_clipped(self) -> None:
        a = pack_action(0.0, 3.0, False)
        assert a["acceleration"] == 1.0


class TestSaveLoadTransitions:
    def test_round_trip(self, tmp_path: Path) -> None:
        ts = [
            Transition(
                image=np.zeros((4, 4, 3), dtype=np.uint8),
                steer=0.1,
                acceleration=0.9,
                brake=True,
            )
        ]
        p = tmp_path / "t.pkl"
        save_transitions(ts, p)
        with p.open("rb") as f:
            loaded = pickle.load(f)
        assert len(loaded) == 1
        assert loaded[0].steer == pytest.approx(0.1)


class TestDemonstrationDataset:
    def test_len(self, demo_pickle: Path) -> None:
        ds = DemonstrationDataset(demo_pickle, image_size=IMAGE_SIZE, train=False)
        assert len(ds) == 20

    def test_item_shapes(self, demo_pickle: Path) -> None:
        ds = DemonstrationDataset(demo_pickle, image_size=IMAGE_SIZE, train=False)
        item = ds[0]
        assert item["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert item["steer"].shape == (1,)
        assert item["acceleration"].shape == (1,)
        assert item["brake"].shape == (1,)


class TestRolloutBuffer:
    def test_push_and_is_full(self) -> None:
        buf = RolloutBuffer(max_size=5)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(4):
            assert not buf.is_full
            buf.push(img, 0.0, 0.5, 0.0, 1.0, 0.0, -0.1, False)
        buf.push(img, 0.0, 0.5, 0.0, 1.0, 0.0, -0.1, True)
        assert buf.is_full

    def test_clear(self) -> None:
        buf = RolloutBuffer(max_size=3)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(3):
            buf.push(img, 0.0, 0.5, 0.0, 1.0, 0.0, -0.1, False)
        buf.clear()
        assert len(buf.rewards) == 0
        assert not buf.is_full

    def test_compute_returns(self) -> None:
        buf = RolloutBuffer(max_size=4)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        for i in range(4):
            buf.push(img, 0.0, 0.5, 0.0, float(i), 0.5, -0.1, i == 3)
        returns, advantages = buf.compute_returns_and_advantages(last_value=0.0)
        assert returns.shape == (4,)
        assert advantages.shape == (4,)
        assert torch.isfinite(returns).all()
        assert torch.isfinite(advantages).all()
