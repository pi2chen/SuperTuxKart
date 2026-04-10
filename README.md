# SuperTuxKart Deep Learning Interest Project

A deep learning project that trains autonomous agents to race in [SuperTuxKart](https://supertuxkart.net/) using imitation learning and reinforcement learning.

## Overview

This project uses [`pystk`](https://github.com/philkr/pystk) — the official Python bindings for SuperTuxKart — to collect gameplay data and train neural network agents that can steer a kart around a track.

Two learning paradigms are supported:

| Mode | Description |
|------|-------------|
| **Imitation Learning** | Train a CNN on human-recorded `(image → action)` pairs |
| **Reinforcement Learning** | Fine-tune the agent end-to-end with PPO rewards |

## Architecture

```
Observation (3 × 128 × 128 RGB image)
        │
  ┌─────▼──────┐
  │  Encoder   │  ResNet-style CNN backbone
  │  (CNN)     │
  └─────┬──────┘
        │  512-d feature vector
  ┌─────▼──────┐
  │  Policy    │  Fully-connected head
  │  Head      │
  └─────┬──────┘
        │
  ┌─────▼──────────────────────┐
  │  Actions                   │
  │  steer ∈ [-1, 1]           │
  │  acceleration ∈ [0, 1]     │
  │  brake ∈ {0, 1}            │
  └────────────────────────────┘
```

An optional **value head** is attached to the same encoder when running PPO.

## Project Structure

```
SuperTuxKart/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py      # Neural network definitions
│   ├── agent.py      # Imitation-learning & RL agent wrappers
│   ├── train.py      # Training entry-point (IL and RL)
│   ├── evaluate.py   # Evaluation / live-play entry-point
│   └── utils.py      # Dataset, transforms, replay buffer, helpers
└── tests/
    └── test_model.py # Unit tests for model shapes & forward passes
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `pystk` requires a working OpenGL environment. On headless servers use
> `xvfb-run` or set `DISPLAY=:0` before running the scripts.

### 2. Collect a demonstration dataset

Drive the kart manually and record `(image, action)` pairs:

```bash
python src/train.py collect --track zengarden --laps 5 --out data/demo.pkl
```

### 3. Train with imitation learning

```bash
python src/train.py il \
    --data data/demo.pkl \
    --epochs 30 \
    --batch-size 64 \
    --lr 1e-3 \
    --save checkpoints/il_agent.pt
```

### 4. Fine-tune with reinforcement learning (PPO)

```bash
python src/train.py rl \
    --checkpoint checkpoints/il_agent.pt \
    --track zengarden \
    --timesteps 500000 \
    --save checkpoints/rl_agent.pt
```

### 5. Watch the agent race

```bash
python src/evaluate.py \
    --checkpoint checkpoints/rl_agent.pt \
    --track zengarden \
    --render
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- pystk ≥ 1.1
- See `requirements.txt` for the full list

## References

- B. Wymann et al., *TORCS: The Open Racing Car Simulator*, 2000.
- P. Krahenbuhl, *Notes on the SuperTuxKart Python Bindings*, 2020.
- J. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- S. Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, 2011.