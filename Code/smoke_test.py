#!/usr/bin/env python3
"""Fast synthetic forward/backward/checkpoint test for the current A-KIT stack."""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import torch


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))


def load_training_module():
    path = CODE_DIR / "A-KIT_lj_agent.py"
    spec = importlib.util.spec_from_file_location("akit_lj_agent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_training_module()
    device = torch.device("cpu")
    batch = 2
    state_dim = 15

    model = module.Network(
        patch_size=10,
        filter_size=10,
        in_chans=9,
        embedding_size=64,
        num_heads=4,
        feedforward_expansion=128,
        k=1,
        blocklayers=2,
        output_size=state_dim,
    ).to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    imu = torch.randn(batch, 9, 100, device=device, dtype=torch.float64)
    phi = torch.eye(state_dim, dtype=torch.float64).repeat(batch, 1, 1)
    # A non-identity accumulated transition catches accidental second state propagation.
    phi[:, 0, 0] = 2.0
    h = torch.zeros(batch, 3, state_dim, dtype=torch.float64)
    h[:, :, :3] = torch.eye(3, dtype=torch.float64)
    r = torch.eye(3, dtype=torch.float64).repeat(batch, 1, 1) * 0.1
    p = torch.eye(state_dim, dtype=torch.float64).repeat(batch, 1, 1)
    innovation = torch.randn(batch, 3, 1, dtype=torch.float64) * 0.01
    state = torch.randn(batch, state_dim, 1, dtype=torch.float64)
    q_origin = torch.eye(state_dim, dtype=torch.float64).repeat(batch, 1, 1) * 1e-6

    q_scale = torch.nn.functional.softplus(model(imu, p.unsqueeze(-1)))
    prediction = module.kalman_filter(phi, h, r, p, innovation, state, q_scale, q_origin)
    zero_innovation_prediction = module.kalman_filter(
        phi, h, r, p, torch.zeros_like(innovation), state, q_scale, q_origin
    )
    target = prediction.detach() + 0.01
    loss = torch.mean((prediction - target) ** 2)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.sqrt(
        sum(torch.sum(parameter.grad.square()) for parameter in model.parameters() if parameter.grad is not None)
    )
    if prediction.shape != (batch, 3):
        raise AssertionError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
    if not torch.allclose(zero_innovation_prediction, state[:, :3, 0]):
        raise AssertionError("state_pred was propagated by Phi_acc more than zero times")
    if not math.isfinite(float(loss)) or not math.isfinite(float(grad_norm)):
        raise AssertionError("Loss or gradient is not finite")
    if float(grad_norm) <= 0.0:
        raise AssertionError("Model received no gradient")
    optimizer.step()

    config = {"mode": "synthetic_smoke", "seed": 42}
    payload = module.checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        config=config,
        epoch=1,
        metrics={"loss": float(loss)},
    )
    with tempfile.TemporaryDirectory(prefix="akit_smoke_") as temp_dir:
        checkpoint = Path(temp_dir) / "checkpoint.pth"
        torch.save(payload, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if loaded["config"] != config or "optimizer_state_dict" not in loaded:
            raise AssertionError("Checkpoint round-trip lost required metadata")

    print(
        f"SMOKE TEST PASSED | prediction={tuple(prediction.shape)} "
        f"loss={float(loss):.6e} grad_norm={float(grad_norm):.6e}"
    )


if __name__ == "__main__":
    main()
