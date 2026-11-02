#!/usr/bin/env python3
"""A-KIT 全链路快速合成测试。

不读取大型真实轨迹，而是构造小批量张量，检查时间窗边界、注册表模型构建、统一模型
输入输出形状、逐步协方差传播、名义位置反馈符号、反向梯度以及 checkpoint 往返。
该测试用于发现代码连接错误，不能替代真实数据验证和 PPredCpp 数值回放。
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import torch
from akit_utils_agent import phi_window_key, sample_window_key
from akit_model_api import build_model, validate_model_io


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
    if [sample_window_key(value) for value in (198207.0, 198207.5, 198207.999, 198208.0)] != [
        198208,
        198208,
        198208,
        198209,
    ]:
        raise AssertionError("Sample window keys do not follow [k-1, k)")
    if [phi_window_key(value) for value in (198207.0, 198207.5, 198207.999, 198208.0)] != [
        198207,
        198208,
        198208,
        198208,
    ]:
        raise AssertionError("Phi window keys do not follow (k-1, k]")

    module = load_training_module()
    device = torch.device("cpu")
    batch = 2
    state_dim = 15

    model = build_model(
        {
            "name": "set_transformer",
            "interface_version": "akit-model-1",
            "parameters": {
                "patch_size": 10,
                "filter_size": 10,
                "in_chans": 9,
                "embedding_size": 64,
                "num_heads": 4,
                "feedforward_expansion": 128,
                "k": 1,
                "blocklayers": 2,
                "output_size": state_dim,
            },
        }
    ).to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    imu = torch.randn(batch, 9, 100, device=device, dtype=torch.float64)
    steps = 4
    phi = torch.eye(state_dim, dtype=torch.float64).repeat(batch, steps, 1, 1)
    phi[:, :, 0, 0] = 1.01
    phi_dt = torch.full((batch, steps), 0.01, dtype=torch.float64)
    phi_mask = torch.ones(batch, steps, dtype=torch.bool)
    phi_mask[0, -1] = False  # exercise variable-length padded propagation
    h = torch.zeros(batch, 3, state_dim, dtype=torch.float64)
    h[:, :, :3] = torch.eye(3, dtype=torch.float64)
    r = torch.eye(3, dtype=torch.float64).repeat(batch, 1, 1) * 0.1
    p = torch.eye(state_dim, dtype=torch.float64).repeat(batch, 1, 1)
    innovation = torch.randn(batch, 3, 1, dtype=torch.float64) * 0.01
    state = torch.randn(batch, state_dim, 1, dtype=torch.float64)
    q_origin = torch.eye(state_dim, dtype=torch.float64).repeat(batch, 1, 1) * 1e-6

    raw_output = model(imu, p.unsqueeze(-1))
    validate_model_io(imu, p, raw_output, input_channels=9, state_dim=state_dim)
    q_scale = torch.nn.functional.softplus(raw_output)
    prediction = module.kalman_filter(phi, phi_dt, h, r, p, innovation, state, q_scale, q_origin, phi_mask)
    zero_innovation_prediction = module.kalman_filter(
        phi, phi_dt, h, r, p, torch.zeros_like(innovation), state, q_scale, q_origin, phi_mask
    )
    target = prediction.detach() + 0.01
    loss = torch.mean((prediction - target) ** 2)

    # Independent reference for stepwise P propagation and subtractive nominal feedback.
    d_scale = torch.diag_embed(torch.sqrt(q_scale))
    qc_pred = d_scale @ q_origin @ d_scale
    p_reference = p.clone()
    for step in range(steps):
        phi_i = phi[:, step]
        q_i = 0.5 * (phi_i @ qc_pred @ phi_i.transpose(-2, -1) + qc_pred) * phi_dt[:, step, None, None]
        p_next = phi_i @ p_reference @ phi_i.transpose(-2, -1) + q_i
        p_reference = torch.where(phi_mask[:, step, None, None], p_next, p_reference)
    innovation_cov = h @ p_reference @ h.transpose(-2, -1) + r
    gain = p_reference @ h.transpose(-2, -1) @ torch.linalg.inv(innovation_cov)
    expected_prediction = state[:, :3, 0] - (gain @ innovation)[:, :3, 0]

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.sqrt(
        sum(torch.sum(parameter.grad.square()) for parameter in model.parameters() if parameter.grad is not None)
    )
    if prediction.shape != (batch, 3):
        raise AssertionError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
    if not torch.allclose(prediction, expected_prediction, rtol=1e-10, atol=1e-12):
        raise AssertionError("Stepwise covariance replay or subtractive feedback differs from the reference")
    if not torch.allclose(zero_innovation_prediction, state[:, :3, 0]):
        raise AssertionError("state_pred must not be propagated by the covariance Phi sequence")
    if torch.allclose(prediction, state[:, :3, 0] + innovation[:, :, 0]):
        raise AssertionError("Nominal position feedback must subtract the estimated INS-GNSS error")
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
