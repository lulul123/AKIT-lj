"""A-KIT 可复用模型接口与构建注册表。

本模块定义稳定的张量语义和接口版本，负责按 JSON 的 ``model`` 段构造模型，并在
新模型接入时检查输入输出形状。它不包含具体网络、优化器、数据读取、softplus、
Kalman 传播或损失计算；这些职责刻意与网络结构解耦。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import nn


MODEL_INTERFACE_VERSION = "akit-model-1"
ModelFactory = Callable[..., nn.Module]
_MODEL_REGISTRY: dict[str, ModelFactory] = {}


@dataclass(frozen=True)
class ModelContract:
    """Tensor-level contract implemented by every registered A-KIT model."""

    interface_version: str = MODEL_INTERFACE_VERSION
    imu_layout: str = "batch,channel,time"
    covariance_layout: str = "batch,state,state"
    output_semantics: str = "raw_qc_scale_logits"


def register_model(name: str, factory: ModelFactory, *, replace: bool = False) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("Model name must be a non-empty string")
    if name in _MODEL_REGISTRY and not replace:
        raise ValueError(f"Model is already registered: {name}")
    _MODEL_REGISTRY[name] = factory


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_REGISTRY))


def validate_model_config(config: Mapping[str, Any]) -> None:
    required = {"name", "interface_version", "parameters"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Model config is missing keys: {missing}")
    if config["interface_version"] != MODEL_INTERFACE_VERSION:
        raise ValueError(
            f"Unsupported model interface {config['interface_version']!r}; "
            f"expected {MODEL_INTERFACE_VERSION!r}"
        )
    if not isinstance(config["parameters"], Mapping):
        raise ValueError("model.parameters must be an object")


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a registered model using only its standardized config section."""
    validate_model_config(config)
    name = str(config["name"])
    try:
        factory = _MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; available: {available_models()}") from exc
    return factory(**dict(config["parameters"]))


def validate_model_io(
    imu: torch.Tensor,
    p_start: torch.Tensor,
    output: torch.Tensor,
    *,
    input_channels: int,
    state_dim: int,
) -> None:
    """Fail early when a new model violates the common training interface."""
    if imu.ndim != 3 or imu.shape[1] != input_channels:
        raise ValueError(f"imu must have shape (B,{input_channels},T), got {tuple(imu.shape)}")
    if p_start.shape != (imu.shape[0], state_dim, state_dim):
        raise ValueError(
            f"p_start must have shape (B,{state_dim},{state_dim}), got {tuple(p_start.shape)}"
        )
    if output.shape != (imu.shape[0], state_dim):
        raise ValueError(f"model output must have shape (B,{state_dim}), got {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise ValueError("model output contains NaN or Inf")
