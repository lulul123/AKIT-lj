#!/usr/bin/env python3
"""训练前检查 A-KIT 标准 JSON 配置的核心约束。

无需安装 jsonschema，检查版本、必需段、模型接口与输出语义、模型维度的一致性、
优化器基本参数及训练/验证 split 隔离。标准 Schema 仍以
schemas/training.schema.json 为完整机器可读定义。本工具只校验配置，不启动训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fail(message: str) -> None:
    raise ValueError(message)


def validate(config: dict) -> None:
    required = {"schema_version", "experiment", "data", "model", "training", "checkpoint"}
    missing = sorted(required - set(config))
    if missing:
        fail(f"missing top-level keys: {missing}")
    if config["schema_version"] != "akit-training-config-1":
        fail("unsupported schema_version")
    model = config["model"]
    for key in ("name", "interface_version", "input_channels", "state_dim", "output_semantics", "parameters"):
        if key not in model:
            fail(f"model.{key} is required")
    if model["interface_version"] != "akit-model-1":
        fail("model.interface_version must be akit-model-1")
    if model["output_semantics"] != "raw_qc_scale_logits":
        fail("model.output_semantics must be raw_qc_scale_logits")
    parameters = model["parameters"]
    if parameters.get("in_chans", model["input_channels"]) != model["input_channels"]:
        fail("model.parameters.in_chans conflicts with model.input_channels")
    if parameters.get("output_size", model["state_dim"]) != model["state_dim"]:
        fail("model.parameters.output_size conflicts with model.state_dim")
    training = config["training"]
    if training.get("epochs", 0) < 1 or training.get("batch_size", 0) < 1:
        fail("training.epochs and training.batch_size must be positive")
    optimizer = training.get("optimizer", {})
    if optimizer.get("name") != "adam" or optimizer.get("learning_rate", 0) <= 0:
        fail("only Adam with a positive learning_rate is currently supported")
    if set(config["data"]["train_splits"]) & set(config["data"]["validation_splits"]):
        fail("training and validation splits must be disjoint")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL {args.config}: {exc}")
        return 1
    print(f"PASS {args.config}: model={config['model']['name']} interface={config['model']['interface_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
