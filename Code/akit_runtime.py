"""Runtime helpers for reproducible A-KIT experiments."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch without requiring CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda and fail clearly when CUDA was explicitly requested."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def resolve_data_dirs(data_root: str, names: list[str]) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    directories = [(root / name).resolve() for name in names]
    missing = [str(path) for path in directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Data directories do not exist: " + ", ".join(missing))
    return [str(path) for path in directories]


def create_output_dir(output_root: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root).expanduser().resolve() / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "git_commit": _git_commit(root),
    }


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if normalization is not None:
        payload["normalization"] = normalization
    return payload
