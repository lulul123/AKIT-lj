#!/usr/bin/env python3
"""验证 Python 协方差传播与 C++ PPredCpp 的数值一致性。

从同一时刻的 PStart 出发，按顺序读取该 GNSS 区间内每个 Phi 和 dt，使用 Qc 重放
C++ KFPrediction 公式，再与 PPredCpp 比较。本工具验证公共 Kalman 传播而不训练模型；
GPS 周回绕形成的非一秒区间需要单独解释或过滤。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


N = 15


def load_matrices(path: Path) -> dict[int, np.ndarray]:
    records: dict[int, np.ndarray] = {}
    with path.open() as stream:
        next(stream)
        rows: list[list[float]] = []
        timestamp = 0
        for line in stream:
            values = line.split()
            if not values:
                continue
            if not rows:
                timestamp = round(float(values[0]))
                values = values[1:]
            rows.append([float(value) for value in values])
            if len(rows) == N:
                records[timestamp] = np.asarray(rows, dtype=np.float64)
                rows = []
    if rows:
        raise ValueError(f"Incomplete matrix at the end of {path}")
    return records


def load_qc(path: Path) -> np.ndarray:
    with path.open() as stream:
        next(stream)
        values = [float(value) for line in stream for value in line.split()]
    if len(values) != N * N:
        raise ValueError(f"{path} contains {len(values)} values, expected {N * N}")
    return np.asarray(values, dtype=np.float64).reshape(N, N)


def replay_selected_phi(
    path: Path,
    selected: set[int],
    p_start: dict[int, np.ndarray],
    qc: np.ndarray,
) -> dict[int, np.ndarray]:
    predictions = {timestamp: p_start[timestamp].copy() for timestamp in selected}
    counts = {timestamp: 0 for timestamp in selected}
    with path.open() as stream:
        next(stream)
        rows: list[list[float]] = []
        timestamp = dt = 0.0
        for line in stream:
            values = line.split()
            if not values:
                continue
            if not rows:
                timestamp, dt = float(values[0]), float(values[1])
                values = values[2:]
            rows.append([float(value) for value in values])
            if len(rows) != N:
                continue
            key = int(np.ceil(timestamp - 1e-7))
            if key in selected:
                phi = np.asarray(rows, dtype=np.float64)
                q_step = 0.5 * (phi @ qc @ phi.T + qc) * dt
                predictions[key] = phi @ predictions[key] @ phi.T + q_step
                counts[key] += 1
            rows = []
    missing = sorted(timestamp for timestamp, count in counts.items() if count == 0)
    if missing:
        raise ValueError(f"No Phi records found for timestamps: {missing}")
    return predictions


def select_evenly(values: list[int], limit: int) -> list[int]:
    if limit <= 0 or limit >= len(values):
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def validate_dataset(data_dir: Path, limit: int) -> dict[str, float | int | str]:
    p_start = load_matrices(data_dir / "PStart.txt")
    cpp = load_matrices(data_dir / "PPredCpp.txt")
    timestamps = select_evenly(sorted(p_start.keys() & cpp.keys()), limit)
    if not timestamps:
        raise ValueError(f"No common PStart/PPredCpp timestamps in {data_dir}")
    predicted = replay_selected_phi(data_dir / "Phi.txt", set(timestamps), p_start, load_qc(data_dir / "Qc.txt"))

    absolute = []
    relative = []
    worst_time = timestamps[0]
    worst_abs = -1.0
    for timestamp in timestamps:
        diff = np.abs(predicted[timestamp] - cpp[timestamp])
        sample_abs = float(diff.max())
        scale = max(float(np.abs(cpp[timestamp]).max()), 1e-30)
        absolute.append(sample_abs)
        relative.append(sample_abs / scale)
        if sample_abs > worst_abs:
            worst_abs = sample_abs
            worst_time = timestamp
    return {
        "dataset": data_dir.name,
        "samples": len(timestamps),
        "max_abs_error": max(absolute),
        "max_relative_error": max(relative),
        "mean_abs_error": float(np.mean(absolute)),
        "worst_timestamp": worst_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dirs", nargs="*", type=Path, help="Trajectory directories (default: all entries in data/datasets.json)")
    parser.add_argument("--samples", type=int, default=20, help="Evenly spaced timestamps per trajectory; 0 means all")
    parser.add_argument("--rtol", type=float, default=2e-6)
    parser.add_argument("--atol", type=float, default=2e-7)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.data_dirs:
        data_dirs = args.data_dirs
    else:
        catalog_path = repo / "data" / "datasets.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        data_dirs = [catalog_path.parent / item["path"] for item in catalog["datasets"]]

    failed = False
    for data_dir in data_dirs:
        result = validate_dataset(data_dir.resolve(), args.samples)
        passed = result["max_abs_error"] <= args.atol or result["max_relative_error"] <= args.rtol
        failed |= not passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} {result['dataset']}: samples={result['samples']} "
            f"max_abs={result['max_abs_error']:.3e} max_rel={result['max_relative_error']:.3e} "
            f"mean_abs={result['mean_abs_error']:.3e} worst_t={result['worst_timestamp']}"
        )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
