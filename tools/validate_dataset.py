#!/usr/bin/env python3
"""根据 data/datasets.json 检查 A-KIT 文本数据的结构完整性。

检查必需文件、列数、矩阵形状、有限数、基础时间顺序、Phi dt 和参考文件 GPS 周。
默认只抽查每个大文件的前 1000 条，``--max-records 0`` 才是全量扫描。本工具不判断
全部物理语义，也不进行模型训练或 C++/Python 数值一致性比较。
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


MATRIX_FILES = {
    "phi": (15, 15, True),
    "p_start": (15, 15, False),
    "measurement_matrix": (3, 15, False),
    "measurement_covariance": (3, 3, False),
    "innovation": (3, 1, False),
}


def sample_window_key(timestamp):
    """Return k for the left-closed, right-open sample window [k-1, k)."""
    return math.floor(timestamp) + 1


def fail(message):
    raise ValueError(message)


def numbers(parts, context):
    try:
        values = [float(value) for value in parts]
    except ValueError as exc:
        fail(f"{context}: non-numeric value ({exc})")
    if not all(math.isfinite(value) for value in values):
        fail(f"{context}: NaN or Inf")
    return values


def check_simple_file(path, value_columns, max_records):
    count = 0
    first_time = last_time = previous = None
    duplicate_times = 0
    seconds = Counter()
    with path.open("r", errors="replace") as stream:
        next(stream, None)
        for line_number, line in enumerate(stream, 2):
            if not line.strip():
                continue
            values = numbers(line.split(), f"{path.name}:{line_number}")
            if len(values) != value_columns + 1:
                fail(f"{path.name}:{line_number}: expected {value_columns + 1} columns, got {len(values)}")
            timestamp = values[0]
            if previous is not None:
                if timestamp < previous:
                    fail(f"{path.name}:{line_number}: timestamp decreased")
                duplicate_times += timestamp == previous
            first_time = timestamp if first_time is None else first_time
            last_time = previous = timestamp
            seconds[sample_window_key(timestamp)] += 1
            count += 1
            if max_records and count >= max_records:
                break
    if not count:
        fail(f"{path.name}: no data records")
    return {
        "records_checked": count,
        "first_time": first_time,
        "last_time": last_time,
        "duplicate_timestamps": duplicate_times,
        "records_per_second_min": min(seconds.values()),
        "records_per_second_max": max(seconds.values()),
        "scan": "partial" if max_records else "full",
    }


def check_matrix_file(path, rows, cols, has_dt, max_records):
    count = 0
    first_time = last_time = previous = None
    dt_min = dt_max = None
    with path.open("r", errors="replace") as stream:
        next(stream, None)
        line_number = 1
        while True:
            line = stream.readline()
            line_number += 1
            if not line:
                break
            if not line.strip():
                continue
            first = numbers(line.split(), f"{path.name}:{line_number}")
            prefix = 2 if has_dt else 1
            if len(first) != prefix + cols:
                fail(f"{path.name}:{line_number}: expected {prefix + cols} values on first matrix row, got {len(first)}")
            timestamp = first[0]
            if previous is not None and timestamp < previous:
                fail(f"{path.name}:{line_number}: timestamp decreased")
            first_time = timestamp if first_time is None else first_time
            last_time = previous = timestamp
            if has_dt:
                dt = first[1]
                if dt <= 0:
                    fail(f"{path.name}:{line_number}: dt must be positive")
                dt_min = dt if dt_min is None else min(dt_min, dt)
                dt_max = dt if dt_max is None else max(dt_max, dt)
            for _ in range(rows - 1):
                row = stream.readline()
                line_number += 1
                if not row:
                    fail(f"{path.name}:{line_number}: truncated matrix")
                values = numbers(row.split(), f"{path.name}:{line_number}")
                if len(values) != cols:
                    fail(f"{path.name}:{line_number}: expected {cols} matrix values, got {len(values)}")
            count += 1
            if max_records and count >= max_records:
                break
    if not count:
        fail(f"{path.name}: no matrix records")
    result = {
        "records_checked": count,
        "first_time": first_time,
        "last_time": last_time,
        "shape": [rows, cols],
        "scan": "partial" if max_records else "full",
    }
    if has_dt:
        result.update({"dt_min": dt_min, "dt_max": dt_max})
    return result


def check_qc(path):
    with path.open("r", errors="replace") as stream:
        next(stream, None)
        values = numbers([token for line in stream for token in line.split()], path.name)
    if len(values) != 225:
        fail(f"{path.name}: expected 225 values, got {len(values)}")
    return {"shape": [15, 15], "finite": True, "scan": "full"}


def check_reference(path, expected_week, max_records):
    count = 0
    first_time = last_time = None
    header_found = False
    with path.open("r", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            if "Week" in line and "GPSTime" in line and "X-ECEF" in line:
                header_found = True
                continue
            if not header_found or not line.strip() or not line.lstrip()[0].isdigit():
                continue
            parts = line.split()
            if len(parts) < 12:
                fail(f"{path.name}:{line_number}: expected at least 12 columns")
            values = numbers(parts[:2] + parts[9:12], f"{path.name}:{line_number}")
            if int(values[0]) != expected_week:
                fail(f"{path.name}:{line_number}: GPS week {int(values[0])}, expected {expected_week}")
            first_time = values[1] if first_time is None else first_time
            last_time = values[1]
            count += 1
            if max_records and count >= max_records:
                break
    if not header_found or not count:
        fail(f"{path.name}: reference header or data missing")
    return {"records_checked": count, "first_time": first_time, "last_time": last_time, "scan": "partial" if max_records else "full"}


def validate_catalog(catalog):
    required = {"schema_version", "status", "project", "defaults", "split_policy", "datasets"}
    missing = sorted(required - catalog.keys())
    if missing:
        fail(f"catalog missing keys: {missing}")
    ids = [item.get("dataset_id") for item in catalog["datasets"]]
    if None in ids or len(ids) != len(set(ids)):
        fail("dataset_id values must exist and be unique")
    for item in catalog["datasets"]:
        for key in ("trajectory_id", "path", "gps_week", "position_reference", "split", "validation_status"):
            if key not in item:
                fail(f"{item['dataset_id']}: missing catalog key {key}")


def validate_one(catalog_path, catalog, entry, max_records):
    data_dir = catalog_path.parent / entry["path"]
    files = dict(catalog["defaults"]["files"])
    files["position_reference"] = entry["position_reference"]
    errors = []
    checks = {}

    for key in catalog["defaults"]["required_file_keys"]:
        path = data_dir / files[key]
        if not path.is_file():
            errors.append(f"missing required file: {path.name}")
    if errors:
        return {"dataset_id": entry["dataset_id"], "status": "fail", "errors": errors, "checks": checks}

    jobs = {
        "imu": lambda: check_simple_file(data_dir / files["imu"], 6, max_records),
        "velocity": lambda: check_simple_file(data_dir / files["velocity"], 3, max_records),
        "nominal_state_prior": lambda: check_simple_file(data_dir / files["nominal_state_prior"], 15, max_records),
        "qc": lambda: check_qc(data_dir / files["qc"]),
        "reference": lambda: check_reference(data_dir / files["position_reference"], entry["gps_week"], max_records),
    }
    for key, (rows, cols, has_dt) in MATRIX_FILES.items():
        jobs[key] = lambda key=key, rows=rows, cols=cols, has_dt=has_dt: check_matrix_file(
            data_dir / files[key], rows, cols, has_dt, max_records
        )

    for name, job in jobs.items():
        try:
            checks[name] = job()
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return {
        "dataset_id": entry["dataset_id"],
        "status": "pass" if not errors else "fail",
        "scan": "partial" if max_records else "full",
        "errors": errors,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/datasets.json")
    parser.add_argument("--dataset", action="append", help="dataset_id; repeat for multiple datasets")
    parser.add_argument("--all", action="store_true", help="validate every catalog entry")
    parser.add_argument("--max-records", type=int, default=1000, help="records per large file; 0 scans complete files")
    parser.add_argument("--report", help="optional JSON report path")
    args = parser.parse_args()

    catalog_path = Path(args.catalog).resolve()
    try:
        catalog = json.loads(catalog_path.read_text())
        validate_catalog(catalog)
        entries = catalog["datasets"]
        requested = set(args.dataset or [])
        if not args.all and not requested:
            parser.error("use --dataset ID or --all")
        selected = entries if args.all else [item for item in entries if item["dataset_id"] in requested]
        found = {item["dataset_id"] for item in selected}
        if requested - found:
            fail(f"unknown dataset_id: {sorted(requested - found)}")
        report = {
            "catalog": str(catalog_path),
            "max_records": args.max_records,
            "results": [validate_one(catalog_path, catalog, item, args.max_records) for item in selected],
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for result in report["results"]:
        print(f"{result['dataset_id']}: {result['status'].upper()} ({result['scan']} scan)")
        for error in result["errors"]:
            print(f"  - {error}")
    return 1 if any(result["status"] == "fail" for result in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
