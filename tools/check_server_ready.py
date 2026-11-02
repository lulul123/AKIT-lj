#!/usr/bin/env python3
"""Check whether a cloned A-KIT repository is ready for validation or training.

The check is read-only. It verifies required repository files, Python packages,
CUDA visibility, catalog paths, and the presence of trajectory directories. Use
``--skip-data`` immediately after cloning, before the large datasets are copied.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/akit-matplotlib")


REQUIRED_FILES = (
    "Code/A-KIT_lj_agent.py",
    "Code/akit_model_api.py",
    "Code/akit_runtime.py",
    "Code/akit_utils_agent.py",
    "configs/akit_set_transformer.json",
    "data/datasets.json",
    "schemas/datasets.schema.json",
    "schemas/training.schema.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-data", action="store_true", help="do not require trajectory directories")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    print(f"repository={repo}")
    print(f"python={platform.python_version()}")

    for relative in REQUIRED_FILES:
        if not (repo / relative).is_file():
            failures.append(f"missing repository file: {relative}")

    for package in ("numpy", "matplotlib", "torch"):
        try:
            module = importlib.import_module(package)
            print(f"{package}={getattr(module, '__version__', 'unknown')}")
        except ImportError:
            failures.append(f"missing Python package: {package}")

    try:
        import torch

        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device={torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    catalog_path = repo / "data" / "datasets.json"
    if catalog_path.is_file() and not args.skip_data:
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            for entry in catalog["datasets"]:
                data_dir = catalog_path.parent / entry["path"]
                if not data_dir.is_dir():
                    failures.append(f"missing trajectory directory: data/{entry['path']}")
        except (KeyError, json.JSONDecodeError) as exc:
            failures.append(f"invalid data catalog: {exc}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: server environment is ready" + (" (code only)" if args.skip_data else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
