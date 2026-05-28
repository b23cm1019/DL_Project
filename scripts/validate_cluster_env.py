from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = [
    ("torch", "import torch"),
    ("torchvision", "import torchvision"),
    ("numpy", "import numpy"),
    ("cv2", "import cv2"),
    ("yaml", "import yaml"),
    ("einops", "from einops import rearrange"),
    ("timm.layers", "from timm.models.layers import DropPath, to_2tuple, trunc_normal_"),
    ("timm.metrics", "from timm.utils.metrics import accuracy"),
    ("fvcore", "import fvcore.nn.weight_init"),
    ("lmdb", "import lmdb"),
    ("h5py", "import h5py"),
    ("mrcfile", "import mrcfile"),
    ("pandas", "import pandas"),
    ("requests", "import requests"),
    ("scipy", "from scipy import interpolate, linalg, special"),
    ("skimage", "import skimage.io"),
    ("sklearn", "import sklearn"),
    ("tensorboard", "from torch.utils.tensorboard import SummaryWriter"),
    ("matplotlib", "import matplotlib"),
    ("seaborn", "import seaborn"),
    ("PIL", "from PIL import Image"),
    ("imageio", "import imageio"),
    ("wandb", "import wandb"),
    ("basicsr", "import basicsr"),
]


def info(message: str) -> None:
    print(f"[INFO] {message}")


def fail(message: str, failures: list[str]) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    failures.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the cn07 virtual environment against the repo requirements."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--expected-cuda")
    parser.add_argument("--expected-torch")
    parser.add_argument("--expected-torchvision")
    parser.add_argument("--check-dataset", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args()


def run_import_checks(failures: list[str]) -> None:
    for label, statement in REQUIRED_IMPORTS:
        try:
            exec(statement, {})
            info(f"import {label}: OK")
        except Exception as exc:  # noqa: BLE001
            fail(f"import {label}: {exc}", failures)


def check_project_registration(project_root: Path, failures: list[str]) -> None:
    basicsr = importlib.import_module("basicsr")
    basicsr_path = Path(basicsr.__file__).resolve()
    if project_root.resolve() not in basicsr_path.parents:
        fail(
            f"basicsr resolves outside project root: {basicsr_path}",
            failures,
        )
    else:
        info(f"basicsr path: {basicsr_path}")

    normalized_sys_path = {Path(entry or os.getcwd()).resolve() for entry in sys.path}
    if project_root.resolve() not in normalized_sys_path:
        fail(
            f"project root is missing from sys.path: {project_root}",
            failures,
        )
    else:
        info("project root registration: OK")


def check_python_location(venv_root: Path, failures: list[str]) -> None:
    executable = Path(sys.executable).resolve()
    if venv_root.resolve() not in executable.parents:
        fail(f"python executable is outside venv: {executable}", failures)
    else:
        info(f"python executable: {executable}")


def check_gpu(
    expected_cuda: str | None,
    expected_torch: str | None,
    expected_torchvision: str | None,
    failures: list[str],
) -> None:
    import torch
    import torchvision

    info(f"torch version: {torch.__version__}")
    info(f"torchvision version: {torchvision.__version__}")
    info(f"torch CUDA: {torch.version.cuda}")

    if expected_torch and torch.__version__ != expected_torch:
        fail(
            f"torch version mismatch: expected {expected_torch}, got {torch.__version__}",
            failures,
        )

    if expected_torchvision and torchvision.__version__ != expected_torchvision:
        fail(
            "torchvision version mismatch: expected "
            f"{expected_torchvision}, got {torchvision.__version__}",
            failures,
        )

    if expected_cuda and torch.version.cuda != expected_cuda:
        fail(
            f"torch CUDA mismatch: expected {expected_cuda}, got {torch.version.cuda}",
            failures,
        )

    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is false", failures)
        return

    info(f"GPU via torch: {torch.cuda.get_device_name(0)}")

    try:
        from torchvision.extension import _check_cuda_version

        _check_cuda_version()
        info("torch/torchvision CUDA compatibility: OK")
    except Exception as exc:  # noqa: BLE001
        fail(f"torchvision CUDA compatibility check failed: {exc}", failures)

    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        info(f"nvidia-smi GPU/driver: {result.stdout.strip()}")
    except Exception as exc:  # noqa: BLE001
        fail(f"nvidia-smi probe failed: {exc}", failures)


def check_dataset_layout(data_root: Path, failures: list[str]) -> None:
    expected_dirs = [
        data_root / "train" / "CDD-11_train" / "clear",
        data_root / "test" / "clear",
    ]
    for path in expected_dirs:
        if not path.is_dir():
            fail(f"missing dataset directory: {path}", failures)
        else:
            info(f"dataset path: {path}")


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    info(f"working directory: {Path.cwd()}")
    info(f"PYTHONNOUSERSITE={os.environ.get('PYTHONNOUSERSITE', '<unset>')}")

    if os.environ.get("PYTHONNOUSERSITE") != "1":
        fail("PYTHONNOUSERSITE is not set to 1", failures)

    check_python_location(args.venv_root, failures)
    run_import_checks(failures)
    if not any(item.startswith("import basicsr:") for item in failures):
        check_project_registration(args.project_root, failures)

    if not args.skip_gpu:
        check_gpu(
            expected_cuda=args.expected_cuda,
            expected_torch=args.expected_torch,
            expected_torchvision=args.expected_torchvision,
            failures=failures,
        )

    if args.check_dataset:
        if not args.data_root:
            fail("--check-dataset requires --data-root", failures)
        else:
            check_dataset_layout(args.data_root, failures)

    if failures:
        print("[RESULT] Environment validation FAILED", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("[RESULT] Environment validation PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
