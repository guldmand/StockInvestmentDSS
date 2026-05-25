from __future__ import annotations

import importlib.metadata
import platform
import sys


def print_version(package_name: str) -> None:
    try:
        version = importlib.metadata.version(package_name)
        print(f"{package_name}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package_name}: NOT INSTALLED")


def main() -> None:
    print("Python:", sys.version)
    print("Platform:", platform.platform())

    packages = [
        "duckdb",
        "numpy",
        "pandas",
        "yfinance",
        "gymnasium",
        "stable-baselines3",
        "torch",
        "finrl",
    ]

    for package in packages:
        print_version(package)

    print("\nImport checks:")

    try:
        import duckdb

        print("duckdb import: OK")
    except Exception as exc:
        print(f"duckdb import: FAILED ({type(exc).__name__}: {exc})")

    try:
        import yfinance

        print("yfinance import: OK")
    except Exception as exc:
        print(f"yfinance import: FAILED ({type(exc).__name__}: {exc})")

    try:
        import gymnasium

        print("gymnasium import: OK")
    except Exception as exc:
        print(f"gymnasium import: FAILED ({type(exc).__name__}: {exc})")

    try:
        import stable_baselines3

        print("stable_baselines3 import: OK")
    except Exception as exc:
        print(f"stable_baselines3 import: FAILED ({type(exc).__name__}: {exc})")

    try:
        import torch

        print("torch import: OK")
        print("torch cuda available:", torch.cuda.is_available())
    except Exception as exc:
        print(f"torch import: FAILED ({type(exc).__name__}: {exc})")

    try:
        import finrl

        print("finrl import: OK")
    except Exception as exc:
        print(f"finrl import: FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
