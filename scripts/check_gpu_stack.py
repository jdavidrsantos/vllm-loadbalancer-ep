"""Build-time CUDA stack guard and read-only runtime GPU compatibility diagnostics.

No driver/GPU is required during docker build. Never log user prompts or keys.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def cuda_tuple(value: str) -> tuple[int, int]:
    try:
        parts = value.split(".")
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError, AttributeError):
        return (0, 0)


def inspect_stack(mode: str) -> dict:
    import torch

    result: dict = {
        "mode": mode,
        "torch": version("torch"),
        "vllm": version("vllm"),
        "fastapi": version("fastapi"),
        "httpx": version("httpx"),
        "huggingface_hub": version("huggingface-hub"),
        "cuda_compiled": torch.version.cuda,
        "compiled_arch_list": torch.cuda.get_arch_list(),
    }
    # The image is already pinned by a Docker manifest digest. Its Python
    # distribution may report a local/dev/build version different from the
    # human-readable Docker tag. That alone is not a CUDA incompatibility.
    # Still report the discrepancy for investigation rather than hiding it.
    installed_vllm = result["vllm"]
    if installed_vllm == "missing":
        raise RuntimeError("vLLM Python distribution is missing from the base image")
    if not shutil.which("vllm"):
        raise RuntimeError("vLLM CLI is missing from PATH in the base image")
    if installed_vllm != "0.11.2":
        result["vllm_version_notice"] = (
            f"Docker tag v0.11.2; installed distribution: {installed_vllm}. "
            "Verify release provenance, but this alone does not prove GPU incompatibility."
        )
    if cuda_tuple(torch.version.cuda or "") < (12, 8):
        raise RuntimeError(
            f"PyTorch was compiled with CUDA {torch.version.cuda!r}; "
            "the target Blackwell GPU requires CUDA >=12.8"
        )
    # Presence of sm_120 in bundled wheel is a useful hint, but device-level
    # verification is authoritative; some kernels use runtime JIT compilation.
    result["sm120_precompiled"] = "sm_120" in result["compiled_arch_list"]
    if mode == "runtime":
        result["gpu_visible"] = torch.cuda.is_available()
        if not result["gpu_visible"]:
            raise RuntimeError("CUDA GPU is not visible to the worker")
        properties = torch.cuda.get_device_properties(0)
        result.update(
            gpu_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            vram_gib=round(properties.total_memory / (1024 ** 3), 2),
        )
        if (properties.major, properties.minor) != (12, 0):
            result["notice"] = "GPU architecture differs from expected Blackwell sm_120"
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,mig.mode.current",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if completed.returncode == 0:
                result["nvidia_smi"] = completed.stdout.strip()[:256]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "runtime"), required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    summary = inspect_stack(args.mode)
    print(json.dumps({"event": "cuda_stack_check", **summary}, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
