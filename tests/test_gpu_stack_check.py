"""Regression tests for the CUDA diagnostic helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_gpu_stack import cuda_tuple


def test_cuda_versions():
    assert cuda_tuple("12.9") >= (12, 8)
    assert cuda_tuple("12.8") >= (12, 8)
    assert cuda_tuple("12.7") < (12, 8)
    assert cuda_tuple(None) < (12, 8)
