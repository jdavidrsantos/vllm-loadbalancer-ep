"""Regression tests for the narrowly allowed inherited vLLM base-image issue."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_base_dependencies import classify_issues


def test_only_known_upstream_pygobject_is_tolerated():
    allowed, fatal = classify_issues(
        "pygobject 3.42.1 requires pycairo, which is not installed.\n"
    )
    assert len(allowed) == 1
    assert not fatal


def test_unrelated_dependency_conflicts_remain_fatal():
    allowed, fatal = classify_issues(
        "pygobject 3.42.1 requires pycairo, which is not installed.\n"
        "torch 2.7 requires some-gpu-package, which is not installed.\n"
    )
    assert len(allowed) == 1
    assert len(fatal) == 1
    assert "torch" in fatal[0]


def test_version_conflict_not_exempted():
    allowed, fatal = classify_issues(
        "pygobject 3.42.1 has requirement pycairo>=1.0, but you have pycairo 0.9.\n"
    )
    assert not allowed
    assert len(fatal) == 1
