"""Prevent regression: locally imported worker modules must be present in Docker."""
from pathlib import Path


def test_worker_python_imports_are_copied_into_docker_image():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text()
    handler = (root / "src" / "handler.py").read_text()

    assert "from warmup_profile import" in handler
    assert (root / "src" / "warmup_profile.py").is_file()
    assert "COPY src/warmup_profile.py /app/warmup_profile.py" in dockerfile
    assert "import warmup_profile" in dockerfile
