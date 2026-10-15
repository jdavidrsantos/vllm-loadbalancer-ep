"""Validate installed Python requirements without masking new dependency errors.

The official vLLM 0.11.2 base ships a non-inference PyGObject package whose
metadata requires pycairo. Installing optional system GUI bindings inside the
CUDA inference image is unnecessary. Allow exactly that pre-existing issue,
and keep every other pip check failure fatal to the Docker build.
"""
from __future__ import annotations

import re
import subprocess
import sys

KNOWN_OPTIONAL_ISSUE = re.compile(
    r"^pygobject \S+ requires pycairo, which is not installed\.$",
    flags=re.IGNORECASE,
)


def classify_issues(output: str) -> tuple[list[str], list[str]]:
    known: list[str] = []
    errors: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if KNOWN_OPTIONAL_ISSUE.fullmatch(line):
            known.append(line)
        else:
            errors.append(line)
    return known, errors


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    known, errors = classify_issues(completed.stdout)

    # pip check can return stderr when something else is broken, even when
    # stdout contains only the known optional PyGObject issue.
    if completed.stderr.strip():
        errors.extend(line.strip() for line in completed.stderr.splitlines() if line.strip())
    if completed.returncode != 0 and not known and not errors:
        errors.append(f"pip check failed with exit code {completed.returncode} and no diagnostic")

    for issue in known:
        print(f"WARNING: tolerated optional upstream dependency: {issue}")
    for issue in errors:
        print(f"ERROR: Python dependency conflict: {issue}", file=sys.stderr)
    if errors:
        return 1
    print("Python dependency check passed (known optional PyGObject issue ignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
