from __future__ import annotations

import sys
from pathlib import Path

FORBIDDEN = "Fraud-data.csv"
ACTIVE_PATHS = [
    Path("app.py"),
    Path("config_file"),
    Path("dvc.yaml"),
    Path("render.yaml"),
    Path("Dockerfile"),
    Path(".github/workflows"),
    Path("scripts"),
    Path("src/FraudGuard/cloud"),
    Path("src/FraudGuard/data"),
    Path("src/FraudGuard/pipeline"),
    Path("tests"),
]
IGNORED_PARTS = {"__pycache__", ".pytest_cache"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_PATHS:
        if not root.exists():
            continue
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_file() and not (set(path.parts) & IGNORED_PARTS):
                files.append(path)
    return files


def main() -> int:
    violations: list[str] = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if FORBIDDEN in text and path.name != Path(__file__).name:
            violations.append(str(path))
    if violations:
        print("Retired legacy dataset path found in active files:")
        for path in violations:
            print(f"- {path}")
        return 1
    print("No retired legacy dataset path found in active files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
