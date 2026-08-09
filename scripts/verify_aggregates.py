from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "aggregates.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        expected[relative] = digest

    actual_files = {
        path.relative_to(ROOT).as_posix(): path
        for path in sorted((ROOT / "results" / "aggregates").glob("*.csv"))
    }
    missing = sorted(set(expected) - set(actual_files))
    unexpected = sorted(set(actual_files) - set(expected))
    mismatched = sorted(
        relative
        for relative, path in actual_files.items()
        if relative in expected and sha256(path) != expected[relative]
    )
    if missing or unexpected or mismatched:
        raise SystemExit(
            f"aggregate verification failed: missing={missing}, "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    print(f"Verified {len(actual_files)} aggregate CSV files.")


if __name__ == "__main__":
    main()
