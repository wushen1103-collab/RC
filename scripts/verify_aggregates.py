from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "final.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    checked = 0
    for raw_line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        observed = sha256(path)
        checked += 1
        if observed != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {checked} final aggregate files.")


if __name__ == "__main__":
    main()
