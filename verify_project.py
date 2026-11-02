"""Verify the extracted delivery manifest; optionally run read-only M4 inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"Missing: {entry['path']}")
        elif path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            failures.append(f"Changed: {entry['path']}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Integrity verified: {len(manifest['files'])} files.")
    if args.simulate:
        sys.path.insert(0, str(ROOT / "work"))
        from run_des_dashboard import Dashboard
        dashboard = Dashboard()
        assert dashboard.controller == "m4", "Five M4 checkpoints are required for this check."
        state = dashboard.step({"count": 1})
        assert state["minute"] == 10 and state["summary"]["steps"] == 1
        assert len(state["cells"]) == 3
        print("Read-only model check passed: five PI-off M4 members; one DES decision; 10 minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
