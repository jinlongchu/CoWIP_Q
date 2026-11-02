"""Build a portable, checksummed project delivery without touching experiments."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_ROOT = "CONWIP_M4_Full_Project"
OMIT_NAMES = {"node_modules", ".venv_madqn_ablation", "__pycache__", ".pytest_cache", ".git"}
OMIT_WORK = {"tmp", "m4_ablation_previews", "m4_no_pi_ablation_previews", "m4_book_qa", "m4_madqn_overview_qa"}
OLD_ROOTS = ()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def local_suffix(value):
    normalized = value.replace("\\", "/")
    for base in OLD_ROOTS:
        if normalized.casefold() == base.casefold():
            return ""
        if normalized.casefold().startswith(base.casefold() + "/"):
            return normalized[len(base) + 1:]
    return None


def relocate_python(data, relative):
    """Replace only literal Path(old-project-root/...) calls, keeping comments."""
    source = data.decode("utf-8-sig")
    tree = ast.parse(source, filename=relative.as_posix())
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(line_number, byte_column):
        prefix = lines[line_number - 1].encode("utf-8")[:byte_column].decode("utf-8")
        return starts[line_number - 1] + len(prefix)

    replacements = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "Path" and len(node.args) == 1 and not node.keywords):
            continue
        argument = node.args[0]
        if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
            continue
        suffix = local_suffix(argument.value)
        if suffix is None:
            continue
        root_expression = f"Path(__file__).resolve().parents[{len(relative.parts) - 1}]"
        replacement = f"({root_expression} / {suffix!r})" if suffix else f"({root_expression})"
        replacements.append((offset(node.lineno, node.col_offset),
                             offset(node.end_lineno, node.end_col_offset), replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    if replacements:
        ast.parse(source, filename=relative.as_posix())
        return source.encode("utf-8"), len(replacements)
    return data, 0


def relocate_workbook_script(data):
    source = data.decode("utf-8")
    pattern = r"const root = (['\"])([^'\"]+)\1;"
    changed = False

    def replace(match):
        nonlocal changed
        if local_suffix(match.group(2)) != "":
            return match.group(0)
        changed = True
        return "const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');"

    source = re.sub(pattern, replace, source)
    if changed:
        source = "import { fileURLToPath } from 'node:url';\n" + source
        return source.encode("utf-8"), 1
    return data, 0


def collect(root, package_directory):
    files, excluded = [], []
    for current, directories, names in os.walk(root, followlinks=False):
        parent = Path(current)
        kept = []
        for name in sorted(directories):
            path = parent / name
            relative = path.relative_to(root)
            omit = (name in OMIT_NAMES or path.is_symlink()
                    or getattr(path, "is_junction", lambda: False)()
                    or (parent == root / "work" and name in OMIT_WORK)
                    or path == package_directory)
            if omit:
                excluded.append(relative.as_posix() + "/")
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            path = parent / name
            relative = path.relative_to(root)
            omit = (path.is_symlink() or path.suffix in {".pyc", ".pyo"}
                    or name.endswith(".inspect.ndjson")
                    or name in {"PACKAGE_MANIFEST.json", "PACKAGE_VERIFICATION.json"}
                    or name.startswith(".env") or path.suffix in {".pem", ".key"}
                    or (parent == root / "outputs" / "des_dashboard" and path.suffix == ".log"))
            if omit:
                excluded.append(relative.as_posix())
            else:
                files.append((relative, path))
    return sorted(files), sorted(excluded)


def required_files():
    paths = ['README_AND_START.txt', "setup_windows.cmd", "verify_project.py", "requirements.txt",
             "work/run_des_dashboard.py", "work/des_dashboard/index.html", "work/des_dashboard/app.js",
             "work/run_controlled_m4_ablation.py", "outputs/m4_no_pi_ablation/comparison_summary.json"]
    for key in ("no_pi", "original_madqn"):
        for seed in range(13101, 13106):
            paths.append(f"outputs/controlled_m4_ablation/formal/{key}/run_{seed}/best_model.pt")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/project_package/CONWIP_M4_Full_Project_2026-11-02.zip")
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError("The package must be created inside the project directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_destination = destination
    version = 2
    while destination.exists() or destination.with_suffix(".zip.partial").exists():
        destination = original_destination.with_name(f"{original_destination.stem}_v{version}.zip")
        version += 1
    files, excluded = collect(ROOT, destination.parent)
    present = {relative.as_posix() for relative, _ in files}
    missing = set(required_files()) - present
    if missing:
        raise FileNotFoundError(f"Missing required package contents: {sorted(missing)}")
    entries, relocations = [], []
    temporary = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for relative, source in files:
            data = source.read_bytes()
            original_sha = digest(data)
            changes = 0
            if relative.parts[0] == "work" and relative.suffix == ".py":
                data, changes = relocate_python(data, relative)
            elif relative.name in {"build_controlled_m4_ablation_workbook.mjs", "build_no_pi_centered_ablation.mjs"}:
                data, changes = relocate_workbook_script(data)
            entry = {"path": relative.as_posix(), "size": len(data), "sha256": digest(data),
                     "source_sha256": original_sha, "path_relocated": bool(changes)}
            entries.append(entry)
            if changes:
                relocations.append({"path": relative.as_posix(), "path_expressions": changes})
            archive.writestr(f"{ARCHIVE_ROOT}/{relative.as_posix()}", data)
        manifest = {
            "project": "CONWIP multi-workshop control / MADQN / M4",
            "packaging_date": "2026-11-02", "timezone": "Pacific/Auckland",
            "primary_strategy": "Five-member independent Q ensemble, M4 without PI residual guidance",
            "file_count": len(entries), "uncompressed_bytes": sum(entry["size"] for entry in entries),
            "model_files": sum(entry["path"].endswith(".pt") for entry in entries),
            "virtual_environment_included": False, "node_modules_included": False,
            "excluded": excluded, "archive_only_path_relocations": relocations,
            "files": entries,
        }
        archive.writestr(f"{ARCHIVE_ROOT}/PACKAGE_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    with zipfile.ZipFile(temporary) as archive:
        invalid = archive.testzip()
        if invalid:
            raise ValueError(f"ZIP CRC failed: {invalid}")
        for entry in entries:
            data = archive.read(f"{ARCHIVE_ROOT}/{entry['path']}")
            if len(data) != entry["size"] or digest(data) != entry["sha256"]:
                raise ValueError(f"Archive hash failed: {entry['path']}")
    temporary.rename(destination)
    archive_digest = hashlib.sha256()
    with destination.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            archive_digest.update(chunk)
    archive_hash = archive_digest.hexdigest()
    checksum_file = destination.with_suffix(".zip.sha256.txt")
    checksum_file.write_text(f"{archive_hash}  {destination.name}\n", encoding="utf-8")
    verification = {"archive": str(destination), "sha256": archive_hash,
                    "archive_bytes": destination.stat().st_size, "file_count": len(entries) + 1,
                    "model_files": manifest["model_files"], "uncompressed_bytes": manifest["uncompressed_bytes"],
                    "crc_verified": True, "all_file_hashes_verified": True,
                    "archive_only_path_relocations": relocations}
    destination.with_suffix(".verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
