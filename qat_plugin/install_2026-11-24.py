#!/usr/bin/env python3
"""Copy code/config into a new QAT project; never modify source, datasets, or weights."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

EXCLUDE = {".git", "runs", "wandb", "datasets", "images", "labels", "weights", "__pycache__", ".venv", ".pytest_cache"}
SUFFIXES = {".py", ".yaml", ".yml", ".json", ".txt", ".md", ".sh", ".toml", ".cfg", ".ini"}


def install(source, destination, plugin):
    source, destination, plugin = source.resolve(), destination.resolve(), plugin.resolve()
    if not (source / "train.py").is_file() or not (source / "models/yolo.py").is_file():
        raise ValueError("Source must contain train.py and models/yolo.py")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate sibling projects")
    if destination.exists():
        raise FileExistsError("Destination already exists; choose a new name. No files overwritten: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_name(destination.name + ".install.lock")
    with lock.open("x"):
        try:
            destination.mkdir()
            files = {}
            for folder, dirs, names in os.walk(str(source), followlinks=False):
                dirs[:] = [d for d in dirs if d not in EXCLUDE and not (Path(folder) / d).is_symlink() and d != "qat_plugin"]
                for name in names:
                    path = Path(folder) / name
                    if path.is_symlink() or (path.suffix not in SUFFIXES and name not in ("LICENSE", ".gitignore")):
                        continue
                    relative = path.relative_to(source)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(path), str(target))
                    files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
            shutil.copytree(str(plugin), str(destination / "qat_plugin"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            manifest = {"source": str(source), "destination": str(destination), "original_source_sha256": files,
                        "datasets_and_weights_copied": False, "source_modified": False}
            (destination / "2026-11-24_QAT_install_manifest.json").write_text(json.dumps(manifest, indent=2))
        finally:
            lock.unlink()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/sunyu/project/gesture_asc/gesture_train_shared"))
    parser.add_argument("--destination", type=Path, default=Path("/home/sunyu/project/gesture_asc/gesture_train_shared_QAT"))
    args = parser.parse_args()
    print("Created:", install(args.source, args.destination, Path(__file__).resolve().parent))
    print("Use an AIMET-compatible environment, then run: python qat_plugin/run_2026-11-24.py --preflight")
