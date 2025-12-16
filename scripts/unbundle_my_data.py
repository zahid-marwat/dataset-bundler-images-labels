#!/usr/bin/env python3
"""Recreate an image dataset from the artefacts produced by ``bundle_my_data.py``.

Edit the configuration variables below before running the script so they point
to the bundled artefacts you want to unpack and the destination directory where
the files should be reconstructed.
"""

from __future__ import annotations

import json
import logging
import shutil
import zipfile
from pathlib import Path

# -- Configuration ---------------------------------------------------------

# Update these paths before running the script.
BUNDLE_PATH = Path("output_bundle/images_bundle.zip")
MANIFEST_PATH = Path("output_bundle/dataset_manifest.json")
OUTPUT_DIR = Path("unpacked_output")

# When True the output directory is cleared before restoring files.
OVERWRITE_OUTPUT = False

# Logging verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
LOG_LEVEL = "INFO"


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if "images" not in data:
        raise ValueError("Manifest missing 'images' section")
    if "labels" not in data:
        data["labels"] = []
    return data


def restore_images(bundle_path: Path, target_dir: Path) -> int:
    with zipfile.ZipFile(bundle_path, "r") as archive:
        archive.extractall(target_dir)
        info_list = archive.infolist()
    logging.info("Extracted %d image files into %s", len(info_list), target_dir)
    return len(info_list)


def restore_labels(labels: list[dict], target_dir: Path) -> int:
    count = 0
    for entry in labels:
        relative_path = entry.get("relative_path")
        if not relative_path:
            logging.debug("Skipping label entry with no relative_path: %s", entry)
            continue
        raw_text = entry.get("raw_text")
        data = entry.get("data")
        label_path = target_dir / Path(relative_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_text is not None:
            label_path.write_text(raw_text, encoding="utf-8")
        elif data is not None:
            label_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            logging.warning(
                "Label %s has no stored content; writing empty file", relative_path
            )
            label_path.touch()
        count += 1
    logging.info("Restored %d label files into %s", count, target_dir)
    return count


def main(
    bundle_path: Path = BUNDLE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Path = OUTPUT_DIR,
    overwrite: bool = OVERWRITE_OUTPUT,
    log_level: str = LOG_LEVEL,
) -> int:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s")

    if not bundle_path.exists():
        logging.error("Bundle file %s does not exist", bundle_path)
        return 1
    if not manifest_path.exists():
        logging.error("Manifest file %s does not exist", manifest_path)
        return 1

    try:
        prepare_output(output_dir, overwrite)
    except FileExistsError as exc:
        logging.error(str(exc))
        return 1

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logging.error("Failed to load manifest: %s", exc)
        return 1

    restored_images = restore_images(bundle_path, images_dir)
    restored_labels = restore_labels(manifest.get("labels", []), labels_dir)

    logging.info(
        "Unbundling finished. Recreated %d images and %d labels in %s",
        restored_images,
        restored_labels,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
