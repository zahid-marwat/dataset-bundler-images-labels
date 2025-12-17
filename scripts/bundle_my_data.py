#!/usr/bin/env python3
"""Bundle an image dataset and its labels into a portable video package.

Edit the configuration variables below to point at the desired source folders
and outputs. Running the script writes two artefacts into the configured output
directory:

* ``images_video.mp4`` – a video that streams the discovered images in order.
* ``dataset_manifest.json`` – a JSON manifest that records the archived image
    frame order and stores the original label payloads (supporting ``.json``,
    ``.xml``, or ``.txt`` inputs).

The manifest embeds the full text of each label file so that
``unbundle_my_data.py`` can faithfully recreate the original label files during
extraction.
"""

from __future__ import annotations

import json
import logging
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_EXTENSIONS = {".json", ".xml", ".txt"}

# -- Configuration ---------------------------------------------------------

# Requires ``opencv-python`` (install with ``pip install opencv-python``).

# Update these paths before running the script.
# Windows tip: use raw strings (r"C:/path") or forward slashes to avoid escape issues.
IMAGES_DIR = Path(r"C:\Users\z-pc\Desktop\dataset-bundler-images-labels\sample data")
LABELS_DIR = Path(r"C:\Users\z-pc\Desktop\dataset-bundler-images-labels\sample data")
OUTPUT_DIR = Path(r"C:\Users\z-pc\Desktop\dataset-bundler-images-labels\output")

# Comma-separated glob patterns used to discover image files.
IMAGE_PATTERNS = "*.jpg,*.jpeg,*.png,*.bmp,*.tif,*.tiff"

# Video output configuration.
VIDEO_FILENAME = "images_video.mp4"
VIDEO_FPS = 24.0

# When True the output directory is cleared before writing new artefacts.
OVERWRITE_OUTPUT = True

# Logging verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
LOG_LEVEL = "INFO"


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {path} already exists. Enable OVERWRITE_OUTPUT to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_files(root: Path, patterns: Iterable[str], allowed_extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        trimmed = pattern.strip()
        if not trimmed:
            continue
        files.extend(path for path in root.rglob(trimmed) if path.is_file())
    if not patterns:
        files = [path for path in root.rglob("*") if path.is_file()]
    unique: dict[str, Path] = {}
    for path in files:
        suffix = path.suffix.lower()
        if allowed_extensions and suffix not in allowed_extensions:
            continue
        key = str(path.resolve())
        if key not in unique:
            unique[key] = path
    ordered = sorted(unique.values(), key=lambda p: relative_path(p, root).as_posix())
    return ordered


def relative_path(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def create_video(
    images: list[Path],
    source_root: Path,
    destination: Path,
    fps: float,
    filename: str,
) -> tuple[list[dict[str, str]], Path]:
    if cv2 is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "opencv-python is required to create the video. Install it with 'pip install opencv-python'."
        )

    video_path = destination / filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    frame_width = frame_height = None
    manifest_entries: list[dict[str, str]] = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.warning("Skipping unreadable image %s", image_path)
            continue

        if writer is None:
            frame_height, frame_width = frame.shape[:2]
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {video_path}")
            logging.debug("Initialised video writer %s with size %sx%s", video_path, frame_width, frame_height)
        else:
            current_height, current_width = frame.shape[:2]
            if (current_width, current_height) != (frame_width, frame_height):
                frame = cv2.resize(frame, (frame_width, frame_height))
                logging.debug(
                    "Resized frame %s from %sx%s to %sx%s",
                    image_path,
                    current_width,
                    current_height,
                    frame_width,
                    frame_height,
                )

        writer.write(frame)
        manifest_entries.append(
            {
                "frame_index": len(manifest_entries),
                "relative_path": relative_path(image_path, source_root).as_posix(),
                "original_path": str(image_path.resolve()),
            }
        )
        logging.debug("Added frame for %s", image_path)

    if writer is None:
        raise RuntimeError("No readable images were found to build the video")

    writer.release()
    logging.info("Stored %d frames in %s", len(manifest_entries), video_path)
    return manifest_entries, video_path


def collect_labels(labels: list[Path], source_root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for label_path in labels:
        relative = relative_path(label_path, source_root).as_posix()
        suffix = label_path.suffix.lower().lstrip(".") or "txt"
        try:
            raw_text = label_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = label_path.read_text(encoding="utf-8", errors="replace")

        sha = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
        entry: dict[str, object] = {
            "relative_path": relative,
            "format": suffix,
            "sha256": sha,
        }

        if suffix == "json":
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict) and "imageData" in parsed:
                    parsed = parsed.copy()
                    removed = parsed.pop("imageData", None)
                    if removed is not None:
                        logging.debug("Stripped imageData from %s", relative)
                entry["data"] = parsed
            except json.JSONDecodeError:
                logging.warning("Label %s could not be parsed as JSON; storing raw text only", relative)
        else:
            entry["raw_text"] = raw_text
        entries.append(entry)
        logging.debug("Captured label %s", relative)
    logging.info("Captured %d label files", len(entries))
    return entries


def build_manifest(
    frame_entries: list[dict[str, str]],
    label_entries: list[dict[str, str]],
    images_root: Path,
    labels_root: Path,
    video_path: Path,
) -> dict[str, object]:
    return {
        "version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_file": video_path.name,
        "video_frames": len(frame_entries),
        "source_images_root": str(images_root.resolve()),
        "source_labels_root": str(labels_root.resolve()),
        "frames": frame_entries,
        "labels": label_entries,
    }


def main(
    images_dir: Path = IMAGES_DIR,
    labels_dir: Path = LABELS_DIR,
    output_dir: Path = OUTPUT_DIR,
    image_patterns: str = IMAGE_PATTERNS,
    overwrite: bool = OVERWRITE_OUTPUT,
    log_level: str = LOG_LEVEL,
) -> int:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format="%(message)s")

    if cv2 is None:
        logging.error(
            "opencv-python is required to build the video (import error: %s)",
            CV2_IMPORT_ERROR,
        )
        return 1

    if not images_dir.exists() or not images_dir.is_dir():
        logging.error("Image directory %s does not exist or is not a folder", images_dir)
        return 1
    if not labels_dir.exists() or not labels_dir.is_dir():
        logging.error("Label directory %s does not exist or is not a folder", labels_dir)
        return 1

    try:
        ensure_output_directory(output_dir, overwrite)
    except FileExistsError as exc:
        logging.error(str(exc))
        return 1

    patterns = [token.strip() for token in image_patterns.split(",") if token.strip()]
    images = discover_files(images_dir, patterns, IMAGE_EXTENSIONS)
    if not images:
        logging.error("No images found in %s matching patterns %s", images_dir, patterns)
        return 1
    logging.info("Discovered %d images", len(images))

    labels = discover_files(labels_dir, ["*"], LABEL_EXTENSIONS)
    if not labels:
        logging.warning("No label files found in %s", labels_dir)
    else:
        logging.info("Discovered %d label files", len(labels))

    try:
        frame_entries, video_path = create_video(images, images_dir, output_dir, VIDEO_FPS, VIDEO_FILENAME)
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1

    label_entries = collect_labels(labels, labels_dir) if labels else []

    manifest = build_manifest(frame_entries, label_entries, images_dir, labels_dir, video_path)
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Wrote manifest to %s", manifest_path)

    logging.info("Bundling complete. Video and manifest written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
