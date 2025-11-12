#!/usr/bin/env python3
"""
Utility for reducing a large folder of image files into a single bundle that
is easier to upload. The default bundle format is a ZIP archive, but an MP4
video can be generated when OpenCV is available and the output path ends with
.mp4. The script can also emit a manifest that records the exact frame order
so downstream annotation steps stay aligned with the bundled media.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable


def _relative_sort_key(root: Path, path: Path) -> str:
    """Return a case-insensitive relative path string for deterministic ordering."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    return relative.as_posix().lower()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bundle a directory of images into a single archive or video file. "
            "All images are sorted alphabetically before bundling to keep "
            "frame ordering deterministic."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the directory that contains the source images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/media_bundle.zip"),
        help=(
            "Path of the bundle to generate. Use .zip (default) or .tar/.tar.gz "
            "for archive outputs. Use .mp4 to encode a video (requires OpenCV). "
            "Defaults to output/media_bundle.zip."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png",  # comma-separated list of glob patterns
        help=(
            "Comma-separated glob patterns for image discovery. "
            "Defaults to '*.jpg,*.jpeg,*.png'."
        ),
    )
    parser.add_argument(
        "--fps",
        default=24.0,
        type=float,
        help="Frames per second to use when producing a video bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output bundle if it already exists.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/frame_manifest.json"),
        help=(
            "Path to write a JSON manifest that captures the bundle ordering. "
            "Defaults to output/frame_manifest.json."
        ),
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing the frame-order manifest file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def expand_patterns(root: Path, pattern_spec: str) -> list[Path]:
    patterns = [pattern.strip() for pattern in pattern_spec.split(",") if pattern.strip()]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    # Remove duplicates while preserving order.
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for file in files:
        if file not in seen and file.is_file():
            seen.add(file)
            unique_files.append(file)
    unique_files.sort(key=lambda candidate: _relative_sort_key(root, candidate))
    return unique_files


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            if path.is_dir():
                raise ValueError(f"Output path {path} exists and is a directory; cannot overwrite.")
            path.unlink()
        else:
            raise FileExistsError(f"Output bundle {path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def bundle_as_archive(files: Iterable[Path], output_path: Path) -> None:
    import tarfile
    import zipfile

    suffix = output_path.suffix.lower()
    if suffix == ".zip":
        compression = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(output_path, mode="w", compression=compression) as archive:
            for image_path in files:
                archive.write(image_path, arcname=image_path.name)
    elif suffix in {".tar", ".tgz", ".gz", ".bz2", ".xz"} or output_path.suffixes[-2:] == [".tar", ".gz"]:
        mode = "w"
        if suffix == ".tgz" or output_path.suffixes[-2:] == [".tar", ".gz"]:
            mode = "w:gz"
        elif suffix == ".gz":
            mode = "w:gz"
        elif suffix == ".bz2":
            mode = "w:bz2"
        elif suffix == ".xz":
            mode = "w:xz"
        with tarfile.open(output_path, mode) as archive:
            for image_path in files:
                archive.add(image_path, arcname=image_path.name)
    else:
        raise ValueError(
            "Unsupported archive extension. Use .zip, .tar, .tar.gz, .tgz, .tar.bz2, or .tar.xz."
        )


def bundle_as_video(files: Iterable[Path], output_path: Path, fps: float) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - only executed when deps missing
        raise RuntimeError(
            "OpenCV (cv2) and NumPy are required for MP4 output. Install them or use a zip/tar archive instead."
        ) from exc

    files = list(files)
    if not files:
        raise ValueError("No input images discovered; cannot create a video bundle.")

    first_frame = cv2.imread(str(files[0]))
    if first_frame is None:
        raise ValueError(f"Unable to read the first frame: {files[0]}")

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on OpenCV build
        raise RuntimeError("Failed to initialise the video writer. Check codec support.")

    try:
        for path in files:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Unable to read frame: {path}")
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def write_manifest(
    files: list[Path],
    manifest_path: Path | None,
    bundle_path: Path,
    source_root: Path,
) -> None:
    """Persist a manifest describing the frame order for downstream consumers."""

    if manifest_path is None:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for index, image_path in enumerate(files):
        try:
            relative = image_path.relative_to(source_root)
            relative_str = relative.as_posix()
        except ValueError:
            relative_str = image_path.name
        entries.append(
            {
                "frame_index": index,
                "relative_path": relative_str,
                "file_name": image_path.name,
            }
        )

    manifest = {
        "version": "1.0",
        "bundle_path": str(bundle_path),
        "source_root": str(source_root),
        "image_count": len(entries),
        "images": entries,
    }

    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    manifest_path: Path | None = None if args.no_manifest else args.manifest

    if not args.input.exists() or not args.input.is_dir():
        logging.error("Input path %s does not exist or is not a directory", args.input)
        return 1

    files = expand_patterns(args.input, args.pattern)
    if not files:
        logging.error("No images found in %s using patterns %s", args.input, args.pattern)
        return 1
    logging.info("Discovered %d images to bundle", len(files))

    try:
        ensure_output_path(args.output, args.overwrite)
    except (FileExistsError, ValueError) as exc:
        logging.error(str(exc))
        return 1

    if args.output.suffix.lower() == ".mp4":
        # Using a dedicated helper keeps the codec-specific logic isolated.
        logging.info("Encoding video bundle at %s", args.output)
        bundle_as_video(files, args.output, args.fps)
    else:
        logging.info("Creating archive bundle at %s", args.output)
        bundle_as_archive(files, args.output)

    if manifest_path is not None:
        logging.info("Writing frame manifest to %s", manifest_path)
    write_manifest(files, manifest_path, args.output, args.input)

    logging.info("Bundle created successfully: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
