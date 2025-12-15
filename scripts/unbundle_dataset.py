#!/usr/bin/env python3
"""Recreate individual images and LabelMe-style annotation files from a bundled
video plus the unified annotation JSON produced by this project."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract individual frames and per-image annotation files from the "
            "pipeline outputs (media bundle + unified annotations)."
        )
    )
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="Path to the bundled media file (MP4).",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to the unified annotation JSON file created by build_annotations.py.",
    )
    parser.add_argument(
        "--output-images",
        type=Path,
        default=Path("output/recovered_images"),
        help="Directory to write the recovered image files (default: output/recovered_images).",
    )
    parser.add_argument(
        "--output-labels",
        type=Path,
        default=Path("output/recovered_labels"),
        help="Directory to write LabelMe-style JSON files (default: output/recovered_labels).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directories if they already contain files.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path {path} exists and is not a directory.")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory {path} already contains files. Use --overwrite to replace them."
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_relative_path(raw: str | None, fallback_name: str) -> Path:
    if not raw:
        return Path(fallback_name)
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts:
        return Path(fallback_name)
    if any(part == ".." for part in parts):
        return Path(fallback_name)
    # Drop potential drive letter on Windows.
    if ":" in parts[0]:
        parts = parts[1:]
        if not parts:
            return Path(fallback_name)
    return Path(*parts)


def load_dataset(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_annotation_lookup(dataset: Dict[str, Any]) -> tuple[
    Dict[int, Dict[str, Any]],
    Dict[int, str],
    Dict[int, List[Dict[str, Any]]],
]:
    images_by_id: Dict[int, Dict[str, Any]] = {}
    categories: Dict[int, str] = {}
    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}

    for image in dataset.get("images", []):
        image_id = image.get("id")
        if image_id is None:
            continue
        images_by_id[int(image_id)] = image

    for category in dataset.get("categories", []):
        category_id = category.get("id")
        name = category.get("name")
        if category_id is None or not name:
            continue
        categories[int(category_id)] = str(name)

    for annotation in dataset.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        annotations_by_image.setdefault(int(image_id), []).append(annotation)

    return images_by_id, categories, annotations_by_image


def segmentation_to_points(segmentation: Iterable[float]) -> List[List[float]]:
    coords = list(segmentation)
    if len(coords) % 2 != 0:
        logging.debug("Odd number of coordinates in segmentation: %s", coords)
        coords = coords[:-1]
    points: List[List[float]] = []
    for idx in range(0, len(coords), 2):
        points.append([coords[idx], coords[idx + 1]])
    return points


def build_labelme_document(
    image_entry: Dict[str, Any],
    category_lookup: Dict[int, str],
    annotations: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    shapes: List[Dict[str, Any]] = []
    for annotation in annotations:
        category_id = annotation.get("category_id")
        label = category_lookup.get(int(category_id)) if category_id is not None else None
        if not label:
            continue
        segmentation = annotation.get("segmentation") or []
        if isinstance(segmentation, dict):  # Unsupported (RLE) for reconstruction
            logging.debug("Skipping non-polygon segmentation for annotation %s", annotation.get("id"))
            continue
        segments = segmentation if isinstance(segmentation, list) else []
        if not segments:
            continue
        for segment in segments:
            if not segment:
                continue
            if isinstance(segment[0], list):
                flat: List[float] = []
                for pair in segment:
                    if len(pair) >= 2:
                        flat.extend(pair[:2])
                segment_points = segmentation_to_points(flat)
            else:
                segment_points = segmentation_to_points(segment)
            if len(segment_points) < 3:
                continue
            shape: Dict[str, Any] = {
                "label": label,
                "points": segment_points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
            frame_index = annotation.get("frame_index")
            if frame_index is not None:
                shape.setdefault("other_data", {})
                shape["other_data"]["frame_index"] = frame_index
            shapes.append(shape)

    width = int(image_entry.get("width", 0))
    height = int(image_entry.get("height", 0))

    document: Dict[str, Any] = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_entry.get("file_name"),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    frame_index = image_entry.get("frame_index")
    if frame_index is not None:
        document.setdefault("other_data", {})
        document["other_data"]["frame_index"] = frame_index
    source_annotation = image_entry.get("source_annotation")
    if source_annotation:
        document.setdefault("other_data", {})
        document["other_data"]["source_annotation"] = source_annotation

    return document


def extract_frames(
    bundle_path: Path,
    ordered_images: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "OpenCV (cv2) is required to extract frames from the bundled video. "
            "Install it with 'pip install opencv-python'."
        ) from exc

    capture = cv2.VideoCapture(str(bundle_path))
    if not capture.isOpened():  # pragma: no cover - depends on environment
        raise RuntimeError(f"Unable to open video bundle {bundle_path}")

    for index, image_entry in enumerate(ordered_images):
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(
                f"Video ended before all frames were recovered (stopped at index {index})."
            )
        target_rel = safe_relative_path(
            image_entry.get("source_image_path"),
            image_entry.get("file_name", f"frame_{index:06d}.png"),
        )
        target_path = output_dir / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target_path), frame):  # pragma: no cover - depends on OpenCV
            capture.release()
            raise RuntimeError(f"Failed to write frame to {target_path}")

    # Check for extra frames in the video.
    extra_success, _ = capture.read()
    if extra_success:
        logging.warning(
            "Bundled video contains more frames than described in annotations; extra frames were ignored."
        )
    capture.release()


def write_label_files(
    ordered_images: List[Dict[str, Any]],
    annotations_by_image: Dict[int, List[Dict[str, Any]]],
    category_lookup: Dict[int, str],
    output_dir: Path,
) -> None:
    for image_entry in ordered_images:
        image_id = int(image_entry["id"])
        annotations = annotations_by_image.get(image_id, [])
        document = build_labelme_document(image_entry, category_lookup, annotations)
        source_annotation = image_entry.get("source_annotation")
        if source_annotation:
            base_rel = safe_relative_path(source_annotation, f"{Path(image_entry['file_name']).stem}.json")
        else:
            base_rel = Path(f"{Path(image_entry['file_name']).stem}.json")
        target_path = output_dir / base_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.bundle.exists():
        logging.error("Bundled media file not found: %s", args.bundle)
        return 1
    if args.bundle.suffix.lower() != ".mp4":
        logging.error("Only MP4 bundles are supported at the moment (received %s).", args.bundle)
        return 1
    if not args.annotations.exists():
        logging.error("Unified annotation file not found: %s", args.annotations)
        return 1

    try:
        prepare_output_dir(args.output_images, args.overwrite)
        prepare_output_dir(args.output_labels, args.overwrite)
    except (ValueError, FileExistsError) as exc:
        logging.error(str(exc))
        return 1

    dataset = load_dataset(args.annotations)
    images_by_id, category_lookup, annotations_by_image = build_annotation_lookup(dataset)

    if not images_by_id:
        logging.error("No image entries found in annotation file %s", args.annotations)
        return 1

    ordered_images = list(images_by_id.values())
    ordered_images.sort(
        key=lambda entry: (
            entry.get("frame_index") if entry.get("frame_index") is not None else float("inf"),
            entry.get("id"),
        )
    )

    logging.info("Extracting %d frames to %s", len(ordered_images), args.output_images)
    try:
        extract_frames(args.bundle, ordered_images, args.output_images)
    except RuntimeError as exc:
        logging.error(str(exc))
        return 1

    logging.info("Writing per-image annotation files to %s", args.output_labels)
    write_label_files(ordered_images, annotations_by_image, category_lookup, args.output_labels)

    logging.info("Dataset successfully reconstructed: %s, %s", args.output_images, args.output_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
