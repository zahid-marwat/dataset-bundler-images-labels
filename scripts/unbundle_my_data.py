#!/usr/bin/env python3
"""Recreate per-frame images and LabelMe annotations from the bundled outputs."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# -- Configuration ---------------------------------------------------------

# Update these paths before running the script.
BUNDLE_PATH = Path(r"C:\Users\z-pc\Desktop\dataset-bundler-images-labels\output\images_video.mp4")
MANIFEST_PATH = Path(r"C:\Users\z-pc\Desktop\dataset-bundler-images-labels\output\dataset_manifest.json")
OUTPUT_DIR = Path("unpacked_output")

# When True the output directory is cleared before restoring files.
OVERWRITE_OUTPUT = True

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


def load_manifest(path: Path) -> Tuple[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict) and "images" in data:
        return "coco", data
    if isinstance(data, dict) and "frames" in data:
        return "dataset", data
    raise ValueError("Manifest missing required sections ('images' or 'frames')")


def safe_relative_path(raw: str | None, fallback_name: str) -> Path:
    if not raw:
        return Path(fallback_name)
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts:
        return Path(fallback_name)
    if any(part == ".." for part in parts):
        return Path(fallback_name)
    if ":" in parts[0]:
        parts = parts[1:]
        if not parts:
            return Path(fallback_name)
    return Path(*parts)


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
        if isinstance(segmentation, dict):
            logging.debug(
                "Skipping non-polygon segmentation for annotation %s", annotation.get("id")
            )
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


def decode_video_frames(bundle_path: Path, target_paths: List[Path]) -> int:
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

    for index, target_path in enumerate(target_paths):
        success, frame = capture.read()
        if not success:
            capture.release()
            raise RuntimeError(
                f"Video ended before all frames were recovered (stopped at index {index})."
            )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target_path), frame):  # pragma: no cover - depends on OpenCV
            capture.release()
            raise RuntimeError(f"Failed to write frame to {target_path}")

    extra_success, _ = capture.read()
    if extra_success:
        logging.warning(
            "Bundled video contains more frames than described in annotations; extra frames were ignored."
        )
    capture.release()
    logging.info("Extracted %d image files", len(target_paths))
    return len(target_paths)


def restore_labelme_annotations(
    ordered_images: List[Dict[str, Any]],
    annotations_by_image: Dict[int, List[Dict[str, Any]]],
    category_lookup: Dict[int, str],
    target_dir: Path,
) -> int:
    count = 0
    for image_entry in ordered_images:
        image_id = int(image_entry["id"])
        annotations = annotations_by_image.get(image_id, [])
        document = build_labelme_document(image_entry, category_lookup, annotations)
        source_annotation = image_entry.get("source_annotation")
        if source_annotation:
            label_rel = safe_relative_path(source_annotation, f"{Path(image_entry['file_name']).stem}.json")
        else:
            label_rel = Path(f"{Path(image_entry['file_name']).stem}.json")
        target_path = target_dir / label_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2)
        count += 1
    logging.info("Restored %d label files into %s", count, target_dir)
    return count


def restore_manifest_labels(labels: List[Dict[str, Any]], target_dir: Path) -> int:
    count = 0
    for index, entry in enumerate(labels):
        relative = entry.get("relative_path")
        file_format = entry.get("format") or "txt"
        fallback_name = f"label_{index:06d}.{file_format}"
        label_rel = safe_relative_path(relative, fallback_name)
        target_path = target_dir / label_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if "data" in entry and entry["data"] is not None:
            text = json.dumps(entry["data"], indent=2, ensure_ascii=False)
        else:
            text = entry.get("raw_text") or ""

        target_path.write_text(text, encoding="utf-8")
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
        manifest_type, manifest = load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logging.error("Failed to load manifest: %s", exc)
        return 1

    if manifest_type == "coco":
        images_by_id, category_lookup, annotations_by_image = build_annotation_lookup(manifest)
        if not images_by_id:
            logging.error("Manifest does not contain any image entries")
            return 1

        ordered_images = list(images_by_id.values())
        ordered_images.sort(
            key=lambda entry: (
                entry.get("frame_index") if entry.get("frame_index") is not None else float("inf"),
                entry.get("id"),
            )
        )

        target_paths: List[Path] = []
        for index, image_entry in enumerate(ordered_images):
            fallback_name = image_entry.get("file_name") or f"frame_{index:06d}.png"
            source_hint = image_entry.get("source_image_path")
            target_rel = safe_relative_path(source_hint, fallback_name)
            target_paths.append(images_dir / target_rel)

        try:
            restored_images = decode_video_frames(bundle_path, target_paths)
        except RuntimeError as exc:
            logging.error(str(exc))
            return 1

        restored_labels = restore_labelme_annotations(
            ordered_images, annotations_by_image, category_lookup, labels_dir
        )
    else:
        frame_entries = manifest.get("frames", [])
        if not frame_entries:
            logging.error("Manifest does not contain any frame entries")
            return 1

        sorted_frames = sorted(
            frame_entries,
            key=lambda entry: entry.get("frame_index", 0),
        )

        target_paths = []
        for index, frame in enumerate(sorted_frames):
            fallback_name = f"frame_{index:06d}.png"
            target_rel = safe_relative_path(frame.get("relative_path"), fallback_name)
            target_paths.append(images_dir / target_rel)

        try:
            restored_images = decode_video_frames(bundle_path, target_paths)
        except RuntimeError as exc:
            logging.error(str(exc))
            return 1

        labels = manifest.get("labels", [])
        restored_labels = restore_manifest_labels(labels, labels_dir)

    logging.info(
        "Unbundling finished. Recreated %d images and %d labels in %s",
        restored_images,
        restored_labels,
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
