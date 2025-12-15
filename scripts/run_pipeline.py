#!/usr/bin/env python3
"""Coordinate the bundling and annotation steps so they run in one command."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bundle_images import expand_patterns  # type: ignore

DEFAULT_MANIFEST_PATH = Path("output/frame_manifest.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    try:
        bool_action = argparse.BooleanOptionalAction  # type: ignore[attr-defined]
    except AttributeError:
        class _BoolAction(argparse.Action):
            def __init__(self, option_strings, dest, default=False, **kwargs):
                super().__init__(option_strings, dest, nargs=0, default=default, **kwargs)

            def __call__(self, parser, namespace, values, option_string=None):
                if option_string is None:
                    return
                setattr(namespace, self.dest, not option_string.startswith("--no-"))

        bool_action = _BoolAction  # type: ignore[assignment]

    parser = argparse.ArgumentParser(
        description=(
            "Run bundle_images.py and build_annotations.py sequentially so the "
            "outputs stay aligned."
        )
    )
    parser.add_argument(
        "--images",
        default=Path("sample data"),
        type=Path,
        help="Directory that contains the raw image files to bundle (default: sample data).",
    )
    
    parser.add_argument(
        "--labels",
        default=Path("sample data"),
        type=Path,
        help="Directory that contains the individual LabelMe JSON label files (default: sample data).",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=Path("output/media_bundle.mp4"),
        help="Destination for the bundled media file (default: output/media_bundle.mp4).",
    )
    parser.add_argument(
        "--annotation-output",
        type=Path,
        default=Path("output/unified_annotation.json"),
        help="Destination for the unified annotation JSON (default: output/unified_annotation.json).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional path to persist the frame-order manifest JSON. When omitted, "
            "the manifest is embedded into the unified annotation output only."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg,*.jpeg,*.png",
        help="Comma-separated glob patterns used to discover images (default: *.jpg,*.jpeg,*.png).",
    )
    parser.add_argument(
        "--fps",
        default=60.0,
        type=float,
        help="Frames per second when encoding videos (passed to bundle_images.py, default: 30).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing bundle output files.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help=(
            "Forwarded to build_annotations.py to skip labels whose source images "
            "cannot be found."
        ),
    )
    parser.add_argument(
        "--auto-generate-placeholders",
        action=bool_action,
        default=True,
        help=(
            "Automatically generate blank placeholder images when no sources are found "
            "in --images. Disable with --no-auto-generate-placeholders."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for this runner and the underlying scripts.",
    )
    return parser.parse_args(argv)


def run_command(command: list[str], description: str, **kwargs: Any) -> None:
    logging.info("%s", description)
    logging.debug("Command: %s", " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True, **kwargs)


def _collect_label_entries(labels_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for label_file in sorted(labels_dir.glob("*.json")):
        with label_file.open("r", encoding="utf-8") as stream:
            data = json.load(stream)

        file_name = data.get("imagePath") or f"{label_file.stem}.jpg"
        entries.append(
            {
                "label_file": label_file,
                "data": data,
                "file_name": Path(file_name).name,
                "width": data.get("imageWidth"),
                "height": data.get("imageHeight"),
            }
        )

    return entries


def _generate_placeholder_images(entries: Iterable[dict[str, Any]], target_dir: Path, overwrite: bool) -> list[Path]:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Generating placeholder images requires Pillow. Install it via 'pip install Pillow' "
            "or disable the feature with --no-auto-generate-placeholders."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for entry in entries:
        width = entry.get("width")
        height = entry.get("height")
        if width is None or height is None:
            continue

        try:
            width_int = int(round(float(width)))
            height_int = int(round(float(height)))
        except (TypeError, ValueError):
            continue

        if width_int <= 0 or height_int <= 0:
            continue

        image_name = entry.get("file_name") or "placeholder.jpg"
        suffix = Path(image_name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            suffix = ".png"
        filename = f"{Path(image_name).stem}{suffix}"
        output_path = target_dir / filename

        if output_path.exists() and not overwrite:
            generated.append(output_path)
            continue

        image = Image.new("RGB", (width_int, height_int), color=(32, 32, 32))
        format_name = "PNG" if output_path.suffix.lower() == ".png" else "JPEG"
        save_kwargs = {"quality": 90} if format_name == "JPEG" else {}
        image.save(output_path, format=format_name, **save_kwargs)
        generated.append(output_path)

    return generated


def _build_manifest_payload(files: list[Path], bundle_path: Path, source_root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
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

    return {
        "version": "1.0",
        "bundle_path": str(bundle_path),
        "source_root": str(source_root),
        "image_count": len(entries),
        "images": entries,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    bundle_script = Path(__file__).with_name("bundle_images.py")
    annotation_script = Path(__file__).with_name("build_annotations.py")

    bundle_output = args.bundle_output
    annotation_output = args.annotation_output
    manifest_path = args.manifest
    label_dir = args.labels

    if not label_dir.exists() or not label_dir.is_dir():
        logging.error("Label directory %s does not exist or is not a folder", label_dir)
        return 1

    for target in [bundle_output, annotation_output]:
        target.parent.mkdir(parents=True, exist_ok=True)

    label_entries = _collect_label_entries(label_dir)
    if not label_entries:
        logging.error("No label files discovered in %s", label_dir)
        return 1

    staging_dir = Path("output/staging_images")
    if staging_dir.exists() and args.overwrite:
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_paths: list[Path] = []
    missing_entries: list[dict[str, Any]] = []
    processed_names: set[str] = set()

    for entry in label_entries:
        file_name = entry.get("file_name")
        if entry.get("width") is None or entry.get("height") is None:
            logging.warning(
                "Skipping label %s because image dimensions are missing",
                entry.get("label_file"),
            )
            continue
        if not file_name or file_name in processed_names:
            continue
        processed_names.add(file_name)

        source_candidates = [args.images / file_name]
        # Some datasets store relative paths inside labels; fall back to base name.
        base_name = Path(file_name).name
        if base_name != file_name:
            source_candidates.append(args.images / base_name)

        source_path = next((candidate for candidate in source_candidates if candidate.exists()), None)
        if source_path is None:
            missing_entries.append(entry)
            continue

        target_path = staging_dir / base_name
        if target_path.exists():
            if args.overwrite:
                target_path.unlink()
            else:
                staged_paths.append(target_path)
                continue

        try:
            os.link(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)

        staged_paths.append(target_path)

    if missing_entries:
        missing_names = [entry.get("file_name") for entry in missing_entries if entry.get("file_name")]
        logging.warning(
            "Missing %d image files referenced by labels: %s",
            len(missing_names),
            ", ".join(sorted(set(missing_names)))[:500],
        )
        if args.auto_generate_placeholders:
            generated = _generate_placeholder_images(missing_entries, staging_dir, args.overwrite)
            if generated:
                staged_paths.extend(generated)
                logging.info(
                    "Generated %d placeholder images for missing entries", len(generated)
                )
            unresolved: list[str] = []
            for entry in missing_entries:
                file_name = entry.get("file_name")
                if not file_name:
                    continue
                staged_path = staging_dir / Path(file_name).name
                if not staged_path.exists():
                    unresolved.append(file_name)
            if unresolved:
                logging.error(
                    "Unable to synthesise placeholders for: %s",
                    ", ".join(sorted(set(filter(None, unresolved))))[:500],
                )
                return 1
        else:
            logging.error(
                "Cannot proceed because some labelled images are missing. Enable placeholder "
                "generation or provide the originals."
            )
            return 1

    image_root = staging_dir
    logging.info("Prepared %d images in staging directory %s", len(staged_paths), image_root)

    pattern_tokens = {token.strip() for token in args.pattern.split(",") if token.strip()}
    suffix_patterns = {f"*{path.suffix.lower()}" for path in staged_paths if path.suffix}
    combined_patterns = pattern_tokens.union(suffix_patterns)
    bundle_pattern = ",".join(sorted(combined_patterns)) if combined_patterns else "*.jpg,*.jpeg,*.png"

    bundle_cmd: list[str] = [
        sys.executable,
        str(bundle_script),
        "--input",
        str(image_root),
        "--output",
        str(bundle_output),
        "--pattern",
    bundle_pattern,
        "--fps",
        str(args.fps),
        "--log-level",
        args.log_level,
        "--no-manifest",
    ]
    if args.overwrite:
        bundle_cmd.append("--overwrite")

    try:
        run_command(bundle_cmd, "Bundling images")
    except subprocess.CalledProcessError as exc:
        logging.error("Image bundling failed with exit code %s", exc.returncode)
        return exc.returncode

    if manifest_path is None and DEFAULT_MANIFEST_PATH.exists():
        try:
            DEFAULT_MANIFEST_PATH.unlink()
            logging.debug("Removed existing manifest at %s", DEFAULT_MANIFEST_PATH)
        except OSError:
            logging.warning(
                "Could not remove existing manifest file at %s", DEFAULT_MANIFEST_PATH
            )

    frame_files = expand_patterns(image_root, bundle_pattern)
    manifest_payload = _build_manifest_payload(frame_files, bundle_output, image_root)
    manifest_json = json.dumps(manifest_payload)

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_json, encoding="utf-8")
        logging.info("Persisted frame manifest to %s", manifest_path)

    annotation_cmd: list[str] = [
        sys.executable,
        str(annotation_script),
        "--labels",
        str(args.labels),
        "--output",
        str(annotation_output),
        "--log-level",
        args.log_level,
    ]
    annotation_cmd.extend(["--manifest", "-", "--require-manifest"])

    annotation_cmd.extend(["--images", str(image_root)])
    if args.skip_missing_images:
        annotation_cmd.append("--skip-missing-images")

    try:
        run_command(annotation_cmd, "Building unified annotations", input=manifest_json, text=True)
    except subprocess.CalledProcessError as exc:
        logging.error("Annotation build failed with exit code %s", exc.returncode)
        return exc.returncode

    logging.info("Pipeline complete. Bundle: %s | Annotations: %s", bundle_output, annotation_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
