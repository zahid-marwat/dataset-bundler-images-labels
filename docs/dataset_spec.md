# Dataset Specification

This document captures the expected structure of the compact dataset that the
pipeline emits. The dataset consists of two artefacts: a media bundle that
contains the raw imagery and a unified annotation file that describes every
object of interest. By default the CLI tools in `scripts/` write both artefacts
into the repository's `output/` directory. A companion `frame_manifest.json`
file links the ordering of bundled frames with the unified annotations.

## Media Bundle

- **Format**: ZIP archive by default. MP4 video is supported when the
  `bundle_images.py` script is executed with an `.mp4` output path and the
  dependencies (`opencv-python`, `numpy`) are installed.
- **Ordering**: Files are added to the bundle in ascending filename order to
  keep the frame indices stable across runs.
- **Extensions**: Image discovery defaults to `*.jpg`, `*.jpeg`, `*.png`. Provide
  a custom pattern via `--pattern` when necessary.

## Unified Annotation File

- **Format**: COCO-style JSON encoded in UTF-8.
- **Structure**:
  - `info`: Metadata describing the dataset build. When a manifest is used the
    section includes `frame_manifest`, the resolved `bundle_path`, and a
    `frames_without_labels` array enumerating frames that do not have
    associated annotations.
  - `images`: Each entry contains `id`, `file_name`, `width`, `height`, and the
    source annotation file. When a manifest is available the `frame_index`
    matches the frame ordering in the bundled media even when a frame does not
    have any shapes.
  - `annotations`: Polygonal annotations that reference `image_id` and
    `category_id`. Bounding boxes (`bbox`) and areas (`area`) are pre-computed.
  - `categories`: Unique labels encountered while parsing the source files.
- **Source**: Each annotation stores the originating label JSON path in the
  `source` field to simplify traceability.

## Assumptions

- Source annotations follow the LabelMe schema and include `imageWidth` and
  `imageHeight` fields.
- Polygon vertices are recorded in pixel coordinates relative to the source
  image dimensions.
- Rectangle annotations are expressed as two points (top-left and bottom-right).
  The conversion logic expands them into polygons before calculating area and
  bounding boxes.
- The manifest file aligns the bundled media with the annotations and captures
  `frame_index`, the relative image path, and the final bundle artifact path.

## Extensibility

- Add new geometry conversions in `build_annotations.py` to support additional
  shape types (e.g., circles or lines).
- Update the bundle script to ingest other modalities (e.g., depth maps) by
  extending the discovery pattern or implementing modality-specific handlers.
- Embed additional metadata (GPS coordinates, capture timestamps) by enriching
  the `images` or `annotations` entries before writing the unified JSON file.
