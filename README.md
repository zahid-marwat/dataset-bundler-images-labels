# Compact Image-Annotation Uploader

*Personal Project by [Your Name]*
**Date:** [Insert Date]

## Problem Statement

Uploading annotated images and their label files (potentially millions of files) to the cloud was extremely slow and inefficient. Each image had a corresponding label, and individually handling so many files created delays, management overhead, and version-control complexity.

## Project Goal

Create a pipeline that converts a large collection of annotated images and labels into **just two files**:

* A bundled media file (e.g., a video or compressed archive of images)
* A unified annotation file (e.g., in COCO or a similar format)
  By doing this, the upload process is streamlined, dataset management is simplified, and compatibility with machine-learning training workflows is maintained.

## How It Works

1. **Bundle Media**: Combine the images into one large file (either a video or archive).
2. **Build Annotations**: Parse all individual label files and merge them into a single JSON (or selected format) that references each image/frame in the bundle.
3. **Upload**: Upload only the media bundle and the annotation file instead of millions of individual files.
4. **Use in ML**: After upload, the dataset is ready for training detection, segmentation, or classification models without additional restructuring.

## Repo Structure

```
/compact-uploader/
│
├── output/
│     ├── media_bundle.mp4       # or .zip/.tar when using archives
│     ├── unified_annotation.json
│     └── frame_manifest.json    # frame order emitted by bundle_images.py
│
├── scripts/
│     ├── bundle_images.py        # convert images → media bundle
│     ├── build_annotations.py    # convert label files → unified JSON
│     ├── run_pipeline.py         # orchestrate bundling + annotation in one call
│     └── README_scripts.md       # how to use the scripts
│
├── docs/
│     └── dataset_spec.md         # specification of formats & field definitions
│
└── README.md                      # this file
```

## Usage Instructions

1. Clone the repo:

	```bash
	git clone https://github.com/<your-username>/compact-uploader.git
	cd compact-uploader
	```
2. Place your raw images and corresponding label files into designated folders (e.g., `raw_images/`, `raw_labels/`).
3. Run the bundling script (writes to `output/media_bundle.mp4` by default):

	```bash
	python3 scripts/bundle_images.py --input raw_images/
	```
	Override `--output` if you need a different name or location.
	The command also emits `output/frame_manifest.json`, which records the exact
	ordering of the bundled frames.
4. Run the annotation build script (writes to `output/unified_annotation.json` by default):

	```bash
	python3 scripts/build_annotations.py --images raw_images/ --labels raw_labels/
	```
	Supply `--output` to redirect the final annotation file elsewhere. The tool
	automatically consumes `output/frame_manifest.json` to keep annotation order
	perfectly aligned with the bundled media (use `--skip-manifest` to ignore it).
5. Upload the media bundle and unified annotation file from the `output/`
	directory to your cloud storage. Keep `frame_manifest.json` alongside them if
	downstream consumers need to validate or reconstruct frame ordering.

### One-Step Pipeline

To execute both steps with a single command, run:

```bash
python3 scripts/run_pipeline.py --images raw_images/ --labels raw_labels/
```

The orchestrator forwards relevant options such as `--overwrite`, `--pattern`,
and `--skip-missing-images` to the underlying tools. Use `--allow-missing-manifest`
if you want the run to proceed even when the bundler does not emit a manifest.
When no images are found in the supplied `--images` directory, the runner
automatically creates grayscale placeholder frames (requires `Pillow`). Disable
this behaviour with `--no-auto-generate-placeholders`.

Example using the bundled sample assets:

```bash
python3 scripts/run_pipeline.py --overwrite
```
The `--overwrite` flag refreshes any existing outputs in `output/`.

## Benefits

* Fewer files to upload = faster uploads, less overhead.
* One standard annotation file that’s easy to integrate with ML pipelines.
* Easier versioning, sharing, and storage tracking.

## Considerations

* The media bundle file may still be large; use compression and efficient storage.
* Must ensure correct mapping between media frames (or archive items) and annotation entries.
* If dataset updates frequently, design a strategy for incremental updates instead of full re-upload.
* Downstream ML workflows must support the chosen media and annotation formats.

## My Reflection

I built this pipeline to address a personal challenge of excessive upload time and file-management overhead. Through this project, I learned about dataset engineering, annotation formats (such as COCO), and the trade-offs between many small files vs. fewer larger files. While the current solution is basic, future enhancements may include incremental updates, support for alternative formats (VOC, YOLO), and streaming data support.

---

If you like, I can also provide **template code stubs** (`bundle_images.py`, `build_annotations.py`) in Python to get you started implementing the pipeline.
