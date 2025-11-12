# Pipeline Scripts

This directory contains the CLI tools that power the compact image-annotation
upload workflow. Each script prints usage instructions when invoked with
`--help`. Unless overridden, both tools emit their results inside the
repository's `output/` directory.

## bundle_images.py

Bundles a directory of image files into a single archive or MP4 video. The
script sorts image names alphabetically to preserve frame order.

Example:

```bash
python scripts/bundle_images.py --input raw_images
```

Use an `.mp4` extension to produce a video bundle (requires OpenCV and NumPy).
The command also writes `output/frame_manifest.json`, which lists every image in
the exact order it was bundled. Pass `--manifest` to change the location or
`--no-manifest` to disable the file.

## build_annotations.py

Merges [LabelMe](https://github.com/wkentaro/labelme) style JSON files into a
single COCO-style annotation file.

Example:

```bash
python scripts/build_annotations.py --labels sample\ data
```

Use `--image-root` to validate that each JSON file references an available image
on disk. Set `--skip-missing-images` to continue when corresponding images are
absent. The script automatically reads `output/frame_manifest.json` (when
present) to align the unified annotations with the bundled media ordering. Use
`--skip-manifest` to ignore the alignment file or `--require-manifest` to enforce
its presence during processing.

When a manifest references frames that have no LabelMe file, the parser now
creates an `images` entry with the correct `frame_index` and omits annotations
for that slot. These frames are listed under `info.frames_without_labels` inside
the unified JSON output so downstream consumers can track empty positions.

## run_pipeline.py

Runs both steps sequentially. It calls `bundle_images.py` first and then
`build_annotations.py`, ensuring the latter consumes the manifest created by the
former. Example:

```bash
python scripts/run_pipeline.py --images raw_images --labels raw_labels
```

Pass `--overwrite`, `--pattern`, `--fps`, and `--skip-missing-images` as needed.
Use `--allow-missing-manifest` to continue even if the bundling step does not
produce a manifest (alignment will be skipped in that case). When no matching
images exist in the `--images` directory, the runner synthesises placeholder
frames from the label metadata (requires Pillow). Disable this auto behaviour
with `--no-auto-generate-placeholders`.

To exercise the full pipeline with the repository's bundled sample data:

```bash
python scripts/run_pipeline.py --overwrite
```
