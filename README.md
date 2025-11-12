# Dataset Bundler for Images and Labels

Uploading annotated images and their label files (potentially millions of files) to the cloud was extremely slow and inefficient. Each image had a corresponding label, and individually handling so many files created delays, management overhead, and version-control complexity.

**Dataset Bundler** solves this problem by combining images and labels into efficient, single-file bundles that are easy to upload, version, and use in ML workflows.

## How It Works

1. **Bundle Media**: Combine images into one large file (either a video or archive format like ZIP/TAR)
2. **Build Annotations**: Parse all individual label files and merge them into a single JSON that references each image/frame in the bundle
3. **Upload**: Upload only the media bundle and the annotation file instead of millions of individual files
4. **Use in ML**: After upload, the dataset is ready for training detection, segmentation, or classification models without additional restructuring

## Features

- 📦 **Multiple Bundle Formats**: Create video bundles (MP4) or archive bundles (ZIP/TAR)
- 🏷️ **Multiple Annotation Formats**: Support for YOLO, COCO, and Pascal VOC formats
- 🚀 **Efficient Storage**: Dramatically reduce file count and improve upload/download speeds
- 🔧 **Easy to Use**: Simple CLI and Python API
- 📊 **ML Ready**: Output format is ready for training without restructuring

## Installation

```bash
# Install from source
git clone https://github.com/zahid-marwat/dataset-bundler-images-labels.git
cd dataset-bundler-images-labels
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

```bash
# Bundle as ZIP with YOLO annotations
dataset-bundler --input ./images --labels ./labels --format yolo --bundle zip

# Bundle as video with YOLO annotations
dataset-bundler --input ./images --labels ./labels --format yolo --bundle video --fps 30

# Bundle with COCO annotations
dataset-bundler --input ./images --coco annotations.json --bundle zip

# Bundle with Pascal VOC annotations
dataset-bundler --input ./images --labels ./labels --format pascal_voc --bundle tar
```

### Python API

```python
from dataset_bundler import DatasetBundler

# Initialize bundler
bundler = DatasetBundler(input_dir="./images", output_dir="./output")

# Bundle as ZIP with YOLO annotations
result = bundler.bundle_complete_dataset(
    bundle_type="zip",
    annotation_format="yolo",
    labels_dir="./labels"
)

# Or bundle as video
result = bundler.bundle_complete_dataset(
    bundle_type="video",
    annotation_format="yolo",
    labels_dir="./labels",
    fps=30
)

print(f"Media bundle: {result['media_bundle']}")
print(f"Annotations: {result['annotations']}")
```

## Supported Formats

### Media Bundle Formats
- **video**: MP4 video file (great for sequential data, easy preview)
- **zip**: ZIP archive (universally compatible)
- **tar**: TAR.GZ archive (efficient for large datasets)

### Annotation Formats
- **YOLO**: Text files with normalized bounding boxes
- **COCO**: JSON format with image and annotation arrays
- **Pascal VOC**: XML files with bounding box coordinates

## Output Structure

The bundler creates two files:
1. **Media Bundle**: Contains all images (as video frames or archive)
2. **Annotations JSON**: Contains all labels in a unified format

Example annotations.json:
```json
{
  "metadata": {
    "format": "yolo",
    "total_images": 1000,
    "input_directory": "./images"
  },
  "annotations": [
    {
      "image": "image001.jpg",
      "width": 1920,
      "height": 1080,
      "objects": [
        {
          "class_id": 0,
          "bbox": [100, 200, 300, 400],
          "bbox_normalized": [0.5, 0.5, 0.2, 0.3]
        }
      ]
    }
  ]
}
```

## Use Cases

- 🌐 **Cloud Upload**: Reduce upload time from hours to minutes
- 🔄 **Version Control**: Track dataset versions with just 2 files
- 📤 **Data Sharing**: Share datasets easily via single download link
- 🎯 **ML Training**: Direct integration with training pipelines
- 💾 **Storage**: Reduce cloud storage costs with efficient bundling

## Example Workflow

```bash
# 1. Organize your data
./my_dataset/
  ├── images/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── labels/
      ├── img1.txt
      ├── img2.txt
      └── ...

# 2. Bundle the dataset
dataset-bundler --input ./my_dataset/images \
                --labels ./my_dataset/labels \
                --format yolo \
                --bundle zip \
                --output ./output

# 3. Upload only 2 files
./output/
  ├── dataset.zip         # All images in one file
  └── annotations.json    # All labels in one file
```

## CLI Options

```
--input, -i       Input directory containing images (required)
--output, -o      Output directory for bundles (default: output)
--bundle, -b      Bundle type: video, zip, tar (default: zip)
--format, -f      Annotation format: yolo, coco, pascal_voc (default: yolo)
--labels, -l      Directory containing label files
--coco            Path to COCO JSON file (for COCO format)
--fps             Frames per second for video (default: 30)
--output-name     Custom name for output bundle
--annotations-name Custom name for annotations file
```

## Requirements

- Python 3.8+
- OpenCV (for video bundling)
- NumPy
- Pillow

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
