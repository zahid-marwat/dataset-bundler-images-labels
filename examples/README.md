# Dataset Bundler Examples

This directory contains example scripts demonstrating how to use the dataset bundler.

## Examples

### basic_usage.py
Demonstrates basic usage with YOLO format annotations:
- Creating ZIP bundles
- Creating video bundles
- Using individual methods

### coco_example.py
Shows how to work with COCO format annotations

### pascal_voc_example.py
Shows how to work with Pascal VOC format annotations

## Running Examples

```bash
# Make sure you have sample data organized
python examples/basic_usage.py
python examples/coco_example.py
python examples/pascal_voc_example.py
```

## Sample Data Structure

Your data should be organized like this:

```
sample_data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/          # For YOLO format
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── annotations/     # For Pascal VOC format
│   ├── image1.xml
│   ├── image2.xml
│   └── ...
└── coco_annotations.json  # For COCO format
```
