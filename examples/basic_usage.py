"""
Example: Bundle a dataset with YOLO annotations
"""
from dataset_bundler import DatasetBundler

# Example 1: Bundle as ZIP with YOLO annotations
print("Example 1: Creating ZIP bundle with YOLO annotations")
bundler = DatasetBundler(input_dir="./sample_data/images", output_dir="./output")

result = bundler.bundle_complete_dataset(
    bundle_type="zip",
    annotation_format="yolo",
    labels_dir="./sample_data/labels",
    output_name="my_dataset.zip"
)

print(f"✓ Media bundle created: {result['media_bundle']}")
print(f"✓ Annotations created: {result['annotations']}")


# Example 2: Bundle as video
print("\nExample 2: Creating video bundle")
result = bundler.bundle_complete_dataset(
    bundle_type="video",
    annotation_format="yolo",
    labels_dir="./sample_data/labels",
    output_name="my_dataset.mp4",
    fps=30
)

print(f"✓ Video bundle created: {result['media_bundle']}")
print(f"✓ Annotations created: {result['annotations']}")


# Example 3: Using individual methods
print("\nExample 3: Using individual methods")
bundler = DatasetBundler(input_dir="./sample_data/images", output_dir="./output")

# Bundle media separately
media_path = bundler.bundle_as_archive(output_name="images_only.zip")
print(f"✓ Images bundled: {media_path}")

# Build annotations separately
annotations_path = bundler.build_annotations_yolo(
    labels_dir="./sample_data/labels",
    output_name="labels_only.json"
)
print(f"✓ Annotations built: {annotations_path}")
