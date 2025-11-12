"""
Example: Working with COCO format annotations
"""
from dataset_bundler import DatasetBundler

# Bundle dataset with COCO annotations
bundler = DatasetBundler(input_dir="./sample_data/images", output_dir="./output")

result = bundler.bundle_complete_dataset(
    bundle_type="zip",
    annotation_format="coco",
    coco_json_path="./sample_data/coco_annotations.json",
    output_name="coco_dataset.zip"
)

print(f"✓ Media bundle: {result['media_bundle']}")
print(f"✓ Annotations: {result['annotations']}")
