"""
Example: Working with Pascal VOC format annotations
"""
from dataset_bundler import DatasetBundler

# Bundle dataset with Pascal VOC annotations
bundler = DatasetBundler(input_dir="./sample_data/images", output_dir="./output")

result = bundler.bundle_complete_dataset(
    bundle_type="tar",
    annotation_format="pascal_voc",
    labels_dir="./sample_data/annotations",
    output_name="voc_dataset.tar.gz"
)

print(f"✓ Media bundle: {result['media_bundle']}")
print(f"✓ Annotations: {result['annotations']}")
