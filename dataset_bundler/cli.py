"""
Command-line interface for dataset bundler
"""
import argparse
import sys
from pathlib import Path

from .bundler import DatasetBundler


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bundle images and labels for efficient ML workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bundle as ZIP with YOLO annotations
  dataset-bundler --input ./images --labels ./labels --format yolo --bundle zip
  
  # Bundle as video with YOLO annotations
  dataset-bundler --input ./images --labels ./labels --format yolo --bundle video --fps 30
  
  # Bundle as TAR with COCO annotations
  dataset-bundler --input ./images --coco annotations.json --bundle tar
  
  # Bundle as ZIP with Pascal VOC annotations
  dataset-bundler --input ./images --labels ./labels --format pascal_voc --bundle zip
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for bundles (default: output)"
    )
    
    parser.add_argument(
        "--bundle", "-b",
        choices=["video", "zip", "tar"],
        default="zip",
        help="Type of media bundle to create (default: zip)"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["yolo", "coco", "pascal_voc"],
        default="yolo",
        help="Annotation format (default: yolo)"
    )
    
    parser.add_argument(
        "--labels", "-l",
        help="Directory containing label files (defaults to input directory)"
    )
    
    parser.add_argument(
        "--coco",
        help="Path to COCO format JSON file (required for COCO format)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video bundle (default: 30)"
    )
    
    parser.add_argument(
        "--output-name",
        help="Name for the output bundle file"
    )
    
    parser.add_argument(
        "--annotations-name",
        default="annotations.json",
        help="Name for the output annotations file (default: annotations.json)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.format == "coco" and not args.coco:
        parser.error("--coco is required when using COCO format")
    
    # Create bundler
    try:
        bundler = DatasetBundler(args.input, args.output)
        
        # Prepare kwargs
        kwargs = {}
        if args.output_name:
            kwargs["output_name"] = args.output_name
        if args.bundle == "video":
            kwargs["fps"] = args.fps
        if args.format == "coco":
            kwargs["coco_json_path"] = args.coco
        
        # Bundle the dataset
        result = bundler.bundle_complete_dataset(
            bundle_type=args.bundle,
            annotation_format=args.format,
            labels_dir=args.labels,
            **kwargs
        )
        
        print("\n✓ Success! Dataset bundled successfully.")
        print(f"  Media: {result['media_bundle']}")
        print(f"  Annotations: {result['annotations']}")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
