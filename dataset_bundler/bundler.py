"""
Main dataset bundler module
"""
import os
import json
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image

from .annotation_parser import AnnotationParser


class DatasetBundler:
    """Bundle images and labels into efficient formats"""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def __init__(self, input_dir: str, output_dir: str = "output"):
        """
        Initialize the dataset bundler
        
        Args:
            input_dir: Directory containing images and labels
            output_dir: Directory to save output bundles
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = AnnotationParser()
        
    def find_images(self) -> List[Path]:
        """Find all images in the input directory"""
        images = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            images.extend(self.input_dir.glob(f"*{ext}"))
            images.extend(self.input_dir.glob(f"*{ext.upper()}"))
        return sorted(images)
    
    def bundle_as_video(self, output_name: str = "dataset.mp4", fps: int = 30, 
                       target_size: Optional[tuple] = None) -> str:
        """
        Bundle images as a video file
        
        Args:
            output_name: Name of the output video file
            fps: Frames per second
            target_size: Optional (width, height) to resize images
            
        Returns:
            Path to the created video file
        """
        images = self.find_images()
        if not images:
            raise ValueError(f"No images found in {self.input_dir}")
        
        output_path = self.output_dir / output_name
        
        # Get dimensions from first image
        first_img = cv2.imread(str(images[0]))
        if target_size:
            height, width = target_size[1], target_size[0]
        else:
            height, width = first_img.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Creating video bundle with {len(images)} images...")
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}, skipping...")
                continue
            
            # Resize if needed
            if target_size and (img.shape[1] != width or img.shape[0] != height):
                img = cv2.resize(img, (width, height))
            
            video_writer.write(img)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} images...")
        
        video_writer.release()
        print(f"Video bundle created: {output_path}")
        return str(output_path)
    
    def bundle_as_archive(self, output_name: str = "dataset.zip", 
                         archive_type: str = "zip") -> str:
        """
        Bundle images as an archive (ZIP or TAR)
        
        Args:
            output_name: Name of the output archive file
            archive_type: Type of archive ('zip' or 'tar')
            
        Returns:
            Path to the created archive file
        """
        images = self.find_images()
        if not images:
            raise ValueError(f"No images found in {self.input_dir}")
        
        output_path = self.output_dir / output_name
        
        print(f"Creating archive bundle with {len(images)} images...")
        
        if archive_type == "zip":
            with zipfile.ZipFile(str(output_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, img_path in enumerate(images):
                    zipf.write(img_path, arcname=img_path.name)
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(images)} images...")
        
        elif archive_type == "tar":
            with tarfile.open(str(output_path), 'w:gz') as tarf:
                for i, img_path in enumerate(images):
                    tarf.add(img_path, arcname=img_path.name)
                    if (i + 1) % 100 == 0:
                        print(f"Processed {i + 1}/{len(images)} images...")
        else:
            raise ValueError(f"Unsupported archive type: {archive_type}")
        
        print(f"Archive bundle created: {output_path}")
        return str(output_path)
    
    def build_annotations_yolo(self, labels_dir: Optional[str] = None, 
                              output_name: str = "annotations.json") -> str:
        """
        Build annotations from YOLO format labels
        
        Args:
            labels_dir: Directory containing label files (defaults to input_dir)
            output_name: Name of the output JSON file
            
        Returns:
            Path to the created annotations file
        """
        if labels_dir is None:
            labels_dir = self.input_dir
        else:
            labels_dir = Path(labels_dir)
        
        images = self.find_images()
        annotations = []
        
        print(f"Parsing YOLO annotations for {len(images)} images...")
        for i, img_path in enumerate(images):
            # Load image to get dimensions
            img = Image.open(img_path)
            width, height = img.size
            
            # Find corresponding label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            annotation = self.parser.parse_yolo_label(
                str(label_path), img_path.name, width, height
            )
            annotations.append(annotation)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} annotations...")
        
        output_path = self.output_dir / output_name
        metadata = {
            "format": "yolo",
            "total_images": len(images),
            "input_directory": str(self.input_dir)
        }
        self.parser.merge_annotations(annotations, str(output_path), metadata)
        print(f"Annotations file created: {output_path}")
        return str(output_path)
    
    def build_annotations_coco(self, coco_json_path: str, 
                              output_name: str = "annotations.json") -> str:
        """
        Build annotations from COCO format JSON
        
        Args:
            coco_json_path: Path to COCO format JSON file
            output_name: Name of the output JSON file
            
        Returns:
            Path to the created annotations file
        """
        print(f"Parsing COCO annotations from {coco_json_path}...")
        annotations = self.parser.parse_coco_annotations(coco_json_path)
        
        output_path = self.output_dir / output_name
        metadata = {
            "format": "coco",
            "total_images": len(annotations),
            "source_file": coco_json_path
        }
        self.parser.merge_annotations(annotations, str(output_path), metadata)
        print(f"Annotations file created: {output_path}")
        return str(output_path)
    
    def build_annotations_pascal_voc(self, labels_dir: Optional[str] = None,
                                    output_name: str = "annotations.json") -> str:
        """
        Build annotations from Pascal VOC format XML files
        
        Args:
            labels_dir: Directory containing XML label files (defaults to input_dir)
            output_name: Name of the output JSON file
            
        Returns:
            Path to the created annotations file
        """
        if labels_dir is None:
            labels_dir = self.input_dir
        else:
            labels_dir = Path(labels_dir)
        
        images = self.find_images()
        annotations = []
        
        print(f"Parsing Pascal VOC annotations for {len(images)} images...")
        for i, img_path in enumerate(images):
            # Find corresponding XML file
            xml_path = labels_dir / f"{img_path.stem}.xml"
            
            if xml_path.exists():
                annotation = self.parser.parse_pascal_voc_xml(str(xml_path))
                # Update image name if needed
                annotation["image"] = img_path.name
                annotations.append(annotation)
            else:
                # Add image without annotations
                img = Image.open(img_path)
                width, height = img.size
                annotations.append({
                    "image": img_path.name,
                    "width": width,
                    "height": height,
                    "objects": []
                })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(images)} annotations...")
        
        output_path = self.output_dir / output_name
        metadata = {
            "format": "pascal_voc",
            "total_images": len(images),
            "input_directory": str(self.input_dir)
        }
        self.parser.merge_annotations(annotations, str(output_path), metadata)
        print(f"Annotations file created: {output_path}")
        return str(output_path)
    
    def bundle_complete_dataset(self, bundle_type: str = "zip", 
                               annotation_format: str = "yolo",
                               labels_dir: Optional[str] = None,
                               **kwargs) -> Dict[str, str]:
        """
        Complete workflow: Bundle media and build annotations
        
        Args:
            bundle_type: Type of bundle ('video', 'zip', or 'tar')
            annotation_format: Format of annotations ('yolo', 'coco', or 'pascal_voc')
            labels_dir: Directory containing label files
            **kwargs: Additional arguments for bundling
            
        Returns:
            Dictionary with paths to created files
        """
        result = {}
        
        # Bundle media
        if bundle_type == "video":
            result["media_bundle"] = self.bundle_as_video(**kwargs)
        elif bundle_type in ["zip", "tar"]:
            result["media_bundle"] = self.bundle_as_archive(
                archive_type=bundle_type, **kwargs
            )
        else:
            raise ValueError(f"Unsupported bundle type: {bundle_type}")
        
        # Build annotations
        if annotation_format == "yolo":
            result["annotations"] = self.build_annotations_yolo(labels_dir)
        elif annotation_format == "coco":
            if "coco_json_path" not in kwargs:
                raise ValueError("coco_json_path required for COCO format")
            result["annotations"] = self.build_annotations_coco(kwargs["coco_json_path"])
        elif annotation_format == "pascal_voc":
            result["annotations"] = self.build_annotations_pascal_voc(labels_dir)
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
        
        print("\nDataset bundling complete!")
        print(f"Media bundle: {result['media_bundle']}")
        print(f"Annotations: {result['annotations']}")
        
        return result
