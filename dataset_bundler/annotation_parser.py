"""
Annotation parser for various label formats
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class AnnotationParser:
    """Parse and merge annotations from various formats"""
    
    def __init__(self):
        self.annotations = []
        
    def parse_yolo_label(self, label_path: str, image_name: str, image_width: int, image_height: int) -> Dict[str, Any]:
        """
        Parse YOLO format label file
        Format: class_id x_center y_center width height (normalized 0-1)
        """
        objects = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to absolute coordinates
                        x_min = int((x_center - width / 2) * image_width)
                        y_min = int((y_center - height / 2) * image_height)
                        x_max = int((x_center + width / 2) * image_width)
                        y_max = int((y_center + height / 2) * image_height)
                        
                        objects.append({
                            "class_id": class_id,
                            "bbox": [x_min, y_min, x_max, y_max],
                            "bbox_normalized": [x_center, y_center, width, height]
                        })
        
        return {
            "image": image_name,
            "width": image_width,
            "height": image_height,
            "objects": objects
        }
    
    def parse_coco_annotations(self, coco_json_path: str) -> List[Dict[str, Any]]:
        """
        Parse COCO format JSON annotations
        """
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        image_map = {img['id']: img for img in coco_data.get('images', [])}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            image_annotations[image_id].append({
                "class_id": ann['category_id'],
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": ann.get('area', w * h),
                "segmentation": ann.get('segmentation', None)
            })
        
        # Build result
        result = []
        for image_id, image_info in image_map.items():
            result.append({
                "image": image_info['file_name'],
                "width": image_info['width'],
                "height": image_info['height'],
                "objects": image_annotations.get(image_id, [])
            })
        
        return result
    
    def parse_pascal_voc_xml(self, xml_path: str) -> Dict[str, Any]:
        """
        Parse Pascal VOC format XML label file
        Note: This is a simplified parser that extracts basic bounding box info
        """
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find('filename').text if root.find('filename') is not None else ""
        size = root.find('size')
        width = int(size.find('width').text) if size is not None else 0
        height = int(size.find('height').text) if size is not None else 0
        
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            if bndbox is not None:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                objects.append({
                    "class_name": name,
                    "bbox": [xmin, ymin, xmax, ymax]
                })
        
        return {
            "image": filename,
            "width": width,
            "height": height,
            "objects": objects
        }
    
    def merge_annotations(self, annotations: List[Dict[str, Any]], output_path: str, metadata: Optional[Dict] = None):
        """
        Merge all annotations into a single JSON file
        """
        output_data = {
            "metadata": metadata or {},
            "annotations": annotations
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_path
    
    def add_annotation(self, annotation: Dict[str, Any]):
        """Add a single annotation to the collection"""
        self.annotations.append(annotation)
    
    def get_annotations(self) -> List[Dict[str, Any]]:
        """Get all collected annotations"""
        return self.annotations
    
    def clear_annotations(self):
        """Clear all collected annotations"""
        self.annotations = []
