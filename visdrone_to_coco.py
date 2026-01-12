#!/usr/bin/env python3
"""
VisDrone2019-DET数据集转换为COCO格式

该脚本用于将VisDrone2019-DET数据集的标注转换为COCO格式，支持训练集和验证集。

VisDrone标注格式：
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

COCO格式输出：
包含images、annotations和categories三个主要部分

使用方法：
python visdrone_to_coco.py --train-img-dir /path/to/train/images --train-ann-dir /path/to/train/annotations --val-img-dir /path/to/val/images --val-ann-dir /path/to/val/annotations --output-dir /path/to/output
"""

import os
import json
import argparse
from PIL import Image
import numpy as np

# VisDrone2019-DET类别映射到COCO类别
# VisDrone类别定义：
# 0: pedestrian
# 1: people
# 2: bicycle
# 3: car
# 4: van
# 5: truck
# 6: tricycle
# 7: awning-tricycle
# 8: bus
# 9: motor
# 10: others
VISDRONE_TO_COCO_CATEGORY = {
    0: {"id": 1, "name": "pedestrian", "supercategory": "person"},
    1: {"id": 2, "name": "people", "supercategory": "person"},
    2: {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
    3: {"id": 4, "name": "car", "supercategory": "vehicle"},
    4: {"id": 5, "name": "van", "supercategory": "vehicle"},
    5: {"id": 6, "name": "truck", "supercategory": "vehicle"},
    6: {"id": 7, "name": "tricycle", "supercategory": "vehicle"},
    7: {"id": 8, "name": "awning-tricycle", "supercategory": "vehicle"},
    8: {"id": 9, "name": "bus", "supercategory": "vehicle"},
    9: {"id": 10, "name": "motor", "supercategory": "vehicle"}
}

# 排除的类别
EXCLUDED_CATEGORIES = [10]  # others类别

class VisDroneToCOCOConverter:
    """VisDrone数据集到COCO格式的转换器"""
    
    def __init__(self):
        self.categories = self._get_categories()
    
    def _get_categories(self):
        """获取COCO类别列表"""
        categories = []
        for cat_id in VISDRONE_TO_COCO_CATEGORY.values():
            categories.append(cat_id)
        return categories
    
    def _get_image_info(self, img_path, img_id):
        """获取图像信息"""
        with Image.open(img_path) as img:
            width, height = img.size
        
        img_name = os.path.basename(img_path)
        
        return {
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "2019-01-01T00:00:00"
        }
    
    def _parse_visdrone_annotation(self, ann_line, img_id, ann_id, img_width, img_height):
        """解析VisDrone标注行"""
        parts = ann_line.strip().split(',')
        if len(parts) != 8:
            raise ValueError(f"Invalid annotation line: {ann_line}")
        
        bbox_left = int(parts[0])
        bbox_top = int(parts[1])
        bbox_width = int(parts[2])
        bbox_height = int(parts[3])
        score = float(parts[4])
        category_id = int(parts[5])
        truncation = int(parts[6])
        occlusion = int(parts[7])
        
        # 跳过排除的类别
        if category_id in EXCLUDED_CATEGORIES:
            return None
        
        # 计算边界框坐标 (x, y, width, height) -> COCO格式
        # 确保边界框不超出图像范围
        x1 = max(0, bbox_left)
        y1 = max(0, bbox_top)
        x2 = min(img_width, bbox_left + bbox_width)
        y2 = min(img_height, bbox_top + bbox_height)
        
        # 计算宽高
        width = x2 - x1
        height = y2 - y1
        
        # 如果宽高为0，跳过
        if width <= 0 or height <= 0:
            return None
        
        # 映射到COCO类别ID
        coco_cat_id = VISDRONE_TO_COCO_CATEGORY[category_id]["id"]
        
        return {
            "id": ann_id,
            "image_id": img_id,
            "category_id": coco_cat_id,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
            "score": score,
            "truncation": truncation,
            "occlusion": occlusion
        }
    
    def convert(self, img_dir, ann_dir, output_file):
        """转换VisDrone数据集到COCO格式"""
        print(f"转换数据集：{img_dir} -> {output_file}")
        
        # 获取所有图像文件
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.jpeg')])
        print(f"找到 {len(img_files)} 张图像")
        
        # 初始化COCO数据结构
        coco_data = {
            "info": {
                "year": 2019,
                "version": "1.0",
                "description": "VisDrone2019-DET Dataset in COCO format",
                "contributor": "VisDrone Dataset Team",
                "url": "http://aiskyeye.com/",
                "date_created": "2019-01-01"
            },
            "licenses": [{
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
                "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"
            }],
            "categories": self.categories,
            "images": [],
            "annotations": []
        }
        
        img_id = 1
        ann_id = 1
        
        for img_file in img_files:
            # 图像信息
            img_path = os.path.join(img_dir, img_file)
            img_info = self._get_image_info(img_path, img_id)
            coco_data["images"].append(img_info)
            
            # 对应的标注文件
            ann_file = os.path.splitext(img_file)[0] + '.txt'
            ann_path = os.path.join(ann_dir, ann_file)
            
            if not os.path.exists(ann_path):
                print(f"警告：找不到标注文件 {ann_path}，跳过")
                img_id += 1
                continue
            
            # 读取并解析标注
            with open(ann_path, 'r') as f:
                ann_lines = f.readlines()
            
            for ann_line in ann_lines:
                ann = self._parse_visdrone_annotation(
                    ann_line, img_id, ann_id, img_info["width"], img_info["height"]
                )
                if ann:
                    coco_data["annotations"].append(ann)
                    ann_id += 1
            
            img_id += 1
        
        # 保存COCO格式标注
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"转换完成，生成 {output_file}")
        print(f"图像数量：{len(coco_data['images'])}")
        print(f"标注数量：{len(coco_data['annotations'])}")
        
        return coco_data
    
    def validate_coco_data(self, coco_data):
        """验证COCO数据格式"""
        print("\n验证COCO数据格式...")
        
        # 检查必要字段
        required_fields = ["info", "licenses", "categories", "images", "annotations"]
        for field in required_fields:
            if field not in coco_data:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 检查图像ID唯一性
        img_ids = [img["id"] for img in coco_data["images"]]
        if len(img_ids) != len(set(img_ids)):
            raise ValueError("图像ID存在重复")
        
        # 检查标注ID唯一性
        ann_ids = [ann["id"] for ann in coco_data["annotations"]]
        if len(ann_ids) != len(set(ann_ids)):
            raise ValueError("标注ID存在重复")
        
        # 检查标注是否引用了有效的图像ID
        img_id_set = set(img_ids)
        for ann in coco_data["annotations"]:
            if ann["image_id"] not in img_id_set:
                raise ValueError(f"标注引用了无效的图像ID: {ann['image_id']}")
        
        # 检查标注是否引用了有效的类别ID
        cat_ids = set(cat["id"] for cat in coco_data["categories"])
        for ann in coco_data["annotations"]:
            if ann["category_id"] not in cat_ids:
                raise ValueError(f"标注引用了无效的类别ID: {ann['category_id']}")
        
        # 检查边界框是否合法
        for ann in coco_data["annotations"]:
            bbox = ann["bbox"]
            if len(bbox) != 4:
                raise ValueError(f"边界框格式错误: {bbox}")
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"边界框宽高必须大于0: {bbox}")
        
        print("COCO数据格式验证通过！")
        return True
    
    def convert_dataset(self, train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, output_dir):
        """转换训练集和验证集"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 转换训练集
        train_output = os.path.join(output_dir, "visdrone2019_det_train.json")
        train_data = self.convert(train_img_dir, train_ann_dir, train_output)
        self.validate_coco_data(train_data)
        
        # 转换验证集
        val_output = os.path.join(output_dir, "visdrone2019_det_val.json")
        val_data = self.convert(val_img_dir, val_ann_dir, val_output)
        self.validate_coco_data(val_data)
        
        print("\n所有数据集转换完成！")


def main():
    parser = argparse.ArgumentParser(description="VisDrone2019-DET to COCO format converter")
    parser.add_argument("--train-img-dir", required=True, help="训练集图像目录")
    parser.add_argument("--train-ann-dir", required=True, help="训练集标注目录")
    parser.add_argument("--val-img-dir", required=True, help="验证集图像目录")
    parser.add_argument("--val-ann-dir", required=True, help="验证集标注目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    
    args = parser.parse_args()
    
    converter = VisDroneToCOCOConverter()
    converter.convert_dataset(
        args.train_img_dir,
        args.train_ann_dir,
        args.val_img_dir,
        args.val_ann_dir,
        args.output_dir
    )


if __name__ == "__main__":
    main()
