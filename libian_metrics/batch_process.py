"""
Batch processing module for analyzing character glyphs in organized folder structures.

This module provides functionality to process multiple characters, each with multiple
glyph images, and compute average metrics for each character.

Module: libian_metrics.batch_process
Author: LiShift Team
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from .io_utils import read_image
from .preprocess import preprocess
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def process_character_folder(
    char_folder: str,
    char_name: str,
    calib: Dict = None
) -> Tuple[Dict, List[str]]:
    """
    Process all images in a character folder and compute average metrics.
    
    处理单个字的所有图片，并计算平均指标。
    
    Args:
        char_folder (str): Path to folder containing images of a single character.
                          单字文件夹路径
        char_name (str): Name of the character (e.g., "乙", "甲").
                        字的名称
        calib (dict): Calibration parameters. If None, use defaults.
                     校准参数，None则使用默认值
    
    Returns:
        tuple: (averaged_metrics_dict, list_of_image_paths)
               (平均指标字典, 处理的图片路径列表)
    
    Example:
        >>> metrics, images = process_character_folder("data/dataset/乙", "乙")
        >>> print(f"Character 乙 LQI: {metrics['LQI']:.2f}")
    """
    if calib is None:
        calib = {}
    
    # 支持的图片格式
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # 收集所有图片文件
    image_files = []
    if os.path.isdir(char_folder):
        for file in os.listdir(char_folder):
            if os.path.splitext(file)[1].lower() in supported_formats:
                image_files.append(os.path.join(char_folder, file))
    
    if not image_files:
        logger.warning(f"No images found in {char_folder}")
        return None, []
    
    # 处理每张图片
    all_metrics = []
    processed_images = []
    quality_flags = []
    
    for img_path in sorted(image_files):
        try:
            img = read_image(img_path)
            if img is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            
            bin_img, skel, meta = preprocess(img)
            metrics = compute_all_metrics(bin_img, skel, calib)
            all_metrics.append(metrics)
            processed_images.append(img_path)
            quality_flags.append(bool(meta.get('quality_flag', False)))
            
            logger.info(f"✓ Processed: {os.path.basename(img_path)} "
                       f"(LQI={metrics['LQI']:.3f})")
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    if not all_metrics:
        logger.warning(f"No metrics computed for character {char_name}")
        return None, []
    
    # 计算平均值
    avg_metrics = {}
    metric_keys = all_metrics[0].keys()
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
        if values:
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f"{key}_std"] = float(np.std(values))
    
    avg_metrics['char'] = char_name
    avg_metrics['image_count'] = len(processed_images)
    # Quality counts are additive fields; all successfully processed images
    # remain included so existing batch behavior and scores are unchanged.
    avg_metrics['quality_pass_count'] = sum(quality_flags)
    avg_metrics['quality_fail_count'] = len(quality_flags) - sum(quality_flags)
    
    return avg_metrics, processed_images


def process_dataset_folder(
    dataset_folder: str,
    calib: Dict = None,
    output_json: str = None
) -> Dict:
    """
    Process entire dataset folder with organized character subfolders.
    
    处理整个数据集文件夹，其中包含多个字的子文件夹。
    
    Args:
        dataset_folder (str): Path to dataset folder structure:
                            数据集文件夹路径，结构为：
                            
                            dataset_folder/
                            ├── 字1/
                            │   ├── image1.jpg
                            │   └── image2.jpg
                            ├── 字2/
                            │   └── ...
                            └── ...
        
        calib (dict): Calibration parameters. If None, use defaults.
                     校准参数
        
        output_json (str): Path to save results JSON. If None, don't save.
                          输出JSON文件路径
    
    Returns:
        dict: Results dictionary with structure:
             结果字典，结构为：
             {
                 "dataset_name": "folder_name",
                 "timestamp": "2024-01-06T23:30:00",
                 "characters": {
                     "字1": {
                         "SSI": 0.71,
                         "SSI_std": 0.05,
                         ...
                         "image_count": 5
                     },
                     ...
                 },
                 "summary": {
                     "total_characters": 10,
                     "total_images": 45,
                     "average_LQI": 0.65
                 }
             }
    
    Example:
        >>> results = process_dataset_folder("data/oracle_bones")
        >>> print(f"Processed {results['summary']['total_characters']} characters")
    """
    if calib is None:
        calib = {}
    
    dataset_folder = os.path.abspath(dataset_folder)
    dataset_name = os.path.basename(dataset_folder)
    
    if not os.path.isdir(dataset_folder):
        raise ValueError(f"Dataset folder not found: {dataset_folder}")
    
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Dataset path: {dataset_folder}")
    print(f"\n{'='*60}")
    print(f"📂 Dataset: {dataset_name}")
    print(f"{'='*60}\n")
    
    # 收集字文件夹
    char_folders = []
    for item in os.listdir(dataset_folder):
        item_path = os.path.join(dataset_folder, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            char_folders.append((item, item_path))
    
    char_folders.sort()
    
    if not char_folders:
        raise ValueError(f"No character folders found in {dataset_folder}")
    
    # 处理每个字
    results = {
        'dataset_name': dataset_name,
        'dataset_path': dataset_folder,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'characters': {}
    }
    
    total_images = 0
    total_quality_pass = 0
    total_quality_fail = 0
    lqi_scores = []
    
    for char_name, char_folder in char_folders:
        print(f"Processing character: {char_name}")
        avg_metrics, image_paths = process_character_folder(
            char_folder, char_name, calib
        )
        
        if avg_metrics:
            results['characters'][char_name] = avg_metrics
            total_images += len(image_paths)
            total_quality_pass += avg_metrics.get('quality_pass_count', 0)
            total_quality_fail += avg_metrics.get('quality_fail_count', 0)
            if 'LQI' in avg_metrics:
                lqi_scores.append(avg_metrics['LQI'])
            
            print(f"  ✓ {char_name}: LQI={avg_metrics['LQI']:.3f} "
                  f"({len(image_paths)} images)")
        else:
            print(f"  ✗ {char_name}: No valid images")
        print()
    
    # 汇总统计
    results['summary'] = {
        'total_characters': len([c for c in results['characters'] if results['characters'][c]]),
        'total_images': total_images,
        'quality_pass_count': total_quality_pass,
        'quality_fail_count': total_quality_fail,
        'average_LQI': float(np.mean(lqi_scores)) if lqi_scores else None,
        'lqi_min': float(np.min(lqi_scores)) if lqi_scores else None,
        'lqi_max': float(np.max(lqi_scores)) if lqi_scores else None,
    }
    
    # 保存结果
    if output_json:
        output_json = os.path.abspath(output_json)
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_json}")
        print(f"✅ Results saved to: {output_json}")
    
    return results


def print_results(results: Dict, detailed: bool = False) -> None:
    """
    Print results in a formatted table.
    
    以表格格式打印结果。
    
    Args:
        results (dict): Results dictionary from process_dataset_folder
        detailed (bool): If True, show detailed metrics for each character
                        如果为True，显示每个字的详细指标
    """
    print(f"\n{'='*70}")
    print(f"📊 Dataset: {results['dataset_name']}")
    print(f"{'='*70}")
    
    summary = results.get('summary', {})
    print(f"\n📈 Summary:")
    print(f"  Total Characters: {summary.get('total_characters', 0)}")
    print(f"  Total Images: {summary.get('total_images', 0)}")
    print(f"  Average LQI: {summary.get('average_LQI', 0):.3f}")
    print(f"  LQI Range: {summary.get('lqi_min', 0):.3f} - {summary.get('lqi_max', 0):.3f}")
    
    if detailed:
        print(f"\n{'='*70}")
        print(f"{'Character':<10} {'LQI':<8} {'SSI':<8} {'GCP':<8} {'SSD':<8} "
              f"{'STR':<8} {'CSI':<8} {'COI':<8} {'Count':<8}")
        print(f"{'-'*70}")
        
        for char_name, metrics in sorted(results['characters'].items()):
            if metrics:
                print(f"{char_name:<10} "
                      f"{metrics.get('LQI', 0):<8.3f} "
                      f"{metrics.get('SSI', 0):<8.3f} "
                      f"{metrics.get('GCP', 0):<8.3f} "
                      f"{metrics.get('SSD', 0):<8.3f} "
                      f"{metrics.get('STR', 0):<8.3f} "
                      f"{metrics.get('CSI', 0):<8.3f} "
                      f"{metrics.get('COI', 0):<8.3f} "
                      f"{metrics.get('image_count', 0):<8}")
    
    print(f"\n{'='*70}\n")
