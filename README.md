# LiBian Metrics / 古文字质量评估工具# LiBian Metrics



[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)**LiBian Metrics** is a Python toolkit for quantitative assessment of ancient character (glyph) image quality. It provides six key metrics and a weighted composite score (LQI) to evaluate the visual characteristics of historical character forms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

**LiBian Metrics** is a Python toolkit for quantitative quality assessment of ancient character (glyph) images. It provides 6 quantitative metrics and a weighted composite quality index (LQI) for analyzing character shape, stroke patterns, and spatial distribution.

### Six Quantitative Metrics

**LiBian指标工具**是用于古文字字形图片量化质量评估的Python工具包。它提供6个量化指标和一个加权综合分（LQI），用于分析字的结体布局、笔画笔势和空间分布。

#### Layout Metrics (总权重 0.55)

## 📋 Features / 功能特性- **SSI (Shape Squareness Index)** [0.20]: Measures how square-shaped the outer contour is

- **GCP (Global Centering of Mass)** [0.10]: Evaluates proximity of foreground center of mass to bounding box center

### Supported Metrics / 支持的指标- **SSD (Spatial Sparsity-Dispersion)** [0.25]: Assesses uniformity of foreground distribution across space



#### Layout & Shape Metrics / 结体布局类 (Weight: 55%)#### Stroke Metrics (总权重 0.45)

- **SSI (Shape Squareness Index)** - 外部轮廓方整度- **STR (Straightness Ratio)** [0.15]: Proportion of straight stroke segments detected via Hough lines

  - Measures how square/rectangular the character outline is- **CSI (Corner Sharpness Index)** [0.15]: Intensity of corner/junction sharpness in strokes

  - 衡量字的外轮廓的方正程度- **COI (Connectivity & Overlap Index)** [0.15]: Complexity of branching and loop structures



- **GCP (Global Centering of Mass)** - 重心居中度### Composite Score

  - Evaluates how centered the character weight distribution is- **LQI (Libian Quality Index)**: Weighted average of all six metrics

  - 评估字的重心位置的居中度  ```

  LQI = 0.20·SSI + 0.10·GCP + 0.25·SSD + 0.15·STR + 0.15·CSI + 0.15·COI

- **SSD (Spatial Sparsity-Dispersion)** - 空间疏密离散度  ```

  - Analyzes the balance of pixel distribution across the character

  - 分析字内像素分布的疏密平衡度## Installation



#### Stroke & Trajectory Metrics / 笔画笔势类 (Weight: 45%)### Prerequisites

- **STR (Straightness Ratio)** - 直线化比例- Python 3.10+

  - Measures the ratio of straight strokes vs curves- pip

  - 衡量笔画的直线化程度

### From Source

- **CSI (Corner Sharpness Index)** - 方折尖锐度

  - Evaluates corner sharpness and angular connections```bash

  - 评估角点的尖锐程度和方折特征cd LiShift

pip install -e .

- **COI (Connectivity & Overlap Index)** - 连接/交重复合指数```

  - Analyzes branching density and intersection patterns

  - 分析笔画分叉和交重的特征This installs the package in editable mode along with all dependencies:

- opencv-python

### Composite Score / 综合分数- numpy

- **LQI (Libian Quality Index)** - 加权综合分- scikit-image

  - Weighted combination of all 6 metrics- scikit-learn

  - 所有指标的加权综合分- joblib

- pandas

## 🚀 Quick Start / 快速开始

## Quick Start

### Installation / 安装

### Command Line Usage

```bash

cd /path/to/LiShift#### Single Image Analysis

pip install -e .```bash

```python -m libian_metrics --image path/to/glyph.jpg

```

Or install dependencies directly:

```bashOutput (JSON):

pip install opencv-python numpy scikit-image scikit-learn joblib pandas matplotlib```json

```{

  "image": "path/to/glyph.jpg",

### Usage Modes / 使用模式  "SSI": 0.71,

  "GCP": 0.88,

#### Mode 1: Single Image Processing / 单图处理  "SSD": 0.63,

  "STR": 0.76,

Process a single character image:  "CSI": 0.54,

```bash  "COI": 0.41,

python -m libian_metrics --image path/to/char.jpg  "LQI": 0.64,

```  "angle": -1.8,

  "bbox": [12, 245, 30, 238],

With calibration and output:  "scale": 0.78,

```bash  "quality_flag": true,

python -m libian_metrics --image char.jpg --calib calibration.json --out result.json  "skel_pixels": 1245,

```  "max_component_area": 8920

}

#### Mode 2: Batch Dataset Processing / 批量数据集处理```



Process an entire dataset folder with organized character subfolders:#### With Calibration

```bash

```bashpython -m libian_metrics --image glyph.jpg --calib calibration.json --out result.json

python -m libian_metrics --dataset data/my_dataset --out results/output.json```

```

#### Batch CSV Output

With calibration and detailed metrics:```bash

```bashpython -m libian_metrics --image glyph1.jpg --csv results.csv

python -m libian_metrics --dataset data/my_dataset --calib calibration.json --out results/output.json --detailedpython -m libian_metrics --image glyph2.jpg --csv results.csv

```# Each call appends to results.csv

```

## 📁 Dataset Structure / 数据集结构

#### All Options

For batch processing, organize your data as follows:```bash

```python -m libian_metrics --help

data/```

└── dataset_name/

    ├── 字1/```

    │   ├── image1.jpgOptions:

    │   ├── image2.jpg  --image PATH           Path to input image (JPG/PNG) [required]

    │   └── image3.png  --calib PATH           Path to calibration JSON file [optional]

    ├── 字2/  --out PATH             Output JSON file path (default: stdout)

    │   ├── image1.jpg  --csv PATH             Output CSV file path (appends if exists)

    │   └── image2.jpg  --debug                Enable debug output

    ├── 字3/```

    │   └── image1.jpg

    └── ...more characters...### Python API

```

```python

**Important notes:**from libian_metrics import preprocess, compute_all_metrics

- Each character has its own folder named with the character (e.g., "甲", "乙", "丙")from libian_metrics.io_utils import read_image

- Images in each folder will be processed and averaged to get the character's metricsimport json

- Supported formats: JPG, JPEG, PNG, BMP, TIFF

- Results will show average metrics ± standard deviation for each character# Load and preprocess image

img = read_image('glyph.jpg')

**重要说明:**bin_img, skel, meta = preprocess(img, target_height=256)

- 每个字的图片放在以该字命名的文件夹中（如"甲", "乙", "丙"）

- 系统会处理每个字的所有图片，并计算平均指标# Compute metrics

- 支持格式：JPG, JPEG, PNG, BMP, TIFFmetrics = compute_all_metrics(bin_img, skel)

- 结果显示每个字的平均指标 ± 标准差

# With calibration

## 📊 Output Format / 输出格式import json

with open('calibration.json') as f:

### Batch Processing Output / 批量处理输出    calib = json.load(f)

metrics = compute_all_metrics(bin_img, skel, calib)

**Single Character Result (单字结果):**

```jsonprint(f"LQI Score: {metrics['LQI']:.3f}")

{```

  "char": "甲",

  "SSI": 0.71,## Calibration

  "SSI_std": 0.05,

  "GCP": 0.88,Generate calibration parameters from a set of sample images:

  "GCP_std": 0.03,

  "SSD": 0.63,```bash

  "SSD_std": 0.08,python -c "

  "STR": 0.76,from libian_metrics.calibrate import calibrate_from_folder, save_calibration

  "STR_std": 0.06,calib = calibrate_from_folder('path/to/samples', sample_n=50)

  "CSI": 0.54,save_calibration(calib, 'calibration.json')

  "CSI_std": 0.07,"

  "COI": 0.41,```

  "COI_std": 0.09,

  "LQI": 0.64,Or use the calibration script:

  "LQI_std": 0.06,

  "image_count": 5```python

}from libian_metrics.calibrate import calibrate_from_folder, save_calibration

```

# Calibrate from sample images

**Batch Dataset Result (数据集结果):**calib = calibrate_from_folder(

```json    'samples/',

{    sample_n=100,

  "dataset_name": "my_dataset",    r_cap_percentile=99.0,

  "dataset_path": "/path/to/data/my_dataset",    c_cap_percentile=95.0

  "timestamp": "2024-01-06T23:30:00",)

  "characters": {

    "甲": {# Save calibration

      "SSI": 0.71,save_calibration(calib, 'calibration.json')

      "SSI_std": 0.05,```

      ...

      "LQI": 0.64,The calibration process:

      "image_count": 51. Loads all images from the specified folder

    },2. Extracts statistical distributions of key parameters

    "乙": {3. Computes percentiles for normalization

      "SSI": 0.68,4. Generates `calibration.json` with optimized parameters

      ...

    }## Project Structure

  },

  "summary": {```

    "total_characters": 10,LiShift/

    "total_images": 45,├── libian_metrics/

    "average_LQI": 0.65,│   ├── __init__.py          # Package initialization

    "lqi_min": 0.52,│   ├── __main__.py          # Module entry point

    "lqi_max": 0.78│   ├── cli.py               # Command-line interface

  }│   ├── io_utils.py          # I/O and utility functions

}│   ├── preprocess.py        # Preprocessing pipeline

```│   ├── skeleton.py          # Skeleton extraction

│   ├── metrics.py           # Core metric computations

## 🛠️ Advanced Usage / 高级用法│   └── calibrate.py         # Calibration utilities

├── tests/

### Command-line Options / 命令行选项│   ├── __init__.py

│   ├── test_metrics.py      # Unit tests

```bash│   ├── generate_samples.py  # Sample image generator

python -m libian_metrics --help│   └── sample_images/       # Test images

```├── runs/                    # Debug outputs (optional)

├── setup.py

**Common options:**├── pyproject.toml

- `--image PATH`: Input image file path└── README.md

- `--dataset PATH`: Dataset folder with character subfolders```

- `--calib PATH`: Calibration JSON file for custom parameters

- `--out PATH`: Output JSON file path## Preprocessing Pipeline

- `--csv PATH`: Output CSV file path (single image mode)

- `--detailed`: Show detailed metrics table in consoleThe preprocessing module includes:

- `--visualize`: Generate visualization charts

- `--viz-dir PATH`: Directory to save visualizations (default: runs/)1. **Grayscale Conversion**: BGR → Grayscale

- `--debug`: Enable debug output2. **Adaptive Binarization**: `cv2.adaptiveThreshold` (GAUSSIAN_C, blockSize=35)

3. **Component Filtering**: Remove small connected components (< 0.02% of image area)

### Creating Custom Calibration / 创建自定义校准文件4. **Skew Correction**: Estimate and correct rotation angle (±5°)

5. **Rescaling**: Normalize to target height (default 256px)

Default calibration values are used if no calibration file is provided.6. **Skeletonization**: Extract skeleton using `skimage.morphology.skeletonize`

7. **Spur Pruning**: Remove short branches (< 6px)

To create a custom calibration file:8. **Quality Checks**: Flag images with insufficient content

```bash

python -c "## Metrics Details

import json

calib = {### SSI (Shape Squareness Index)

    'r_cap': 3.0,      # Shape squareness cap- Uses minimum area rotated rectangle from largest component

    'c_cap': 1.0,      # Sparsity cap- Aspect ratio: `r = max(w,h) / min(w,h)`

    # Add more parameters as needed- Formula: `SSI = 1 - |log(r)| / |log(r_cap)|`

}- Higher values indicate more square-shaped characters

with open('calibration.json', 'w') as f:

    json.dump(calib, f, indent=2)### GCP (Global Centering of Mass)

"- Compares center of mass with bounding box center

```- Formula: `GCP = 1 - distance / diagonal`

- Closer to 1.0 indicates better centering

## 📈 Processing Examples / 处理示例

### SSD (Spatial Sparsity-Dispersion)

### Example 1: Process Single Image- Divides image into 5×5 grid

```bash- Calculates coefficient of variation (CV) of pixel density

python -m libian_metrics --image sample.jpg --out sample_result.json- Formula: `SSD = 1 - min(CV, c_cap) / c_cap`

```- Higher values indicate more uniform distribution



### Example 2: Batch Process Oracle Bone Inscriptions### STR (Straightness Ratio)

```bash- Uses probabilistic Hough line detection on skeleton

# Assuming data/oracle_bones/ contains character subfolders- Filters by dominant angles (0°, 45°, 90°, 135°)

python -m libian_metrics --dataset data/oracle_bones --out results/oracle_results.json --detailed- Formula: `STR = straight_pixels / total_skeleton_pixels`

```- Higher values indicate more linear strokes



### Example 3: Batch Process with Visualization### CSI (Corner Sharpness Index)

```bash- Analyzes turning angles in skeleton paths

python -m libian_metrics \- Identifies sharp corners (angle changes > threshold)

  --dataset data/bronze_inscriptions \- Combines angle sharpness with corner density

  --calib calibration.json \- Higher values indicate sharper, more angular characters

  --out results/bronze_results.json \

  --visualize \### COI (Connectivity & Overlap Index)

  --viz-dir results/viz- Counts branching points (degree ≥ 3)

```- Estimates loops and overlapping regions

- Formula: `COI = α·branch_density + (1-α)·loop_density`

## 🔍 Understanding the Metrics / 指标解释- Higher values indicate more complex connectivity



### SSI (0-1) / 方整度## Quality Flags

- **1.0**: Perfect square/rectangle outline

- **0.5-0.8**: Normal characters with varied aspect ratioImages are marked with `quality_flag=false` if:

- **<0.5**: Very elongated or irregular outlines- Largest component area < 1% of image area

- Skeleton pixels < 200

### GCP (0-1) / 重心居中度- Hough line coverage < 5% of skeleton (noise indicator)

- **1.0**: Weight perfectly centered

- **0.7-0.9**: Well-centered characterSuch images should be filtered out in downstream analysis.

- **<0.7**: Off-center weight distribution

## Testing

### SSD (0-1) / 疏密离散度

- **1.0**: Perfect uniform pixel distribution### Generate Sample Images

- **0.6-0.8**: Well-balanced density```bash

- **<0.6**: Highly variable pixel distributioncd tests

python generate_samples.py

### STR (0-1) / 直线化比例```

- **1.0**: Mostly straight strokes

- **0.5-0.8**: Mix of straight and curved strokesThis creates three sample images:

- **<0.5**: Mostly curved strokes- `sample_1.png`: Well-formed square character

- `sample_2.png`: Character with regular strokes

### CSI (0-1) / 方折尖锐度- `sample_3.png`: Asymmetric character

- **1.0**: Very sharp corners

- **0.4-0.7**: Normal corner sharpness### Run Unit Tests

- **<0.4**: Rounded, smooth connections```bash

python -m pytest tests/ -v

### COI (0-1) / 连接交重度```

- **0.7-1.0**: High connectivity/overlap

- **0.3-0.7**: Medium connectivityOr:

- **<0.3**: Low connectivity```bash

python -m unittest discover tests/ -v

## 📝 Configuration / 配置说明```



### Default Parameters / 默认参数### End-to-End Test

```bash

| Parameter | Default | Description / 说明 |# Single image

|-----------|---------|-------------|python -m libian_metrics --image tests/sample_images/sample_1.png

| `target_height` | 256 | Normalized height of character / 字的标准化高度 |

| `small_comp_ratio` | 2e-4 | Threshold for removing small components / 小连通域阈值 |# Batch processing

| `grid_size` | 5 | Grid size for sparsity analysis / 疏密分析网格大小 |for img in tests/sample_images/*.png; do

| `angle_thresh_deg` | 30 | Angle threshold for corner detection / 角点检测角度阈值 |    python -m libian_metrics --image "$img" --csv results.csv

done

All parameters can be customized through the calibration file.

# View results

## 🧪 Testing / 测试cat results.csv

```

To test with sample images:

```bash## Example Output

# Create sample dataset / 创建示例数据集

mkdir -p data/test_chars/{甲,乙,丙}### JSON Format (Single Image)

```json

# Copy your test images / 复制测试图片{

# cp your_images/*.jpg data/test_chars/甲/  "image": "tests/sample_images/sample_1.png",

  "SSI": 0.82,

# Process the dataset / 处理数据集  "GCP": 0.91,

python -m libian_metrics --dataset data/test_chars --out results/test_results.json --detailed  "SSD": 0.74,

```  "STR": 0.68,

  "CSI": 0.59,

## ⚠️ Quality Control Flags / 质量控制  "COI": 0.45,

  "LQI": 0.71,

The system tracks a `quality_flag` for each image. This flag is set to `False` when:  "angle": 0.3,

- Maximum connected component area < 1% of image area  "bbox": [40, 216, 40, 216],

- Skeleton pixels < 200  "scale": 1.0,

- Hough line coverage < 5% of skeleton  "quality_flag": true,

  "skel_pixels": 425,

结果中的 `quality_flag` 为 `False` 表示图片可能有质量问题，下游可进行过滤。  "max_component_area": 30976

}

## 📚 API Reference / API 参考```



### Main Processing Functions / 主要处理函数### CSV Format (Multiple Images)

```csv

#### Single Image / 单图处理image,SSI,GCP,SSD,STR,CSI,COI,LQI,angle,bbox,scale,quality_flag,skel_pixels,max_component_area

```pythonsample_1.png,0.82,0.91,0.74,0.68,0.59,0.45,0.71,0.3,"[40, 216, 40, 216]",1.0,True,425,30976

from libian_metrics.preprocess import preprocesssample_2.png,0.75,0.88,0.71,0.72,0.61,0.48,0.69,0.1,"[38, 218, 38, 218]",1.0,True,512,31500

from libian_metrics.metrics import compute_all_metrics```



# Preprocess image## Configuration Files

bin_img, skel, meta = preprocess(img_bgr)

### calibration.json

# Compute metricsGenerated from sample images using `calibrate_from_folder()`:

metrics = compute_all_metrics(bin_img, skel)

# Returns: {SSI, GCP, SSD, STR, CSI, COI, LQI}```json

```{

  "r_cap": 2.85,

#### Batch Processing / 批量处理  "c_cap": 0.92,

```python  "angle_thresh_deg": 28.5,

from libian_metrics.batch_process import process_dataset_folder, print_results  "density_alpha": 0.6,

  "num_samples": 50,

# Process entire dataset  "r_values_percentiles": {

results = process_dataset_folder(    "min": 1.02,

    'data/my_dataset',    "p25": 1.15,

    calib=None,    "p50": 1.28,

    output_json='results/output.json'    "p75": 1.45,

)    "max": 2.98

  },

# Print formatted results  "cv_values_percentiles": {

print_results(results, detailed=True)    "min": 0.15,

```    "p25": 0.32,

    "p50": 0.58,

## 🐛 Troubleshooting / 故障排除    "p75": 0.78,

    "max": 1.08

**Issue**: "No images found in folder"  }

- **Solution**: Check that images are in the correct subdirectories and have supported extensions}

```

**Issue**: Metrics are all very low (< 0.1)

- **Solution**: Image may be upside down or inverted. Check `quality_flag` is True## Notes



**Issue**: "Module not found" error- All six metrics are **single-image computable** and independent

- **Solution**: Install package with `pip install -e .` from the project root- Suitable for comparative analysis across time periods (e.g., Chu vs. Han)

- Preprocessing is deterministic (no random components)

**问题**: "找不到图片"- All metrics are normalized to [0, 1] range

- **解决**: 检查图片是否在正确的子文件夹中，且文件扩展名支持- Output is JSON by default, CSV append mode for batch processing



**问题**: 所有指标都很低 (< 0.1)## Citation

- **解决**: 图片可能颠倒或反色。检查 `quality_flag` 是否为 True

If you use LiBian Metrics in academic work, please cite:

**问题**: "模块未找到"错误

- **解决**: 从项目根目录用 `pip install -e .` 安装包```

LiBian Metrics: A Python Toolkit for Ancient Character Glyph Quality Assessment

## 📄 License / 许可证```



MIT License - See LICENSE file for details## License



## 🤝 Contributing / 贡献[Specify your license here]



Contributions are welcome! Please feel free to submit issues or pull requests.## Author



## 📞 Support / 支持LiShift Team



For bugs, feature requests, or questions, please open an issue on GitHub.## Support



---For issues, feature requests, or contributions, please contact the development team.


**Version**: 1.0.0  
**Last Updated**: 2024-01-06  
**Author**: LiShift Team

---

## File Structure / 文件结构

```
LiShift/
├── libian_metrics/              # Main package / 主包
│   ├── __init__.py
│   ├── __main__.py             # CLI entry point / CLI入口
│   ├── cli.py                  # Command-line interface / 命令行界面
│   ├── preprocess.py           # Image preprocessing / 图像预处理
│   ├── metrics.py              # Metric computation / 指标计算
│   ├── batch_process.py        # Batch processing / 批量处理
│   ├── calibrate.py            # Calibration utilities / 校准工具
│   ├── io_utils.py             # I/O utilities / 输入输出工具
│   ├── skeleton.py             # Skeleton utilities / 骨架工具
│   └── visualize.py            # Visualization / 可视化
├── data/                        # Data folder (put your datasets here) / 数据文件夹
│   └── README.md
├── results/                     # Results folder (output saved here) / 结果文件夹
├── pyproject.toml
├── setup.py
└── README.md                    # This file / 本文件
```

## Quick Reference / 快速参考

```bash
# Process single image / 处理单张图片
python -m libian_metrics --image char.jpg

# Batch process / 批量处理
python -m libian_metrics --dataset data/my_dataset --detailed

# Save output to file / 输出到文件
python -m libian_metrics --dataset data/my_dataset --out results/output.json

# Use calibration file / 使用校准文件
python -m libian_metrics --dataset data/my_dataset --calib calibration.json

# Show help / 显示帮助
python -m libian_metrics --help
```

## 使用流程总结 / Workflow Summary

1. **准备数据 / Prepare Data**
   ```bash
   mkdir -p data/my_dataset/{甲,乙,丙}
   # Copy images to character folders
   ```

2. **运行处理 / Run Processing**
   ```bash
   python -m libian_metrics --dataset data/my_dataset --out results/output.json --detailed
   ```

3. **查看结果 / View Results**
   ```bash
   cat results/output.json
   ```

4. **分析结果 / Analyze Results**
   - Check `summary.average_LQI` for overall quality
   - Compare metrics across characters
   - Identify outliers or quality issues


