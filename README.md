# Stereo Depth and Visual Odometry on KITTI

Classical computer vision implementation of dense stereo depth estimation and visual odometry using the KITTI dataset.

## Dependencies

### Required Python Packages
```bash
pip install numpy opencv-python scipy matplotlib pyyaml
```

### Python Version
- Python 3.8 or higher

### Dataset
- KITTI Odometry Dataset (sequences 00-10)
- KITTI Ground Truth Poses (for evaluation)
- KITTI Stereo 2015 dataset

Download from: 
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip


## Project Structure
```
CV Project/
├── configs/
│   └── config.yaml          # Configuration parameters
├── Dataset/                 # KITTI dataset (not included)
│   ├── poses/              # Ground truth poses
│   ├── sequences/          # Image sequences 00-10
│   └──Dataset_Stereo/
├── outputs/                # Generated results
├── scripts/                # Executable scripts
│   ├── run_stereo.py
│   ├── run_vo.py
│   ├── evaluate_depth_stereo2015.py
│   └── evaluate_vo.py
└── src/                    # Source code modules
    ├── stereo/
    ├── vo/
    ├── evaluation/
    ├── utils/
    └── visualization/
```

## Configuration

Edit `configs/config.yaml` to set your dataset path:
```yaml
dataset:
  kitti_odometry_path: "path/to/your/dataset"
```

## Usage

### 1. Stereo Depth Estimation

Process stereo images to generate disparity and depth maps:
```bash
# Process frames 0-10 from sequence 00
python scripts/run_stereo.py --sequence 0 --frames 0-10

# Process specific frames
python scripts/run_stereo.py --sequence 5 --frames 0,5,10,15,20

# Process frames from sequence 05
python scripts/run_stereo.py --sequence 5 --frames 0-50

# Generate SGBM results
python scripts/run_stereo.py --sequence 0 --frames 0-10 --method sgbm

# Generate block_matching results 
python scripts/run_stereo.py --sequence 0 --frames 0-10 --method block_matching
```

**Outputs:**
- `outputs/disparity/seq_XX/*.npy` - Disparity maps
- `outputs/depth/seq_XX/*.npy` - Depth maps
- `outputs/visualizations/seq_XX/disparity_*.png` - Disparity visualizations
- `outputs/visualizations/seq_XX/depth_*.png` - Depth visualizations

### 2. Visual Odometry

Estimate camera trajectory from monocular images with stereo-based scale:
```bash
# Process 50 frames from sequence 00
python scripts/run_vo.py --sequence 0 --frames 50

# Process 100 frames
python scripts/run_vo.py --sequence 0 --frames 100

# Process different sequence
python scripts/run_vo.py --sequence 5 --frames 200
```

**Outputs:**
- `outputs/trajectory/XX.txt` - Estimated trajectory (KITTI format)
- `outputs/visualizations/vo_seq_XX/matches_*.png` - Feature match visualizations
- `outputs/visualizations/vo_seq_XX/trajectory_2d_seq_XX.png` - 2D trajectory plot
- `outputs/visualizations/vo_seq_XX/trajectory_3d_seq_XX.png` - 3D trajectory plot

### 3. Evaluate Visual Odometry

Compare estimated trajectory against ground truth:
```bash
# Evaluate sequence 00
python scripts/evaluate_vo.py --trajectory outputs/trajectory/00.txt --sequence 0

# Evaluate sequence 05
python scripts/evaluate_vo.py --trajectory outputs/trajectory/05.txt --sequence 5
```

**Outputs:**
- `outputs/evaluation/eval_seq_XX/trajectory_comparison_seq_XX.png` - Trajectory comparison plots
- `outputs/evaluation/eval_seq_XX/trajectory_error_seq_XX.png` - Error over time plot
- `outputs/evaluation/eval_seq_XX/metrics_seq_XX.txt` - Evaluation metrics (ATE, RPE)

### 4. Evaluate Depth (Template)

# Evaluate on first 10 images with SGBM
python scripts/evaluate_depth_stereo2015.py --images 0-9 --method sgbm

# Evaluate with block matching
python scripts/evaluate_depth_stereo2015.py --images 0-9 --method block_matching

# Evaluate specific images
python scripts/evaluate_depth_stereo2015.py --images 0,5,10,15,20 --method sgbm
```

## Expected Output
```
Average Metrics Across 10 Images
Disparity Metrics:
  MAE: 2.5-4.0 pixels (SGBM)
  MAE: 4.0-8.0 pixels (block matching)
  Bad-pixel rate: 15-25% (SGBM)
  Bad-pixel rate: 30-50% (block matching)
## Algorithm Overview

### Stereo Depth Pipeline
1. **Disparity Estimation**: Semi-Global Block Matching (SGBM)
2. **Post-processing**: Median filtering, hole filling, speckle removal
3. **Depth Conversion**: Z = f × B / disparity

### Visual Odometry Pipeline
1. **Feature Detection**: ORB features
2. **Feature Matching**: Descriptor matching with ratio test
3. **3D Point Triangulation**: Using stereo depth
4. **Pose Estimation**: PnP with RANSAC for metric scale
5. **Trajectory Chaining**: Sequential pose composition

## Evaluation Metrics

### Depth Metrics
- **MAE** (Mean Absolute Error): Average disparity error in pixels
- **Bad-pixel rate**: Percentage of pixels with error > 3 pixels

### Visual Odometry Metrics
- **ATE** (Absolute Trajectory Error): RMSE of position errors
- **RPE** (Relative Pose Error): Drift per unit distance/rotation

## Configuration Parameters

Edit `configs/config.yaml` to adjust:
```yaml
stereo:
  window_size: 11          # SGBM block size
  max_disparity: 64        # Maximum disparity search range
  cost_function: "SAD"     # Not used (SGBM has fixed cost)
  
vo:
  feature_detector: "ORB"  # Feature type (ORB or SIFT)
  ransac_threshold: 1.0    # RANSAC inlier threshold (pixels)
  min_inliers: 50          # Minimum inliers for valid pose
```

## Example Results (Sequence 00, 50 frames)
```
Depth Statistics:
  Valid pixels: ~460,000 per frame
  Depth range: 6-80 meters
  Mean depth: ~16 meters

Visual Odometry Performance:
  Trajectory length: 44.04m (GT: 45.84m)
  ATE RMSE: 1.41 meters
  RPE (1 frame): 0.067m translation, 0.12° rotation
  RPE (10 frames): 0.40m translation, 0.62° rotation
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root directory
cd "CV Project"
python scripts/run_stereo.py --sequence 0 --frames 0-10
```

### Dataset Path Issues
Update `configs/config.yaml` with your actual dataset path (use forward slashes or double backslashes on Windows).

### Ground Truth Not Found
Download ground truth poses from KITTI website and place in `Dataset/poses/00.txt`, `01.txt`, etc.

## Notes

- This implementation uses **classical computer vision methods only** (no deep learning)
- SGBM (Semi-Global Block Matching) is used for stereo instead of basic block matching for better quality
- All methods comply with classical geometric computer vision constraints
- The system achieves ~3-4% trajectory drift over short sequences

## Authors
Soheil Taghavi, Ferdowsi University of Mashhad
