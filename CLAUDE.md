# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python implementation for generating improved 3D Ken Burns effects from single images, based on the paper "3D Ken Burns Effect from a Single Image". The project extends the original work with training capabilities for neural networks, improved depth estimation, better image inpainting, and a novel dolly zoom effect.

## Essential Commands

### Setup and Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (required for inference)
chmod +x download.sh
./download.sh
```

### Running 3D Ken Burns Effect

```bash
# Basic usage
python kbe.py --in /path/to/image.jpg --out /path/to/output/

# Full example with custom parameters
CUDA_AVAILABLE_DEVICES=0 python kbe.py \
  --in ./images/test.png \
  --out ./images/kbe/ \
  --estim-path ./models/trained/disparity-estimation.tar \
  --refine-path ./models/trained/disparity-refinement.tar \
  --inpaint-path ./models/trained/inpainting-color.tar \
  --write-frames \
  --startU 512 --startV 512 --endU 600 --endV 600 \
  --startW 400 --startH 200 --endW 300 --endH 150

# Dolly zoom effect
python kbe.py --in /path/to/image.jpg --out /path/to/output/ --dolly
```

### Training Networks

```bash
# Train disparity estimation network
CUDA_AVAILABLE_DEVICES=0 python train.py \
  --training-mode estimation \
  --batch-size 8 \
  --lr-estimation 0.0001 \
  --save-name my_model

# Train inpainting network with partial convolution
CUDA_AVAILABLE_DEVICES=0 python train.py \
  --training-mode inpainting \
  --batch-size 2 \
  --lr-inpaint 0.0005 \
  --save-name test \
  --partial-conv
```

## Architecture

### Core Components

1. **Disparity Estimation Network** (`models/disparity_estimation.py`)
   - Estimates depth from single images
   - Uses semantic segmentation for improved accuracy

2. **Disparity Refinement Network** (`models/disparity_refinement.py`)
   - Increases resolution and quality of depth maps
   - Compatible with pre-trained networks from original paper

3. **Inpainting Networks**
   - `models/pointcloud_inpainting.py`: Standard inpainting
   - `models/partial_inpainting.py`: Partial convolution-based inpainting
   - Handles disocclusion artifacts in Ken Burns effect

4. **Pipeline** (`utils/pipeline.py`)
   - Main processing pipeline orchestrating all components
   - Handles both 3D Ken Burns and dolly zoom effects
   - Manages GPU memory and model loading

### Key Processing Steps

1. **Depth Estimation**: Single image → disparity map using semantic guidance
2. **Refinement**: Low-res disparity → high-res refined disparity
3. **3D Reconstruction**: Disparity → point cloud using camera parameters
4. **Ken Burns Rendering**: Point cloud warping + perspective projection
5. **Inpainting**: Fill disoccluded regions using neural networks
6. **Video Generation**: Frame sequence → MP4 output

### CUDA Integration

- Custom CUDA kernels in `utils/common.py` for point cloud rendering
- Requires `CUDA_HOME` environment variable set correctly
- Path to `helper_math.h` must be updated in `utils/common.py` line 14

## Important Configuration

### CUDA Setup

- Set `CUDA_HOME` environment variable to CUDA installation path
- Update `path_to_math_helper` in `utils/common.py:14` to match your system
- Minimum 6GB GPU memory recommended

### Dataset Configuration

Training requires datasets with paired (image, depth) data:

- Update dataset paths in `train.py`
- Ensure proper folder structure: `images/` and `depth/` subdirectories
- Camera parameters (focal length, baseline) must be specified per dataset

### Model Paths

Default model paths in `kbe.py`:

- `./models/trained/disparity-estimation-no-mask.tar`
- `./models/trained/disparity-refinement.tar`
- `./models/trained/inpainting-color.tar`

## Key Parameters

### Ken Burns Effect Parameters

- `startU/startV`: Starting crop window center coordinates
- `startW/startH`: Starting crop window dimensions
- `endU/endV`: Ending crop window center coordinates
- `endW/endH`: Ending crop window dimensions
- `--dolly`: Enable dolly zoom effect (focal length changes)
- `--2d`: 2D Ken Burns (no depth-based warping)

### Training Parameters

- `--training-mode`: `estimation`, `refinement`, `inpainting`, `inpainting_ref`
- `--mask-loss`: Type of mask loss (`none`, `same`, `other`)
- `--partial-conv`: Use partial convolution for inpainting
