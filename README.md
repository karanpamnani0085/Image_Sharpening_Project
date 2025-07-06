# Image Sharpening Project

This project implements an efficient image sharpening system using knowledge distillation from a teacher model to a lightweight student model.

## Features

- Teacher-student architecture for efficient inference
- Multiple loss functions (Perceptual, SSIM, Knowledge Distillation)
- Efficient student model using depthwise separable convolutions
- Comprehensive visualization tools

## Dataset

This project uses the DIV2K dataset for high-quality image super-resolution and sharpening tasks:
- Training Data: DIV2K_train_HR.zip
- Validation Data: DIV2K_valid_HR.zip
Download and extract these datasets, and update the dataset paths in the configuration file accordingly.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

### Demo

```bash
python demo.py --image path/to/image.jpg
```

## Project Structure

- `model/`: Neural network model implementations
- `loss/`: Loss function implementations
- `utils/`: Utility functions and dataset handling
- `data/`: Training and validation data
- `checkpoints/`: Saved model weights

## Configuration

Adjust training parameters in `config.json`:
- Dataset paths
- Model architecture
- Training hyperparameters
- Loss weights

## Results

The student model achieves comparable performance to the teacher while being significantly lighter:
- Faster inference time
- Reduced memory footprint
- Maintains image qualitypython train.py
