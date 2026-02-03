# ResNet50 CIFAKE Classification

Deep learning project for binary classification of CIFAKE images using ResNet50 architecture with TensorFlow GPU acceleration.

## Project Overview

This project implements a ResNet50-based classifier to distinguish between real and fake images from the CIFAKE dataset. The model is optimized for GPU training using TensorFlow with CUDA support, mixed precision training, and XLA compilation.

## Features

- **ResNet50 Architecture**: Pre-trained ResNet50 model fine-tuned for binary classification
- **GPU Acceleration**: CUDA-enabled TensorFlow for faster training
- **Mixed Precision Training**: Float16 precision for improved performance on Tensor Core GPUs
- **XLA Compilation**: Accelerated Linear Algebra for optimized execution
- **Data Augmentation**: Image preprocessing and augmentation pipelines
- **Model Evaluation**: Comprehensive metrics including confusion matrix and classification reports

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for GPU acceleration)
- WSL2 Ubuntu (for Windows users)

## Installation

### Quick Start

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Finals
```

2. Create a virtual environment:
```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate  # On Windows WSL
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Detailed Setup

For detailed setup instructions, especially for WSL2 GPU configuration, see [WSL-GPU-Setup-Guide.md](WSL-GPU-Setup-Guide.md).

## Usage

1. Ensure your dataset is in the `archive/` directory with the following structure:
```
archive/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

2. Open the Jupyter notebook:
```bash
jupyter notebook resnet50-cifake-classification1.ipynb
```

3. Run all cells to train and evaluate the model.

## Project Structure

```
Finals/
├── archive/              # Dataset directory (not included in repo)
├── models/              # Saved model files (not included in repo)
├── resnet50-cifake-classification1.ipynb  # Main notebook
├── requirements.txt     # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── WSL-GPU-Setup-Guide.md  # Setup instructions
```

## Dependencies

See `requirements.txt` for the complete list. Key dependencies include:

- TensorFlow 2.20.0+ (with CUDA support)
- NumPy 1.24.0+
- scikit-learn 1.3.0+
- Matplotlib 3.7.0+
- Seaborn 0.12.0+
- Jupyter Notebook 6.4.0+

## Notes

- Model files (`.h5`) and dataset directories are excluded from the repository due to size constraints
- Ensure you have sufficient GPU memory for training
- The notebook includes GPU detection and configuration code

