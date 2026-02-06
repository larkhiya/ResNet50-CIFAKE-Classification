# WSL2 GPU Setup Guide for ResNet Classification

This guide will help you set up your WSL2 Ubuntu environment with TensorFlow GPU support to run the ResNet101 CIFAKE classification notebook.

---

## Prerequisites

Before starting, ensure you have:

- **Windows 10/11** with WSL2 installed
- **NVIDIA GPU** with compatible drivers installed on Windows
- **WSL2 Ubuntu** distribution installed
- **CUDA-compatible GPU** (for TensorFlow GPU acceleration)

---

## Step 1: Update System and Install Python Tools

Open your WSL2 Ubuntu terminal and update the package manager:

```bash
sudo apt update
```

Install Python pip and virtual environment tools:

```bash
sudo apt install -y python3-pip python3-venv
```

**Why:** This ensures you have the latest package information and the tools needed to create isolated Python environments.

---

## Step 2: Navigate to Your Project Directory

Change to your project directory (update the path to match your setup):


---

## Step 3: Create a Virtual Environment

Create a virtual environment named `.venv-wsl`:

```bash
python3 -m venv .venv-wsl
```

**Why:** Virtual environments isolate project dependencies and prevent conflicts with system-wide Python packages.

---

## Step 4: Activate the Virtual Environment

Activate the newly created virtual environment:

```bash
source .venv-wsl/bin/activate
```

You should see `(.venv-wsl)` appear at the beginning of your terminal prompt.

---

## Step 5: Upgrade pip

Ensure you have the latest version of pip:

```bash
pip install --upgrade pip
```

---

## Step 6: Install Dependencies

Install all required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `tensorflow[and-cuda]` - TensorFlow with CUDA and cuDNN libraries for GPU acceleration
- `numpy` - Numerical computing library
- `scikit-learn` - Machine learning utilities (metrics, preprocessing)
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `jupyter` and `notebook` - Jupyter Notebook environment

**Why:** Using `requirements.txt` ensures consistent dependency versions across different environments and makes it easier to reproduce the setup.

**Note:** If you prefer JupyterLab over Jupyter Notebook, you can install it separately:

```bash
pip install jupyterlab
```

---

## Step 9: Verify GPU Setup

Verify that TensorFlow can detect your GPU:

```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

**Expected output:**
```
TensorFlow version: 2.x.x
GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If you see an empty list `[]` for GPU Available, your GPU is not detected. Check your NVIDIA drivers and WSL2 CUDA setup.

---

## Step 10: Launch Jupyter Notebook

With the virtual environment still activated, launch Jupyter Notebook:

```bash
jupyter notebook
```

**Or** launch JupyterLab:

```bash
jupyter lab
```

This will open a browser window where you can navigate to and open `resnet101-cifake-classification1.ipynb`.
