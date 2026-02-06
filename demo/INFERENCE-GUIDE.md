# CIFAKE (Real vs AI/Fake) – Export + Use Your ResNet Model (Demo)

This guide shows how to:

- **Restore** pre-trained models (if you cloned the repo)
- **Train** a ResNet model in your notebook (optional, if you want to train from scratch)
- **Export** it as a `.h5` file
- **Use the demo program** in the `demo/` folder where you input an image and it outputs **REAL** or **FAKE**

> Important: your CIFAKE dataset images are **32×32**, so the demo program expects **32×32** inputs.

---

## 1) Restore pre-trained models (if cloning the repo)

If you cloned this repository, the model files are tracked with **Git LFS** (Large File Storage). By default, Git only downloads pointer files, not the actual model weights.

### 1.1 Check if Git LFS is installed

```bash
git lfs version
```

If not installed, install it first:

- **Ubuntu/Debian (WSL)**:
  ```bash
  sudo apt update && sudo apt install git-lfs
  git lfs install
  ```

- **Windows**:
  Download from https://git-lfs.com/ or use `winget install GitHub.GitLFS`

### 1.2 Pull the actual model files

From the project root, run:

```bash
git lfs pull
```

This downloads the actual model weights:
- `models/best_model_weights.h5` (checkpoint)
- `models/resnet50_binary.h5` (final exported model)

### 1.3 Verify the models were restored

```bash
ls -lh models/
```

You should see files that are several MB in size (not just a few bytes). If the files are only ~130 bytes, they are still LFS pointers—run `git lfs pull` again.

Once restored, skip to **Section 4** to run the demo program.

---

## 2) Train and export your model (`.h5`) — Optional

1. Open the notebook:

```bash
jupyter notebook resnet50-cifake-classification.ipynb
```

2. Run all cells top-to-bottom until the training cell:

- The training cell calls `model.fit(...)`
- After training, the notebook saves:
  - `./models/best_model_weights.h5` (checkpoint)
  - `./models/resnet50_binary.h5` (final exported model)

3. Confirm the model file exists:

```bash
ls -lh models/
```

If you see `resnet50_binary.h5`, you’re ready for inference.

---

## 3) Understand the output (REAL vs FAKE)

Your model ends with:

- **1 sigmoid output** in the range \([0,1]\)

By default, this project interprets it like this:

- **p(REAL) ≥ 0.5 → REAL**
- **p(REAL) < 0.5 → FAKE**

Educational note: sigmoid is commonly used for **binary classification**. It behaves like a “confidence score” for the positive class (here: **REAL**).

Why REAL is treated as “1”: with CIFAKE folders `FAKE/` and `REAL/`, Keras typically assigns labels alphabetically:

- `FAKE = 0`
- `REAL = 1`

---

## 4) Run the demo program (CLI)

The `demo/` folder contains:

- `demo/predict_image.py` – CLI program
- `demo/image-ai-test.jpg` – example AI/fake image (32×32)
- `demo/image-real-test.jpg` – example real image (32×32)

> Make sure you have moved your sample images into `demo/` (e.g. `git mv image-ai-test.jpg demo/` and `git mv image-real-test.jpg demo/`).

### 4.1 Activate your environment

In WSL/Linux:

```bash
source .venv-wsl/bin/activate
```

### 4.2 Run prediction on a 32×32 image

From the project root:

```bash
python3 demo/predict_image.py --image demo/image-ai-test.jpg
```

You’ll see output like:

```text
Step 1/3: Loading model...
Step 2/3: Loading and preparing image...
Step 3/3: Inference complete.
Prediction: FAKE
p(REAL) = 0.000071 (threshold=0.5)
```

### 4.3 If your image is NOT 32×32

The requirement says your input *must* be 32×32. If you still want the script to auto-resize, run:

```bash
python3 demo/predict_image.py --image path/to/any_size.jpg --resize
```

---

## 5) (Optional) Programmatic usage (import as a module)

You can also import the functions from `demo/predict_image.py` inside another Python app if you want, since all the logic lives in that single file.

Example:

```python
from demo.predict_image import load_keras_model, predict_image

model = load_keras_model("models/resnet50_binary.h5")
result = predict_image(model, "demo/image-real-test.jpg", resize_if_needed=False)

print(result.label, result.probability_real)
```

---

## 6) Basic tests (recommended)

To validate preprocessing + label mapping:

1. Install dev deps:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
```

2. Run tests (if you add any in the future):

```bash
pytest -q
```

---

## Troubleshooting

- **“Image must be 32x32 …”**
  - Provide a 32×32 image or use `--resize`.
- **GPU not being used**
  - In another terminal, run `nvidia-smi` while prediction is running (it should show a Python process if GPU is used).
- **Jupyter cell below `model.fit(...)` won’t run**
  - If `model.fit` shows `[*]`, it’s still running. Interrupt or wait for completion.

