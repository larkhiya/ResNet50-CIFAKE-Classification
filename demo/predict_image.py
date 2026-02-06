"""
Demo CLI program to load the exported CIFAKE Keras model (.h5)
and classify a user-provided image as REAL vs FAKE.

Usage (WSL / Linux) from project root:
  source .venv-wsl/bin/activate
  python3 demo/predict_image.py --image demo/image-ai-test.jpg
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Silence most TensorFlow / CUDA logs for a clean CLI output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all, 3=only fatal

import tensorflow as tf  # noqa: E402

try:
    from absl import logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    # If absl is not available, just skip; TF_CPP_MIN_LOG_LEVEL will still help.
    pass


Label = Literal["REAL", "FAKE"]


@dataclass(frozen=True)
class PredictionResult:
    label: Label
    probability_real: float


def load_keras_model(model_path: str | Path) -> tf.keras.Model:
    """
    Load a Keras model saved as .h5 (architecture + weights).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.suffix.lower() != ".h5":
        raise ValueError(f"Expected a .h5 model file, got: {path.name}")

    return tf.keras.models.load_model(str(path))


def load_image_rgb(image_path: str | Path) -> tf.Tensor:
    """
    Load an image file into a uint8 tensor of shape (H, W, 3).

    Supports common formats that TensorFlow can decode (PNG/JPEG).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    raw = tf.io.read_file(str(path))
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    # decode_image returns dtype=uint8 and shape=(H, W, C)
    if img.shape.rank != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape={img.shape} for {path}")
    return img


def prepare_model_input(
    img_rgb_uint8: tf.Tensor,
    *,
    expected_size: tuple[int, int] = (32, 32),
    resize_if_needed: bool = False,
) -> tf.Tensor:
    """
    Prepare a single image tensor for model.predict().

    - Enforces 32x32 by default (to match CIFAKE dataset).
    - If resize_if_needed=True, resizes instead of failing.
    - Returns float32 tensor of shape (1, H, W, 3).

    Note: The training notebook fed raw pixel values (0..255) into the model.
    For inference consistency, we keep the same scaling (no /255 normalization).
    """
    if img_rgb_uint8.shape.rank != 3 or img_rgb_uint8.shape[-1] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {img_rgb_uint8.shape}")

    height = int(img_rgb_uint8.shape[0]) if img_rgb_uint8.shape[0] is not None else None
    width = int(img_rgb_uint8.shape[1]) if img_rgb_uint8.shape[1] is not None else None
    exp_h, exp_w = expected_size

    if (height, width) != (exp_h, exp_w):
        if not resize_if_needed:
            raise ValueError(
                f"Image must be {exp_h}x{exp_w}, got {height}x{width}. "
                f"Either provide a 32x32 image or run with --resize."
            )
        img_rgb_uint8 = tf.image.resize(img_rgb_uint8, [exp_h, exp_w], method="bilinear")
        img_rgb_uint8 = tf.cast(tf.clip_by_value(img_rgb_uint8, 0, 255), tf.uint8)

    img = tf.cast(img_rgb_uint8, tf.float32)
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img


def interpret_sigmoid_output(probability_real: float, *, threshold: float = 0.5) -> PredictionResult:
    """
    Interpret the model output (sigmoid) as REAL vs FAKE.

    CIFAKE directory names are typically: FAKE, REAL.
    Keras label assignment is alphabetical => FAKE=0, REAL=1.
    So sigmoid output closer to 1.0 means more likely REAL.
    """
    if not (0.0 <= probability_real <= 1.0):
        raise ValueError(f"Expected probability in [0, 1], got {probability_real}")
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Threshold must be between 0 and 1 (exclusive), got {threshold}")

    label: Label = "REAL" if probability_real >= threshold else "FAKE"
    return PredictionResult(label=label, probability_real=float(probability_real))


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    *,
    threshold: float = 0.5,
    resize_if_needed: bool = False,
    expected_size: tuple[int, int] = (32, 32),
) -> PredictionResult:
    img = load_image_rgb(image_path)
    x = prepare_model_input(img, expected_size=expected_size, resize_if_needed=resize_if_needed)
    preds = model.predict(x, verbose=0)

    try:
        probability_real = float(preds.reshape(-1)[0])
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unexpected model output shape: {getattr(preds, 'shape', None)}") from exc

    return interpret_sigmoid_output(probability_real, threshold=threshold)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CIFAKE inference demo: classify a 32x32 image as REAL vs FAKE using a saved Keras .h5 model."
    )
    parser.add_argument(
        "--model",
        default="models/resnet50_binary.h5",
        help="Path to the exported Keras model (.h5). Default: models/resnet50_binary.h5 (run from project root)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a 32x32 input image (PNG/JPG).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for REAL (sigmoid output). Default: 0.5",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="If set, automatically resize the image to 32x32 instead of erroring.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    print("Step 1/3: Loading model...")
    model = load_keras_model(args.model)

    print("Step 2/3: Loading and preparing image...")
    result = None
    try:
        result = predict_image(
            model,
            args.image,
            threshold=args.threshold,
            resize_if_needed=args.resize,
        )
    except Exception as exc:
        # Surface a clear, user-facing error while keeping the output minimal.
        print(f"Error: {exc}")
        return

    print("Step 3/3: Inference complete.")

    # Clean, minimal output
    print(f"Prediction: {result.label}")
    print(f"p(REAL) = {result.probability_real:.6f} (threshold={args.threshold})")


if __name__ == "__main__":
    main()

