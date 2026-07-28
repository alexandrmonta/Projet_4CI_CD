"""30 descripteurs de couleur. Partage par train.py et inference.py."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

IMAGE_SIZE = 64

_CHROMA_BINS = 8
_HUE_BINS = 8
_GREY_CHROMA_MAX = 0.06
_VIVID_CHROMA_MIN = 0.25


def load_image(source):
    """chemin, fichier ou octets bruts -> image RGB 64x64"""
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = io.BytesIO(bytes(source))

    img = Image.open(source)
    if img.size != (IMAGE_SIZE, IMAGE_SIZE):
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return img.convert("RGB")


def _hue(red, green, blue, maximum, chroma):
    safe_chroma = np.where(chroma < 1e-6, 1.0, chroma)
    hue = np.zeros_like(red)

    is_red = maximum == red
    is_green = (maximum == green) & ~is_red
    is_blue = ~is_red & ~is_green

    hue[is_red] = ((green - blue)[is_red] / safe_chroma[is_red]) % 6.0
    hue[is_green] = ((blue - red)[is_green] / safe_chroma[is_green]) + 2.0
    hue[is_blue] = ((red - green)[is_blue] / safe_chroma[is_blue]) + 4.0

    return hue / 6.0


def _describe(img):
    array = np.asarray(img, dtype=np.float32) / 255.0
    red, green, blue = array[..., 0], array[..., 1], array[..., 2]

    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    chroma = maximum - minimum
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    values: list[float] = []
    names: list[str] = []

    for label, channel in (("red", red), ("green", green), ("blue", blue), ("luminance", luminance)):
        values += [float(channel.mean()), float(channel.std())]
        names += [f"{label}_mean", f"{label}_std"]

    values += [
        float(chroma.mean()),
        float(chroma.std()),
        float(np.percentile(chroma, 50)),
        float(np.percentile(chroma, 90)),
        float((chroma < _GREY_CHROMA_MAX).mean()),
        float((chroma > _VIVID_CHROMA_MIN).mean()),
    ]
    names += [
        "chroma_mean",
        "chroma_std",
        "chroma_p50",
        "chroma_p90",
        "grey_pixel_ratio",
        "vivid_pixel_ratio",
    ]

    histogram, _ = np.histogram(chroma, bins=_CHROMA_BINS, range=(0.0, 1.0))
    values += (histogram / chroma.size).tolist()
    names += [f"chroma_hist_{i:02d}" for i in range(_CHROMA_BINS)]

    # pondere par la chroma : la teinte d'un pixel gris n'a pas de sens
    hue = _hue(red, green, blue, maximum, chroma)
    histogram, _ = np.histogram(hue, bins=_HUE_BINS, range=(0.0, 1.0), weights=chroma)
    values += (histogram / max(float(chroma.sum()), 1e-6)).tolist()
    names += [f"hue_hist_{i:02d}" for i in range(_HUE_BINS)]

    return values, names


def extract_features(source):
    values, _ = _describe(load_image(source))
    return np.asarray(values, dtype=np.float64)


_FEATURE_NAMES: list[str] | None = None


def feature_names():
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        _, names = _describe(Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)))
        _FEATURE_NAMES = names
    return list(_FEATURE_NAMES)
