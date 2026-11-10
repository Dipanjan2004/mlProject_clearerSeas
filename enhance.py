"""Task 1 - Preprocessing: gray-world white balance and non-local means denoising.
Images are OpenCV BGR arrays (uint8)."""

import cv2
import numpy as np


def resize(image, width=1280):
    """Brackish frames are 1920 x 1080; we work at 1280 pixels wide."""
    height = round(image.shape[0] * width / image.shape[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def white_balance(image):
    """Gray world: multiply B, G and R so that the three channels get the same mean.
    Water absorbs red light first, so red gets the largest gain and the blue-green cast disappears."""
    pixels = image.astype(np.float32)
    means = pixels.reshape(-1, 3).mean(axis=0)
    gains = np.clip(means.mean() / means, 0.25, 4.0)
    return np.clip(pixels * gains, 0, 255).astype(np.uint8)


def denoise(image, h=2):
    """Non-local means: each pixel becomes a weighted average of the pixels whose 7 x 7 neighbourhood looks
    similar (searched in a 21 x 21 window). A small strength h removes noise but keeps fine texture."""
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
