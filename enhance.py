"""Task 1 - Preprocessing: gray-world white balance, non-local means denoising, dark channel prior dehazing
and CLAHE. Images are OpenCV BGR arrays (uint8)."""

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


def dark_channel(image, patch=15):
    """Darkest of the three colour channels at every pixel, then the minimum over a 15 x 15 patch (erosion)."""
    return cv2.erode(image.min(axis=2), np.ones((patch, patch), np.uint8))


def dehaze(image, omega=0.8, t_min=0.3):
    """Dark channel prior (He et al., 2009). Haze model: I = J t + A (1 - t), where I is the observed image,
    J the clear image, A the colour of the water light and t the transmission (share of light that reaches
    the camera). In clear water the dark channel is close to 0, so a bright dark channel means haze."""
    f = image.astype(np.float32) / 255
    dark = dark_channel(f)
    haziest = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    water_light = np.maximum(f.reshape(-1, 3)[haziest].mean(axis=0), 0.05)
    transmission = 1 - omega * dark_channel(f / water_light)
    transmission = cv2.GaussianBlur(transmission, (0, 0), 15)
    t = np.maximum(transmission, t_min)[..., None]
    clear = (f - water_light) / t + water_light
    return np.clip(clear * 255, 0, 255).astype(np.uint8), dark, transmission


def clahe(image, clip_limit=2.0, tiles=8):
    """Contrast Limited Adaptive Histogram Equalization on the L (lightness) channel of Lab only,
    so local contrast improves without changing the colours."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clip_limit, (tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance(image):
    """Run the four steps in order. Returns every intermediate image; the last one ("CLAHE") is the result."""
    steps = {"Raw": image}
    steps["White balance"] = white_balance(steps["Raw"])
    steps["Denoised"] = denoise(steps["White balance"])
    steps["Dehazed"] = dehaze(steps["Denoised"])[0]
    steps["CLAHE"] = clahe(steps["Dehazed"])
    return steps
