"""Task 3 - Feature extraction: colour and texture numbers that describe one segmented animal."""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def colour_features(crop, mask):
    """25 numbers: mean and standard deviation of B, G, R and of L, a, b (Lab), an 8-bin hue histogram,
    a 4-bin saturation histogram and the colour contrast (Delta E) between the animal and its surroundings."""
    bgr = crop[mask].astype(np.float32)
    lab_image = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lab = lab_image[mask]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[mask]
    hue = np.histogram(hsv[:, 0], bins=8, range=(0, 180))[0] / len(hsv)
    saturation = np.histogram(hsv[:, 1], bins=4, range=(0, 256))[0] / len(hsv)
    surroundings = lab_image[~mask] if (~mask).any() else lab
    contrast = np.linalg.norm(lab.mean(axis=0) - np.median(surroundings, axis=0))
    return np.hstack([bgr.mean(axis=0), bgr.std(axis=0), lab.mean(axis=0), lab.std(axis=0), hue, saturation, contrast])


def texture_features(crop, mask):
    """10 numbers: GLCM contrast, homogeneity, energy and correlation at distances 1 and 2 (each averaged over
    0, 45, 90 and 135 degrees), plus the entropy and uniformity of the grey-level histogram."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    levels = np.where(mask, gray // 8 + 1, 0).astype(np.uint8)
    glcm = graycomatrix(levels, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33, symmetric=True)
    glcm = glcm[1:, 1:]
    properties = ("contrast", "homogeneity", "energy", "correlation")
    glcm_values = [graycoprops(glcm, name).mean(axis=1) for name in properties]
    p = np.bincount(gray[mask], minlength=256) / mask.sum()
    entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()
    uniformity = (p ** 2).sum()
    return np.hstack(glcm_values + [entropy, uniformity])
