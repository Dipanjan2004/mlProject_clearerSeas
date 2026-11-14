"""Task 3 - Feature extraction: 58 colour, texture and shape numbers that describe one segmented animal."""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

CHAIN_CODE = {(1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3, (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7}


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


def shape_features(mask):
    """23 numbers: area, perimeter, length, width, aspect ratio, circularity, extent, solidity, the 7 Hu
    moments and the histogram of the 8-direction Freeman chain code of the boundary."""
    binary = mask.astype(np.uint8)
    contour = max(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    area = float(binary.sum())
    perimeter = max(cv2.arcLength(contour, True), 1.0)
    (_, _), (side1, side2), _ = cv2.minAreaRect(contour)
    length, width = max(side1, side2, 1.0), max(min(side1, side2), 1.0)
    _, _, box_w, box_h = cv2.boundingRect(contour)
    hull_area = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    region = [area, perimeter, length, width, length / width, 4 * np.pi * area / perimeter ** 2,
              area / (box_w * box_h), min(area / hull_area, 1.0)]

    hu = cv2.HuMoments(cv2.moments(binary, binaryImage=True)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    points = contour[:, 0, :]
    steps = np.roll(points, -1, axis=0) - points
    codes = [CHAIN_CODE[(dx, dy)] for dx, dy in steps if (dx, dy) in CHAIN_CODE]
    chain = np.bincount(codes, minlength=8) / max(len(codes), 1)
    return np.hstack([region, hu, chain])


def extract_features(crop, mask):
    """The full feature vector (58 numbers) of the animal inside `mask`."""
    mask = mask.astype(bool)
    vector = np.hstack([colour_features(crop, mask), texture_features(crop, mask), shape_features(mask)])
    return np.nan_to_num(vector)
