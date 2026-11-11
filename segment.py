"""Task 2 - Segmentation: separate the animal from the water inside a crop around its bounding box."""

import cv2
import numpy as np


def crop_with_margin(image, box, margin=0.25):
    """Cut the box (x, y, w, h) plus 25 % extra on every side, so the crop also shows the water around the
    animal. Returns the crop and the position of the box inside the crop."""
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = max(0, x - int(margin * w)), max(0, y - int(margin * h))
    x1 = min(image.shape[1], x + w + int(margin * w))
    y1 = min(image.shape[0], y + h + int(margin * h))
    return image[y0:y1, x0:x1], (x - x0, y - y0, min(w, x1 - x), min(h, y1 - y))


def box_mask(shape, box):
    x, y, w, h = box
    mask = np.zeros(shape[:2], bool)
    mask[y:y + h, x:x + w] = True
    return mask


def segment(crop, box, method="otsu"):
    """Returns the animal mask (True = animal) and the colour difference image.

    1. Colour difference (Delta E): distance in Lab colour space between each pixel and the median colour
       of the water around the box. The animal has a large Delta E, the water a small one.
    2. Threshold: Otsu's method on Delta E, or k-means (k = 2) on the Lab colours as an alternative.
    3. Opening removes small specks, closing fills small gaps.
    4. Only pixels inside the box are kept, then the largest connected region (filled, so holes disappear).
    If almost nothing is left, the whole box is used instead (fallback)."""
    inside = box_mask(crop.shape, box)
    if inside.all():
        return inside, np.zeros(crop.shape[:2], np.float32)
    lab = cv2.cvtColor(crop.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (0, 0), 1)
    water_colour = np.median(lab[~inside], axis=0)
    difference = np.linalg.norm(lab - water_colour, axis=2)

    if method == "otsu":
        scaled = cv2.normalize(difference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(scaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
        _, labels, centres = cv2.kmeans(lab.reshape(-1, 3), 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        animal = np.argmax(np.linalg.norm(centres - water_colour, axis=1))
        mask = (labels.reshape(crop.shape[:2]) == animal).astype(np.uint8)

    size = max(3, int(0.08 * min(box[2], box[3])) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask[~inside] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(mask)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    if mask.sum() < 0.06 * box[2] * box[3]:
        return inside, difference
    return mask.astype(bool), difference
