"""Evaluation of the image processing steps.

  1. Enhancement on UIEB: PSNR and SSIM against the reference images, entropy and contrast for visible detail.
  2. Segmentation on SUIM: IoU, Dice and precision against the hand-drawn fish masks.
  3. Segmentation of the Brackish crops: how often Otsu fails, and the mean colour of the segmented animals.
"""

from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from enhance import enhance
from segment import box_mask, crop_with_margin, segment


def shrink(image, side):
    """Resize so that the longer side is at most `side` pixels."""
    scale = side / max(image.shape[:2])
    if scale >= 1:
        return image
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def detail(image):
    """Entropy of the grey-level histogram (bits) and RMS contrast: both grow with the visible detail."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    return {"Entropy": -(p[p > 0] * np.log2(p[p > 0])).sum(), "Contrast": gray.std() / 255}


def uieb_image(name):
    """Score the image after each preprocessing step against the reference version of the same image."""
    raw = shrink(cv2.imread(f"data/uieb/raw-890/{name}"), 512)
    reference = cv2.imread(f"data/uieb/reference-890/{name}")
    reference = cv2.resize(reference, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
    rows = [{"step": step, "PSNR": peak_signal_noise_ratio(reference, image),
             "SSIM": structural_similarity(reference, image, channel_axis=2), **detail(image)}
            for step, image in enhance(raw).items()]
    return rows + [{"step": "Reference", **detail(reference)}]


def overlap(mask, truth):
    """IoU and Dice measure the overlap with the true mask; precision is the share of the mask that is fish."""
    intersection, union = (mask & truth).sum(), (mask | truth).sum()
    return {"IoU": intersection / union, "Dice": 2 * intersection / (mask.sum() + truth.sum()),
            "Precision": intersection / mask.sum()}


def suim_image(path):
    """Segment every fish of one SUIM image, raw and enhanced, and compare with the hand-drawn mask."""
    image = shrink(cv2.imread(str(path)), 960)
    truth = cv2.imread(str(path.parent.parent / "masks" / f"{path.stem}.bmp"), cv2.IMREAD_GRAYSCALE)
    truth = cv2.resize(truth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) == 6
    enhanced = enhance(image)["CLAHE"]
    height, width = truth.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(truth.astype(np.uint8), connectivity=8)
    rows = []
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        too_small = area < 400 or min(w, h) < 20
        too_big_or_cut = w * h > 0.25 * height * width or x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1
        if too_small or too_big_or_cut:
            continue
        fish, box = crop_with_margin(labels == k, (x, y, w, h))
        rows.append({"method": "Bounding box only", **overlap(box_mask(fish.shape, box), fish)})
        for source, picture in (("raw", image), ("enhanced", enhanced)):
            crop, _ = crop_with_margin(picture, (x, y, w, h))
            for method in ("otsu", "kmeans"):
                rows.append({"method": f"{method}, {source}", **overlap(segment(crop, box, method)[0], fish)})
    return rows


def brackish_crop(item):
    """Otsu segmentation of one Brackish crop, raw and enhanced: did it fall back to the box, and the mean colour."""
    row = {"species": item["label"]}
    box = (item["bx"], item["by"], item["bw"], item["bh"])
    for source in ("raw", "enhanced"):
        crop = cv2.imread(f"data/crops/{source}/{item['box_id']}.png")
        mask, _ = segment(crop, box)
        b, g, r = crop[mask].mean(axis=0)
        row.update({f"fallback {source}": np.array_equal(mask, box_mask(crop.shape, box)),
                    f"R {source}": r, f"G {source}": g, f"B {source}": b})
    return row


if __name__ == "__main__":
    with Pool() as pool:
        print("=== 1. Enhancement quality on UIEB ===")
        names = sorted(p.name for p in Path("data/uieb/raw-890").glob("*.png"))
        uieb = pd.DataFrame([row for rows in pool.map(uieb_image, names) for row in rows])
        uieb = uieb.groupby("step", sort=False).mean()
        print(f"{len(names)} images\n{uieb.round(3).to_string()}")
        uieb.round(4).to_csv("results/uieb.csv")

        print("\n=== 2. Segmentation quality on SUIM ===")
        paths = sorted(Path("data/SUIM").glob("*/images/*.jpg"))
        suim = pd.DataFrame([row for rows in pool.map(suim_image, paths) for row in rows])
        table = suim.groupby("method", sort=False).mean()
        table["Fish"] = suim.groupby("method", sort=False).size()
        print(f"{len(paths)} images\n{table.round(3).to_string()}")
        table.round(4).to_csv("results/suim.csv")

        print("\n=== 3. Segmentation of the Brackish crops ===")
        crops = pd.read_csv("data/crops.csv").to_dict("records")
        brackish = pd.DataFrame(pool.map(brackish_crop, crops))
        table = brackish.groupby("species").mean()
        table.loc["All"] = brackish.drop(columns="species").mean()
        print(table.round(3).to_string())
        table.round(4).to_csv("results/brackish_segmentation.csv")
