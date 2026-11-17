"""Classify one animal and show every step of the pipeline.

    python predict.py                                  first demo frame (from a test video)
    python predict.py image.jpg 998 26 134 114         any image and the animal's box: x y width height
"""

import json
import sys

import cv2
import joblib
import matplotlib.pyplot as plt

from enhance import enhance, resize
from features import extract_features
from segment import crop_with_margin, segment


def classify(enhanced_frame, box, model):
    """Crop the animal from the enhanced frame, segment it, extract its features and predict the species."""
    crop, inner_box = crop_with_margin(enhanced_frame, box)
    mask, difference = segment(crop, inner_box)
    species = model.predict([extract_features(crop, mask)])[0]
    return {"crop": crop, "mask": mask, "difference": difference, "species": species}


def outline(crop, mask):
    """Draw the boundary of the mask in yellow on a copy of the crop."""
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(crop.copy(), contours, -1, (0, 255, 255), 2)


if __name__ == "__main__":
    if len(sys.argv) == 6:
        path, box, truth = sys.argv[1], [float(v) for v in sys.argv[2:]], "unknown"
    else:
        demo = json.load(open("data/samples/frames.json"))[0]
        animal = demo["boxes"][0]
        path, box, truth = f"data/samples/{demo['image']}", [animal[k] for k in "xywh"], animal["label"]

    frame = cv2.imread(path)
    box = [v * 1280 / frame.shape[1] for v in box]
    steps = enhance(resize(frame))
    result = classify(steps["CLAHE"], box, joblib.load("models/model.pkl"))
    raw_crop, _ = crop_with_margin(steps["Raw"], box)

    images = [raw_crop[..., ::-1], result["crop"][..., ::-1], result["difference"],
              outline(result["crop"], result["mask"])[..., ::-1]]
    titles = ["Raw crop", "Enhanced crop", "Colour difference (Delta E)", f"Predicted: {result['species']}"]
    plt.figure(figsize=(14, 4))
    for i, (image, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(1, 4, i)
        plt.imshow(image, cmap="magma")
        plt.title(title)
        plt.axis("off")
    plt.suptitle(f"True species: {truth}")
    plt.tight_layout()
    plt.show()
