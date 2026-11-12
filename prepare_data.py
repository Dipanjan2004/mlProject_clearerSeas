"""Download the three datasets and cut the animal crops used for training.

  Brackish - underwater video frames with labelled boxes of six species  (classification)
  UIEB     - 890 underwater images with reference (clear) versions        (enhancement quality)
  SUIM     - underwater images with hand-drawn fish masks                  (segmentation quality)

Output: data/crops/raw/*.png, data/crops/enhanced/*.png, data/crops.csv and a few demo frames in data/samples/.
"""

import json
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import disable_progress_bars
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from enhance import enhance, resize
from segment import crop_with_margin

DATA = Path("data")
BRACKISH = DATA / "brackish"
CLASSES = ["crab", "fish", "jellyfish", "shrimp", "small_fish", "starfish"]


def download(repo, path, folder):
    """Fetch one file of a Hugging Face dataset, unless it is already on disk."""
    if not (folder / path).exists():
        hf_hub_download(repo, path, repo_type="dataset", local_dir=folder)
    return folder / path


def load_boxes():
    """One row per labelled animal: frame, video, species and box (x, y, w, h) in the 1920 x 1080 frame."""
    rows = []
    for split in ("train", "valid", "test"):
        metadata = download("dronefreak/Brackish", f"data/{split}/metadata.jsonl", BRACKISH)
        for line in metadata.read_text().splitlines():
            record = json.loads(line)
            video = Path(record["file_name"]).name.split("to")[0]
            for (x, y, w, h), category in zip(record["objects"]["bbox"], record["objects"]["categories"]):
                rows.append({"frame": f"data/{split}/{record['file_name']}", "video": video,
                             "label": CLASSES[category], "x": x, "y": y, "w": w, "h": h})
    return pd.DataFrame(rows)


def usable_boxes(boxes):
    """Boxes of at least 24 pixels that are not cut by the frame border."""
    inside = (boxes.x > 2) & (boxes.y > 2) & (boxes.x + boxes.w < 1918) & (boxes.y + boxes.h < 1078)
    return boxes[inside & (boxes[["w", "h"]].min(axis=1) >= 24)]


def select_boxes(usable):
    """Take 300 random boxes per species. Frames of one video are almost identical, so whole videos go to the
    test set (the dataset's own split mixes them): the videos are divided into 5 groups and the group with the
    most even classes is used for testing."""
    chosen = usable.groupby("label").sample(n=300, random_state=42).reset_index(drop=True)
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=42).split(chosen, chosen.label, chosen.video)
    test = min((test for _, test in folds), key=lambda test: chosen.label[test].value_counts(normalize=True).max())
    chosen["split"] = "train"
    chosen.loc[test, "split"] = "test"
    chosen.insert(0, "box_id", [f"b{i:04d}" for i in range(len(chosen))])
    return chosen


def cut_crops(job):
    """Enhance one whole frame, then cut a raw and an enhanced crop for every selected box in it."""
    frame, boxes = job
    raw = resize(cv2.imread(str(BRACKISH / frame)))
    enhanced = enhance(raw)["CLAHE"]
    scale = raw.shape[1] / 1920
    rows = []
    for box in boxes:
        xywh = [box[k] * scale for k in "xywh"]
        raw_crop, (bx, by, bw, bh) = crop_with_margin(raw, xywh)
        enhanced_crop, _ = crop_with_margin(enhanced, xywh)
        cv2.imwrite(str(DATA / "crops" / "raw" / f"{box['box_id']}.png"), raw_crop)
        cv2.imwrite(str(DATA / "crops" / "enhanced" / f"{box['box_id']}.png"), enhanced_crop)
        rows.append({**box, "bx": bx, "by": by, "bw": bw, "bh": bh})
    return rows


def save_demo_frames(boxes, usable):
    """Copy two frames per species from the test videos (frames with several species first), with all their
    labelled animals, for the app and predict.py."""
    test = usable[usable.video.isin(boxes[boxes.split == "test"].video)]
    test = test.assign(species=test.groupby("frame").label.transform("nunique")).sort_values("species", ascending=False)
    frames = test.drop_duplicates(["label", "video"]).groupby("label").head(2).frame.unique()
    samples = DATA / "samples"
    samples.mkdir(exist_ok=True)
    demo = []
    for frame in frames:
        shutil.copy(download("dronefreak/Brackish", frame, BRACKISH), samples / Path(frame).name)
        animals = test[test.frame == frame][["x", "y", "w", "h", "label"]].to_dict("records")
        demo.append({"image": Path(frame).name, "boxes": animals})
    (samples / "frames.json").write_text(json.dumps(demo, indent=1))


if __name__ == "__main__":
    disable_progress_bars()
    print("=== 1. Brackish: choosing 300 boxes per species ===")
    usable = usable_boxes(load_boxes())
    boxes = select_boxes(usable)
    print(boxes.groupby(["label", "split"]).size().unstack())
    print("videos:", boxes.groupby("split").video.nunique().to_dict())

    print("\n=== 2. Downloading the frames that contain them ===")
    frames = boxes.frame.unique()
    with ThreadPoolExecutor(8) as pool:
        list(tqdm(pool.map(lambda f: download("dronefreak/Brackish", f, BRACKISH), frames), total=len(frames)))

    print("\n=== 3. Enhancing every frame and cutting raw and enhanced crops ===")
    for kind in ("raw", "enhanced"):
        (DATA / "crops" / kind).mkdir(parents=True, exist_ok=True)
    jobs = [(frame, group.to_dict("records")) for frame, group in boxes.groupby("frame")]
    with Pool() as pool:
        rows = [row for result in tqdm(pool.imap(cut_crops, jobs), total=len(jobs)) for row in result]
    pd.DataFrame(rows).sort_values("box_id").to_csv(DATA / "crops.csv", index=False)
    save_demo_frames(boxes, usable)

    print("\n=== 4. UIEB and SUIM for the evaluation ===")
    if not (DATA / "uieb").exists():
        snapshot_download("Edddddd8787/UIEB", repo_type="dataset", local_dir=DATA / "uieb",
                          allow_patterns=["raw-890/*", "reference-890/*"])
    if not (DATA / "SUIM").exists():
        zipfile.ZipFile(download("SatwikKambham/suim", "SUIM.zip", DATA)).extractall(DATA)
    print("done")
