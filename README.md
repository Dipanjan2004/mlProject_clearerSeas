# Clearer Seas

Underwater image enhancement, segmentation and species classification using digital image processing and
machine learning. Course project for ETE551 *Digital Image Analysis Using Machine Learning*.

Dipanjan Chowdhury (1MS24ET015) and Devansh Gupta (1MS24ET012), Department of Electronics and Telecommunication
Engineering, Ramaiah Institute of Technology, 2026-27.

## What it does

Underwater photos are blue-green, hazy and low in contrast, so animals are hard to recognise. The project
classifies six species of the Brackish dataset (crab, fish, jellyfish, shrimp, small fish, starfish) in four
steps:

1. **Preprocessing:** gray-world white balance, non-local means denoising, dark channel prior dehazing and CLAHE.
2. **Segmentation:** colour difference (ΔE) from the water, Otsu threshold (k-means as an alternative), opening
   and closing, largest connected region.
3. **Features:** 58 numbers per animal: colour (25), texture with the GLCM (10) and shape with Hu moments and a
   chain code (23).
4. **Classification:** SVM, Random Forest and KNN, with standardisation and PCA for SVM and KNN.

An ablation study measures how much preprocessing and segmentation improve the classification.

## Files

| File | What it does |
|---|---|
| `enhance.py` | Task 1: preprocessing |
| `segment.py` | Task 2: segmentation |
| `features.py` | Task 3: feature extraction |
| `prepare_data.py` | Downloads the datasets, picks 300 animals per species, splits them by video and cuts raw and enhanced crops |
| `train.py` | Task 4: ablation study, model choice by cross-validation, final test; saves `models/model.pkl` |
| `evaluate.py` | Enhancement quality on UIEB, segmentation quality on SUIM and on the Brackish crops |
| `predict.py` | Classifies one animal and shows every step |
| `app.py` | Streamlit demo |
| `docs/` | Synopsis, syllabus, project report, syllabus mapping and viva questions |

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py     # downloads about 2.5 GB and cuts the crops (10 to 20 minutes)
python train.py            # ablation study and final model (about 30 seconds)
python evaluate.py         # UIEB and SUIM evaluation (about 4 minutes)
python predict.py          # one animal, step by step
streamlit run app.py       # demo in the browser
```

The trained model, the results and a few demo frames are already in the repository, so `predict.py` and
`app.py` work without running the other steps.

## Results

**Ablation study** (accuracy in 5-fold cross-validation on the 68 training videos; every fold holds whole videos):

| Setup | SVM | Random Forest | KNN | Best macro-F1 |
|---|---|---|---|---|
| Raw image, no segmentation | 57.9 % | 61.6 % | 51.0 % | 0.561 |
| Raw image + segmentation | 53.7 % | 57.5 % | 53.1 % | 0.542 |
| Preprocessed, no segmentation | 63.3 % | 66.5 % | 59.2 % | 0.629 |
| Preprocessing + segmentation (full) | 65.8 % | 64.0 % | 59.2 % | 0.622 |

Preprocessing gives the largest gain. Segmentation does not help on raw crops, where the masks are unreliable,
but it helps the SVM on preprocessed crops.

**Final model:** the SVM on the full pipeline (best cross-validation macro-F1), trained on the 68 training videos
and tested once on 16 other videos: **52.9 % accuracy**, macro-F1 0.518 (guessing gives 16.7 %). With a random
split of the same crops the accuracy would be 88.6 %, because almost identical frames of one video end up in both
sets.

**Enhancement on UIEB** (890 images): SSIM against the reference images 0.766 → 0.772, entropy 6.87 → 7.41 bits
(reference images 7.51), RMS contrast 0.140 → 0.199 (reference images 0.208).

**Segmentation:** on 3,048 hand-labelled SUIM fish the Otsu masks reach an IoU of 0.546 and are 84 % fish, against
56 % for the plain bounding box. On the Brackish crops segmentation falls back to the whole box for 8.7 % of the
raw crops but only 3.7 % of the enhanced crops.
