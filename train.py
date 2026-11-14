"""Task 4 - Classification. Ablation study: SVM, Random Forest and KNN on four setups (raw or preprocessed crops,
with or without segmentation), compared by 5-fold cross-validation on the training videos. The best classifier of
the full pipeline is then trained on all training videos and tested once on the unseen test videos."""

import json

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features
from segment import box_mask, segment

SETUPS = {
    "Raw image, no segmentation": ("raw", False),
    "Raw image + segmentation": ("raw", True),
    "Preprocessed, no segmentation": ("enhanced", False),
    "Preprocessing + segmentation (full)": ("enhanced", True),
}
FULL = "Preprocessing + segmentation (full)"


def classifiers():
    """SVM and KNN compare distances, so the features are standardised (z-scores) and reduced with PCA to the
    components that keep 95 % of the variance. Random Forest looks at one feature at a time and needs neither."""
    return {
        "SVM": make_pipeline(StandardScaler(), PCA(0.95), SVC(C=10, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "KNN": make_pipeline(StandardScaler(), PCA(0.95), KNeighborsClassifier(n_neighbors=7, weights="distance")),
    }


def load_features(crops, source, segmented):
    """One feature row per crop: raw or enhanced crop, described inside the segmented mask or the whole box."""
    rows = []
    for item in crops.itertuples():
        crop = cv2.imread(f"data/crops/{source}/{item.box_id}.png")
        box = (item.bx, item.by, item.bw, item.bh)
        mask = segment(crop, box)[0] if segmented else box_mask(crop.shape, box)
        rows.append(extract_features(crop, mask))
    return np.array(rows)


if __name__ == "__main__":
    crops = pd.read_csv("data/crops.csv")
    y, videos = crops.label.to_numpy(), crops.video.to_numpy()
    train, test = (crops.split == "train").to_numpy(), (crops.split == "test").to_numpy()
    print(f"{train.sum()} training crops from {len(set(videos[train]))} videos, "
          f"{test.sum()} test crops from {len(set(videos[test]))} other videos")

    print("\n=== 1. Ablation study: 5-fold cross-validation on the training videos ===")
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    rows, features = [], {}
    for setup, (source, segmented) in SETUPS.items():
        X = load_features(crops, source, segmented)
        features[setup] = X
        for name, model in classifiers().items():
            cv = cross_validate(model, X[train], y[train], groups=videos[train], cv=folds,
                                scoring=["accuracy", "f1_macro"])
            rows.append({"setup": setup, "classifier": name, "accuracy": cv["test_accuracy"].mean(),
                         "f1": cv["test_f1_macro"].mean()})
    ablation = pd.DataFrame(rows)
    ablation.to_csv("results/ablation.csv", index=False)
    table = ablation.pivot(index="setup", columns="classifier", values="accuracy").loc[list(SETUPS)]
    table = table[["SVM", "Random Forest", "KNN"]]
    table["Best F1"] = ablation.groupby("setup").f1.max()
    print(table.round(3).to_string())

    full = ablation[ablation.setup == FULL]
    best = full.loc[full.f1.idxmax(), "classifier"]

    print(f"\n=== 2. Final model: {best} (best F1 on the full pipeline), tested on the test videos ===")
    X = features[FULL]
    model = classifiers()[best].fit(X[train], y[train])
    predicted = model.predict(X[test])
    print(classification_report(y[test], predicted, digits=3))
    report = classification_report(y[test], predicted, output_dict=True)
    pd.DataFrame(report).T.drop(index="accuracy").to_csv("results/classification_report.csv")
    joblib.dump(model, "models/model.pkl")
    ConfusionMatrixDisplay.from_predictions(y[test], predicted, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title(f"{best}, full pipeline, unseen test videos")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")

    print("\n=== 3. Why we split by video: the same model with a random split of the crops ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    random_split = accuracy_score(y_test, classifiers()[best].fit(X_train, y_train).predict(X_test))
    print(f"split by video: {report['accuracy']:.1%}   random split: {random_split:.1%}")

    pca = PCA(0.95).fit(StandardScaler().fit_transform(X[train]))
    summary = {"final_model": best, "test_accuracy": report["accuracy"], "test_f1": report["macro avg"]["f1-score"],
               "random_split_accuracy": random_split, "features": X.shape[1], "pca_components": int(pca.n_components_)}
    with open("results/summary.json", "w") as file:
        json.dump(summary, file, indent=2)
