"""Clearer Seas demo app.  Run with:  streamlit run app.py"""

import json

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from enhance import enhance, resize
from predict import classify, outline
from segment import crop_with_margin

st.set_page_config(page_title="Clearer Seas", layout="wide")
st.title("Clearer Seas")
st.write("Underwater image enhancement, segmentation and species classification "
         "using digital image processing and machine learning.")


@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")


@st.cache_data
def preprocess(frame):
    return enhance(resize(frame))


demo, results = st.tabs(["Run the pipeline", "Results"])

with demo:
    source = st.radio("Image", ["Frame from a test video", "Upload your own"], horizontal=True)
    if source == "Frame from a test video":
        frames = json.load(open("data/samples/frames.json"))
        index = st.selectbox("Frame", range(len(frames)), format_func=lambda i: f"Frame {i + 1}: "
                             + ", ".join(sorted({animal["label"] for animal in frames[i]["boxes"]})))
        frame = cv2.imread(f"data/samples/{frames[index]['image']}")
        animals = frames[index]["boxes"]
    else:
        upload = st.file_uploader("Underwater image", type=["jpg", "jpeg", "png"])
        if upload is None:
            st.stop()
        frame = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        st.write("Move the two sliders so that the box surrounds the animal.")
        left, right = st.columns(2)
        xs = left.slider("Left to right (%)", 0, 100, (20, 80))
        ys = right.slider("Top to bottom (%)", 0, 100, (20, 80))
        height, width = frame.shape[:2]
        animals = [{"x": width * xs[0] / 100, "y": height * ys[0] / 100, "w": width * (xs[1] - xs[0]) / 100,
                    "h": height * (ys[1] - ys[0]) / 100, "label": "unknown"}]

    steps = preprocess(frame)
    scale = 1280 / frame.shape[1]
    st.subheader("1. Preprocessing")
    for column, (name, image) in zip(st.columns(5), steps.items()):
        column.image(image, caption=name, channels="BGR")

    st.subheader("2. Segmentation, 3. Features and 4. Classification")
    model = load_model()
    for number, animal in enumerate(animals, start=1):
        box = [animal[k] * scale for k in "xywh"]
        result = classify(steps["CLAHE"], box, model)
        raw_crop, _ = crop_with_margin(steps["Raw"], box)
        difference = cv2.normalize(result["difference"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        columns = st.columns(5)
        columns[0].image(raw_crop, caption="Raw crop", channels="BGR", width="stretch")
        columns[1].image(result["crop"], caption="Enhanced crop", channels="BGR", width="stretch")
        columns[2].image(difference, caption="Colour difference ΔE", width="stretch")
        columns[3].image(outline(result["crop"], result["mask"]), caption="Segmented animal", channels="BGR",
                         width="stretch")
        columns[4].markdown(f"**Animal {number}: {result['species']}**  \nTrue species: {animal['label']}")

with results:
    ablation = pd.read_csv("results/ablation.csv")
    summary = json.load(open("results/summary.json"))
    st.subheader("Ablation study: accuracy in 5-fold cross-validation on the training videos")
    table = ablation.pivot(index="setup", columns="classifier", values="accuracy")
    st.dataframe(table.loc[ablation.setup.unique(), ablation.classifier.unique()].style.format("{:.1%}"))
    st.subheader(f"Final model ({summary['final_model']}, full pipeline) on the unseen test videos")
    st.write(f"Accuracy {summary['test_accuracy']:.1%}, macro F1-score {summary['test_f1']:.3f}")
    left, right = st.columns(2)
    left.image("results/confusion_matrix.png", caption="Confusion matrix of the final model")
    right.dataframe(pd.read_csv("results/classification_report.csv", index_col=0).round(3))
    st.subheader("Enhancement quality on UIEB")
    st.dataframe(pd.read_csv("results/uieb.csv", index_col=0).round(3))
    st.subheader("Segmentation quality on SUIM")
    st.dataframe(pd.read_csv("results/suim.csv", index_col=0).round(3))
