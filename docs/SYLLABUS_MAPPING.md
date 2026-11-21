# Syllabus mapping: ETE551 Digital Image Analysis Using Machine Learning

Where each unit of the syllabus is used in the project. Locations are written as `file.py: function`.

## Unit I: Introduction and digital image fundamentals

| Topic | Where | How it is used |
|---|---|---|
| Digital image as a function f(x, y) | every file | An image is an M × N × 3 array of 8-bit values (OpenCV stores the channels as B, G, R) |
| Fundamental steps of image processing | `prepare_data.py` → `train.py` | Acquisition → enhancement → segmentation → description (features) → recognition (classification) |
| Machine learning workflow | `prepare_data.py`, `train.py`, `app.py` | Collect data, split by video, extract features, train, evaluate, use the model in the app |
| Image sensing and acquisition | `prepare_data.py` | Brackish frames come from a fixed underwater video camera |
| Sampling | `enhance.py: resize` | Frames are resampled from 1920 × 1080 to 1280 × 720 |
| Quantization | `features.py: texture_features` | The 256 grey levels are reduced to 32 levels for the GLCM |
| Neighbours, adjacency and connectivity | `segment.py: segment` | The mask keeps the largest 8-connected region |
| Regions and boundaries | `segment.py: segment`, `features.py: shape_features` | The region's outer boundary is traced, filled (so holes disappear) and used for the shape features |
| Distance measures | `segment.py: segment`, `train.py` (KNN) | ΔE is the Euclidean distance between two colours in Lab; KNN uses the Euclidean distance between feature vectors |
| Basics of colour images | `enhance.py: clahe`, `features.py: colour_features` | BGR, HSV and Lab colour spaces |

## Unit II: Intensity transformations and spatial filtering

| Topic | Where | How it is used |
|---|---|---|
| Intensity transformations | `enhance.py: white_balance`, `enhance.py: dehaze` | White balance multiplies each channel by a gain; dehazing maps every pixel with J = (I − A) / t + A |
| Histogram processing | `enhance.py: clahe`, `features.py: texture_features` | CLAHE equalizes the histogram in 8 × 8 tiles; histogram entropy and uniformity are texture features |
| Smoothing spatial filters | `enhance.py: dark_channel`, `enhance.py: dehaze`, `segment.py: segment` | Minimum filter (erosion) for the dark channel, Gaussian blur for the transmission map and before ΔE |
| Types of noise and denoising | `enhance.py: denoise` | Non-local means removes the sensor noise of the video frames |

## Unit III: Image segmentation

| Topic | Where | How it is used |
|---|---|---|
| Thresholding | `segment.py: segment` | Otsu's method on the colour difference ΔE (main method) |
| K-means clustering | `segment.py: segment` (`method="kmeans"`) | Alternative method: two colour clusters, the one farther from the water is the animal |
| Morphology (opening, closing) | `segment.py: segment` | Removes specks and fills small gaps in the mask |
| Applications of segmentation | `train.py`, `evaluate.py` | The mask decides which pixels the features describe; IoU and Dice on SUIM measure its quality |

## Unit IV: Feature descriptors

| Topic | Where | How it is used |
|---|---|---|
| Chain codes | `features.py: shape_features` | Histogram of the 8 Freeman directions along the boundary |
| Boundary descriptors | `features.py: shape_features` | Perimeter and circularity 4πA / P² |
| Statistical moments | `features.py: colour_features`, `features.py: shape_features` | Mean and standard deviation of every colour channel; Hu's seven moment invariants |
| Region descriptors | `features.py: shape_features` | Area, length, width, aspect ratio, extent, solidity |
| Texture: histogram based | `features.py: texture_features` | Entropy and uniformity of the grey-level histogram |
| Texture: GLCM | `features.py: texture_features` | Contrast, homogeneity, energy and correlation at distances 1 and 2 |

## Unit V: Machine learning

| Topic | Where | How it is used |
|---|---|---|
| Supervised and unsupervised learning | `train.py`, `segment.py` | SVM, Random Forest and KNN are supervised; k-means and PCA are unsupervised |
| Principal component analysis | `train.py: classifiers` | Keeps 95 % of the variance before SVM and KNN |
| Support vector machine | `train.py: classifiers` | RBF kernel, C = 10 |
| Confusion matrix and classification metrics | `train.py` | Accuracy, precision, recall, F1-score, confusion matrix |

## Syllabus topics the pipeline does not need

| Topic | Reason |
|---|---|
| Sharpening filters (first and second derivatives) | CLAHE already restores local contrast; sharpening would also amplify the noise |
| Point, line and edge detectors | The animal is found by its colour difference from the water, not by its edges |
| Region growing, region splitting and merging | Otsu thresholding and k-means are enough inside a small crop |
| Image augmentation | The classes are already balanced (300 animals each) |
