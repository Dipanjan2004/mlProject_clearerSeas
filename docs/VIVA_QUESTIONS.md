# Viva questions and answers

Short answers you can say aloud. Numbers are this project's results.

## About the project

**1. What does your project do, in one sentence?**
It cleans up underwater images, cuts each animal out of the water, describes it with 58 colour, texture and shape
numbers, and classifies six species with SVM, Random Forest and KNN.

**2. Why are underwater images blue-green and hazy?**
Water absorbs red light first, so blue-green remains. Particles in the water scatter light, which adds a bright
veil and lowers contrast. The model is I = J·t + A·(1 − t): I is the observed image, J the clear scene, A the
colour of the water light and t the transmission.

**3. How is the code organised?**
One file per task: `enhance.py` (preprocessing), `segment.py` (segmentation), `features.py` (features) and
`train.py` (classification). `prepare_data.py` downloads the data and cuts the crops, `evaluate.py` measures the
enhancement and segmentation, `predict.py` classifies one animal and `app.py` is the Streamlit demo.

**4. What is the main result?**
Preprocessing gives the largest gain. In cross-validation on the training videos the best macro-F1 rises from
0.561 on raw crops to 0.629 with preprocessing (0.622 with segmentation as well). The final SVM reaches 52.9 %
accuracy on 16 videos it never saw.

**5. Isn't 52.9 % accuracy low?**
Guessing would give 16.7 % for six species, and the test videos were never seen in training, in very murky water.
With a random split of the same crops the accuracy is 88.6 %, but only because almost identical frames of one
video end up in both sets. 52.9 % is the honest number.

**6. What is an ablation study?**
Switching one part on or off to measure what it adds. We compared four setups (raw or preprocessed crops, each with
or without segmentation) with all three classifiers.

## Unit I: Image fundamentals

**7. What is a digital image?**
A function f(x, y) with discrete coordinates and values, stored as an M × N matrix of pixels; a colour image has
three such matrices. OpenCV stores them in the order B, G, R.

**8. Where do sampling and quantization appear?**
Sampling: the frames are resized from 1920 × 1080 to 1280 × 720. Quantization: the GLCM reduces the 256 grey
levels to 32 so that the 32 × 32 matrix is not too sparse.

**9. Where do you use connectivity?**
After thresholding we keep only the largest connected region inside the box. Its outer boundary is traced as an
8-connected contour and filled, which also removes holes.

**10. Which distance measures do you use?**
The colour difference ΔE is the Euclidean distance between two colours in Lab. KNN uses the Euclidean distance
between feature vectors.

**11. Which colour models do you use and why?**
BGR to read and save images, HSV for the hue and saturation histograms, and Lab for ΔE and CLAHE because Lab
separates lightness (L) from colour (a, b) and its distances match how people see colour differences.

**12. What is the machine-learning workflow?**
Collect data, split it into training and test sets, extract features, train, tune or choose with cross-validation,
test once, and deploy (our Streamlit app).

## Unit II: Preprocessing

**13. Explain gray-world white balance.**
It assumes that the average colour of a scene is grey. Each channel is multiplied by mean(B, G, R) / mean(channel).
Red is weak underwater, so it gets the largest gain and the blue-green cast disappears.

**14. What is non-local means denoising?**
Each pixel becomes a weighted average of pixels whose 7 × 7 neighbourhood looks similar, searched in a 21 × 21
window. Unlike a Gaussian blur it keeps edges. We use a small strength, h = 2, because stronger denoising also
removes the fine texture of the animals.

**15. Why denoise before dehazing?**
Dehazing divides by the transmission t, which can be as small as 0.3, so any noise would be amplified up to 3.3
times.

**16. Explain the dark channel prior.**
In clear images, almost every small patch has a pixel that is very dark in at least one colour channel. The dark
channel is the minimum over the three channels and over a 15 × 15 patch. Haze lifts this minimum, so a bright dark
channel means thick haze.

**17. How do you dehaze with it?**
The water light A is the mean colour of the 0.1 % haziest pixels. The transmission is t = 1 − ω · dark(I / A) with
ω = 0.8, smoothed with a Gaussian blur. The clear image is J = (I − A) / max(t, 0.3) + A.

**18. Why smooth the transmission map?**
The minimum filter makes it blocky, and blocks would show up as squares in the result. A Gaussian blur removes
them.

**19. What are ω and t_min?**
ω = 0.8 keeps a little haze so the image looks natural. t_min = 0.3 stops the division from blowing up noise in
very hazy areas.

**20. What is CLAHE and why only on L?**
Contrast-limited adaptive histogram equalization equalizes the histogram in 8 × 8 tiles and clips each tile's
histogram at 2.0 so noise is not amplified. It is applied only to the lightness channel so the colours that white
balance and dehazing corrected stay the same.

**21. What is histogram equalization?**
It maps grey levels through the cumulative histogram, s_k = (L − 1) Σ p(r_j), so the levels spread over the whole
range and contrast increases.

**22. Which smoothing filters do you use?**
A minimum filter (erosion) for the dark channel, Gaussian blurs for the transmission map and before ΔE, and
non-local means for noise.

**23. Does the enhancement have any problems?**
The lamp in the Brackish scene lights the water unevenly, so the open water can turn violet after dehazing. The
frames are compressed video, so stretching the smooth water also shows small steps (banding). The animals are
affected much less.

## Unit III: Segmentation

**24. Why segment at all?**
So that the features describe the animal and not the water around it.

**25. Why crop with a 25 % margin?**
The margin is mostly water, so its median colour gives the water colour that the animal is compared with.

**26. What is ΔE?**
The Euclidean distance in Lab between a pixel's colour and the water colour. The animal has a large ΔE, the water
a small one.

**27. Explain Otsu's method.**
It tries every threshold and keeps the one that maximises the between-class variance P1·P2·(m1 − m2)², where P1,
P2 are the fractions of pixels below and above the threshold and m1, m2 their means. It needs no manual
threshold.

**28. Why threshold ΔE instead of the grey image?**
An animal can be as bright as the water but a different colour; ΔE uses all three Lab channels.

**29. How does the k-means alternative work?**
It groups the crop's pixel colours into two clusters, and the cluster whose centre is farther from the water colour
is the animal.

**30. What do opening and closing do?**
Opening (erosion then dilation) removes small specks. Closing (dilation then erosion) fills small gaps. The
structuring element is a disk about 8 % of the box's shorter side.

**31. What if segmentation fails?**
If the mask covers less than 6 % of the box, the whole box is used instead. This happens for 8.7 % of the raw
crops but only 3.7 % of the enhanced crops.

**32. How good is your segmentation?**
On 3,048 hand-labelled SUIM fish, Otsu reaches a mean IoU of 0.546 and Dice of 0.672 on enhanced images. The plain
box scores a similar IoU (0.562) because the fish fill most of their box, but only 56 % of the box is fish, while
84 % of our mask is fish.

**33. IoU versus Dice?**
IoU = |A ∩ B| / |A ∪ B|. Dice = 2|A ∩ B| / (|A| + |B|). Dice is always at least as large as IoU.

## Unit IV: Features

**34. Which features do you use?**
25 colour features (mean and standard deviation of B, G, R and L, a, b, hue and saturation histograms, contrast to
the surroundings), 10 texture features (GLCM and histogram entropy and uniformity) and 23 shape features (region
descriptors, Hu moments and a chain-code histogram).

**35. What is a chain code?**
We walk along the boundary pixel by pixel; each step goes in one of 8 directions, numbered 0 to 7 (Freeman code).
The histogram of these directions describes the outline.

**36. Name your region descriptors.**
Area, perimeter, length and width of the rotated bounding rectangle, aspect ratio, circularity 4πA / P² (1 for a
circle), extent (area / bounding box area) and solidity (area / convex hull area).

**37. What are Hu moments and why the log?**
Seven combinations of normalized central moments that stay the same when the shape is moved, scaled or rotated.
They span many orders of magnitude, so we use −sign(h) · log10|h|.

**38. What is a GLCM?**
The grey-level co-occurrence matrix counts how often a pixel with grey level i has a neighbour with level j at a
given distance and direction. Contrast, homogeneity, energy and correlation summarise it.

**39. How did you compute the GLCM only on the animal?**
Animal pixels get levels 1 to 32 and water pixels level 0. After counting, row 0 and column 0 are dropped, so
pairs that touch the water are ignored.

**40. What are entropy and uniformity of the histogram?**
Entropy −Σ p log2 p is high when many grey levels occur (rough texture); uniformity Σ p² is high when a few levels
dominate (smooth texture).

## Unit V: Machine learning

**41. Supervised versus unsupervised learning?**
Supervised learning uses labels: SVM, Random Forest and KNN. Unsupervised learning finds structure without
labels: k-means segmentation and PCA.

**42. Why standardise the features?**
Area is in hundreds of pixels while circularity is below 1. z = (x − μ) / σ gives every feature the same scale,
otherwise the large ones dominate the distances.

**43. What is PCA and why only for SVM and KNN?**
PCA turns correlated features into uncorrelated components ordered by variance; 25 components keep 95 % of the
variance of the 58 features. SVM and KNN use distances and benefit from it. Random Forest splits on one feature at
a time and does not need scaling or PCA.

**44. Explain the SVM.**
It finds the boundary with the largest margin between classes; six classes are handled in pairs. The RBF kernel
lets the boundary curve. C = 10 controls how strictly training errors are punished.

**45. Explain Random Forest and KNN.**
Random Forest trains 300 decision trees on random samples and random feature subsets and takes a vote. KNN gives a
new animal the majority species of its 7 nearest training animals, closer ones counting more.

**46. What is a confusion matrix?**
Rows are the true species, columns the predicted ones; the diagonal counts correct predictions. Our biggest
mistakes are shrimp predicted as small fish (31) and jellyfish predicted as fish (29).

**47. Define accuracy, precision, recall and F1.**
Accuracy = correct / all. Precision = TP / (TP + FP). Recall = TP / (TP + FN). F1 = 2PR / (P + R). The macro
average gives each species the same weight.

**48. What is cross-validation and how did you use it?**
The training data is split into 5 parts; each part is tested once by a model trained on the other four. Our parts
contain whole videos. We used it to compare the setups and to pick the final classifier: SVM had the best macro-F1
on the full pipeline (0.622, against 0.604 for Random Forest and 0.562 for KNN).

## Design decisions

**49. Why split by video instead of randomly?**
Neighbouring frames of a video are almost identical. A random split puts near copies in both sets and the model
looks better than it is (88.6 % instead of 52.9 %). The dataset's own split has the same problem.

**50. How did you choose the 1,800 animals?**
Boxes smaller than 24 pixels or cut by the frame border were removed, then 300 random boxes were taken per
species so that every class has the same size.

**51. Why is the test accuracy (52.9 %) lower than the cross-validation accuracy (65.8 %)?**
The 16 test videos have their own water and lighting, and 16 videos is a small sample: one video can change the
result a lot (for example, almost half of the test jellyfish were predicted as fish). Cross-validation over 68 videos is the
more stable number for comparing setups.

**52. Why fixed classifier settings instead of a grid search?**
Common settings (C = 10, 300 trees, k = 7) already work well, and fewer choices keep the project easy to explain and
reduce the risk of fitting the validation data.

**53. Why does segmentation not help on raw crops?**
In murky water the animal barely differs from the water, so the masks are unreliable. After preprocessing the
masks are much better (fallbacks drop from 8.7 % to 3.7 %) and segmentation helps the SVM.

**54. Why does preprocessing help so much?**
In raw crops every animal is greenish. White balance and dehazing bring back the real colours, so the colour
features can separate the species.

**55. Why is PSNR lower after enhancement on UIEB?**
PSNR compares pixel values directly, and dehazing changes the overall brightness. SSIM, which compares structure,
is slightly higher (0.766 → 0.772), and the entropy and contrast move close to the reference images.

**56. Why hand-crafted features instead of deep learning?**
Every step can be explained and visualised, and it trains in seconds on a laptop with only 1,460 training animals.

**57. What would you improve?**
More training videos, features learned by a small neural network, and following an animal over several frames
instead of classifying a single frame.

**58. How long does everything take?**
`prepare_data.py` 10 to 20 minutes depending on the internet speed (it downloads about 2.5 GB and enhances
1,710 frames), `train.py` about 30 seconds, `evaluate.py` about 4 minutes.
