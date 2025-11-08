EMNIST Letters Classification using HOG and SVM

1. Data Processing

Data processing is a steps taken to prepare raw input data so it can be efficiently used by a machine learning model.
In my current project, the dataset used is emnist-letters-train.csv. Each row, from what ive learned, Each row represents one image with 785 columns, where the first column is the label (1–26), and the remaining 784 columns are pixel values (0–255) of a 28×28 grayscale image.
During processing, the data is split into features (X) and labels (y). The pixel values are normalized to the range [0, 1] and reshaped into 28×28 format before feature extraction using HOG.



2. LOOCV Evaluation Method

Leave-One-Out Cross-Validation (LOOCV) is an evaluation technique in which each sample is used once as a test case while all remaining samples are used for training.
This means that if the dataset contains 13,000 samples, the model will be trained and tested 13,000 times, once for every sample.
LOOCV ensures that every data point contributes to both training and evaluation, resulting in a nearly unbiased performance estimate.
Although this method is computationally expensive and time-consuming, it provides a highly reliable measure of how well the model generalizes to unseen data.



3. HOG and SVM Parameters

Parameters are specific configuration values that control how algorithms operate and affect the accuracy, speed, and generalization of a model.

HOG Parameters:
Orientations: 12 → determines the number of gradient directions considered when computing edges.

Pixels per cell: (4, 4) → defines the cell size used for gradient computation.

Cells per block: (3, 3) → controls local normalization, improving feature stability and contrast adjustment.

These parameters balance between feature detail and computation efficiency, ensuring that enough visual information is extracted without creating excessively large feature vectors.

SVM Parameters:
Kernel: Linear → separates data classes using a straight hyperplane.

C = 10.0 → controls the balance between maximizing the margin width and minimizing classification errors.

A higher C value allows fewer misclassifications, while the linear kernel keeps the model simple and effective for high-dimensional data like HOG features.



4. Evaluation Results

Evaluation metrics are quantitative measures used to assess how well a model performs on classification tasks.

After performing LOOCV, the model achieved the following metrics:

Accuracy: 80.73%
Precision: 80.64%
F1-score: 80.65%
Total runtime: 13.75 hours

5. Conclusion

The system successfully classified handwritten English letters from the EMNIST dataset using HOG for feature extraction and SVM for classification.
Although LOOCV required a long computation time, it provided a thorough and unbiased performance evaluation.
This project demonstrates that traditional machine learning techniques, when supported by effective feature extraction and parameter tuning, can still achieve strong and reliable results in image recognition tasks.
