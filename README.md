# BreastCancerClassification
# Breast Cancer Classification: Comparing Random Forest and SVM

This project explores the application of machine learning for breast cancer classification using the Breast Cancer Wisconsin (Diagnostic) Data Set. It focuses on comparing the performance of two popular algorithms: Random Forest and Support Vector Machine (SVM).

## Project Goal

The primary goal is to build and evaluate machine learning models that can accurately classify breast tumor samples as malignant or benign based on features extracted from digitized images of fine needle aspirates (FNAs). The project also investigates the impact of different SVM kernels and the importance of feature scaling.

## Dataset

The project utilizes the Breast Cancer Wisconsin (Diagnostic) Data Set from the UCI Machine Learning Repository. This dataset contains 569 samples, each with 30 features computed from FNA images, including characteristics like radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry of the cell nuclei.

## Model Selection

### Random Forest Classifier

Random Forests were chosen for this project due to their several advantages in handling this type of data:

* **High Dimensionality:**  The dataset has 30 features, and Random Forests can effectively handle high-dimensional data without requiring extensive feature selection.
* **Robustness:** Random Forests are robust to noise and outliers, which can be present in real-world medical data.
* **Interpretability:** They provide insights into feature importance, helping to understand which features are most relevant for classification.
* **Non-parametric:** Random Forests make no assumptions about the underlying data distribution, making them suitable for a variety of datasets.

### Support Vector Machine (SVM)

SVMs were chosen for their ability to create flexible decision boundaries and their effectiveness in various classification tasks. The project explores different SVM kernels:

* **Linear Kernel:** Suitable for linearly separable data.
* **RBF Kernel:** Can capture non-linear relationships in the data.

The project also investigates the impact of feature scaling on SVM performance, particularly with the RBF kernel.

## Methodology

The project follows these steps:

1. **Data Loading and Preparation:** The dataset is loaded and prepared for model training.
2. **Model Training:** Random Forest and SVM models are trained on the data.
3. **Hyperparameter Tuning:** Grid search is used to find the best hyperparameters for each model.
4. **Feature Scaling:**  The effect of feature scaling on SVM performance is examined.
5. **Model Evaluation:** The models are evaluated using accuracy, F1 score, and confusion matrix.
6. **Feature Importance Analysis:** The importance of different features in the Random Forest model is analyzed.

## Results

The project demonstrates the effectiveness of both Random Forest and SVM for breast cancer classification. The results highlight the importance of choosing the appropriate kernel for SVM and the impact of feature scaling on model performance.

* **Random Forest:** Achieved high accuracy (around 0.965) and F1 score (around 0.952).
* **SVM (linear kernel):** Showed good performance with accuracy around 0.947 to 0.956.
* **SVM (RBF kernel):** Performed poorly initially but improved significantly after feature scaling, achieving accuracy of 0.982.

The project provides valuable insights into applying machine learning for breast cancer classification and the factors that can influence model performance.
