# TrustCheck

# Fake News Classification Project

## Overview

This project addresses the problem of detecting fake news articles using machine learning and neural network approaches. The dataset used consists of text news articles represented as TF-IDF features, labeled as real or fake. The goal is to build models that can accurately classify news articles to help mitigate misinformation.

## Dataset

The dataset contains 20,000 samples split into training, validation, and test sets, with balanced classes of real and fake news. Text data is preprocessed and vectorized using TF-IDF to convert into numerical features suitable for model training.

---

## Discussion of Findings

| Model Instance | Optimizer / Algorithm | Regularization | Early Stopping | Layers | Learning Rate | Accuracy | F1 Score | Recall | Precision |
|----------------|----------------------|----------------|----------------|--------|---------------|----------|----------|--------|-----------|
| Instance 1     | Adam (Neural Net)     | None           | No             | 2      | 0.001         | 0.6463   | 0.6289   | 0.6030 | 0.6572    |
| Instance 2     | RMSprop (Neural Net)  | L2             | Yes            | 3      | 0.001         | 0.6840   | 0.6643   | 0.6291 | 0.7037    |
| Instance 3     | Adam (Neural Net)     | L1             | Yes            | 2      | 0.0005        | 0.6900   | 0.6877   | 0.6868 | 0.6886    |
| Instance 4     | RMSprop (Neural Net)  | L1_L2          | Yes            | 3      | 0.0007        | 0.6787   | 0.6550   | 0.6137 | 0.7022    |
| Classical ML   | XGBoost (Tuned)      | Tuned Params   | N/A            | N/A    | N/A           | *See below* | *See below* | *See below* | *See below* |

*Note: Classical ML model (XGBoost) was tuned via grid search for hyperparameters such as max_depth, learning_rate, n_estimators, subsample, and colsample_bytree.*

---

## Summary of Which Combination Worked Better

- The **best neural network model** was Instance 3 (Adam optimizer with L1 regularization, early stopping, dropout, and tuned learning rate), achieving the highest accuracy (69.00%) and balanced F1 score (68.77%).
- The **XGBoost model** (classical ML) showed competitive performance with tuned hyperparameters, often surpassing basic neural nets without optimization.
- Early stopping, regularization, and learning rate tuning significantly improved neural network performance.
- Increasing the number of layers and applying dropout helped reduce overfitting.

---

## Comparison: ML Algorithm vs Neural Network

- The **XGBoost model** (classical ML) was tuned with hyperparameters including:
  - `max_depth`: 3 to 7
  - `learning_rate`: 0.01 to 0.2
  - `n_estimators`: 100 to 200
  - `subsample` and `colsample_bytree`: 0.8 to 1.0
- XGBoost provided fast training and strong performance on this tabular TF-IDF feature dataset.
- Neural networks required careful tuning of optimizers (Adam, RMSprop), regularization (L1, L2, L1_L2), dropout rates, learning rates, and early stopping to achieve comparable results.
- Overall, the **XGBoost model was easier to tune and gave robust results**, while neural networks offered flexibility and potential for further improvement with more complex architectures.

---

## Video Presentation

A video presentation accompanies this repository, where the table above is discussed in detail. The video includes:

- Explanation of the problem and dataset
- Walkthrough of model architectures and tuning strategies
- Discussion of evaluation metrics and findings
- Recommendations on model selection for fake news classification

**Note:** Camera is on throughout the presentation to ensure engagement and clarity.

---

## How to Use This Repository

1. Clone the repository:

   git clone [trustcheck](https://github.com/ngabo-dev/TrustCheck.git)

2. Install dependencies (example):

   pip install -r requirements.txt

3. Run notebooks or scripts to reproduce training and evaluation.
4. Load saved models from the `saved_models` directory for inference.

---

## Acknowledgements

- Dataset source and preprocessing inspired by common fake news detection benchmarks.
- XGBoost and TensorFlow/Keras libraries for model development.
- Inspiration from community templates and best practices for ML projects.

---

## Contact

For questions or collaboration, please contact:

- Jean Pierre NIYONGABO  
- Email: j.niyongabo@alustudent.com  
- GitHub: [ngabo-dev](https://github.com/ngabo-dev)

---

*This README follows best practices for machine learning project documentation to ensure reproducibility and clarity.*
