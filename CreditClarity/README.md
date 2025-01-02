# Loan Approval Prediction System

## Project Overview
A machine learning project aimed at predicting loan approval decisions using customer data. This project includes data preprocessing, model training, evaluation, and deployment via a Flask web app.

## Key Features
- **Interactive Web App**: Real-time predictions using a simple interface.
- **Machine Learning Model**: Support Vector Machine (SVM) optimized through GridSearchCV.
- **Data Processing**: Handling missing values, outliers, and encoding categorical data.

## Results
- **Accuracy**: 78.86%
- **Best Hyperparameters**:
  - `C`: 10
  - `gamma`: 0.01
  - `kernel`: RBF

- **Classification Metrics**:
           precision    recall  f1-score   support

       0       0.95      0.42      0.58        43
       1       0.76      0.99      0.86        80

accuracy                           0.79       123

## Challenges
- Handling missing and skewed data.
- Managing imbalanced classes in the target variable.
- Computational overhead during hyperparameter tuning.

## Future Enhancements
- Experiment with advanced models like Gradient Boosting.
- Implement batch prediction capabilities.

---
