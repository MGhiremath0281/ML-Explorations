# Loan Approval Prediction System

## Project Overview

This project will involve developing a machine learning model to predict loan approval decisions based on customer data. The system involves data preprocessing, model training, evaluation, and deployment through a Flask web app.

## Key Features

- **Interactive Web App**: Real-time predictions through a simple interface.
- **Machine Learning Model**: K-Nearest Neighbors (KNN) optimized through hyperparameter tuning.
- **Data Processing**: Handling missing values, outliers, and encoding categorical data.
- **Model Comparison**: SVM vs KNN

## Model Comparison: SVM vs KNN

| Metric                               | Support Vector Machine (SVM) | K-Nearest Neighbors (KNN) |
|--------------------------------------|------------------------------|---------------------------|
| **Accuracy**                         | 77.24%                       | 71%                       |
| **Accuracy (Class 0)**               | 0.86                         | 0.59                      |
| Recall (Class 0)                     | 0.42                         | 0.53                      |
| F1-Score (Class 0)                   | 0.56                         | 0.56                      |
| **Accuracy (Class 1)**               | 0.75                         | 0.76                      |
| Recalls of Class 1                   | 0.96                         | 0.80                      |
| **F1-Score (Class 1)**               | 0.85                         | 0.78                      |
| **Macro Average Precision**          | 0.81                         | 0.68                      |
| **Macro Average Recall**             | 0.69                         | 0.67                      |
| **Macro Average F1-Score**           | 0.70                         | 0.67                      |
| **Weighted Average Precision**       | 0.79                         | 0.70                      |
| **Weighted Average Recall**          | 0.77                         | 0.71                      |
| Weighted Average F1-Score            | 0.75                         | 0.70                      |

### Key Takeaways from the Comparison:

- **SVM performed better overall in terms of accuracy (77.24%)** compared to KNN (71%). However, **KNN achieved a more balanced result** across both classes, with better performance in precision for Class 1 (loan approval).
- **Class 1 (loan approval) recall** is much better for SVM (0.96) than for KNN (0.80), but this comes at the expense of a much lower recall for Class 0 (denied loan). SVM had a harder time predicting Class 0, as seen in the much lower recall for Class 0 (0.42).
- **KNN** is generally better balanced for both classes, although SVM remains very slightly better when the overall F1-Score is considered for Class 1 (loan approval).
 
- **KNN provides a much more interpretable model** and is computationally simpler, while SVM can be more computationally expensive and harder to explain, particularly when using the kernel trick.

## Results

- **Accuracy**: 71%
- **Best Hyperparameters**:  n_neighbors: 3
- **Best F1-Score**: 0.804 for tuned KNN model

### Classification Metrics:

| Metric           | Precision | Recall  | F1-Score | Support |
|------------------|-----------|---------|----------|---------|
| **0 (Denied)**   | 0.59      | 0.53    | 0.56     | 43      |
| **1 (Approved)** | 0.76      | 0.80    | 0.78     | 80      |
| **Accuracy**     |           |         | 0.71     | 123     |
| **Macro avg**    | 0.68      | 0.67    | 0.67     | 123     |
| **Weighted avg** | 0.70      | 0.71    | 0.70     | 123     |

## Why the Switch from SVM to KNN?

1. **More Equal Representation Among Classes**: 
   The SVM model was doing a good job with Class 1 (positive class), but had quite a low recall for Class 0 (negative class). Recall is 0.42; therefore, it could lead to biased predictions in real life, especially when the distribution of loan approvals is not uniformly distributed.
KNN, with optimized hyperparameters (n_neighbors = 3), gives more balanced results for both classes. This makes it a better fit for real-world loan approval predictions, where you want fair and accurate predictions for both approved and denied loans.

2. **Lower Computational Overhead**:
SVM was a more computationally expensive algorithm, primarily because the algorithm used for hyperparameter tuning, GridSearchCV, makes it a very long process. So, it made the whole process of training and evaluation lengthy. 
   KNN is simpler and, thus, computationally less expensive with lower complexities of training and predicting. It saves system resources especially when the model is scaled.

3. **Interpretability and Simplicity:**
KNN is a more interpretable algorithm since it only depends on the proximity of neighboring data points for prediction. This approach makes it easier to explain to stakeholders and debug as compared to the SVM algorithm that involves more complex decision boundaries and kernel tricks.  
SVM can be more difficult to interpret and explain, especially when applying the RBF non-linear kernel.

4. **Sufficiency of Model Performance**:
Although SVM had higher overall accuracy and a better F1-Score for Class 1, the KNN model was more balanced. It gave a reasonable F1-Score of 0.804, improving recall for both classes and giving a better balance between precision and recall. This tradeoff in accuracy was acceptable in favor of a more robust and fair model.

