# Predictive Maintenance Pipeline with Machine Learning

## Introduction
This repository contains a comprehensive machine learning pipeline developed to predict equipment maintenance needs using the `predictive_maintenance.csv` dataset. Predictive maintenance is crucial in industrial applications, enabling organizations to anticipate and prevent equipment failures, reduce downtime, and improve operational efficiency. 

The project aims to predict two outcomes:
1. **Failure or Not**: Binary classification to determine whether equipment maintenance is required.
2. **Failure Type**: Multi-class classification to identify the type of failure when maintenance is needed.

This project highlights advanced data preprocessing, feature selection, hyperparameter tuning, and the integration of cutting-edge machine learning models, showcasing practical and scalable solutions for real-world predictive maintenance problems.

---

## Tools and Libraries Used

### Programming Language
- **Python 3.x**: The primary language for data analysis and machine learning.

### Libraries and Frameworks
- **NumPy**: For numerical computing and array operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization and exploratory analysis.
- **Scikit-learn**: For preprocessing, model evaluation, and feature selection.
- **Imbalanced-learn (SMOTE)**: For handling imbalanced datasets by oversampling minority classes.
- **CatBoost**: Gradient boosting algorithm tailored for categorical data, used for binary classification.
- **XGBoost**: A high-performance gradient boosting framework, used for multi-class classification.

---

## Dataset Description
The repository includes the `predictive_maintenance.csv` dataset, which consists of **10,000 rows** and **14 features**. Below is a detailed description of the dataset:

### Features
1. **UID**: Unique identifier for each observation (excluded from modeling).
2. **productID**: Encodes product quality:
   - **L**: Low quality (50% of products).
   - **M**: Medium quality (30%).
   - **H**: High quality (20%).
3. **air temperature [K]**: Generated using a random walk process, normalized to a standard deviation of 2 K around 300 K.
4. **process temperature [K]**: Derived by adding 10 K to the air temperature, normalized with a standard deviation of 1 K.
5. **rotational speed [rpm]**: Calculated based on a power of 2860 W, overlaid with normally distributed noise.
6. **torque [Nm]**: Normally distributed around 40 Nm with a standard deviation of 10 Nm.
7. **tool wear [min]**: Tool wear increases by 5/3/2 minutes for High/Medium/Low quality products, respectively.

### Targets
- **Failure or Not**: A binary target indicating if a machine failure occurred.
- **Failure Type**: A multi-class target specifying the type of failure when it occurred.

### Failure Modes
Failures are attributed to one or more of the following modes:
- Overstrain Failure
- Power Failure
- Tool Wear Failure
- Heat Dissipation Failure
- Random Failures

**Important**: Targets are not used as features to avoid data leakage.
**Dataset is from kaggle**: [https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification]

---

## Key Features and Workflow

### 1. Data Preprocessing
- **Encoding**: `LabelEncoder` was used to convert categorical variables to numerical format.
- **Scaling**: Features were standardized using `StandardScaler` for models sensitive to feature magnitudes.
- **Handling Imbalanced Data**: SMOTE was applied to balance class distributions in both binary and multi-class datasets.

### 2. Train-Test Splitting
- Data was split into training and testing sets using `train_test_split`, with a test size of 25%.
- Stratification was used to maintain class balance.

### 3. Models Used
#### **CatBoost Classifier** (Binary Classification)
- Predicts whether maintenance is required.
- Automatically balances classes using `auto_class_weights='Balanced'`.
- Configured with `random_state=42` for reproducibility.

#### **XGBoost Classifier** (Multi-Class Classification)
- Identifies the failure type when maintenance is needed.
- Tuned hyperparameters include:
  - `subsample=0.6`, `reg_lambda=0.5`, `reg_alpha=0.1`
  - `n_estimators=100`, `max_depth=5`, `learning_rate=0.3`
  - `gamma=0.9`, `colsample_bytree=1.0`

### 4. Model Evaluation
- **Cross-Validation**: 10-fold cross-validation was performed, with metrics averaged across folds.
- **Classification Reports**: Provided precision, recall, F1-score, and support for all classes.
- **Confusion Matrices**: Visualized to understand misclassifications.

### 5. Prediction Flow
1. **Step 1**: Predict if equipment failure is likely (binary classification).
2. **Step 2**: If failure is predicted, identify the failure type (multi-class classification).
3. **Mapping**: Failure type predictions were converted from numeric values to descriptive labels.

---

## Additional Notebooks
This repository also includes two Jupyter notebooks for feature selection and hyperparameter optimization:
1. **Binary Classification Notebook (machines.ipynb)**:
   - Focuses on selecting the best features and tuning hyperparameters for the CatBoost model using `RandomizedSearchCV`.
2. **Multi-Class Classification Notebook(machines2.ipynb)**:
   - Focuses on selecting the best features and tuning hyperparameters for the XGBoost model using `RandomizedSearchCV`.

These notebooks demonstrate the iterative experimentation and parameter optimization that informed the final pipeline design.

---

## Future Improvements
- **Enhanced Feature Engineering**: Introduce domain-specific features to improve predictions.
- **Hyperparameter Optimization**: Extend the search space with `GridSearchCV` or advanced tuning techniques.
- **Deployment**: Package the pipeline as a web application or API for real-time predictions.

---

## Conclusion
This project highlights the use of machine learning for predictive maintenance, demonstrating proficiency in handling imbalanced data, feature selection, model optimization, and evaluation. By providing a robust and scalable pipeline, it offers a practical solution for reducing downtime and improving operational efficiency in industrial environments.

---

## Contact
For questions, suggestions, or feedback, please reach out via:
- **Email**: [pharezayodele6@gmail.com]
- **LinkedIn**: [www.linkedin.com/in/pharez-ayodele-73b13021b]

---
