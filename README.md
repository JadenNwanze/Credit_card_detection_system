#  Credit Card Fraud Detection System (CCFDS)

An end-to-end **Credit Card Fraud Detection System** that leverages machine learning algorithms to detect fraudulent transactions in real time. This project also implements modern **MLOps best practices** to ensure reproducibility, scalability, and maintainability throughout the machine learning lifecycle.

---

##  Project Overview

Credit card fraud is a growing concern in the financial sector, resulting in billions of dollars in losses annually. This system uses historical credit card transaction data to train classification models that can predict whether a transaction is fraudulent or legitimate.

The goal is to:
- Detect fraudulent activity with high precision and recall.
- Minimize false positives that may inconvenience customers.
- Deploy a scalable and reliable fraud detection system using MLOps principles.

---

##  Dataset

The system uses the **[Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)**. It contains 284,807 transactions with 492 frauds (0.172% of all transactions), making it a highly imbalanced dataset.

**Properties:**
- 28 anonymized principal components (V1–V28) from PCA
- `Time`, `Amount`
- `Class`: Target label (1 = fraud, 0 = legit)

---

##  Machine Learning Models and tools utilized

Ensembled models were trained to improve prediction performance:

- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Voting Classifier Ensemble**
-  **SHAP Interpretability** to provide magnitude of influence of features
- **SMOTE Technique** for handling data imbalance which is natural in eal anomaly datasets
- **Pickle** for file serialization


---

## Performance Metrics

Evaluation was conducted using stratified train-test splits and cross-validation. Metrics included:

- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **AUC-ROC Curve**

Due to the class imbalance, special attention was paid to recall and precision on the fraud class.

---

##  MLOps Methodologies Implemented

This project incorporates the following MLOps tools and practices:

### Version Control
- **Git**: All source code, notebooks, and experiments are version-controlled.
- .gitignore is properly configured for data, models, and logs.

###  Dependency Management
- requirements.txt is used to manage all Python package dependencies for reproducibility.

###  Modular Project Structure

The codebase follows a modular programming paradigm

### Experiment Tracking
- **DVC (Data Version Control)** is used for:
  - Tracking raw and processed data
  - Managing model artifacts (model.pkl)
  - Reproducing experiments
  - Versioning metrics.json

### Model Evaluation
- Evaluation metrics are tracked and stored with DVC to ensure transparency and reproducibility.

### Containerization
- **Docker** is used to build and package the application with all its dependencies for consistent deployment across different computing devices.

### Monitoring 
Although no monitoring was done, but future integration of monitoring tools will be used
- Integrating **Prometheus** and **Grafana** for real-time monitoring.
- Adding **MLflow** for experiment tracking.

---

