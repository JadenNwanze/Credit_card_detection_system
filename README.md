# Credit Card Fraud Detection System (CCFDS)

This is a machine learning web application that predicts whether a credit card transaction is fraudulent or legitimate. The system uses an ensemble model trained on a real-world dataset and includes SHAP-based explanations to make predictions transparent and understandable.

Live Demo: https://creditcarddetectionsystem-egseuj4mjysbwzs6benygd.streamlit.app/

---

## Problem Statement

Financial fraud detection involves identifying suspicious transactions that may indicate fraudulent behavior. As digital financial systems grow, the need to proactively detect such anomalies has become increasingly important.

This project aims to support this need by building a model that can accurately classify credit card transactions as either legitimate or fraudulent. The goal is to provide an interpretable, deployable fraud detection system for demonstration and educational purposes.

---

## Dataset

The dataset used was obtained from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013. The data is highly imbalanced, with fraudulent transactions accounting for only 0.172% of the total.

Due to its size, the dataset is tracked in the repository using Git Large File Storage (Git LFS).

---

## Methodology

### Preprocessing Techniques

- **SMOTE** (Synthetic Minority Over-sampling Technique) was used to address class imbalance.
- **StandardScaler** was applied to normalize the feature space.
- **Pickle** was used to serialize the trained model.
- **Git LFS** was used for tracking large files like the dataset and model.

### Models Used

- Logistic Regression  
- Random Forest  
- XGBoost  
- A **Voting Classifier** was used to combine these models and improve performance.

---

## Tools and Libraries

- Python
- scikit-learn
- XGBoost
- SHAP
- Pandas and NumPy
- Streamlit (for deployment)
- Git & Git LFS

---

## SHAP Interpretability

The application includes SHAP (SHapley Additive exPlanations) visualizations, which highlight which features contributed most to each model decision. This allows for a deeper understanding of how individual predictions are made.

---

## Running the App Locally

```bash
# Step 1: Clone the repository
git clone https://github.com/JadenNwanze/Credit_card_detection_system.git
cd Credit_card_detection_system

# Step 2: Install required packages
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py
