# Credit Card Fraud Detection System (CCFDS)

This is a machine learning web application that predicts whether a credit card transaction present in used dataset is fraudulent or legitimate. The system uses an ensemble model trained on a real-world dataset.
 
## Problem Statement

Financial fraud detection involves identifying suspicious transactions that may indicate fraudulent behavior. As digital financial systems grow, the need to detect such  has become increasingly important.

This project aims to support this need by building a model that can accurately classify credit card transactions as either legitimate or fraudulent. The goal is to provide an interpretable, deployable fraud detection system for demonstration purposes.

---

## Dataset

The dataset used was obtained from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013. The data is highly imbalanced, with fraudulent transactions accounting for only 0.172% of the total.

Note:
Due to its size, the dataset is tracked in the repository using Git Large File Storage (Git LFS).

---

## Methodology utilized

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

The application includes SHAP (SHapley Additive exPlanations) visualizations, which highlight which features contributed most to each of the model's decisions. This allows for a better understanding of how individual predictions are made.

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
```

## Try the Live App
A Streamlit version of the app is hosted here:
https://creditcarddetectionsystem-egseuj4mjysbwzs6benygd.streamlit.app/

## About Me
I’m a dedicated Computer Science student with a strong passion for Artificial Intelligence. I thrive on solving complex problems and am continually seeking opportunities to apply and expand my machine learning skills in practical, real-world projects.

I developed this Credit Card Fraud Detection System (CCFDS) to explore how machine learning can be applied to high-stakes problems like financial fraud prevention. This project allowed me to dive deep into ensemble modeling, handling imbalanced data, model interpretability using SHAP, and deploying an interactive AI system using Streamlit.


Contact info:

GitHub: https://github.com/JadenNwanze

Email: Jadennwanze@gmail.com

LinkedIn: [https://www.linkedin.com/in/jaden-nwanze-32579b29b/](https://www.linkedin.com/in/jadennwanze/)

## 🔗 Related Projects

- **LeNet-5: CNN for Handwritten Digit Recognition**  
  A faithful implementation of the classic LeNet-5 architecture from the paper *"Gradient-Based Learning Applied to Document Recognition"*. This project reinforces my understanding of convolutional neural networks (CNNs) and showcases my ability to reproduce foundational deep learning research.  
   [GitHub Repository](github.com/JadenNwanze/lenet5-mnist-reproduction)


## License
This project is open-source and licensed under the MIT License.
Feel free to use, modify, and distribute it with proper attributions.


