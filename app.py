import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load trained model saved with pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

st.title("Credit Card Fraud Detection system")
st.markdown("Predict whether a transaction is fraudulent or innocuous using a trained machine learning model.")

# Plot class distribution (Improved)
st.subheader("Transaction Class Distribution")

# Calculate counts and percentages
class_counts = df["Class"].value_counts().sort_index()
class_labels = ["Legit (0)", "Fraud (1)"]
class_percentages = (class_counts / class_counts.sum()) * 100

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(class_labels, class_counts, color="steelblue")

# Add value labels above bars
for bar, count, pct in zip(bars, class_counts, class_percentages):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 500,
            f"{count:,} ({pct:.2f}%)", ha='center', fontsize=10, fontweight='bold')

# Axis labels and title
ax.set_ylabel("Number of Transactions", fontsize=11)
ax.set_title("Class Balance: Legit vs. Fraud", fontsize=13, fontweight='bold')
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

st.pyplot(fig)


# Choose class to filter by
label_choice = st.radio("Choose transaction type to explore:", ["Legit", "Fraud"])

# Filter data
if label_choice == "Fraud":
    filtered_df = df[df["Class"] == 1]
else:
    filtered_df = df[df["Class"] == 0]

# Ensure at least one transaction exists
if len(filtered_df) == 0:
    st.warning("No transactions of this type found.")
else:
    # Choose a transaction
    index = st.slider("Select transaction index", 0, len(filtered_df) - 1, 0)
    selected_txn = filtered_df.iloc[index]

    st.subheader("🔍 Transaction Details")
    st.write(selected_txn.drop("Class"))  # hide true label

    # Prepare input
    input_data = selected_txn.drop("Class").values.reshape(1, -1)

    # Predict
    if st.button("Predict Fraud"):
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"Fraud Detected with {prob:.2%} confidence!")
        else:
            st.success(f"Legitimate Transaction with {1 - prob:.2%} confidence.")

        # Show actual label
        st.info(f"Actual Label: {'Fraud' if selected_txn['Class'] == 1 else 'Legit'}")
