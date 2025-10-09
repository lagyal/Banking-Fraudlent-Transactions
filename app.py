import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Bank Fraud Detection", layout="wide")
st.title("Bank Fraud Detection App")
st.write("Upload a transaction dataset and predict fraudulent activities using a trained Random Forest model.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())
    st.write("**Shape:**", data.shape)
    st.write("**Missing Values:**")
    st.write(data.isnull().sum())

    if 'isFraud' in data.columns:
        x = data.drop('isFraud', axis=1)
        y = data['isFraud']
    else:
        st.warning("No 'isFraud' column found — assuming this is new data for prediction.")
        x = data
        y = None

    x = pd.get_dummies(x)

    model_loaded = False
    try:
        with open("fraud_model.pkl", "rb") as f:
            model = pickle.load(f)
        if hasattr(model, "predict"):
            st.success("✅ Loaded existing model (fraud_model.pkl)")
            model_loaded = True
        else:
            st.warning("Loaded file is not a valid model. Retraining...")
    except FileNotFoundError:
        st.warning("No pre-trained model found. Training a new Random Forest model...")

    if not model_loaded:
        if y is not None:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            st.write("### Model Evaluation:")
            st.write(f"**Accuracy:** {acc:.4f}")
            st.text(classification_report(y_test, y_pred))

            with open("fraud_model.pkl", "wb") as f:
                pickle.dump(model, f)
                st.success("💾 Model saved as fraud_model.pkl")
        else:
            st.error("Cannot train model: dataset does not have labels ('isFraud'). Please upload a labeled dataset first.")
            st.stop()

    if st.button("Predict Fraud"):
        preds = model.predict(x)
        st.subheader("Prediction Results (Is Fraud/ Not Fraud)")
        result_df = pd.DataFrame({
            "Transaction_ID": range(len(preds)),
            "Fraud_Prediction": ["Is Fraud" if p == 1 else "Not Fraud" for p in preds]
        })
        st.dataframe(result_df)

        st.download_button(
            label="Download Predictions as CSV",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name='fraud_predictions.csv',
            mime='text/csv'
        )

else:
    st.info("Please upload a CSV file to begin.")
