import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Streamlit App Configuration

st.set_page_config(page_title="SecurePay Credit Card Anomaly Detection", layout="wide")
st.title("💳 SecurePay Credit Card Anomaly Detection Dashboard")

st.markdown("""
This app detects **fraudulent transactions** using an **Isolation Forest** anomaly detection model.  
You can upload a CSV file or manually enter a single transaction.
""")


# Helper Functions

@st.cache_data
def train_model(df):
    """Train Isolation Forest on numeric features"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = IsolationForest(contamination=0.002, random_state=42)
    model.fit(X_scaled)
    scores = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)

    df_results = df.copy()
    df_results["anomaly_score"] = scores
    df_results["anomaly_label"] = np.where(preds == -1, "Fraud", "Normal")

    return model, scaler, df_results

def plot_pca(df, labels):
    """Visualize PCA reduction with anomaly labels"""
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
    pca_df['Label'] = labels

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Label',
        data=pca_df, palette={'Normal': 'blue', 'Fraud': 'red'}, alpha=0.6, ax=ax
    )
    plt.title("PCA Visualization of Transactions (Fraud vs Normal)")
    st.pyplot(fig)


# File Upload or Manual Input

uploaded_file = st.file_uploader("📂 Upload your SecurePay transactions CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded dataset with shape: {df.shape}")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Detect if dataset contains target or text columns
    non_numeric = [col for col in df.columns if not np.issubdtype(df[col].dtype, np.number)]
    if non_numeric:
        st.warning(f"⚠️ Non-numeric columns detected and removed: {non_numeric}")
        df = df.drop(columns=non_numeric)

    # Handle missing values
    df = df.fillna(df.mean())

    # Train model and show results
    model, scaler, df_results = train_model(df)

    fraud_count = (df_results["anomaly_label"] == "Fraud").sum()
    st.metric("Detected Anomalies (Fraud)", fraud_count)
    st.metric("Total Transactions", len(df_results))

    st.subheader("🔍 Anomaly Summary")
    st.dataframe(df_results[df_results["anomaly_label"] == "Fraud"].head(20))

    st.subheader("📉 PCA Visualization")
    plot_pca(df, df_results["anomaly_label"])

    # Save detected anomalies
    df_results.to_csv("detected_anomalies.csv", index=False)
    st.success("💾 Anomaly results saved as 'detected_anomalies.csv'")

else:
    st.info("👆 Upload a CSV file above to start anomaly detection.")


# Single Transaction

st.markdown("---")
st.header("🧮 Test a Single Transaction")

with st.expander("Enter transaction details manually"):
    cols = st.columns(4)
    with cols[0]: time = st.number_input("Transaction Time", 0.0)
    with cols[1]: amount = st.number_input("Transaction Amount", 0.0)
    features = {"Time": time, "Amount": amount}

    for i in range(1, 29):
        col = cols[i % 4]
        with col:
            features[f"V{i}"] = st.number_input(f"V{i}", 0.0)

    if st.button("🔎 Predict This Transaction"):
        input_df = pd.DataFrame([features])
        try:
            model, scaler, df_results = train_model(input_df)
            pred_label = df_results["anomaly_label"].iloc[0]
            score = df_results["anomaly_score"].iloc[0]
            st.success(f"Prediction: **{pred_label}** (score = {score:.4f})")
        except Exception as e:
            st.error(f"Error: {e}")
