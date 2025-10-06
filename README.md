# Anomaly-Detection-in-Transactions-SecurePay-Credit-Card-Fraud-Detection-
This project builds a **hybrid anomaly detection system** using multiple unsupervised algorithms —   **Isolation Forest**, **Local Outlier Factor (LOF)**, and **Elliptic Envelope** —   to detect **fraudulent transactions** from the `SecurePay` / `creditcard.csv` dataset.


##  Dataset Information

**Dataset:** SecurePay: Credit Card Fraud Detection Data - https://www.kaggle.com/datasets/eshummalik/securepay-credit-card-fraud-detection-data
**Records:** 284,807  
**Features:** 31 (`Time, Amount, V1–V28, Class`)  
**Target:**  
- `0` → Normal transaction  
- `1` → Fraudulent transaction (Abnormal)

The dataset is highly **imbalanced** — frauds represent only ~0.17% of all transactions.  
This makes it ideal for **unsupervised anomaly detection**.


##  Environment Setup

###  Python Version
- **Recommended:** Python **3.11.x**
-  Avoid using Python 3.12+ (TensorFlow/NumPy DLL compatibility issues)

pip install pandas numpy scikit-learn seaborn matplotlib streamlit


1. Command-Line Pipeline Usage

   Run the anomaly detection pipeline:

   python anomaly_pipeline.py --input creditcard.csv --output_dir outputs

Example Output

Loading data from: creditcard.csv

Loaded shape: (284807, 31)

Dropping columns: ['Class']

Preprocessed: numeric features = 30

Training IsolationForest...

Fitting LocalOutlierFactor (for scoring)...

Training EllipticEnvelope...

EllipticEnvelope failed: The 'contamination' parameter of EllipticEnvelope must be a float in the range (0.0, 0.5]. Got 'auto' instead.

iso_score: threshold at quantile 0.98 = 0.0341

lof_score: threshold at quantile 0.98 = 1.7863

ee_score: threshold at quantile 0.98 = 0.0000

Anomalies detected: 11137/284807 → saved to outputs/detected_anomalies.csv

Running PCA for 2D visualization...

Saved: outputs/pca_scatter.png

Saved: outputs/pca_explained_variance.png

Saved: outputs/score_histograms.png

Saved model: outputs/isoforest.pkl

Saved model: outputs/lof_fitted.pkl

Saved scaler: outputs/scaler.pkl

Saved report: outputs/pipeline_report.txt

Total runtime: 0:09:40.706862


2. Streamlit Interactive Dashboard

Once the models are generated, you can explore results interactively via Streamlit.

Run the app

streamlit run app.py

## Streamlit Features

Upload a CSV (e.g., creditcard.csv)
Automatic anomaly detection
PCA 2D visualization (Fraud vs Normal)
Anomaly summary table
CSV export of detected anomalies
Manual transaction testing panel
Works fully offline

## Techniques Used

| Algorithm                      | Purpose                          | Notes                                      |
| ------------------------------ | -------------------------------- | ------------------------------------------ |
| **Isolation Forest**           | Tree-based anomaly detection     | Best general performance                   |
| **Local Outlier Factor (LOF)** | Density-based detection          | Useful for local anomalies                 |
| **Elliptic Envelope**          | Gaussian assumption (optional)   | Used when data follows normal distribution |
| **PCA (2D)**                   | Visualization of latent features | Highlights fraud clusters                  |


Visualizations

Example output images saved in outputs/:

PCA Scatter Plot → Fraud vs Normal clusters

Score Histograms → Anomaly score distributions

Explained Variance Plot → PCA variance contribution

You can view them directly:

start outputs/pca_scatter.png

## Insights

The model detected ~11K anomalous transactions out of 284K.

Most frauds cluster in lower PCA components, far from the dense normal region.

Isolation Forest consistently outperformed others in stability and speed.

## Future Enhancements

Integrate Autoencoder-based Deep Anomaly Detection (TensorFlow)

Add RAG-style explainability to anomalies

Deploy on AWS / GCP for real-time fraud monitoring

## Author

Ittyavira C Abraham

MCA (AI) — Amrita Vishwa Vidyapeetham
