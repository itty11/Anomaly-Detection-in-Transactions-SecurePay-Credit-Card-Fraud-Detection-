import os
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


def load_data(path):
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print("Loaded shape:", df.shape)
    return df


def preprocess(df, drop_cols=None):
    """
    Preprocess the credit card dataframe. Returns processed_df, features_list.
    - drop_cols: columns to drop (target etc). By default will drop 'Class' if present.
    """
    df = df.copy()
    if drop_cols is None:
        drop_cols = []
        if "Class" in df.columns:
            drop_cols.append("Class")

    # Drop non-feature columns if present
    existing_drop = [c for c in drop_cols if c in df.columns]
    if existing_drop:
        print("Dropping columns:", existing_drop)
        df = df.drop(columns=existing_drop)

    # Common dataset has columns: Time, V1..V28, Amount
    # We'll keep all numeric columns only (robust for other variants)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise RuntimeError("No numeric columns found in dataset after dropping target columns.")

    # Fill remaining missing values (if any) with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Scale features with RobustScaler (robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[numeric_cols])

    print(f"Preprocessed: numeric features = {len(numeric_cols)}")
    return pd.DataFrame(X_scaled, columns=numeric_cols), numeric_cols, scaler


def train_models(X, random_state=42):
    """
    Train a set of unsupervised anomaly detectors and return them with scores.
    Returns:
      models: dict of trained objects
      scores_df: DataFrame with score columns (higher = more anomalous for our convention)
    """
    models = {}
    scores = pd.DataFrame(index=X.index)

    # Isolation Forest (lower score = more normal; use -score to make 'higher = more anomaly')
    print("Training IsolationForest...")
    iso = IsolationForest(n_estimators=200, contamination='auto', random_state=random_state, n_jobs=-1)
    iso.fit(X)
    iso_score = -iso.decision_function(X)  # make higher => more anomalous
    models["isoforest"] = iso
    scores["iso_score"] = iso_score

    # Local Outlier Factor (LOF) - fit_predict returns -1 for outliers.
    # To get a continuous score we use negative_outlier_factor_ (lower = more abnormal), so invert sign.
    print("Fitting LocalOutlierFactor (for scoring)...")
    lof = LocalOutlierFactor(n_neighbors=35, contamination='auto', novelty=False, n_jobs=-1)
    # LOF cannot be used with novelty=True for fit_predict when data is the same; use fit_predict to compute scores
    lof_fit_pred = lof.fit_predict(X)
    lof_score = -lof.negative_outlier_factor_  # invert so higher = more anomalous
    models["lof_fitted"] = lof  # note: fitted but not usable for predict on new data unless novelty=True
    scores["lof_score"] = lof_score

    # EllipticEnvelope - robust covariance (assumes Gaussian-like normal data)
    print("Training EllipticEnvelope...")
    try:
        ee = EllipticEnvelope(support_fraction=0.85, contamination='auto', random_state=random_state)
        ee.fit(X)
        ee_score = -ee.decision_function(X)
        models["elliptic"] = ee
        scores["ee_score"] = ee_score
    except Exception as e:
        print("EllipticEnvelope failed:", e)
        scores["ee_score"] = 0.0
        models["elliptic"] = None

    return models, scores


def consensus_anomaly_flag(scores_df, threshold_quantile=0.98):
    """
    Produce a final anomaly boolean flag based on consensus of models.
    Approach:
      - For each model score column, mark top quantile (threshold_quantile) as anomaly
      - Count votes across models; require at least 2 votes (or majority) to final-flag
    Returns df with added 'anomaly_votes' and 'anomaly_final' boolean.
    """
    df = scores_df.copy()
    score_cols = [c for c in df.columns if c.endswith("_score")]
    votes = pd.DataFrame(index=df.index)

    for col in score_cols:
        thr = df[col].quantile(threshold_quantile)
        votes[col.replace("_score", "_vote")] = (df[col] >= thr).astype(int)
        print(f"{col}: threshold at quantile {threshold_quantile} = {thr:.4f}")

    votes["anomaly_votes"] = votes.sum(axis=1)
    # final: flagged as anomaly if >= 2 models agree OR voting count >= ceil(n_models/2)
    n_models = len(score_cols)
    min_votes = max(2, (n_models // 2) + 1)
    votes["anomaly_final"] = (votes["anomaly_votes"] >= min_votes).astype(int)
    df = pd.concat([df, votes], axis=1)
    return df


def visualize(X_scaled_df, scores_df, output_dir):
    """
    Create PCA scatter plot and score histograms and save them.
    X_scaled_df: DataFrame of scaled features
    scores_df: DataFrame including iso_score, lof_score, ee_score, anomaly flags
    """
    os.makedirs(output_dir, exist_ok=True)

    # PCA for 2D visualization
    print("Running PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled_df.values)
    pc_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=X_scaled_df.index)
    explained = pca.explained_variance_ratio_

    merged = pc_df.join(scores_df, how="left")
    merged["anomaly_final"] = merged["anomaly_final"].fillna(0)

    # PCA scatter
    plt.figure(figsize=(10, 7))
    palette = {0: "#7fc97f", 1: "#fb8072"}
    sns.scatterplot(data=merged, x="PC1", y="PC2", hue="anomaly_final",
                    palette=palette, alpha=0.6, s=20)
    plt.title("PCA scatter: anomalies (red) vs normal (green)")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    plt.legend(title="Anomaly final", loc="best")
    pca_scatter_path = os.path.join(output_dir, "pca_scatter.png")
    plt.tight_layout()
    plt.savefig(pca_scatter_path, dpi=150)
    plt.close()
    print("Saved:", pca_scatter_path)

    # explained variance plot
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(explained) + 1), explained * 100)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA Explained Variance")
    pca_ev_path = os.path.join(output_dir, "pca_explained_variance.png")
    plt.tight_layout()
    plt.savefig(pca_ev_path, dpi=150)
    plt.close()
    print("Saved:", pca_ev_path)

    # Score histograms
    score_cols = [c for c in scores_df.columns if c.endswith("_score")]
    plt.figure(figsize=(12, 4 * len(score_cols)))
    for i, col in enumerate(score_cols, start=1):
        plt.subplot(len(score_cols), 1, i)
        sns.histplot(scores_df[col], bins=80, kde=True, color="steelblue")
        plt.axvline(scores_df[col].quantile(0.98), color="red", linestyle="--", label="98% quantile")
        plt.title(f"Distribution of {col}")
        plt.legend()
    score_hist_path = os.path.join(output_dir, "score_histograms.png")
    plt.tight_layout()
    plt.savefig(score_hist_path, dpi=150)
    plt.close()
    print("Saved:", score_hist_path)


def save_models(models, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, m in models.items():
        if m is not None:
            path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(m, path)
            print("Saved model:", path)
    # save scaler too
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print("Saved scaler:", scaler_path)


def run_pipeline(input_csv, output_dir, random_state=42):
    t0 = datetime.now()

    df = load_data(input_csv)

    # Detect typical column names pattern and drop 'Class' if present
    drop_cols = []
    if "Class" in df.columns:
        drop_cols.append("Class")

    X_scaled_df, feature_cols, scaler = preprocess(df, drop_cols)

    models, scores = train_models(X_scaled_df, random_state=random_state)

    # Build final consensus
    scores_with_votes = consensus_anomaly_flag(scores, threshold_quantile=0.98)

    # Merge scores back to original (unscaled) records for output
    out_df = df.copy().reset_index(drop=True)
    out_df = pd.concat([out_df, scores_with_votes.reset_index(drop=True)], axis=1)

    # Save anomalies (final consensus flagged)
    anomalies_df = out_df[out_df["anomaly_final"] == 1].copy()
    anomalies_path = os.path.join(output_dir, "detected_anomalies.csv")
    os.makedirs(output_dir, exist_ok=True)
    anomalies_df.to_csv(anomalies_path, index=False)
    print(f"Anomalies detected: {len(anomalies_df)}/{len(out_df)} → saved to {anomalies_path}")

    # Visualize and save models
    visualize(X_scaled_df, scores_with_votes, output_dir)
    save_models(models, scaler, output_dir)

    # Save summary report
    report_lines = [
        f"Pipeline run at {t0.isoformat()}",
        f"Input file: {input_csv}",
        f"Rows processed: {len(df)}",
        f"Features used: {len(feature_cols)}",
        f"Detected anomalies (final): {len(anomalies_df)}",
        "",
        "Models trained: " + ", ".join([k for k in models.keys() if models[k] is not None])
    ]
    report_path = os.path.join(output_dir, "pipeline_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("Saved report:", report_path)

    t1 = datetime.now()
    print("Total runtime:", (t1 - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection pipeline for SecurePay / Credit Card Fraud dataset")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV (credit card dataset)")
    parser.add_argument("--output_dir", "-o", default="outputs", help="Where to save models & anomalies")
    parser.add_argument("--random_state", "-r", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(args.input, args.output_dir, random_state=args.random_state)
