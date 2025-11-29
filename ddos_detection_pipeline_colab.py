"""DDoS Detection Pipeline for CICDDoS2019 Dataset

This script demonstrates how to reproduce the core machine‑learning baseline
from the paper “A Machine Learning Approach for DDoS Prevention System in
Cloud Computing Environment” and add a feature‑reduction improvement. The
pipeline downloads the CICDDoS2019 dataset from Kaggle using the Kaggle API,
prepares the data, trains three classifiers (Logistic Regression, Support
Vector Machine and Random Forest) and evaluates their performance. It also
computes feature importances from the Random Forest and retrains all models
using only the top 20 features to show how feature reduction affects
accuracy and training time.

To run this script in Google Colab:

1. Create a Kaggle API token (Settings → API → Create New Token). The token
   will look like ``KGAT_<...>`` and can be set via an environment
   variable. Do **NOT** commit your token into version control.
2. At the top of your Colab notebook set the environment variable
   ``KAGGLE_API_TOKEN`` to your token string. E.g.:

   ``os.environ['KAGGLE_API_TOKEN'] = 'KGAT_yourtokenhere'``

3. Install the Kaggle CLI inside the Colab environment:

   ``!pip install kaggle -q``

4. Run this script. It will download the 5 % sample of CICDDoS2019 from Kaggle
   (dataset slug ``manmandes/ddos2019-5percent``), extract the CSV files,
   choose one of them for analysis, and train/test the models.

The actual dataset consists of multiple CSV files. Each file corresponds to
one day or attack scenario and contains dozens of traffic features plus a
label column indicating benign or attack traffic. The script assumes there
is a column named ``Label`` (case‑insensitive) that contains the class
labels; adjust ``LABEL_COLS`` if your file names the label differently.

Note: Running this code requires approximately 4–8 GB of memory depending
on the size of the CSV file. If your Colab runtime cannot handle the full
dataset, consider sampling a fraction of the rows using ``pd.read_csv(...,
sample``) or manually selecting a smaller subset.
"""

import os
import zipfile
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def download_dataset(dataset_slug: str, download_dir: Path) -> Path:
    """Download a dataset from Kaggle using the Kaggle CLI.

    Args:
        dataset_slug: The Kaggle dataset identifier (e.g. ``manmandes/ddos2019-5percent``).
        download_dir: Directory where the dataset ZIP file and extracted files should live.

    Returns:
        Path to the extracted dataset directory.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    # Use the Kaggle CLI to download the dataset. The token must be set via
    # the KAGGLE_API_TOKEN environment variable before running this script.
    print(f"Downloading Kaggle dataset {dataset_slug}…")
    import subprocess
    result = subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(download_dir),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to download dataset {dataset_slug}:\n{result.stdout}\n{result.stderr}"
        )
    # Determine downloaded ZIP file name (Kaggle names it ``{slug}.zip``)
    zip_path = None
    for f in download_dir.iterdir():
        if f.suffix == ".zip":
            zip_path = f
            break
    if zip_path is None:
        raise FileNotFoundError("ZIP file not found after download.")
    # Extract the zip archive
    extracted_dir = download_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting files…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)
    return extracted_dir


def load_first_csv(data_dir: Path, label_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the first CSV file found in ``data_dir`` and separate features and labels.

    Args:
        data_dir: Directory containing one or more CSV files.
        label_cols: Optional list of possible label column names (case‑insensitive).

    Returns:
        Tuple of (features DataFrame, label Series).
    """
    if label_cols is None:
        label_cols = ["label", "Label", " Label", "attack", "Attack"]
    csv_files = sorted(p for p in data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    csv_path = csv_files[0]
    print(f"Loading {csv_path.name}…")
    df = pd.read_csv(csv_path)
    # Identify the label column
    label_col = None
    for col in df.columns:
        if col.strip().lower() in [c.lower().strip() for c in label_cols]:
            label_col = col
            break
    if label_col is None:
        raise ValueError(
            f"Could not find a label column in {csv_path}. Tried {label_cols}."
        )
    y = df[label_col].copy()
    # Convert labels to binary 0/1: benign = 0, attack = 1
    y_binary = (~y.astype(str).str.contains("benign", case=False)).astype(int)

    # Drop label column and non‑numeric columns
    X = df.drop(columns=[label_col])
    # Convert all columns to numeric where possible
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    X_numeric = X_numeric.fillna(0.0)
    return X_numeric, y_binary


def train_and_evaluate(X, y, top_k: int = 20) -> None:
    """Train LR, SVM and RF models on full feature set and reduced feature set.

    Args:
        X: Feature DataFrame or ndarray.
        y: Binary label Series/ndarray.
        top_k: Number of top features to select using Random Forest importances.
    """
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize features for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    print("Training Logistic Regression on all features…")
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_time = time.time() - t0
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)

    # Train SVM
    print("Training SVM on all features…")
    t0 = time.time()
    svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm.fit(X_train_scaled, y_train)
    svm_time = time.time() - t0
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(y_test, svm_pred)

    # Train Random Forest
    print("Training Random Forest on all features…")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)

    # Compute feature importances and select top_k features
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    top_feature_names = X.columns[indices]
    print(f"Top {top_k} features selected by RF: {list(top_feature_names)[:10]}…")

    # Create reduced datasets
    X_train_red = X_train.iloc[:, indices]
    X_test_red = X_test.iloc[:, indices]
    scaler_red = StandardScaler()
    X_train_red_scaled = scaler_red.fit_transform(X_train_red)
    X_test_red_scaled = scaler_red.transform(X_test_red)

    # Retrain models on reduced feature set
    print("\nTraining models on reduced feature set…")
    # LR reduced
    t0 = time.time()
    lr_red = LogisticRegression(max_iter=1000)
    lr_red.fit(X_train_red_scaled, y_train)
    lr_red_time = time.time() - t0
    lr_red_pred = lr_red.predict(X_test_red_scaled)
    lr_red_acc = accuracy_score(y_test, lr_red_pred)
    lr_red_f1 = f1_score(y_test, lr_red_pred)

    # SVM reduced
    t0 = time.time()
    svm_red = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm_red.fit(X_train_red_scaled, y_train)
    svm_red_time = time.time() - t0
    svm_red_pred = svm_red.predict(X_test_red_scaled)
    svm_red_acc = accuracy_score(y_test, svm_red_pred)
    svm_red_f1 = f1_score(y_test, svm_red_pred)

    # RF reduced
    t0 = time.time()
    rf_red = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_red.fit(X_train_red, y_train)
    rf_red_time = time.time() - t0
    rf_red_pred = rf_red.predict(X_test_red)
    rf_red_acc = accuracy_score(y_test, rf_red_pred)
    rf_red_f1 = f1_score(y_test, rf_red_pred)

    # Print results
    print("\n=== Performance Summary ===")
    header = (
        "Model\t\t| Full Acc | Full F1 | Train Time (s) | Reduced Acc | Reduced F1 | Train Time (s)"
    )
    print(header)
    print("LR\t\t| {0:.4f}\t {1:.4f}\t {2:.3f}\t\t {3:.4f}\t {4:.4f}\t {5:.3f}".format(
        lr_acc, lr_f1, lr_time, lr_red_acc, lr_red_f1, lr_red_time
    ))
    print("SVM\t\t| {0:.4f}\t {1:.4f}\t {2:.3f}\t\t {3:.4f}\t {4:.4f}\t {5:.3f}".format(
        svm_acc, svm_f1, svm_time, svm_red_acc, svm_red_f1, svm_red_time
    ))
    print("RF\t\t| {0:.4f}\t {1:.4f}\t {2:.3f}\t\t {3:.4f}\t {4:.4f}\t {5:.3f}".format(
        rf_acc, rf_f1, rf_time, rf_red_acc, rf_red_f1, rf_red_time
    ))


def main():
    # Ensure Kaggle API token is available
    if "KAGGLE_API_TOKEN" not in os.environ:
        raise EnvironmentError(
            "KAGGLE_API_TOKEN environment variable is not set. "
            "Obtain an API token from Kaggle and set it before running."
        )
    # Define dataset slug (5 % sample to reduce memory usage). You can change
    # this to ``dhoogla/cicddos2019`` or another slug if you want the full dataset.
    dataset_slug = "manmandes/ddos2019-5percent"
    data_root = Path("./cicddos2019")
    extracted_dir = download_dataset(dataset_slug, data_root)
    # Load features and labels from the first CSV file found
    X, y = load_first_csv(extracted_dir)
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    # Train and evaluate models
    train_and_evaluate(X, y, top_k=20)


if __name__ == "__main__":
    main()