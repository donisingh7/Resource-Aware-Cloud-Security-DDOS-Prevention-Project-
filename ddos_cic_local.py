"""
CICDDoS2019 local baseline + feature-reduction experiment.

- Reads all .parquet files from a local folder (CIC-DDoS2019 cleaned parquet).
- Builds a binary classifier: 0 = benign/normal, 1 = attack.
- Trains Logistic Regression, SVM (RBF), Random Forest on all features.
- Uses RF feature importances to select Top-K features and retrains models.
- Prints accuracy, F1, training time, inference time.

Author: Doni + ChatGPT helper :)
"""

import os
import glob
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


# ==== CONFIGURE HERE (your local path) =====================
BASE_DIR = r"D:\SVNIT\M.tech\3rd sem\odd sem\archive"
MAX_ROWS = 300_000          # max rows to keep (to control RAM/time)
TOP_K_FEATURES = 20         # RF top-K features for Option 2
# ==========================================================


def load_parquet_folder(folder: str) -> pd.DataFrame:
    """Load and concatenate all .parquet files in a folder."""
    parquet_paths = glob.glob(os.path.join(folder, "*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No .parquet files found in {folder}")

    print(f"Found {len(parquet_paths)} parquet files:")
    for p in parquet_paths:
        print("  -", os.path.basename(p))

    dfs = []
    for p in parquet_paths:
        try:
            df_part = pd.read_parquet(p)  # requires pyarrow or fastparquet
            dfs.append(df_part)
        except Exception as e:
            print(f"[WARN] Skipping {p} due to read error: {e}")

    if not dfs:
        raise RuntimeError("Could not load any parquet file.")

    df = pd.concat(dfs, ignore_index=True)
    print("Combined shape:", df.shape)
    return df


def detect_label_column(df: pd.DataFrame) -> str:
    """Guess label column name (Label/Attack/Class/etc)."""
    candidates = []
    for c in df.columns:
        cname = c.strip().lower()
        if cname in ("label", "attack", "class", "classification", "target"):
            candidates.append(c)

    if not candidates:
        print("Available columns:\n", df.columns.tolist())
        raise RuntimeError(
            "Could not detect label column automatically. "
            "Edit detect_label_column() and set it manually."
        )

    # Prefer 'Label' if present
    for c in candidates:
        if c.strip().lower() == "label":
            print(f"Using label column: {c}")
            return c

    print(f"Using label column: {candidates[0]}")
    return candidates[0]


def binarize_labels(series: pd.Series) -> pd.Series:
    """
    Convert multiclass labels to binary:
    0 = BENIGN/NORMAL, 1 = any attack.
    """
    def _map(v: str) -> int:
        v_low = str(v).strip().lower()
        if "benign" in v_low or "normal" in v_low:
            return 0
        return 1

    return series.astype(str).map(_map)


def prepare_data(df: pd.DataFrame):
    """Full pipeline: detect label, build X,y, sample, split, scale."""
    label_col = detect_label_column(df)

    y_raw = df[label_col]
    y = binarize_labels(y_raw)

    # keep only numeric columns as features
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=["number"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"Numeric feature count: {X.shape[1]}")

    # Optional sampling to avoid memory issues
    if len(X) > MAX_ROWS:
        print(f"Sampling {MAX_ROWS} rows from {len(X)}...")
        tmp = X.copy()
        tmp["__y__"] = y.values
        tmp = tmp.sample(MAX_ROWS, random_state=42)
        y = tmp["__y__"]
        X = tmp.drop(columns=["__y__"])

    X_arr = X.values
    y_arr = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr,
        y_arr,
        test_size=0.3,
        random_state=42,
        stratify=y_arr,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y_arr, X_train_scaled, X_test_scaled, y_train, y_test


def train_and_eval(name, model, Xtr, ytr, Xte, yte, setting):
    """Train a model and print metrics and timings."""
    start = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(Xte)
    infer_time = time.perf_counter() - start

    acc = accuracy_score(yte, y_pred)
    f1 = f1_score(yte, y_pred)

    print(f"\n=== {name} ({setting}) ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(f"Train time   : {train_time:.2f} s")
    print(f"Infer time   : {infer_time:.4f} s")

    return {
        "setting": setting,
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "estimator": model,
    }


def main():
    # 1) Load dataset
    print(f"Loading parquet files from: {BASE_DIR}")
    df = load_parquet_folder(BASE_DIR)

    # 2) Prepare data
    X_full, y_all, X_train_full, X_test_full, y_train, y_test = prepare_data(df)

    results = []

    # 3) Base models on ALL features  (Base paper)
    lr = LogisticRegression(max_iter=500, n_jobs=-1)
    results.append(
        train_and_eval("Logistic Regression", lr,
                       X_train_full, y_train, X_test_full, y_test,
                       setting="All features")
    )

    svm = SVC(kernel="rbf", gamma="scale")
    results.append(
        train_and_eval("SVM (RBF kernel)", svm,
                       X_train_full, y_train, X_test_full, y_test,
                       setting="All features")
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    results.append(
        train_and_eval("Random Forest", rf,
                       X_train_full, y_train, X_test_full, y_test,
                       setting="All features")
    )

    # 4) Feature importance from RF → Top-K features (Option 2)
    importances = rf.feature_importances_
    feature_names = X_full.columns.tolist()
    feat_imp = sorted(zip(feature_names, importances),
                      key=lambda x: x[1],
                      reverse=True)

    top_feats = [f for f, _ in feat_imp[:TOP_K_FEATURES]]

    print(f"\nTop {TOP_K_FEATURES} features (from RF):")
    for i, (fname, score) in enumerate(feat_imp[:TOP_K_FEATURES], start=1):
        print(f"{i:2d}. {fname}  ({score:.6f})")

    X_top = X_full[top_feats].values

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_top,
        y_all,
        test_size=0.3,
        random_state=42,
        stratify=y_all,
    )

    scaler_top = StandardScaler()
    X_train_top = scaler_top.fit_transform(X_train_t)
    X_test_top = scaler_top.transform(X_test_t)

    # 5) Models on reduced features
    lr_top = LogisticRegression(max_iter=500, n_jobs=-1)
    results.append(
        train_and_eval("Logistic Regression", lr_top,
                       X_train_top, y_train_t, X_test_top, y_test_t,
                       setting=f"Top-{TOP_K_FEATURES} features")
    )

    svm_top = SVC(kernel="rbf", gamma="scale")
    results.append(
        train_and_eval("SVM (RBF kernel)", svm_top,
                       X_train_top, y_train_t, X_test_top, y_test_t,
                       setting=f"Top-{TOP_K_FEATURES} features")
    )

    rf_top = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    results.append(
        train_and_eval("Random Forest", rf_top,
                       X_train_top, y_train_t, X_test_top, y_test_t,
                       setting=f"Top-{TOP_K_FEATURES} features")
    )

    # 6) Summary table
    summary_df = pd.DataFrame(results)[
        ["setting", "model", "accuracy", "f1", "train_time_s", "infer_time_s"]
    ].sort_values(["model", "setting"])

    print("\n\n============ SUMMARY ============")
    print(summary_df.to_string(index=False))

    # Optional: save to CSV for report
    out_path = os.path.join(BASE_DIR, "ddos_results_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
