
"""
CICDDoS2019 – Base ML models + Feature-Reduction Advancement (Option 2)

- Base paper style:
    Logistic Regression, SVM, Random Forest on CICDDoS2019.
- Advancement:
    Use Random Forest feature importances to select Top-K features,
    retrain models, compare accuracy + timing (full vs reduced features).

Run:
    CICDDOS_DATA_ROOT=/path/to/dataset python main.py
"""

import os
import time
import zipfile
from pathlib import Path
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


# =====================================================================
#  CONFIG
# =====================================================================
# Set CICDDOS_DATA_ROOT to the folder containing CICDDoS2019 CSV/Parquet files.
DATA_ROOT = os.getenv("CICDDOS_DATA_ROOT", "")

# Maximum number of rows to load per file (to avoid RAM crash)
MAX_ROWS_PER_FILE = 200_000

# Total maximum rows (after concatenation)
MAX_TOTAL_ROWS = 400_000

# How many top features we keep for advancement
TOP_K_FEATURES = 20
# =====================================================================


def unzip_archives(root: Path) -> None:
    """Unzip any .zip files under root (once)."""
    for z in root.rglob("*.zip"):
        target_dir = z.with_suffix("")
        if target_dir.exists():
            # Already extracted earlier
            continue
        print(f"[+] Extracting {z} -> {target_dir}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(target_dir)


def find_data_files(root: Path):
    """Find CSV and Parquet files, prefer *training* files if present."""
    csv_files = list(root.rglob("*.csv"))
    parquet_files = list(root.rglob("*.parquet"))

    def prefer_training(files):
        training = [f for f in files if "train" in f.name.lower() or "training" in f.name.lower()]
        return training if training else files

    csv_files = prefer_training(csv_files)
    parquet_files = prefer_training(parquet_files)

    print(f"[i] Found {len(csv_files)} CSV and {len(parquet_files)} Parquet files.")
    if not csv_files and not parquet_files:
        raise RuntimeError("No CSV/Parquet files found. Check DATA_ROOT or unzip step.")

    return csv_files, parquet_files


def load_dataset(root: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load & combine data from CSV / parquet files with light sampling."""
    unzip_archives(root)
    csv_files, parquet_files = find_data_files(root)

    dfs = []
    total_rows = 0

    def read_limited_csv(path: Path):
        nonlocal total_rows
        remaining = MAX_TOTAL_ROWS - total_rows
        if remaining <= 0:
            return None
        nrows = min(MAX_ROWS_PER_FILE, remaining)
        print(f"[+] Reading CSV {path.name} (max {nrows} rows)…")
        try:
            df_part = pd.read_csv(path, nrows=nrows, low_memory=False)
            total_rows += len(df_part)
            return df_part
        except Exception as e:
            print(f"[!] Skipping {path} due to error: {e}")
            return None

    def read_limited_parquet(path: Path):
        nonlocal total_rows
        remaining = MAX_TOTAL_ROWS - total_rows
        if remaining <= 0:
            return None
        print(f"[+] Reading Parquet {path.name} (may take time)…")
        try:
            df_part = pd.read_parquet(path)
            if remaining < len(df_part):
                df_part = df_part.sample(remaining, random_state=42)
            total_rows += len(df_part)
            return df_part
        except Exception as e:
            print(f"[!] Skipping {path} due to error: {e}")
            return None

    for p in csv_files:
        if total_rows >= MAX_TOTAL_ROWS:
            break
        dfp = read_limited_csv(p)
        if dfp is not None:
            dfs.append(dfp)

    for p in parquet_files:
        if total_rows >= MAX_TOTAL_ROWS:
            break
        dfp = read_limited_parquet(p)
        if dfp is not None:
            dfs.append(dfp)

    if not dfs:
        raise RuntimeError("Could not read any data files.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[i] Combined data shape: {df.shape}")

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Replace +/-inf and then fill NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Detect label column
    label_col = None
    for c in df.columns:
        cname = c.strip().lower()
        if cname in ("label", "attack", "class", "classification", "target"):
            label_col = c
            break

    if label_col is None:
        print("[!] Could not auto-detect label column.")
        print("    Columns sample:", df.columns[:30].tolist())
        raise RuntimeError(
            "Set label_col manually inside load_dataset() based on your file."
        )

    print(f"[i] Using label column: {label_col}")
    y_raw = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])

    # Binary labels: 0 = benign/normal, 1 = attack
    def binarize(lbl: str) -> int:
        low = lbl.lower()
        if "benign" in low or "normal" in low:
            return 0
        return 1

    y = y_raw.apply(binarize)

    # Numeric features only
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    X = X[numeric_cols].astype("float32")

    print(f"[i] Numeric features: {len(numeric_cols)}")
    print("[i] Example numeric columns:", numeric_cols[:10])

    return X, y, numeric_cols


def train_and_eval(name, model, Xtr, ytr, Xte, yte):
    """Train model and print metrics + timing."""
    start = time.perf_counter()
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(Xte)
    infer_time = time.perf_counter() - start

    acc = accuracy_score(yte, y_pred)
    f1 = f1_score(yte, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1-score   : {f1:.4f}")
    print(f"Train time : {train_time:.2f} s")
    print(f"Infer time : {infer_time:.4f} s (on full test set)")

    return {
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "train_time_s": train_time,
        "infer_time_s": infer_time,
        "estimator": model,
    }


def main():
    if not DATA_ROOT:
        raise EnvironmentError(
            "CICDDOS_DATA_ROOT is not set. Point it to the folder containing "
            "your CICDDoS2019 CSV/Parquet files."
        )
    root = Path(DATA_ROOT).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"CICDDOS_DATA_ROOT does not exist: {root}")

    print(f"[i] Using DATA_ROOT = {root}")

    # 1) Load and preprocess dataset
    X, y, feature_names = load_dataset(root)

    # 2) Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=0.3,
        random_state=42,
        stratify=y.values,
    )

    # Scale for LR & SVM (RF technically doesn't need it, but it's ok)
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(X_train)
    X_test_full = scaler_full.transform(X_test)

    print("\n================ BASE MODELS (All features) ================")
    results_full = []

    lr = LogisticRegression(max_iter=500, n_jobs=-1)
    results_full.append(
        train_and_eval("Logistic Regression (all)", lr, X_train_full, y_train, X_test_full, y_test)
    )

    svm = SVC(kernel="rbf", gamma="scale")
    results_full.append(
        train_and_eval("SVM RBF (all)", svm, X_train_full, y_train, X_test_full, y_test)
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    results_full.append(
        train_and_eval("Random Forest (all)", rf, X_train_full, y_train, X_test_full, y_test)
    )

    # 3) Feature importance from Random Forest (Advancement part)
    rf_model = results_full[-1]["estimator"]
    importances = rf_model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    top_feats = [name for name, _ in feat_imp[:TOP_K_FEATURES]]

    print(f"\n================ TOP {TOP_K_FEATURES} FEATURES (by RF) ================")
    for i, (name, score) in enumerate(feat_imp[:TOP_K_FEATURES], start=1):
        print(f"{i:2d}. {name:<40} {score:.6f}")

    # 4) Re-train models on reduced feature set
    X_reduced = X[top_feats].values
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reduced,
        y.values,
        test_size=0.3,
        random_state=42,
        stratify=y.values,
    )

    scaler_red = StandardScaler()
    X_tr_red = scaler_red.fit_transform(X_tr_r)
    X_te_red = scaler_red.transform(X_te_r)

    print(f"\n================ MODELS (Top-{TOP_K_FEATURES} features) ================")
    results_reduced = []

    lr_r = LogisticRegression(max_iter=500, n_jobs=-1)
    results_reduced.append(
        train_and_eval(
            f"Logistic Regression (top-{TOP_K_FEATURES})",
            lr_r,
            X_tr_red,
            y_tr_r,
            X_te_red,
            y_te_r,
        )
    )

    svm_r = SVC(kernel="rbf", gamma="scale")
    results_reduced.append(
        train_and_eval(
            f"SVM RBF (top-{TOP_K_FEATURES})",
            svm_r,
            X_tr_red,
            y_tr_r,
            X_te_red,
            y_te_r,
        )
    )

    rf_r = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    results_reduced.append(
        train_and_eval(
            f"Random Forest (top-{TOP_K_FEATURES})",
            rf_r,
            X_tr_red,
            y_tr_r,
            X_te_red,
            y_te_r,
        )
    )

    # 5) Summary table
    def to_row(setting, r):
        return {
            "setting": setting,
            "model": r["model"],
            "accuracy": r["accuracy"],
            "f1": r["f1"],
            "train_time_s": r["train_time_s"],
            "infer_time_s": r["infer_time_s"],
        }

    rows = [to_row("all_features", r) for r in results_full] + [
        to_row(f"top_{TOP_K_FEATURES}", r) for r in results_reduced
    ]
    summary = pd.DataFrame(rows)
    print("\n================ SUMMARY ================")
    print(summary.sort_values(["model", "setting"]).to_string(index=False))


if __name__ == "__main__":
    main()
