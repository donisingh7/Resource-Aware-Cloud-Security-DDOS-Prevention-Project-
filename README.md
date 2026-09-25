# Resource-Aware Cloud Security — DDoS Detection Experiments

Machine-learning experiments for **CICDDoS2019** focused on two practical questions:

1. Can standard classifiers detect benign vs. DDoS traffic effectively?
2. Can feature reduction and resource-aware model selection reduce computation while preserving useful predictive performance?

This repository compares full-feature and reduced-feature pipelines using **Logistic Regression, SVM, and Random Forest**.

## Repository map

| File | Purpose |
| --- | --- |
| `main.py` | Baseline models + Random-Forest feature reduction |
| `additional_novelty.py` | Extends the experiment with resource-aware model selection |
| `ddos_cic_local.py` | Local Parquet workflow with full-vs-top-K comparison |
| `ddos_detection_pipeline_colab.py` | Colab/Kaggle workflow for a smaller CICDDoS2019 sample |

## Experiment flow

```text
CICDDoS2019 CSV / Parquet
        │
        ▼
Label normalization
BENIGN/NORMAL -> 0
Attack        -> 1
        │
        ▼
Numeric feature cleanup
        │
        ▼
Train/test split
        │
        ├── Logistic Regression
        ├── SVM (RBF)
        └── Random Forest
        │
        ▼
Random-Forest feature importance
        │
        ▼
Top-K feature subset
        │
        ▼
Retrain + compare
accuracy / F1 / train time / inference time
```

## Setup

```bash
python -m venv .venv
pip install -r requirements.txt
```

For local runs, point `CICDDOS_DATA_ROOT` to the folder containing your dataset.

Linux/macOS:

```bash
export CICDDOS_DATA_ROOT=/path/to/cicddos2019
python main.py
```

PowerShell:

```powershell
$env:CICDDOS_DATA_ROOT="D:\path\to\cicddos2019"
python main.py
```

## Colab / Kaggle workflow

`ddos_detection_pipeline_colab.py` can download a CICDDoS2019 sample through the Kaggle CLI. Set `KAGGLE_API_TOKEN` in the environment and never commit it.

## Outputs

The scripts report combinations of:

- accuracy
- F1 score
- training time
- inference time
- selected top-K features
- full-feature vs. reduced-feature comparisons

`ddos_cic_local.py` can also write a `ddos_results_summary.csv` file into the configured data folder.

## Reproducibility

- Dataset files are intentionally not committed.
- Random seeds are fixed where the current implementation supports them.
- Results depend on dataset subset, sampling limits, preprocessing, hardware, and package versions.
- This README intentionally does not invent or hard-code benchmark numbers. Re-run the scripts and report the metrics for the exact experiment configuration you used.

## Tech stack

Python, pandas, NumPy, scikit-learn, PyArrow, Kaggle CLI.

## Author

**Doni Singh Agrawal**  
M.Tech — Information Security & Privacy, SVNIT Surat

Research/engineering proof for cloud security and resource-aware machine-learning experimentation.
