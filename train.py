
# Run: python train.py

from __future__ import annotations
import json
from pathlib import Path
from urllib.request import urlretrieve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "results"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
FEATURES = ["age_years", "sex", "bmi", "waist_cm"]


def read_nhanes_file(name: str, columns: list[str]) -> pd.DataFrame:
    """ 
    Description
    """
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{name}.XPT"
    if not path.exists():
        print(f"Downloading {name} from CDC...")
        urlretrieve(f"{BASE}/{name}.XPT", path)
    # pandas reads the SAS transport files published by CDC directly.
    frame = pd.read_sas(path, format="xport")
    absent = set(columns).difference(frame.columns)
    if absent:
        raise ValueError(f"{name} is missing expected fields: {sorted(absent)}")
    return frame[columns]


def prepare_data() -> pd.DataFrame:
    """ 
    Description
    """
    demographics = read_nhanes_file("DEMO_J", ["SEQN", "RIDAGEYR", "RIAGENDR"])
    body = read_nhanes_file("BMX_J", ["SEQN", "BMXBMI", "BMXWAIST"])
    diabetes = read_nhanes_file("DIQ_J", ["SEQN", "DIQ010"])
    df = demographics.merge(body, on="SEQN", validate="one_to_one")
    df = df.merge(diabetes, on="SEQN", validate="one_to_one")
    df = df.loc[(df.RIDAGEYR >= 20) & df.DIQ010.isin([1, 2])].copy()
    # DIQ010: 1=yes; 2=no; 3=borderline. Other codes are not a clear answer.
    df["diagnosed_diabetes"] = (df.DIQ010 == 1).astype(int)
    df = df.rename(columns={
        "RIDAGEYR": "age_years", "RIAGENDR": "sex",
        "BMXBMI": "bmi", "BMXWAIST": "waist_cm",
    })
    df["sex"] = df.sex.map({1: "male", 2: "female"})
    if df.diagnosed_diabetes.nunique() != 2:
        raise ValueError("Expected both diagnosis classes in the selected sample")
    return df


def main() -> None:
    """ 
    Description
    """
    OUTPUT.mkdir(exist_ok=True)
    df = prepare_data()
    X = df[FEATURES]
    y = df.diagnosed_diabetes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    
    # Preprocessing is inside the pipeline so test data never informs imputation.
    preprocess = ColumnTransformer([
        ("numbers", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), ["age_years", "bmi", "waist_cm"]),
        ("sex", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["sex"]),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = model.predict(X_test)
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    baseline_probability = baseline.predict_proba(X_test)[:, 1]

    metrics = {
        "outcome": "Self-reported prior clinician diagnosis (DIQ010=1 vs DIQ010=2)",
        "participants": int(len(df)),
        "training_participants": int(len(X_train)),
        "test_participants": int(len(X_test)),
        "diagnosed_in_sample": int(y.sum()),
        "sample_fraction_diagnosed": round(float(y.mean()), 4),
        "test_roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "test_average_precision": round(float(average_precision_score(y_test, probability)), 4),
        "test_brier_score": round(float(brier_score_loss(y_test, probability)), 4),
        "baseline_test_average_precision": round(float(average_precision_score(y_test, baseline_probability)), 4),
        "confusion_matrix_0_1": confusion_matrix(y_test, prediction).tolist(),
        "threshold": 0.5,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUTPUT / "classification_report.txt").write_text(
        classification_report(y_test, prediction, target_names=["No report", "Reported diagnosis"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y_test, probability)
    axes[0].plot(fpr, tpr, label=f"Model (AUC {metrics['test_roc_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[0].legend()
    ConfusionMatrixDisplay.from_predictions(
        y_test, prediction, labels=[0, 1], display_labels=["No", "Yes"],
        ax=axes[1], colorbar=False
    )
    axes[1].set(title="Predictions at a 0.5 threshold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "evaluation.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results in {OUTPUT}")


if __name__ == "__main__":
    main()
