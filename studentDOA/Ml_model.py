"""Latih dan simpan model klasifikasi status mahasiswa."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "studentDOA_prepared.csv"
MODEL_FILE = BASE_DIR / "studentDOA_model.joblib"
TARGET = "Target"
CATEGORICAL_COLUMNS = [
    "Marital status", "Application mode", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Jalankan 'python Data_Prev.py' terlebih dahulu.")

    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(str)
    categorical = [column for column in CATEGORICAL_COLUMNS if column in X.columns]
    numeric = [column for column in X.columns if column not in categorical]

    preprocess = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numbers", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    joblib.dump({
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "prediction": prediction,
        "accuracy": accuracy_score(y_test, prediction),
        "report": classification_report(
            y_test, prediction, output_dict=True, zero_division=0
        ),
    }, MODEL_FILE)
    print(f"Accuracy test: {accuracy_score(y_test, prediction):.3f}")
    print(f"Model tersimpan: {MODEL_FILE}")


if __name__ == "__main__":
    main()
