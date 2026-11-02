"""Train and compare models on the feature store, track in MLflow, save the best one."""
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_PATH = Path("feature_store/credit_features.csv")
MODEL_PATH = Path("models/model.joblib")
TARGET = "risk"
MIN_ACCURACY = 0.70  # quality gate

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
}


def build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = pd.read_csv(FEATURE_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk")

    best_name, best_f1, best_pipe = None, -1.0, None
    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipe = build_pipeline(model, X_train).fit(X_train, y_train)
            preds = pipe.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds),
                "roc_auc": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
            }
            mlflow.log_params({"model": name, **model.get_params()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
            print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if metrics["f1"] > best_f1:
                best_name, best_f1, best_pipe, best_metrics = name, metrics["f1"], pipe, metrics

    if best_metrics["accuracy"] < MIN_ACCURACY:
        raise SystemExit(f"Quality gate failed: accuracy {best_metrics['accuracy']:.3f} < {MIN_ACCURACY}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    Path("models/metrics.json").write_text(json.dumps({"model": best_name, **best_metrics}, indent=2))
    print(f"Best model: {best_name} (f1={best_f1:.4f}) saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
