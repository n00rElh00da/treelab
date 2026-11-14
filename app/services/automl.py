from __future__ import annotations
import time
from typing import Any
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from .training import TrainingError, _make_preprocessor, _feature_names


def run_automl(frame: pd.DataFrame, config: dict[str, Any], mode: str, time_budget: int, metric_label: str) -> dict[str, Any]:
    if not 1 <= time_budget <= 90:
        raise TrainingError("AutoML time budget must be between 1 and 90 seconds")
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise TrainingError("FLAML is not installed") from exc
    target = config["target"]
    features = config["features"]
    frame = frame.dropna(subset=[target]).copy()
    if config.get("duplicates") == "Drop duplicates":
        frame = frame.drop_duplicates()
    X = frame[features]
    y = frame[target]
    if y.dropna().drop_duplicates().shape[0] < 2:
        raise TrainingError("The target must contain at least two classes")
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - int(config["split"])) / 100, random_state=42, stratify=stratify)
    except ValueError as exc:
        raise TrainingError(f"Unable to split dataset: {exc}") from exc
    preprocessor = _make_preprocessor(frame, features, config["missing"], config["encoding"], config["scaling"])
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)
    automl = AutoML()
    search_space = ["rf", "extra_tree"] if mode == "Controlled" else ["rf", "extra_tree", "lgbm", "xgboost", "histgb", "svc", "lrl1"]
    metric = {"Accuracy": "accuracy", "F1 score": "f1", "Precision": "precision", "Recall": "recall"}.get(metric_label, "f1")
    started = time.perf_counter()
    automl.fit(X_train=transformed_train, y_train=y_train, task="classification", time_budget=time_budget, metric=metric, estimator_list=search_space, verbose=0)
    duration_ms = int((time.perf_counter() - started) * 1000)
    predictions = automl.predict(transformed_test)
    y_true_labels = [str(value) for value in y_test]
    prediction_labels = [str(value) for value in predictions]
    labels = sorted(set(y_true_labels + prediction_labels))
    average = "weighted"
    metrics = {
        "accuracy": float(accuracy_score(y_true_labels, prediction_labels)),
        "precision": float(precision_score(y_true_labels, prediction_labels, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_labels, prediction_labels, average=average, zero_division=0)),
        "f1": float(f1_score(y_true_labels, prediction_labels, average=average, zero_division=0)),
        "labels": labels,
        "classification_report": classification_report(y_true_labels, prediction_labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true_labels, prediction_labels, labels=labels).tolist(),
        "best_loss": float(automl.best_loss) if automl.best_loss is not None else None,
    }
    best = str(automl.best_estimator)
    chosen = automl.model.estimator if hasattr(automl.model, "estimator") else automl.model
    names = _feature_names(preprocessor)
    native = getattr(chosen, "feature_importances_", [])
    importance = sorted([{"feature": name, "importance": float(value)} for name, value in zip(names, native)], key=lambda item: item["importance"], reverse=True)[:30]
    sample_predictions = []
    for index, (row_index, row) in enumerate(X_test.head(8).iterrows()):
        sample_predictions.append({"id": f"row_{row_index}", "values": {feature: (None if pd.isna(row[feature]) else str(row[feature])) for feature in features}, "predicted": str(predictions[index]), "actual": str(y_test.iloc[index])})
    explainability = {"feature_importance": importance, "permutation_importance": [], "sample_predictions": sample_predictions, "tree_text": f"FLAML selected {best}; detailed estimator explanation is available through its persisted configuration."}
    return {"pipeline": (preprocessor, automl), "metrics": metrics, "explainability": explainability, "duration_ms": duration_ms, "parameters": {"mode": mode, "time_budget": time_budget, "metric": metric_label, "search_space": search_space, "best_estimator": best, "best_config": automl.best_config or {}} , "model_type": f"AutoML · {best}"}
