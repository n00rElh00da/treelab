from __future__ import annotations
import io
import pickle
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier


class TrainingError(ValueError):
    pass


def _class_weight(value: str) -> str | None:
    return "balanced" if value.lower().startswith("balanced") else None


def _make_preprocessor(frame: pd.DataFrame, features: list[str], missing: str, encoding: str, scaling: str) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    impute = "median" if missing.startswith("Median") else "mean" if missing.startswith("Mean") else "most_frequent"
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=impute))]
    if scaling == "Standard scaler":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaling == "Robust scaler":
        numeric_steps.append(("scaler", RobustScaler()))
    elif scaling == "Min-max scaler":
        numeric_steps.append(("scaler", MinMaxScaler()))
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) if encoding == "Ordinal encoding" else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", cat_encoder)]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _model(model_type: str, parameters: dict[str, Any]):
    weight = _class_weight(str(parameters.get("classWeight", parameters.get("class_weight", "Balanced"))))
    if model_type == "Decision Tree":
        criterion = str(parameters.get("criterion", "gini")).lower()
        if criterion not in {"gini", "entropy"}:
            raise TrainingError("Decision Tree criterion must be gini or entropy")
        depth = parameters.get("maxDepth", parameters.get("max_depth", 5))
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=None if str(depth) == "None" else int(depth),
            min_samples_split=int(parameters.get("minSplit", parameters.get("min_samples_split", 2))),
            min_samples_leaf=int(parameters.get("minLeaf", parameters.get("min_samples_leaf", 1))),
            class_weight=weight,
            random_state=42,
        )
    if model_type == "Random Forest":
        depth = parameters.get("maxDepth", parameters.get("max_depth", 8))
        return RandomForestClassifier(
            n_estimators=int(parameters.get("estimators", parameters.get("n_estimators", 100))),
            max_depth=None if str(depth) == "None" else int(depth),
            max_features=parameters.get("maxFeatures", parameters.get("max_features", "sqrt")),
            min_samples_split=int(parameters.get("minSplit", parameters.get("min_samples_split", 2))),
            class_weight=weight,
            random_state=42,
            n_jobs=-1,
        )
    raise TrainingError(f"Unsupported model type: {model_type}")


def _feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def train_classifier(frame: pd.DataFrame, model_type: str, config: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    target = config["target"]
    features = config["features"]
    if target not in frame.columns:
        raise TrainingError(f"Target column '{target}' does not exist")
    if not features:
        raise TrainingError("Select at least one feature")
    if any(feature not in frame.columns for feature in features):
        raise TrainingError("One or more selected features do not exist")
    frame = frame.dropna(subset=[target]).copy()
    if config.get("duplicates") == "Drop duplicates":
        frame = frame.drop_duplicates()
    y = frame[target]
    if y.nunique() < 2:
        raise TrainingError("The target must contain at least two classes")
    X = frame[features]
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - int(config["split"])) / 100, random_state=42, stratify=stratify)
    except ValueError as exc:
        raise TrainingError(f"Unable to split dataset: {exc}") from exc
    preprocessor = _make_preprocessor(frame, features, config["missing"], config["encoding"], config["scaling"])
    estimator = _model(model_type, parameters)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    duration_ms = int((time.perf_counter() - started) * 1000)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    estimator_classes = pipeline.named_steps["model"].classes_
    labels = [str(label) for label in estimator_classes]
    average = "weighted"
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, average=average, zero_division=0)),
        "labels": labels,
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=estimator_classes).tolist(),
    }
    if probabilities is not None and len(labels) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
        except ValueError:
            pass
    model = pipeline.named_steps["model"]
    feature_names = _feature_names(pipeline.named_steps["preprocessor"])
    native = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    importance = sorted([{"feature": name, "importance": float(value)} for name, value in zip(feature_names, native)], key=lambda item: item["importance"], reverse=True)[:30]
    transformed_test = pipeline.named_steps["preprocessor"].transform(X_test)
    try:
        perm = permutation_importance(pipeline.named_steps["model"], transformed_test, y_test, n_repeats=4, random_state=42, scoring="f1_weighted")
        permutation = sorted([{"feature": name, "importance": float(value)} for name, value in zip(feature_names, perm.importances_mean)], key=lambda item: item["importance"], reverse=True)[:30]
    except Exception:
        permutation = []
    sample_predictions = []
    for index, (row_index, row) in enumerate(X_test.head(8).iterrows()):
        item: dict[str, Any] = {"id": f"row_{row_index}", "values": {feature: (None if pd.isna(row[feature]) else str(row[feature])) for feature in features}, "predicted": str(predictions[index]), "actual": str(y_test.iloc[index])}
        if probabilities is not None:
            item["probabilities"] = {label: float(probabilities[index][position]) for position, label in enumerate(labels)}
        sample_predictions.append(item)
    tree_text = export_text(model, feature_names=feature_names, max_depth=5) if model_type == "Decision Tree" else "Random Forest explainability is represented by aggregate feature importance and permutation importance."
    explainability = {"feature_importance": importance, "permutation_importance": permutation, "sample_predictions": sample_predictions, "tree_text": tree_text}
    return {"pipeline": pipeline, "metrics": metrics, "explainability": explainability, "duration_ms": duration_ms, "parameters": parameters}
