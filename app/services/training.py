from __future__ import annotations
import io
import logging
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
from ..config import get_settings

logger = logging.getLogger(__name__)

class TrainingError(ValueError):
    pass


def _permutation_importance(
    estimator: Any,
    transformed: Any,
    target: Any,
    feature_names: list[str],
    sample_limit: int,
    target_transformer: Any = None,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        sample_count = min(len(target), max(1, sample_limit))
        sample_indices = np.arange(sample_count)
        sampled_target = target.iloc[sample_indices] if hasattr(target, "iloc") else target[sample_indices]
        if target_transformer is not None:
            sampled_target = target_transformer.transform(sampled_target)
        permutation = permutation_importance(
            estimator,
            transformed[sample_indices],
            sampled_target,
            n_repeats=4,
            random_state=42,
            scoring="f1_weighted",
        )
        return (
            sorted(
                [{"feature": name, "importance": float(value)} for name, value in zip(feature_names, permutation.importances_mean)],
                key=lambda item: item["importance"],
                reverse=True,
            )[:30],
            None,
        )
    except Exception as exc:
        logger.exception("Explainability permutation importance failed")
        return [], f"Permutation importance unavailable: {type(exc).__name__}: {exc}"


def _prediction_analysis(
    predictions: Any,
    actual: Any,
    probabilities: Any,
    labels: list[str],
    row_indices: Any,
    rows: pd.DataFrame,
    features: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_labels = [str(value) for value in predictions]
    actual_labels = [str(value) for value in actual]
    sample_predictions: list[dict[str, Any]] = []
    if probabilities is None or not labels:
        for index, (row_index, row) in enumerate(rows.head(8).iterrows()):
            sample_predictions.append(
                {
                    "id": f"row_{row_index}",
                    "values": {feature: (None if pd.isna(row[feature]) else str(row[feature])) for feature in features},
                    "predicted": prediction_labels[index],
                    "actual": actual_labels[index],
                    "correct": prediction_labels[index] == actual_labels[index],
                }
            )
        return sample_predictions, {"available": False, "reason": "The selected estimator does not provide class probabilities."}

    probability_rows = [[float(value) for value in row] for row in probabilities]
    confidences = [max(row) for row in probability_rows]
    correct = [predicted == expected for predicted, expected in zip(prediction_labels, actual_labels)]
    bin_edges = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.000001]
    correct_counts = [0] * (len(bin_edges) - 1)
    incorrect_counts = [0] * (len(bin_edges) - 1)
    for confidence, is_correct in zip(confidences, correct):
        bucket = next((index for index in range(len(bin_edges) - 1) if bin_edges[index] <= confidence < bin_edges[index + 1]), len(correct_counts) - 1)
        (correct_counts if is_correct else incorrect_counts)[bucket] += 1
    for index, (row_index, row) in enumerate(rows.head(8).iterrows()):
        sample_predictions.append(
            {
                "id": f"row_{row_index}",
                "values": {feature: (None if pd.isna(row[feature]) else str(row[feature])) for feature in features},
                "predicted": prediction_labels[index],
                "actual": actual_labels[index],
                "correct": correct[index],
                "probabilities": {label: probability_rows[index][position] for position, label in enumerate(labels)},
                "max_probability": confidences[index],
            }
        )
    high_confidence_errors = [
        {
            "id": f"row_{row_index}",
            "predicted": prediction,
            "actual": expected,
            "max_probability": confidence,
            "correct": False,
        }
        for row_index, prediction, expected, confidence, is_correct in zip(row_indices, prediction_labels, actual_labels, confidences, correct)
        if not is_correct and confidence >= 0.8
    ]
    return sample_predictions, {
        "available": True,
        "bin_edges": bin_edges[:-1] + [1],
        "correct": correct_counts,
        "incorrect": incorrect_counts,
        "high_confidence_errors": high_confidence_errors[:50],
        "threshold": 0.8,
    }


def _class_weight(value: str) -> str | None:
    return "balanced" if value.lower().startswith("balanced") else None


def _integer_parameter(parameters: dict[str, Any], key: str, default: int, minimum: int, allow_none: bool = False) -> int | None:
    value = parameters.get(key, default)
    if allow_none and (value is None or value == "None"):
        return None
    if isinstance(value, bool):
        raise TrainingError(f"Model parameter '{key}' must be an integer of at least {minimum}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"Model parameter '{key}' must be an integer of at least {minimum}") from exc
    if integer < minimum:
        raise TrainingError(f"Model parameter '{key}' must be an integer of at least {minimum}")
    return integer


def _make_preprocessor(frame: pd.DataFrame, features: list[str], missing: str, encoding: str, scaling: str) -> ColumnTransformer:
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in features if column not in numeric]
    impute = "median" if missing.startswith("Median") else "mean" if missing.startswith("Mean") else "most_frequent"
    add_indicator = missing == "Flag as feature"
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy=impute, add_indicator=add_indicator))]
    if scaling == "Standard scaler":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaling == "Robust scaler":
        numeric_steps.append(("scaler", RobustScaler()))
    elif scaling == "Min-max scaler":
        numeric_steps.append(("scaler", MinMaxScaler()))
    cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) if encoding == "Ordinal encoding" else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers: list[tuple[str, Any, list[str]]] = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", add_indicator=add_indicator)), ("encoder", cat_encoder)]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _prepare_frame(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    target = config.get("target")
    features = list(dict.fromkeys(config.get("features") or []))
    if not target or target not in frame.columns:
        raise TrainingError(f"Target column '{target}' does not exist")
    if not features:
        raise TrainingError("Select at least one feature")
    if target in features:
        raise TrainingError("The target column cannot also be selected as a feature")
    if any(feature not in frame.columns for feature in features):
        raise TrainingError("One or more selected features do not exist")
    allowed = {
        "missing": {"Median imputation", "Mean imputation", "Drop rows", "Flag as feature"},
        "encoding": {"One-hot encoding", "Ordinal encoding"},
        "scaling": {"Standard scaler", "Robust scaler", "Min-max scaler", "No scaling"},
        "duplicates": {"Keep first row", "Drop duplicates"},
        "imbalance": {"Balanced class weights", "No adjustment"},
    }
    for key, choices in allowed.items():
        if config.get(key) not in choices:
            raise TrainingError(f"Unsupported {key} option: {config.get(key)}")
    frame = frame.dropna(subset=[target]).copy()
    if config.get("duplicates") == "Drop duplicates":
        frame = frame.drop_duplicates()
    if config.get("missing") == "Drop rows":
        frame = frame.dropna(subset=features)
    if frame.empty:
        raise TrainingError("No rows remain after the selected preprocessing rules")
    return frame, features


def _model(model_type: str, parameters: dict[str, Any]):
    weight = _class_weight(str(parameters.get("classWeight", parameters.get("class_weight", "Balanced"))))
    if model_type == "Decision Tree":
        criterion = str(parameters.get("criterion", "gini")).lower()
        if criterion not in {"gini", "entropy"}:
            raise TrainingError("Decision Tree criterion must be gini or entropy")
        depth = parameters.get("maxDepth", parameters.get("max_depth", 5))
        return DecisionTreeClassifier(
            criterion=criterion,
            max_depth=_integer_parameter({"maxDepth": depth}, "maxDepth", 5, 1, allow_none=True),
            min_samples_split=_integer_parameter({"minSplit": parameters.get("minSplit", parameters.get("min_samples_split", 2))}, "minSplit", 2, 2),
            min_samples_leaf=_integer_parameter({"minLeaf": parameters.get("minLeaf", parameters.get("min_samples_leaf", 1))}, "minLeaf", 1, 1),
            class_weight=weight,
            random_state=42,
        )
    if model_type == "Random Forest":
        depth = parameters.get("maxDepth", parameters.get("max_depth", 8))
        return RandomForestClassifier(
            n_estimators=_integer_parameter({"estimators": parameters.get("estimators", parameters.get("n_estimators", 100))}, "estimators", 100, 1),
            max_depth=_integer_parameter({"maxDepth": depth}, "maxDepth", 8, 1, allow_none=True),
            max_features=parameters.get("maxFeatures", parameters.get("max_features", "sqrt")),
            min_samples_split=_integer_parameter({"minSplit": parameters.get("minSplit", parameters.get("min_samples_split", 2))}, "minSplit", 2, 2),
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
    frame, features = _prepare_frame(frame, config)
    target = config["target"]
    y = frame[target]
    if y.dropna().drop_duplicates().shape[0] < 2:
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
        "evaluation": {"average": average, "split": int(config["split"]), "random_state": 42},
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
    permutation, permutation_error = _permutation_importance(
        pipeline.named_steps["model"],
        transformed_test,
        y_test,
        feature_names,
        get_settings().explainability_max_rows,
    )
    sample_predictions, prediction_analysis = _prediction_analysis(
        predictions,
        y_test,
        probabilities,
        labels,
        X_test.index,
        X_test,
        features,
    )
    tree_text = export_text(model, feature_names=feature_names, max_depth=5) if model_type == "Decision Tree" else "Random Forest explainability is represented by aggregate feature importance and permutation importance."
    explainability = {
        "feature_importance": importance,
        "permutation_importance": permutation,
        "sample_predictions": sample_predictions,
        "prediction_analysis": prediction_analysis,
        "tree_text": tree_text,
        "status": "failed" if permutation_error else "completed",
        "error": permutation_error,
    }
    return {"pipeline": pipeline, "metrics": metrics, "explainability": explainability, "duration_ms": duration_ms, "parameters": parameters}
