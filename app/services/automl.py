from __future__ import annotations
import logging
import time
from typing import Any
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from .training import TrainingError, _feature_names, _make_preprocessor, _model, _permutation_importance, _prediction_analysis, _prepare_frame
from ..config import get_settings
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


CONTROLLED_ESTIMATORS = ["dt", "rf"]
EXTENDED_ESTIMATORS = ["rf", "extra_tree", "lgbm", "xgboost", "histgb", "svc", "lrl1"]


class _ControlledModel:
    def __init__(self, estimator: Any, algorithm: str, score: float, evaluated_scores: dict[str, float]):
        self.model = estimator
        self.best_estimator = algorithm
        self.best_loss = 1 - score
        self.best_config = estimator.get_params()
        self.best_result = {"evaluated_trials": len(evaluated_scores)}
        self.evaluated_scores = evaluated_scores
        self._label_transformer = None

    def predict(self, values: Any) -> Any:
        return self.model.predict(values)

    def predict_proba(self, values: Any) -> Any:
        return self.model.predict_proba(values)

    @property
    def classes_(self) -> Any:
        return self.model.classes_


def _metric_score(actual: Any, predicted: Any, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(actual, predicted))
    if metric == "precision":
        return float(precision_score(actual, predicted, average="weighted", zero_division=0))
    if metric == "recall":
        return float(recall_score(actual, predicted, average="weighted", zero_division=0))
    return float(f1_score(actual, predicted, average="weighted", zero_division=0))


def _fit_controlled(
    transformed_train: Any,
    y_train: pd.Series,
    metric: str,
    class_weight: str | None,
) -> _ControlledModel:
    validation_stratify = y_train if y_train.value_counts().min() >= 2 else None
    candidate_train, candidate_validation, target_train, target_validation = train_test_split(
        transformed_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=validation_stratify,
    )
    candidates = {
        "dt": _model("Decision Tree", {"classWeight": "Balanced" if class_weight else "None"}),
        "rf": _model("Random Forest", {"classWeight": "Balanced" if class_weight else "None"}),
    }
    scores: dict[str, float] = {}
    for name, candidate in candidates.items():
        candidate.fit(candidate_train, target_train)
        scores[name] = _metric_score(target_validation, candidate.predict(candidate_validation), metric)
    winner = max(scores, key=scores.get)
    fitted_winner = clone(candidates[winner]).fit(transformed_train, y_train)
    return _ControlledModel(fitted_winner, winner, scores[winner], scores)
def run_automl(frame: pd.DataFrame, config: dict[str, Any], mode: str, time_budget: int, metric_label: str) -> dict[str, Any]:
    if not 1 <= time_budget <= 90:
        raise TrainingError("AutoML time budget must be between 1 and 90 seconds")
    operation_started = time.perf_counter()
    logger.info("AutoML stage=preprocessing mode=%s rows=%s columns=%s", mode, len(frame), len(frame.columns))
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise TrainingError("FLAML is not installed") from exc
    frame, features = _prepare_frame(frame, config)
    target = config["target"]
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
    logger.info("AutoML stage=preprocessing_complete mode=%s features=%s", mode, len(features))
    automl = AutoML()
    search_space = CONTROLLED_ESTIMATORS if mode == "Controlled" else EXTENDED_ESTIMATORS
    metric = {"Accuracy": "accuracy", "F1 score": "f1", "Precision": "precision", "Recall": "recall"}.get(metric_label, "f1")
    validation = config.get("validation", "Stratified 5-fold")
    min_class_count = int(y_train.value_counts().min()) if not y_train.empty else 0
    if validation.startswith("Stratified") and min_class_count >= 2:
        requested_splits = 3 if "3-fold" in validation else 5
        fit_options = {"eval_method": "cv", "n_splits": min(requested_splits, min_class_count)}
    else:
        fit_options = {"eval_method": "holdout", "split_ratio": 0.8}
    search_started = time.perf_counter()
    logger.info("AutoML stage=search_start mode=%s validation=%s", mode, validation)
    if mode == "Controlled":
        class_weight = "balanced" if config["imbalance"] == "Balanced class weights" else None
        automl = _fit_controlled(transformed_train, y_train, metric, class_weight)
    else:
        automl.fit(X_train=transformed_train, y_train=y_train, task="classification", time_budget=time_budget, metric=metric, estimator_list=search_space, verbose=0, **fit_options)
    search_duration_ms = int((time.perf_counter() - search_started) * 1000)
    logger.info("AutoML stage=search_complete mode=%s duration_ms=%s", mode, search_duration_ms)
    predictions = automl.predict(transformed_test)
    probabilities = None
    probability_labels: list[str] = []
    if hasattr(automl, "predict_proba"):
        try:
            candidate = automl.predict_proba(transformed_test)
            class_values = getattr(automl, "classes_", None)
            if class_values is None and hasattr(automl, "_label_transformer"):
                transformer = automl._label_transformer
                encoded_classes = range(len(candidate[0]))
                class_values = transformer.inverse_transform(encoded_classes)
            if class_values is not None and len(class_values) == len(candidate[0]):
                probabilities = candidate
                probability_labels = [str(value) for value in class_values]
        except (AttributeError, IndexError, TypeError, ValueError):
            probabilities = None
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
        "best_validation_score": 1 - float(automl.best_loss) if automl.best_loss is not None else None,
        "evaluation": {"average": average, "split": int(config["split"]), "random_state": 42, "validation": validation},
    }
    logger.info("AutoML stage=evaluation_complete mode=%s", mode)
    best = str(automl.best_estimator)
    chosen = automl.model.estimator if hasattr(automl.model, "estimator") else automl.model
    names = _feature_names(preprocessor)
    native = getattr(chosen, "feature_importances_", [])
    importance = sorted([{"feature": name, "importance": float(value)} for name, value in zip(names, native)], key=lambda item: item["importance"], reverse=True)[:30]
    sample_predictions, prediction_analysis = _prediction_analysis(
        predictions,
        y_test,
        probabilities,
        probability_labels,
        X_test.index,
        X_test,
        features,
    )
    # FLAML internally label-encodes string targets before fitting its estimator.
    # Score the raw estimator against that encoded target so permutation
    # importance is not silently discarded for normal string-class datasets.
    permutation, permutation_error = _permutation_importance(
        chosen,
        transformed_test,
        y_test,
        names,
        get_settings().explainability_max_rows,
        getattr(automl, "_label_transformer", None),
    )
    explainability = {
        "feature_importance": importance,
        "permutation_importance": permutation,
        "sample_predictions": sample_predictions,
        "prediction_analysis": prediction_analysis,
        "tree_text": f"FLAML selected {best}; detailed estimator explanation is available through its persisted configuration.",
        "status": "failed" if permutation_error else "completed",
        "error": permutation_error,
    }
    logger.info("AutoML stage=explainability_complete mode=%s status=%s", mode, explainability["status"])
    total_duration_ms = int((time.perf_counter() - operation_started) * 1000)
    best_result = automl.best_result or {}
    evaluated_trials = next((best_result.get(key) for key in ("evaluated_trials", "num_trials") if best_result.get(key) is not None), None)
    parameters = {
        "mode": mode,
        "time_budget": time_budget,
        "metric": metric_label,
        "search_space": search_space,
        "best_estimator": best,
        "best_config": automl.best_config or {},
        "evaluated_trials": evaluated_trials,
        "evaluated_estimators": getattr(automl, "evaluated_scores", None),
        "search_duration_ms": search_duration_ms,
        "total_duration_ms": total_duration_ms,
    }
    return {"pipeline": (preprocessor, automl), "metrics": metrics, "explainability": explainability, "duration_ms": total_duration_ms, "parameters": parameters, "model_type": f"AutoML · {best}"}
