from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.experiments import model_parameters, to_out, validate_json_payload
from app.config import Settings
from app.models import Experiment
from app.schemas import TrainIn
from app.services.profiling import parse_csv
from app.services import training
from app.services.automl import CONTROLLED_ESTIMATORS, EXTENDED_ESTIMATORS, run_automl
from app.services.training import TrainingError, _model, train_classifier


def train_payload(**overrides: object) -> TrainIn:
    values = {
        "dataset_id": "dataset-1",
        "model_type": "Decision Tree",
        "target": "target",
        "features": ["feature"],
        "parameters": {"criterion": "gini", "maxDepth": "3"},
    }
    values.update(overrides)
    return TrainIn(**values)


def test_invalid_numeric_model_parameter_is_rejected_by_schema() -> None:
    with pytest.raises(ValidationError, match="parameters.maxDepth"):
        train_payload(parameters={"maxDepth": "not-a-number"})


def test_invalid_numeric_model_parameter_is_a_training_error() -> None:
    with pytest.raises(TrainingError, match="maxDepth"):
        _model("Decision Tree", {"maxDepth": "not-a-number"})


def test_imbalance_preprocessing_controls_model_class_weight() -> None:
    balanced = train_payload(imbalance="Balanced class weights")
    unbalanced = train_payload(imbalance="No adjustment")

    assert model_parameters(balanced, {"imbalance": balanced.imbalance})["classWeight"] == "Balanced"
    assert model_parameters(unbalanced, {"imbalance": unbalanced.imbalance})["classWeight"] == "None"


def test_training_applies_no_adjustment() -> None:
    frame = pd.DataFrame({"target": ["no", "yes"] * 4, "feature": range(8)})
    config = {
        "target": "target",
        "features": ["feature"],
        "missing": "Median imputation",
        "encoding": "One-hot encoding",
        "scaling": "No scaling",
        "duplicates": "Keep first row",
        "imbalance": "No adjustment",
        "split": 75,
    }

    result = train_classifier(frame, "Decision Tree", config, {"classWeight": "None"})

    assert result["pipeline"].named_steps["model"].class_weight is None
    analysis = result["explainability"]["prediction_analysis"]
    assert analysis["available"] is True
    assert sum(analysis["correct"]) + sum(analysis["incorrect"]) == 2
    assert all("max_probability" in row for row in result["explainability"]["sample_predictions"])
    assert result["metrics"]["evaluation"] == {"average": "weighted", "split": 75, "random_state": 42}


def test_training_preserves_multiclass_labels_and_probability_mapping() -> None:
    frame = pd.DataFrame(
        {
            "target": ["red", "green", "blue"] * 4,
            "feature": list(range(12)),
        }
    )
    config = {
        "target": "target",
        "features": ["feature"],
        "missing": "Median imputation",
        "encoding": "One-hot encoding",
        "scaling": "No scaling",
        "duplicates": "Keep first row",
        "imbalance": "Balanced class weights",
        "split": 75,
    }

    result = train_classifier(frame, "Decision Tree", config, {"classWeight": "Balanced"})
    metrics = result["metrics"]
    samples = result["explainability"]["sample_predictions"]

    assert set(metrics["labels"]) == {"blue", "green", "red"}
    assert len(metrics["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in metrics["confusion_matrix"])
    assert all(sum(sample["probabilities"].values()) == pytest.approx(1) for sample in samples)


def classification_config() -> dict[str, object]:
    return {
        "target": "target",
        "features": ["feature"],
        "missing": "Median imputation",
        "encoding": "One-hot encoding",
        "scaling": "No scaling",
        "duplicates": "Keep first row",
        "imbalance": "Balanced class weights",
        "split": 75,
        "validation": "Holdout 20%",
    }


def test_automl_estimator_lists_are_explicit() -> None:
    assert CONTROLLED_ESTIMATORS == ["dt", "rf"]
    assert EXTENDED_ESTIMATORS == ["rf", "extra_tree", "lgbm", "xgboost", "histgb", "svc", "lrl1"]


def test_automl_returns_winning_configuration_and_separate_durations() -> None:
    frame = pd.DataFrame({"target": ["no", "yes"] * 6, "feature": range(12)})

    result = run_automl(frame, classification_config(), "Extended", 1, "F1 score")

    parameters = result["parameters"]
    assert parameters["best_estimator"] in EXTENDED_ESTIMATORS
    assert parameters["best_config"]
    assert parameters["time_budget"] == 1
    assert parameters["search_duration_ms"] >= 0
    assert parameters["total_duration_ms"] >= parameters["search_duration_ms"]
    assert result["duration_ms"] == parameters["total_duration_ms"]
    assert result["metrics"]["best_validation_score"] == pytest.approx(1 - result["metrics"]["best_loss"])


def test_controlled_automl_fits_and_scores_decision_tree_and_random_forest() -> None:
    frame = pd.DataFrame({"target": ["no", "yes"] * 6, "feature": range(12)})

    result = run_automl(frame, classification_config(), "Controlled", 1, "F1 score")

    parameters = result["parameters"]
    scores = parameters["evaluated_estimators"]
    assert set(scores) == {"dt", "rf"}
    assert parameters["best_estimator"] in {"dt", "rf"}
    assert parameters["best_config"]
    assert result["metrics"]["best_validation_score"] == pytest.approx(scores[parameters["best_estimator"]])
    if parameters["best_estimator"] == "dt":
        assert "max_depth" in parameters["best_config"]
    else:
        assert "n_estimators" in parameters["best_config"]


def test_explainability_failure_keeps_training_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("explainability test failure")

    monkeypatch.setattr(training, "permutation_importance", fail)
    frame = pd.DataFrame({"target": ["no", "yes"] * 4, "feature": range(8)})

    result = train_classifier(frame, "Decision Tree", classification_config(), {"classWeight": "None"})

    assert result["metrics"]["accuracy"] >= 0
    assert result["explainability"]["status"] == "failed"
    assert "explainability test failure" in result["explainability"]["error"]


def test_non_finite_persisted_payload_is_rejected() -> None:
    with pytest.raises(TrainingError, match="non-serializable or non-finite"):
        validate_json_payload({"metric": float("nan")}, "AutoML metrics")


def test_numpy_scalar_payload_is_cleaned_before_validation() -> None:
    payload = validate_json_payload({"metric": pd.Series([0.75]).iloc[0]}, "AutoML metrics")

    assert payload == {"metric": pytest.approx(0.75)}


def test_settings_reject_insecure_production_cookies() -> None:
    production_database = "postgresql+psycopg://user:password@db.example.com:5432/postgres?sslmode=require"

    with pytest.raises(ValidationError, match="COOKIE_SECURE"):
        Settings(app_env="production", cookie_secure=False, database_url=production_database, session_secret="x" * 32)

    assert Settings(app_env="production", cookie_secure=True, database_url=production_database, session_secret="x" * 32).cookie_secure is True


def test_settings_reject_insecure_production_database() -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        Settings(app_env="production", cookie_secure=True, session_secret="x" * 32)


def test_experiment_response_serializes_created_at() -> None:
    experiment = Experiment(
        id="experiment-1",
        dataset_id="dataset-1",
        dataset_version_id="version-1",
        model_type="Decision Tree",
        status="completed",
        parameters={},
        metrics={},
        explainability={},
        duration_ms=1,
        created_at=datetime(2026, 11, 14, 16, 59, tzinfo=timezone.utc),
    )

    response = to_out(experiment)

    assert response.created_at == "2026-11-14T16:59:00+00:00"


@pytest.mark.parametrize(
    ("delimiter", "filename"),
    [(",", "comma.csv"), (";", "semicolon.csv"), ("\t", "tab.csv"), ("|", "pipe.csv")],
)
def test_csv_delimiters_are_profiled(delimiter: str, filename: str) -> None:
    content = f"target{delimiter}feature\nno{delimiter}1\nyes{delimiter}2\n".encode()

    profile = parse_csv(content, filename)

    assert profile["target"] == "target"
    assert profile["column_count"] == 2
