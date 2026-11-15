from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from app.api.experiments import model_parameters, to_out
from app.config import Settings
from app.models import Experiment
from app.schemas import TrainIn
from app.services.profiling import parse_csv
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


def test_settings_reject_insecure_production_cookies() -> None:
    with pytest.raises(ValidationError, match="COOKIE_SECURE"):
        Settings(app_env="production", cookie_secure=False)

    assert Settings(app_env="production", cookie_secure=True).cookie_secure is True


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
