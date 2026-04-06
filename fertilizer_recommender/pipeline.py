from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from fertilizer_recommender.config import (
    CATEGORICAL_ENCODERS_PATH,
    DATA_PATH,
    FEATURE_ORDER,
    MODEL_DIR,
    MODEL_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
    TARGET_ENCODER_PATH,
)


@dataclass
class TrainingArtifacts:
    model: object
    scaler: StandardScaler
    target_encoder: LabelEncoder
    categorical_encoders: dict[str, LabelEncoder]
    metrics: dict[str, dict[str, float]]


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _encode_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, dict[str, LabelEncoder]]:
    encoded = df.copy()

    target_encoder = LabelEncoder()
    encoded[TARGET_COLUMN] = target_encoder.fit_transform(encoded[TARGET_COLUMN])

    categorical_columns = encoded.select_dtypes(include=["object", "string"]).columns
    categorical_encoders: dict[str, LabelEncoder] = {}
    for column in categorical_columns:
        encoder = LabelEncoder()
        encoded[column] = encoder.fit_transform(encoded[column])
        categorical_encoders[column] = encoder

    return encoded, target_encoder, categorical_encoders


def train_pipeline(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingArtifacts:
    dataset = load_dataset() if df is None else df.copy()
    encoded_df, target_encoder, categorical_encoders = _encode_dataframe(dataset)

    X = encoded_df[FEATURE_ORDER]
    y = encoded_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "knn": KNeighborsClassifier(),
        "svm": SVC(),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    metrics: dict[str, dict[str, float]] = {}
    trained_models: dict[str, object] = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        metrics[name] = {"accuracy": accuracy}
        trained_models[name] = model

    best_model_name = max(metrics, key=lambda name: metrics[name]["accuracy"])
    return TrainingArtifacts(
        model=trained_models[best_model_name],
        scaler=scaler,
        target_encoder=target_encoder,
        categorical_encoders=categorical_encoders,
        metrics=metrics,
    )


def save_artifacts(artifacts: TrainingArtifacts) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, MODEL_PATH)
    joblib.dump(artifacts.scaler, SCALER_PATH)
    joblib.dump(artifacts.target_encoder, TARGET_ENCODER_PATH)
    joblib.dump(artifacts.categorical_encoders, CATEGORICAL_ENCODERS_PATH)


def train_and_save(df: pd.DataFrame | None = None) -> TrainingArtifacts:
    artifacts = train_pipeline(df=df)
    save_artifacts(artifacts)
    return artifacts


def load_artifacts() -> TrainingArtifacts:
    required_paths = [
        MODEL_PATH,
        SCALER_PATH,
        TARGET_ENCODER_PATH,
        CATEGORICAL_ENCODERS_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing model artifacts. Run `python train_model.py` first. "
            f"Missing: {', '.join(missing_paths)}"
        )

    return TrainingArtifacts(
        model=joblib.load(MODEL_PATH),
        scaler=joblib.load(SCALER_PATH),
        target_encoder=joblib.load(TARGET_ENCODER_PATH),
        categorical_encoders=joblib.load(CATEGORICAL_ENCODERS_PATH),
        metrics={},
    )


def prepare_input_frame(
    raw_inputs: dict[str, object],
    categorical_encoders: dict[str, LabelEncoder],
) -> pd.DataFrame:
    encoded_inputs = raw_inputs.copy()

    for column, encoder in categorical_encoders.items():
        if column not in encoded_inputs:
            continue
        value = str(encoded_inputs[column])
        known_values = set(encoder.classes_)
        if value not in known_values:
            raise ValueError(
                f"Unexpected value '{value}' for '{column}'. Allowed values: {sorted(known_values)}"
            )
        encoded_inputs[column] = int(encoder.transform([value])[0])

    frame = pd.DataFrame([[encoded_inputs[column] for column in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return frame


def predict_recommendation(
    raw_inputs: dict[str, object],
    artifacts: TrainingArtifacts | None = None,
) -> str:
    loaded_artifacts = artifacts or load_artifacts()
    input_frame = prepare_input_frame(raw_inputs, loaded_artifacts.categorical_encoders)
    scaled_input = loaded_artifacts.scaler.transform(input_frame)
    prediction = int(loaded_artifacts.model.predict(scaled_input)[0])
    return str(loaded_artifacts.target_encoder.inverse_transform([prediction])[0])


def get_feature_options(df: pd.DataFrame | None = None) -> dict[str, list[str]]:
    dataset = load_dataset() if df is None else df
    option_columns = [
        "Soil_Type",
        "Crop_Type",
        "Crop_Growth_Stage",
        "Season",
        "Irrigation_Type",
        "Previous_Crop",
        "Region",
    ]
    return {column: sorted(dataset[column].dropna().astype(str).unique().tolist()) for column in option_columns}


def get_target_labels(df: pd.DataFrame | None = None) -> list[str]:
    dataset = load_dataset() if df is None else df
    return sorted(dataset[TARGET_COLUMN].dropna().astype(str).unique().tolist())
