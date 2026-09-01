"""Generate leakage-safe predictions for the latest available season."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "model.pkl"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "latest_predictions.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "models"
    / "latest_prediction_metrics.json"
)


def require_file(path: Path) -> None:
    """Raise a clear error when a required artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def format_euros(value: float) -> str:
    """Format a euro amount in millions."""
    return f"€{value / 1_000_000:,.2f}M"


def main() -> None:
    """Train chronologically and predict the latest season."""
    print("=" * 72)
    print("LATEST-SEASON PREDICTION")
    print("=" * 72)

    for path in (
        FEATURES_PATH,
        MODEL_PATH,
        FEATURE_COLUMNS_PATH,
    ):
        require_file(path)

    print("Loading features and saved model configuration...")

    features = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
    )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict):
        raise TypeError(
            "models/model.pkl must contain a model bundle dictionary."
        )

    required_bundle_keys = {
        "model",
        "feature_columns",
        "feature_defaults",
        "metadata",
    }

    missing_bundle_keys = (
        required_bundle_keys - set(bundle)
    )

    if missing_bundle_keys:
        raise KeyError(
            "Model bundle is missing keys: "
            f"{sorted(missing_bundle_keys)}"
        )

    saved_feature_columns = json.loads(
        FEATURE_COLUMNS_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    bundle_feature_columns = list(
        bundle["feature_columns"]
    )

    if saved_feature_columns != bundle_feature_columns:
        raise ValueError(
            "Feature order mismatch between model.pkl and "
            "feature_columns.json."
        )

    feature_columns = saved_feature_columns

    required_data_columns = {
        "player_id",
        "name",
        "season",
        "position",
        "primary_competition_id",
        "competition_name",
        "primary_club_id",
        "target_valuation_date",
        "target_market_value_eur",
        "y",
    }

    missing_data_columns = (
        required_data_columns - set(features.columns)
    )

    if missing_data_columns:
        raise KeyError(
            "Feature table is missing columns: "
            f"{sorted(missing_data_columns)}"
        )

    features["season"] = pd.to_numeric(
        features["season"],
        errors="coerce",
    )

    features["y"] = pd.to_numeric(
        features["y"],
        errors="coerce",
    )

    features["target_market_value_eur"] = pd.to_numeric(
        features["target_market_value_eur"],
        errors="coerce",
    )

    features = features.dropna(
        subset=[
            "season",
            "y",
            "target_market_value_eur",
        ]
    ).copy()

    features["season"] = features["season"].astype(int)

    latest_season = int(features["season"].max())
    earliest_season = int(features["season"].min())

    train_mask = features["season"] < latest_season
    latest_mask = features["season"] == latest_season

    train_rows = features.loc[train_mask].copy()
    latest_rows = features.loc[latest_mask].copy()

    if train_rows.empty:
        raise ValueError(
            "No earlier seasons are available for training."
        )

    if latest_rows.empty:
        raise ValueError(
            "No rows exist for the latest season."
        )

    if latest_rows["player_id"].duplicated().any():
        duplicate_count = int(
            latest_rows["player_id"].duplicated().sum()
        )
        raise ValueError(
            "Latest season contains duplicate players: "
            f"{duplicate_count:,}"
        )

    # Reindexing guarantees the exact training column order.
    numeric_features = features.reindex(
        columns=feature_columns
    ).apply(
        pd.to_numeric,
        errors="coerce",
    )

    x_train = numeric_features.loc[
        train_rows.index
    ].copy()

    x_latest = numeric_features.loc[
        latest_rows.index
    ].copy()

    feature_defaults = bundle.get(
        "feature_defaults",
        {},
    )

    # Defaults are calculated using training seasons only.
    # This prevents information from the held-out season
    # influencing its own prediction.
    chronological_defaults: dict[str, float] = {}

    for column in feature_columns:
        median = x_train[column].median()

        if pd.isna(median):
            median = feature_defaults.get(column, 0.0)

        if pd.isna(median):
            median = 0.0

        chronological_defaults[column] = float(median)

    x_train = x_train.fillna(
        chronological_defaults
    )

    x_latest = x_latest.fillna(
        chronological_defaults
    )

    if not np.isfinite(x_train.to_numpy()).all():
        raise ValueError(
            "Training features contain infinite values."
        )

    if not np.isfinite(x_latest.to_numpy()).all():
        raise ValueError(
            "Latest-season features contain infinite values."
        )

    y_train = train_rows["y"].astype(float)

    # clone() copies the tuned configuration but not the
    # fitted trees. The model is therefore retrained cleanly.
    holdout_model = clone(bundle["model"])

    print(
        f"Training seasons: {earliest_season}–"
        f"{latest_season - 1}"
    )
    print(f"Latest held-out season: {latest_season}")
    print(f"Training rows: {len(train_rows):,}")
    print(f"Prediction rows: {len(latest_rows):,}")
    print(f"Feature count: {len(feature_columns):,}")
    print("\nTraining chronological holdout model...")

    holdout_model.fit(
        x_train,
        y_train,
    )

    predicted_log_values = holdout_model.predict(
        x_latest
    )

    predicted_values = np.expm1(
        predicted_log_values
    )

    # Market values cannot be negative.
    predicted_values = np.clip(
        predicted_values,
        a_min=0.0,
        a_max=None,
    )

    actual_values = latest_rows[
        "target_market_value_eur"
    ].to_numpy(dtype=float)

    mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values,
        )
    )

    r2 = r2_score(
        actual_values,
        predicted_values,
    )

    prediction_gap = (
        predicted_values - actual_values
    )

    output = pd.DataFrame(
        {
            "player_id": latest_rows[
                "player_id"
            ].to_numpy(),
            "name": latest_rows[
                "name"
            ].to_numpy(),
            "season": latest_rows[
                "season"
            ].to_numpy(),
            "position": latest_rows[
                "position"
            ].to_numpy(),
            "primary_competition_id": latest_rows[
                "primary_competition_id"
            ].to_numpy(),
            "competition_name": latest_rows[
                "competition_name"
            ].to_numpy(),
            "primary_club_id": latest_rows[
                "primary_club_id"
            ].to_numpy(),
            "target_valuation_date": latest_rows[
                "target_valuation_date"
            ].to_numpy(),
            "actual_market_value_eur": actual_values,
            "predicted_market_value_eur": predicted_values,
            "prediction_gap_eur": prediction_gap,
            "absolute_error_eur": np.abs(
                prediction_gap
            ),
            "model": (
                "XGBoost chronological holdout"
            ),
        }
    )


    # Compatibility aliases expected by build_player_catalog.py.
    output["actual_value_eur"] = output[
        "actual_market_value_eur"
    ]
    output["predicted_value_eur"] = output[
        "predicted_market_value_eur"
    ]

    output = output.sort_values(
        ["name", "player_id"],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    metrics = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "prediction_type": (
            "chronological_latest_season_holdout"
        ),
        "model": "XGBoost",
        "training_start_season": earliest_season,
        "training_end_season": latest_season - 1,
        "test_season": latest_season,
        "training_rows": int(len(train_rows)),
        "test_rows": int(len(latest_rows)),
        "feature_count": int(len(feature_columns)),
        "mae_eur": float(mae),
        "rmse_eur": float(rmse),
        "r2": float(r2),
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("LATEST-SEASON RESULTS")
    print("=" * 72)
    print(f"MAE:  {format_euros(mae)}")
    print(f"RMSE: {format_euros(rmse)}")
    print(f"R²:   {r2:.4f}")
    print(
        "Meaning: on average, predictions miss by "
        f"approximately {format_euros(mae)}."
    )
    print(f"\nPredictions: {OUTPUT_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
