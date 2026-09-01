"""Tune XGBoost, evaluate it on 2025, and save deployment artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SEASON = 2025

BASE_FEATURES = [
    "goals",
    "assists",
    "minutes_played",
    "age",
    "age_squared",
    "goals_per_90",
    "assists_per_90",
    "goal_contributions_per_90",
    "appearances",
    "minutes_share",
    "is_prime_age",
    "club_strength_log",
    "club_strength_available",
]

FEATURE_PREFIXES = (
    "position_",
    "league_",
    "club_tier_",
)

# We deliberately exclude these from the historical model because
# reliable historical contract snapshots are unavailable for most rows.
EXCLUDED_CONTRACT_FEATURES = [
    "contract_years_remaining",
    "contract_info_available",
]


def format_euros(value: float) -> str:
    """Format a euro value in readable millions or thousands."""
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:,.2f}M"

    if abs(value) >= 1_000:
        return f"€{value / 1_000:,.0f}K"

    return f"€{value:,.0f}"


def load_features() -> pd.DataFrame:
    """Load and validate the engineered player-season data."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run python src/features.py first."
        )

    frame = pd.read_csv(FEATURES_PATH, low_memory=False)

    required_columns = {
        "season",
        "y",
        "target_market_value_eur",
    }

    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(
            "The feature file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["season"] = pd.to_numeric(
        frame["season"],
        errors="raise",
    ).astype(int)

    frame["y"] = pd.to_numeric(
        frame["y"],
        errors="raise",
    )

    frame["target_market_value_eur"] = pd.to_numeric(
        frame["target_market_value_eur"],
        errors="raise",
    )

    return frame


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the exact ordered feature list used by the model."""
    feature_columns = [
        column
        for column in frame.columns
        if (
            column in BASE_FEATURES
            or column.startswith(FEATURE_PREFIXES)
        )
        and column not in EXCLUDED_CONTRACT_FEATURES
    ]

    if not feature_columns:
        raise ValueError("No model feature columns were found.")

    missing_base_features = [
        feature
        for feature in BASE_FEATURES
        if feature not in feature_columns
    ]

    if missing_base_features:
        raise ValueError(
            "Missing expected model features: "
            + ", ".join(missing_base_features)
        )

    return feature_columns


def prepare_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    defaults: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create a numeric matrix and replace any missing feature values."""
    matrix = frame[feature_columns].copy()

    for column in feature_columns:
        matrix[column] = pd.to_numeric(
            matrix[column],
            errors="coerce",
        )

    if defaults is None:
        defaults = {
            column: float(matrix[column].median())
            if matrix[column].notna().any()
            else 0.0
            for column in feature_columns
        }

    matrix = matrix.fillna(defaults)

    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError(
            "The feature matrix contains infinite or invalid values."
        )

    return matrix, defaults


def log_predictions_to_euros(
    log_predictions: np.ndarray,
) -> np.ndarray:
    """Invert log1p predictions and prevent negative market values."""
    predictions = np.expm1(log_predictions)
    return np.clip(predictions, a_min=0.0, a_max=None)


def euro_mae_scorer(
    estimator: XGBRegressor,
    features: pd.DataFrame,
    log_targets: pd.Series,
) -> float:
    """Return negative MAE in euros for RandomizedSearchCV."""
    log_predictions = estimator.predict(features)

    actual_euros = np.expm1(
        np.asarray(log_targets, dtype=float)
    )

    predicted_euros = log_predictions_to_euros(
        np.asarray(log_predictions, dtype=float)
    )

    return -mean_absolute_error(
        actual_euros,
        predicted_euros,
    )


def make_time_folds(
    seasons: pd.Series,
    number_of_folds: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window folds that never train on the future."""
    unique_seasons = sorted(seasons.unique())

    if len(unique_seasons) <= number_of_folds:
        raise ValueError(
            "Not enough seasons to create time-based CV folds."
        )

    validation_seasons = unique_seasons[-number_of_folds:]
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    print("\nTime-aware cross-validation folds:")

    for validation_season in validation_seasons:
        train_indices = np.flatnonzero(
            seasons.to_numpy() < validation_season
        )

        validation_indices = np.flatnonzero(
            seasons.to_numpy() == validation_season
        )

        if len(train_indices) == 0 or len(validation_indices) == 0:
            continue

        training_season_min = int(
            seasons.iloc[train_indices].min()
        )

        training_season_max = int(
            seasons.iloc[train_indices].max()
        )

        print(
            f"  Train {training_season_min}–{training_season_max} "
            f"({len(train_indices):,} rows) -> "
            f"validate {validation_season} "
            f"({len(validation_indices):,} rows)"
        )

        folds.append(
            (
                train_indices,
                validation_indices,
            )
        )

    if len(folds) != number_of_folds:
        raise ValueError(
            f"Expected {number_of_folds} folds, created {len(folds)}."
        )

    return folds


def calculate_metrics(
    actual_euros: np.ndarray,
    predicted_euros: np.ndarray,
) -> dict[str, float]:
    """Calculate evaluation metrics in real euros."""
    return {
        "mae_eur": float(
            mean_absolute_error(
                actual_euros,
                predicted_euros,
            )
        ),
        "rmse_eur": float(
            mean_squared_error(
                actual_euros,
                predicted_euros,
            )
            ** 0.5
        ),
        "r2": float(
            r2_score(
                actual_euros,
                predicted_euros,
            )
        ),
    }


def json_ready(value):
    """Convert NumPy values into normal Python values for JSON."""
    if isinstance(value, dict):
        return {
            str(key): json_ready(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def write_model_card(
    metadata: dict,
    feature_columns: list[str],
) -> None:
    """Write transparent documentation for the saved model."""
    metrics = metadata["evaluation_metrics"]

    feature_lines = "\n".join(
        f"{index}. `{feature}`"
        for index, feature in enumerate(
            feature_columns,
            start=1,
        )
    )

    parameter_lines = "\n".join(
        f"- `{name}`: {value}"
        for name, value in sorted(
            metadata["best_parameters"].items()
        )
    )

    card = f"""# PL Transfer Value — Model Card

## Model overview

This model estimates a professional football player's market value in euros
from a completed domestic-league season.

The estimator is an XGBoost gradient-boosted tree regressor trained on
`log1p(market_value_in_eur)`. Predictions are converted back to euros with
`expm1`.

## Intended use

The model is intended for:

- exploratory football analytics
- preliminary player valuation
- scouting support
- identifying players who may deserve further investigation

It should not be treated as a transfer-fee quote or used as the only basis
for a recruitment decision.

## Training and evaluation window

- Final training seasons: {metadata["training_window"]}
- Evaluation training seasons: {metadata["evaluation_training_window"]}
- Held-out test season: {metadata["test_season"]}
- Final training rows: {metadata["final_training_rows"]:,}
- Held-out test rows: {metadata["test_rows"]:,}

The held-out test season was not used during hyperparameter tuning.

## Held-out performance

- MAE: {format_euros(metrics["mae_eur"])}
- RMSE: {format_euros(metrics["rmse_eur"])}
- R²: {metrics["r2"]:.4f}

The MAE is the most readable headline: predictions differ from the observed
market value by approximately {format_euros(metrics["mae_eur"])} on average.

The larger RMSE shows that errors are substantially greater for some
high-value players.

## Target

`y = log1p(market_value_in_eur)`

Training on the logarithmic target reduces the influence of a small number
of extremely valuable players. Deployment predictions are inverted with
`expm1`.

## Exact feature order

The following order must be preserved during prediction:

{feature_lines}

A feature-order mismatch can produce plausible-looking but incorrect
predictions without necessarily raising an error. The same ordered list is
therefore stored inside `model.pkl` and in `feature_columns.json`.

## Best hyperparameters

{parameter_lines}

## Contract-data policy

Contract features are excluded from the primary historical model.

The dataset contains current contract-expiration information, but does not
provide reliable contract snapshots for every historical player-season.
Attaching a current contract date to an old season would leak future
information into training.

Contract information may be displayed as contextual or experimental
information in the application, but this saved model does not silently
pretend that complete historical contract data exists.

## Important limitations

- Transfermarkt values are estimates, not confirmed transfer fees.
- Historical contract information is incomplete.
- Injuries and injury history are not fully represented.
- Reputation, nationality, commercial value, and negotiation conditions are
  not modeled directly.
- The data has stronger coverage for well-documented competitions.
- Market value changes over time, so model performance can decay.
- Predictions for unusual player profiles are less reliable.
- Holding-out one recent season is useful but is not the same as continuous
  production monitoring.

## Responsible interpretation

This model should generate evidence for discussion, not replace scouts,
analysts, medical staff, or contract specialists.

Generated: {metadata["saved_at_utc"]}
"""

    (MODELS_DIR / "model_card.md").write_text(
        card,
        encoding="utf-8",
    )


def main() -> None:
    """Run tuning, final evaluation, retraining, and artifact saving."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("XGBOOST TUNING AND FINAL MODEL")
    print("=" * 72)
    print("Loading engineered features...")

    data = load_features()
    feature_columns = get_feature_columns(data)

    train_mask = data["season"] < TEST_SEASON
    test_mask = data["season"] == TEST_SEASON

    if not train_mask.any():
        raise ValueError("No training rows were found.")

    if not test_mask.any():
        raise ValueError(
            f"No rows were found for test season {TEST_SEASON}."
        )

    training_data = data.loc[train_mask].copy()
    test_data = data.loc[test_mask].copy()

    X_train, training_defaults = prepare_matrix(
        training_data,
        feature_columns,
    )

    X_test, _ = prepare_matrix(
        test_data,
        feature_columns,
        defaults=training_defaults,
    )

    y_train = training_data["y"].copy()
    y_test = test_data["y"].copy()

    print(
        f"Training rows: {len(training_data):,}"
    )
    print(
        f"Test rows: {len(test_data):,}"
    )
    print(
        f"Features: {len(feature_columns):,}"
    )
    print(
        f"Training seasons: "
        f"{training_data['season'].min()}–"
        f"{training_data['season'].max()}"
    )
    print(
        f"Untouched test season: {TEST_SEASON}"
    )
    print(
        "Contract columns excluded: "
        + ", ".join(EXCLUDED_CONTRACT_FEATURES)
    )

    time_folds = make_time_folds(
        training_data["season"].reset_index(drop=True),
        number_of_folds=3,
    )

    parameter_distributions = {
        "n_estimators": [300, 450, 600, 800],
        "learning_rate": [
            0.02,
            0.035,
            0.05,
            0.075,
            0.10,
        ],
        "max_depth": [3, 4, 5, 6, 7],
        "min_child_weight": [1, 3, 5, 8],
        "subsample": [0.70, 0.80, 0.90, 1.00],
        "colsample_bytree": [
            0.65,
            0.75,
            0.85,
            1.00,
        ],
        "reg_alpha": [0.0, 0.01, 0.10, 0.50],
        "reg_lambda": [0.50, 1.0, 2.0, 5.0, 10.0],
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_distributions,
        n_iter=10,
        scoring=euro_mae_scorer,
        cv=time_folds,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    print("\n" + "=" * 72)
    print("RANDOMIZED SEARCH")
    print("=" * 72)
    print("Candidates: 10")
    print("Time-aware folds: 3")
    print("Total model fits: 30")
    print("Optimization metric: MAE in real euros")

    search_start = perf_counter()
    search.fit(X_train, y_train)
    search_seconds = perf_counter() - search_start

    best_parameters = json_ready(search.best_params_)
    best_cv_mae = float(-search.best_score_)

    print("\n" + "=" * 72)
    print("BEST SEARCH RESULT")
    print("=" * 72)
    print(
        f"Cross-validation MAE: {format_euros(best_cv_mae)}"
    )
    print(
        f"Search time: {search_seconds:,.1f} seconds"
    )
    print("Best parameters:")

    for name, value in sorted(best_parameters.items()):
        print(f"  {name}: {value}")

    tuned_model = search.best_estimator_

    test_log_predictions = tuned_model.predict(X_test)
    test_predictions_eur = log_predictions_to_euros(
        test_log_predictions
    )
    actual_test_eur = np.expm1(y_test.to_numpy())

    test_metrics = calculate_metrics(
        actual_test_eur,
        test_predictions_eur,
    )

    print("\n" + "=" * 72)
    print("TUNED MODEL — UNTOUCHED 2025 TEST SEASON")
    print("=" * 72)
    print(
        f"MAE:  {format_euros(test_metrics['mae_eur'])}"
    )
    print(
        f"RMSE: {format_euros(test_metrics['rmse_eur'])}"
    )
    print(
        f"R²:   {test_metrics['r2']:.4f}"
    )
    print(
        "Meaning: on average, predictions miss by approximately "
        f"{format_euros(test_metrics['mae_eur'])}."
    )

    prediction_columns = [
        column
        for column in [
            "player_id",
            "player_name",
            "season",
            "position",
            "league",
            "competition_id",
            "primary_club_id",
            "target_date",
        ]
        if column in test_data.columns
    ]

    test_predictions = test_data[prediction_columns].copy()
    test_predictions["actual_value_eur"] = actual_test_eur
    test_predictions["predicted_value_eur"] = (
        test_predictions_eur
    )
    test_predictions["error_eur"] = (
        test_predictions["predicted_value_eur"]
        - test_predictions["actual_value_eur"]
    )
    test_predictions["absolute_error_eur"] = (
        test_predictions["error_eur"].abs()
    )

    test_predictions.to_csv(
        PROCESSED_DIR / "tuned_test_predictions.csv",
        index=False,
    )

    tuning_results = pd.DataFrame(search.cv_results_).copy()

    tuning_results["mean_validation_mae_eur"] = (
        -tuning_results["mean_test_score"]
    )

    result_columns = [
        "rank_test_score",
        "mean_validation_mae_eur",
        "std_test_score",
        "mean_fit_time",
        "params",
    ]

    tuning_results[result_columns].sort_values(
        "rank_test_score"
    ).to_csv(
        MODELS_DIR / "tuning_results.csv",
        index=False,
    )

    print("\nRetraining the selected configuration on all seasons...")

    X_all, final_defaults = prepare_matrix(
        data,
        feature_columns,
    )
    y_all = data["y"].copy()

    final_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_parameters,
    )

    final_training_start = perf_counter()
    final_model.fit(X_all, y_all)
    final_training_seconds = (
        perf_counter() - final_training_start
    )

    saved_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_type": "XGBRegressor",
        "model_filename": "model.pkl",
        "target": "market_value_in_eur",
        "target_transform": "log1p",
        "inverse_transform": "expm1",
        "training_window": (
            f"{int(data['season'].min())}–"
            f"{int(data['season'].max())}"
        ),
        "evaluation_training_window": (
            f"{int(training_data['season'].min())}–"
            f"{int(training_data['season'].max())}"
        ),
        "test_season": TEST_SEASON,
        "final_training_rows": int(len(data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(feature_columns)),
        "best_parameters": best_parameters,
        "cross_validation_mae_eur": best_cv_mae,
        "evaluation_metrics": test_metrics,
        "search_seconds": float(search_seconds),
        "final_training_seconds": float(
            final_training_seconds
        ),
        "saved_at_utc": saved_at,
        "contract_policy": (
            "Historical contract features excluded because reliable "
            "season-level contract snapshots are unavailable."
        ),
    }

    model_bundle = {
        "model": final_model,
        "feature_columns": feature_columns,
        "feature_defaults": final_defaults,
        "metadata": metadata,
    }

    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(model_bundle, model_path)

    with (MODELS_DIR / "feature_columns.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            feature_columns,
            file,
            indent=2,
        )

    with (MODELS_DIR / "model_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    write_model_card(
        metadata,
        feature_columns,
    )

    # Reload immediately to confirm that the deployment artifact works.
    reloaded_bundle = joblib.load(model_path)

    if reloaded_bundle["feature_columns"] != feature_columns:
        raise RuntimeError(
            "Saved feature order does not match training order."
        )

    sample_prediction_log = reloaded_bundle["model"].predict(
        X_all.iloc[[0]][
            reloaded_bundle["feature_columns"]
        ]
    )

    sample_prediction_eur = float(
        log_predictions_to_euros(
            sample_prediction_log
        )[0]
    )

    if not np.isfinite(sample_prediction_eur):
        raise RuntimeError(
            "Reloaded model produced an invalid prediction."
        )

    print("\n" + "=" * 72)
    print("FINAL MODEL SAVED")
    print("=" * 72)
    print(f"Model: {model_path}")
    print(
        "Feature order: "
        f"{MODELS_DIR / 'feature_columns.json'}"
    )
    print(
        "Metadata: "
        f"{MODELS_DIR / 'model_metadata.json'}"
    )
    print(
        "Model card: "
        f"{MODELS_DIR / 'model_card.md'}"
    )
    print(
        "Tuning results: "
        f"{MODELS_DIR / 'tuning_results.csv'}"
    )
    print(
        f"Final training rows: {len(data):,}"
    )
    print(
        f"Exact feature count: {len(feature_columns)}"
    )
    print(
        f"Reload test prediction: "
        f"{format_euros(sample_prediction_eur)}"
    )
    print(
        f"Model file size: "
        f"{model_path.stat().st_size / 1_000_000:.2f} MB"
    )


if __name__ == "__main__":
    main()
