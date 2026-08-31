"""Train and compare baseline market-value regression models.

Models train on log1p(market value), but all reported metrics are
calculated in real euros after applying expm1.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)
MODELS_DIR = PROJECT_ROOT / "models"
COMPARISON_PATH = (
    MODELS_DIR / "model_comparison.csv"
)
PLOT_PATH = (
    MODELS_DIR
    / "predicted_vs_actual.html"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_predictions.csv"
)

RANDOM_STATE = 42
ACCENT = "#7C3AED"

# These columns identify rows or contain the target.
# They must never be passed to the model.
METADATA_COLUMNS = {
    "player_id",
    "name",
    "season",
    "season_start_date",
    "season_end_date",
    "target_valuation_date",
    "position",
    "primary_competition_id",
    "competition_name",
    "primary_club_id",
    "club_tier",
    "target_market_value_eur",
    "y",
}

# Current contract data is available only near the latest
# snapshot. Earlier seasons do not contain comparable values.
# Including these in the historical comparison would create an
# unfair train/test mismatch.
CONTRACT_COLUMNS_EXCLUDED_FROM_PRIMARY_MODEL = {
    "contract_years_remaining",
    "contract_info_available",
}


def format_euros(value: float) -> str:
    """Format a euro value for readable terminal output."""

    if value >= 1_000_000_000:
        return f"€{value / 1_000_000_000:,.2f}B"

    if value >= 1_000_000:
        return f"€{value / 1_000_000:,.2f}M"

    if value >= 1_000:
        return f"€{value / 1_000:,.1f}K"

    return f"€{value:,.0f}"


def load_features() -> tuple[
    pd.DataFrame,
    list[str],
]:
    """Load the engineered player-season dataset."""

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. "
            "Run python src/features.py first."
        )

    features = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
    )

    excluded_columns = (
        METADATA_COLUMNS
        | CONTRACT_COLUMNS_EXCLUDED_FROM_PRIMARY_MODEL
    )

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded_columns
    ]

    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            features[column]
        )
    ]

    if non_numeric_features:
        raise TypeError(
            "Non-numeric model features detected: "
            f"{non_numeric_features}"
        )

    missing_values = (
        features[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_values:
        raise ValueError(
            "Unexpected missing values in the "
            f"primary model matrix: {missing_values:,}"
        )

    return features, feature_columns


def make_models() -> list[
    tuple[str, object]
]:
    """Return models in the requested comparison order."""

    linear_model = Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    random_forest = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_leaf=2,
        max_features=0.80,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    xgboost_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    return [
        (
            "Linear Regression",
            linear_model,
        ),
        (
            "Random Forest",
            random_forest,
        ),
        (
            "XGBoost",
            xgboost_model,
        ),
    ]


def evaluate_model(
    model_name: str,
    model: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    actual_euros: np.ndarray,
) -> tuple[
    dict[str, float | str],
    np.ndarray,
]:
    """Fit one model and evaluate it in real euros."""

    print("\n" + "-" * 72)
    print(f"Training {model_name}")
    print("-" * 72)

    start_time = time.perf_counter()

    model.fit(
        x_train,
        y_train,
    )

    predicted_log_values = model.predict(
        x_test
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    # Convert predictions from log space back into euros.
    predicted_euros = np.expm1(
        predicted_log_values
    )

    # A market value cannot be negative.
    predicted_euros = np.maximum(
        predicted_euros,
        0,
    )

    mae = mean_absolute_error(
        actual_euros,
        predicted_euros,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_euros,
            predicted_euros,
        )
    )

    r_squared = r2_score(
        actual_euros,
        predicted_euros,
    )

    print(
        f"MAE:  {format_euros(mae)}"
    )
    print(
        f"RMSE: {format_euros(rmse)}"
    )
    print(
        f"R²:   {r_squared:.4f}"
    )
    print(
        "Training time: "
        f"{training_seconds:.1f} seconds"
    )

    result = {
        "Model": model_name,
        "MAE_EUR": mae,
        "RMSE_EUR": rmse,
        "R2": r_squared,
        "Training_seconds":
            training_seconds,
    }

    return result, predicted_euros


def save_comparison(
    results: list[dict[str, float | str]],
) -> pd.DataFrame:
    """Print and save the model comparison table."""

    comparison = pd.DataFrame(results)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    printable = comparison.copy()

    printable["MAE"] = printable[
        "MAE_EUR"
    ].map(format_euros)

    printable["RMSE"] = printable[
        "RMSE_EUR"
    ].map(format_euros)

    printable["R²"] = printable[
        "R2"
    ].map(lambda value: f"{value:.4f}")

    printable["Time"] = printable[
        "Training_seconds"
    ].map(
        lambda value: f"{value:.1f}s"
    )

    printable = printable[
        [
            "Model",
            "MAE",
            "RMSE",
            "R²",
            "Time",
        ]
    ]

    print("\n" + "=" * 72)
    print("MODEL COMPARISON — METRICS IN REAL EUROS")
    print("=" * 72)
    print(
        printable.to_string(
            index=False
        )
    )

    return comparison


def save_predictions(
    test_metadata: pd.DataFrame,
    actual_euros: np.ndarray,
    predicted_euros: np.ndarray,
    winner_name: str,
) -> None:
    """Save test predictions for later bargain analysis."""

    predictions = (
        test_metadata.copy()
        .reset_index(drop=True)
    )

    predictions[
        "actual_market_value_eur"
    ] = actual_euros

    predictions[
        "predicted_market_value_eur"
    ] = predicted_euros

    predictions["prediction_gap_eur"] = (
        predicted_euros - actual_euros
    )

    predictions[
        "absolute_error_eur"
    ] = np.abs(
        predictions[
            "prediction_gap_eur"
        ]
    )

    predictions["model"] = winner_name

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


def create_scatter_plot(
    test_metadata: pd.DataFrame,
    actual_euros: np.ndarray,
    predicted_euros: np.ndarray,
    winner_name: str,
    test_season: int,
) -> None:
    """Save a predicted-versus-actual Plotly chart."""

    actual_millions = np.maximum(
        actual_euros / 1_000_000,
        0.01,
    )

    predicted_millions = np.maximum(
        predicted_euros / 1_000_000,
        0.01,
    )

    axis_min = max(
        min(
            actual_millions.min(),
            predicted_millions.min(),
        )
        * 0.80,
        0.01,
    )

    axis_max = max(
        actual_millions.max(),
        predicted_millions.max(),
    ) * 1.20

    customdata = np.column_stack(
        [
            test_metadata[
                "name"
            ].fillna("Unknown"),
            test_metadata[
                "position"
            ].fillna("Unknown"),
            test_metadata[
                "competition_name"
            ].fillna("Unknown"),
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scattergl(
            x=actual_millions,
            y=predicted_millions,
            mode="markers",
            name="Players",
            marker={
                "color": ACCENT,
                "size": 7,
                "opacity": 0.52,
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Position: "
                "%{customdata[1]}"
                "<br>Competition: "
                "%{customdata[2]}"
                "<br>Actual: €%{x:.2f}M"
                "<br>Predicted: €%{y:.2f}M"
                "<extra></extra>"
            ),
        )
    )

    # A point on this diagonal has a perfect prediction.
    fig.add_trace(
        go.Scatter(
            x=[
                axis_min,
                axis_max,
            ],
            y=[
                axis_min,
                axis_max,
            ],
            mode="lines",
            name="Perfect prediction",
            line={
                "color": "#A3A3A3",
                "width": 2,
                "dash": "dash",
            },
            hoverinfo="skip",
        )
    )

    log_range = [
        np.log10(axis_min),
        np.log10(axis_max),
    ]

    fig.update_layout(
        template="plotly_dark",
        title={
            "text": (
                f"{winner_name}: predicted vs "
                f"actual values — season "
                f"{test_season}"
            ),
            "x": 0.02,
        },
        xaxis_title=(
            "Actual market value "
            "(€ millions, log scale)"
        ),
        yaxis_title=(
            "Predicted market value "
            "(€ millions, log scale)"
        ),
        height=700,
        legend_title_text="",
        hovermode="closest",
    )

    fig.update_xaxes(
        type="log",
        range=log_range,
    )

    fig.update_yaxes(
        type="log",
        range=log_range,
        scaleanchor="x",
        scaleratio=1,
    )

    fig.write_html(
        PLOT_PATH,
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )


def main() -> None:
    """Train, compare, and report all three models."""

    features, feature_columns = (
        load_features()
    )

    test_season = int(
        features["season"].max()
    )

    train_mask = (
        features["season"] < test_season
    )

    test_mask = (
        features["season"] == test_season
    )

    train_data = features.loc[
        train_mask
    ].copy()

    test_data = features.loc[
        test_mask
    ].copy()

    if train_data.empty or test_data.empty:
        raise ValueError(
            "The time-based split produced "
            "an empty train or test set."
        )

    x_train = train_data[
        feature_columns
    ].astype("float32")

    x_test = test_data[
        feature_columns
    ].astype("float32")

    y_train = train_data[
        "y"
    ].astype("float64")

    actual_euros = test_data[
        "target_market_value_eur"
    ].to_numpy(dtype="float64")

    print("=" * 72)
    print("TIME-BASED TRAIN/TEST SPLIT")
    print("=" * 72)
    print(
        "Training seasons: "
        f"{train_data['season'].min()} "
        f"to {train_data['season'].max()}"
    )
    print(
        f"Test season: {test_season}"
    )
    print(
        f"Training rows: {len(train_data):,}"
    )
    print(
        f"Test rows: {len(test_data):,}"
    )
    print(
        f"Model features: "
        f"{len(feature_columns):,}"
    )
    print(
        "Test median market value: "
        f"{format_euros(np.median(actual_euros))}"
    )
    print(
        "Test mean market value: "
        f"{format_euros(np.mean(actual_euros))}"
    )
    print(
        "\nContract columns excluded from "
        "the primary historical comparison:"
    )

    for column in sorted(
        CONTRACT_COLUMNS_EXCLUDED_FROM_PRIMARY_MODEL
    ):
        print(f"  - {column}")

    results = []
    predictions_by_model = {}

    for model_name, model in make_models():
        result, predicted_euros = (
            evaluate_model(
                model_name=model_name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                actual_euros=actual_euros,
            )
        )

        results.append(result)
        predictions_by_model[
            model_name
        ] = predicted_euros

        del model
        gc.collect()

    comparison = save_comparison(
        results
    )

    winner_row = comparison.loc[
        comparison["MAE_EUR"].idxmin()
    ]

    winner_name = str(
        winner_row["Model"]
    )

    winner_predictions = (
        predictions_by_model[
            winner_name
        ]
    )

    print("\n" + "=" * 72)
    print("WINNER BEFORE TUNING")
    print("=" * 72)
    print(f"Model: {winner_name}")
    print(
        "Headline MAE: "
        f"{format_euros(winner_row['MAE_EUR'])}"
    )
    print(
        "Meaning: on average, predictions "
        "miss the test value by approximately "
        f"{format_euros(winner_row['MAE_EUR'])}."
    )

    test_metadata = test_data[
        [
            "player_id",
            "name",
            "season",
            "position",
            "primary_competition_id",
            "competition_name",
            "primary_club_id",
            "target_valuation_date",
        ]
    ].copy()

    save_predictions(
        test_metadata=test_metadata,
        actual_euros=actual_euros,
        predicted_euros=winner_predictions,
        winner_name=winner_name,
    )

    create_scatter_plot(
        test_metadata=test_metadata,
        actual_euros=actual_euros,
        predicted_euros=winner_predictions,
        winner_name=winner_name,
        test_season=test_season,
    )

    print(
        "\nNumeric comparison saved to:"
    )
    print(f"  {COMPARISON_PATH}")

    print(
        "Test predictions saved to:"
    )
    print(f"  {PREDICTIONS_PATH}")

    print(
        "Interactive scatter plot saved to:"
    )
    print(f"  {PLOT_PATH}")


if __name__ == "__main__":
    main()
