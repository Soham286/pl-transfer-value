"""Build a deployment-safe catalog of real players and held-out predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tuned_test_predictions.csv"
)

PLAYERS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "players.csv"
)

CLUBS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "clubs.csv"
)

FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_columns.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "player_catalog.csv"
)

LEAGUE_NAMES = {
    "GB1": "Premier League",
    "ES1": "La Liga",
    "L1": "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
    "NL1": "Eredivisie",
    "PO1": "Liga Portugal",
    "BE1": "Belgian Pro League",
    "DK1": "Danish Superliga",
    "GR1": "Greek Super League",
    "RU1": "Russian Premier League",
    "SC1": "Scottish Premiership",
    "TR1": "Süper Lig",
    "UKR1": "Ukrainian Premier League",
}


def require_file(path: Path) -> None:
    """Raise a clear error when a required artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Complete the earlier pipeline first."
        )


def decode_one_hot(
    frame: pd.DataFrame,
    prefix: str,
    fallback: str,
) -> pd.Series:
    """Convert one-hot columns back into readable categories."""
    columns = [
        column
        for column in frame.columns
        if column.startswith(prefix)
    ]

    if not columns:
        return pd.Series(
            fallback,
            index=frame.index,
            dtype="object",
        )

    values = frame[columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)

    maximum_columns = values.idxmax(axis=1)
    maximum_values = values.max(axis=1)

    decoded = maximum_columns.str.removeprefix(prefix)
    decoded = decoded.where(
        maximum_values > 0,
        fallback,
    )

    return decoded


def main() -> None:
    """Create a compact catalog used by the deployed application."""
    required_paths = [
        FEATURES_PATH,
        PREDICTIONS_PATH,
        PLAYERS_PATH,
        CLUBS_PATH,
        FEATURE_COLUMNS_PATH,
    ]

    for path in required_paths:
        require_file(path)

    print("=" * 72)
    print("BUILDING REAL-PLAYER CATALOG")
    print("=" * 72)

    feature_columns = json.loads(
        FEATURE_COLUMNS_PATH.read_text(
            encoding="utf-8",
        )
    )

    features = pd.read_csv(
        FEATURES_PATH,
        low_memory=False,
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
    )

    players = pd.read_csv(
        PLAYERS_PATH,
        low_memory=False,
    )

    clubs = pd.read_csv(
        CLUBS_PATH,
        low_memory=False,
    )

    latest_season = int(
        pd.to_numeric(
            features["season"],
            errors="raise",
        ).max()
    )

    latest_features = features.loc[
        pd.to_numeric(
            features["season"],
            errors="coerce",
        )
        == latest_season
    ].copy()

    print(
        f"Latest feature season: {latest_season}"
    )
    print(
        f"Latest player-season rows: "
        f"{len(latest_features):,}"
    )
    print(
        f"Held-out prediction rows: "
        f"{len(predictions):,}"
    )

    required_prediction_columns = {
        "player_id",
        "actual_value_eur",
        "predicted_value_eur",
    }

    missing_prediction_columns = (
        required_prediction_columns
        .difference(predictions.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction file is missing: "
            + ", ".join(
                sorted(missing_prediction_columns)
            )
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in latest_features.columns
    ]

    if missing_features:
        raise ValueError(
            "Latest features are missing model columns: "
            + ", ".join(missing_features)
        )

    latest_features["player_id"] = pd.to_numeric(
        latest_features["player_id"],
        errors="raise",
    ).astype("int64")

    predictions["player_id"] = pd.to_numeric(
        predictions["player_id"],
        errors="raise",
    ).astype("int64")

    players["player_id"] = pd.to_numeric(
        players["player_id"],
        errors="raise",
    ).astype("int64")

    clubs["club_id"] = pd.to_numeric(
        clubs["club_id"],
        errors="raise",
    ).astype("int64")

    if latest_features.duplicated(
        subset=["player_id"]
    ).any():
        duplicate_count = int(
            latest_features.duplicated(
                subset=["player_id"]
            ).sum()
        )

        raise ValueError(
            "Latest feature data contains "
            f"{duplicate_count:,} duplicate players."
        )

    prediction_join_keys = ["player_id"]

    if (
        "season" in predictions.columns
        and "season" in latest_features.columns
    ):
        predictions["season"] = pd.to_numeric(
            predictions["season"],
            errors="raise",
        ).astype(int)

        latest_features["season"] = pd.to_numeric(
            latest_features["season"],
            errors="raise",
        ).astype(int)

        prediction_join_keys.append("season")

    prediction_columns = (
        prediction_join_keys
        + [
            "actual_value_eur",
            "predicted_value_eur",
        ]
    )

    predictions = predictions[
        prediction_columns
    ].drop_duplicates(
        subset=prediction_join_keys,
        keep="last",
    )

    catalog = latest_features.merge(
        predictions,
        on=prediction_join_keys,
        how="inner",
        validate="one_to_one",
    )

    coverage = len(catalog) / max(
        len(latest_features),
        1,
    )

    print(
        f"Prediction coverage: {coverage:.1%}"
    )

    if coverage < 0.95:
        raise ValueError(
            "Less than 95% of latest players matched held-out "
            "predictions. Check the prediction file."
        )

    player_metadata_columns = [
        column
        for column in [
            "player_id",
            "name",
            "image_url",
            "current_club_name",
            "country_of_citizenship",
            "sub_position",
        ]
        if column in players.columns
    ]

    player_metadata = players[
        player_metadata_columns
    ].copy()

    player_metadata = player_metadata.rename(
        columns={
            "name": "player_name",
        }
    )

    catalog = catalog.merge(
        player_metadata,
        on="player_id",
        how="left",
        validate="one_to_one",
    )

    primary_club_id_column = next(
        (
            column
            for column in [
                "primary_club_id",
                "player_club_id",
                "club_id",
            ]
            if column in catalog.columns
        ),
        None,
    )

    if primary_club_id_column is not None:
        club_names = clubs[
            [
                "club_id",
                "name",
            ]
        ].rename(
            columns={
                "club_id": primary_club_id_column,
                "name": "season_club_name",
            }
        )

        club_names[
            primary_club_id_column
        ] = pd.to_numeric(
            club_names[primary_club_id_column],
            errors="coerce",
        )

        catalog[
            primary_club_id_column
        ] = pd.to_numeric(
            catalog[primary_club_id_column],
            errors="coerce",
        )

        catalog = catalog.merge(
            club_names,
            on=primary_club_id_column,
            how="left",
            validate="many_to_one",
        )
    else:
        catalog["season_club_name"] = np.nan

    catalog["player_name"] = (
        catalog["player_name"]
        .fillna(
            catalog["player_id"].map(
                lambda value: f"Player {value}"
            )
        )
        .astype(str)
        .str.strip()
    )

    catalog["club_name"] = (
        catalog["season_club_name"]
        .fillna(
            catalog.get(
                "current_club_name",
                pd.Series(
                    index=catalog.index,
                    dtype="object",
                ),
            )
        )
        .fillna("Unknown club")
        .astype(str)
        .str.strip()
    )

    if "league" in catalog.columns:
        catalog["league_code"] = (
            catalog["league"]
            .fillna("Unknown")
            .astype(str)
        )
    elif "competition_id" in catalog.columns:
        catalog["league_code"] = (
            catalog["competition_id"]
            .fillna("Unknown")
            .astype(str)
        )
    else:
        catalog["league_code"] = decode_one_hot(
            catalog,
            prefix="league_",
            fallback="Unknown",
        )

    catalog["league_name"] = (
        catalog["league_code"]
        .map(LEAGUE_NAMES)
        .fillna(catalog["league_code"])
    )

    if "position" not in catalog.columns:
        catalog["position"] = decode_one_hot(
            catalog,
            prefix="position_",
            fallback="Unknown",
        )

    if "club_tier" not in catalog.columns:
        catalog["club_tier"] = decode_one_hot(
            catalog,
            prefix="club_tier_",
            fallback="Unknown",
        )

    if "image_url" not in catalog.columns:
        catalog["image_url"] = ""

    catalog["image_url"] = (
        catalog["image_url"]
        .fillna("")
        .astype(str)
    )

    catalog["actual_value_eur"] = pd.to_numeric(
        catalog["actual_value_eur"],
        errors="raise",
    )

    catalog["predicted_value_eur"] = pd.to_numeric(
        catalog["predicted_value_eur"],
        errors="raise",
    )

    catalog["valuation_gap_eur"] = (
        catalog["predicted_value_eur"]
        - catalog["actual_value_eur"]
    )

    catalog["valuation_gap_pct"] = np.where(
        catalog["actual_value_eur"] > 0,
        (
            catalog["valuation_gap_eur"]
            / catalog["actual_value_eur"]
        )
        * 100,
        np.nan,
    )

    catalog["prediction_source"] = (
        f"Held-out {latest_season} prediction; "
        f"model trained on seasons before {latest_season}"
    )

    catalog["search_label"] = (
        catalog["player_name"]
        + " · "
        + catalog["club_name"]
        + " · "
        + catalog["position"].astype(str)
        + " · #"
        + catalog["player_id"].astype(str)
    )

    for column in feature_columns:
        catalog[column] = pd.to_numeric(
            catalog[column],
            errors="coerce",
        )

    missing_model_values = int(
        catalog[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    infinite_model_values = int(
        np.isinf(
            catalog[feature_columns].to_numpy(
                dtype=float
            )
        ).sum()
    )

    if missing_model_values:
        raise ValueError(
            "Catalog contains "
            f"{missing_model_values:,} missing model values."
        )

    if infinite_model_values:
        raise ValueError(
            "Catalog contains "
            f"{infinite_model_values:,} infinite model values."
        )

    preferred_columns = [
        "player_id",
        "search_label",
        "player_name",
        "image_url",
        "club_name",
        "position",
        "sub_position",
        "country_of_citizenship",
        "league_code",
        "league_name",
        "season",
        "club_tier",
        "age",
        "goals",
        "assists",
        "minutes_played",
        "appearances",
        "actual_value_eur",
        "predicted_value_eur",
        "valuation_gap_eur",
        "valuation_gap_pct",
        "prediction_source",
    ]

    output_columns = []

    for column in preferred_columns + feature_columns:
        if (
            column in catalog.columns
            and column not in output_columns
        ):
            output_columns.append(column)

    catalog = catalog[
        output_columns
    ].sort_values(
        by=[
            "player_name",
            "player_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    missing_images = int(
        catalog["image_url"]
        .str.strip()
        .eq("")
        .sum()
    )

    print("\n" + "=" * 72)
    print("PLAYER CATALOG CREATED")
    print("=" * 72)
    print(f"Rows: {len(catalog):,}")
    print(f"Columns: {len(catalog.columns):,}")
    print(f"Duplicate players: {catalog['player_id'].duplicated().sum():,}")
    print(f"Missing player images: {missing_images:,}")
    print(
        "Mean absolute held-out error: "
        f"€{catalog['valuation_gap_eur'].abs().mean() / 1_000_000:,.2f}M"
    )
    print(
        "Potentially undervalued players: "
        f"{(catalog['valuation_gap_eur'] > 0).sum():,}"
    )
    print(f"Output: {OUTPUT_PATH}")
    print(
        "File size: "
        f"{OUTPUT_PATH.stat().st_size / 1_000_000:,.2f} MB"
    )

    print("\nSample players:")
    print(
        catalog[
            [
                "player_name",
                "club_name",
                "position",
                "actual_value_eur",
                "predicted_value_eur",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
