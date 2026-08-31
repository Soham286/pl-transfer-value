"""Create leakage-aware player-season modelling features.

The output contains one row per player per season and is written to
data/processed/features.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DATA_DIR / "features.csv"

MIN_MINUTES = 300
TARGET_TOLERANCE_DAYS = 120
CONTRACT_SNAPSHOT_TOLERANCE_DAYS = 120

VALID_POSITIONS = [
    "Attack",
    "Midfield",
    "Defender",
    "Goalkeeper",
]


def load_inputs() -> dict[str, pd.DataFrame]:
    """Load only the columns required by feature engineering."""

    print("Loading raw data...")

    players = pd.read_csv(
        RAW_DATA_DIR / "players.csv",
        usecols=[
            "player_id",
            "name",
            "date_of_birth",
            "position",
            "contract_expiration_date",
        ],
        parse_dates=[
            "date_of_birth",
            "contract_expiration_date",
        ],
        low_memory=False,
    )

    appearances = pd.read_csv(
        RAW_DATA_DIR / "appearances.csv",
        usecols=[
            "game_id",
            "player_id",
            "player_club_id",
            "competition_id",
            "goals",
            "assists",
            "minutes_played",
        ],
        low_memory=False,
    )

    games = pd.read_csv(
        RAW_DATA_DIR / "games.csv",
        usecols=[
            "game_id",
            "competition_id",
            "season",
            "date",
            "home_club_id",
            "away_club_id",
            "competition_type",
        ],
        parse_dates=["date"],
        low_memory=False,
    )

    competitions = pd.read_csv(
        RAW_DATA_DIR / "competitions.csv",
        usecols=[
            "competition_id",
            "name",
            "type",
            "sub_type",
        ],
        low_memory=False,
    )

    valuations = pd.read_csv(
        RAW_DATA_DIR / "player_valuations.csv",
        usecols=[
            "player_id",
            "date",
            "market_value_in_eur",
        ],
        parse_dates=["date"],
        low_memory=False,
    )

    games["season"] = pd.to_numeric(
        games["season"],
        errors="coerce",
    )

    games = games.dropna(
        subset=[
            "season",
            "date",
            "competition_id",
        ]
    ).copy()

    games["season"] = games["season"].astype(int)

    print(f"  Players: {len(players):,}")
    print(f"  Appearances: {len(appearances):,}")
    print(f"  Games: {len(games):,}")
    print(f"  Competitions: {len(competitions):,}")
    print(f"  Valuations: {len(valuations):,}")

    return {
        "players": players,
        "appearances": appearances,
        "games": games,
        "competitions": competitions,
        "valuations": valuations,
    }


def find_domestic_leagues(
    competitions: pd.DataFrame,
    games: pd.DataFrame,
) -> set[str]:
    """Return competition IDs classified as domestic leagues."""

    competition_type = (
        competitions["type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    domestic_ids = set(
        competitions.loc[
            competition_type == "domestic_league",
            "competition_id",
        ]
    )

    # Fall back to games.csv if the competition table uses
    # a different label in a future dataset version.
    if not domestic_ids:
        game_type = (
            games["competition_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        domestic_ids = set(
            games.loc[
                game_type == "domestic_league",
                "competition_id",
            ]
        )

    if not domestic_ids:
        raise ValueError(
            "No domestic-league competitions were found."
        )

    print(
        "Domestic leagues found: "
        f"{len(domestic_ids):,}"
    )

    return domestic_ids


def build_player_seasons(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    domestic_ids: set[str],
) -> pd.DataFrame:
    """Aggregate match appearances into player-season rows."""

    games_for_merge = games[
        [
            "game_id",
            "competition_id",
            "season",
            "date",
        ]
    ].rename(
        columns={
            "competition_id":
                "game_competition_id",
        }
    )

    match_rows = appearances.merge(
        games_for_merge,
        on="game_id",
        how="inner",
        validate="many_to_one",
    )

    competition_mismatches = (
        match_rows["competition_id"]
        != match_rows["game_competition_id"]
    ).sum()

    print(
        "Appearance/game competition mismatches: "
        f"{competition_mismatches:,}"
    )

    # The competition recorded in games.csv is authoritative.
    match_rows = (
        match_rows
        .drop(columns=["competition_id"])
        .rename(
            columns={
                "game_competition_id":
                    "competition_id",
            }
        )
    )

    domestic_rows = match_rows[
        match_rows["competition_id"].isin(
            domestic_ids
        )
    ].copy()

    if domestic_rows.empty:
        raise ValueError(
            "No domestic-league appearances remain "
            "after filtering."
        )

    print(
        "Domestic-league appearance rows: "
        f"{len(domestic_rows):,}"
    )

    # Aggregate all domestic-league performance for one player
    # during one season.
    player_seasons = (
        domestic_rows
        .groupby(
            ["player_id", "season"],
            as_index=False,
        )
        .agg(
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            minutes_played=(
                "minutes_played",
                "sum",
            ),
            appearances=(
                "game_id",
                "nunique",
            ),
            first_appearance_date=(
                "date",
                "min",
            ),
            last_appearance_date=(
                "date",
                "max",
            ),
        )
    )

    # If a player changed clubs or leagues during a season,
    # use the club/competition pair where they played the
    # greatest number of minutes as their primary context.
    primary_context = (
        domestic_rows
        .groupby(
            [
                "player_id",
                "season",
                "competition_id",
                "player_club_id",
            ],
            as_index=False,
        )
        .agg(
            context_minutes=(
                "minutes_played",
                "sum",
            ),
            context_appearances=(
                "game_id",
                "nunique",
            ),
        )
        .sort_values(
            [
                "player_id",
                "season",
                "context_minutes",
                "context_appearances",
                "competition_id",
                "player_club_id",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                True,
                True,
            ],
        )
        .drop_duplicates(
            subset=[
                "player_id",
                "season",
            ],
            keep="first",
        )
        .rename(
            columns={
                "competition_id":
                    "primary_competition_id",
                "player_club_id":
                    "primary_club_id",
            }
        )
    )

    player_seasons = player_seasons.merge(
        primary_context[
            [
                "player_id",
                "season",
                "primary_competition_id",
                "primary_club_id",
            ]
        ],
        on=[
            "player_id",
            "season",
        ],
        how="inner",
        validate="one_to_one",
    )

    return player_seasons


def add_season_context(
    player_seasons: pd.DataFrame,
    games: pd.DataFrame,
    domestic_ids: set[str],
) -> pd.DataFrame:
    """Add competition dates and available season minutes."""

    domestic_games = games[
        games["competition_id"].isin(
            domestic_ids
        )
    ].copy()

    season_calendar = (
        domestic_games
        .groupby(
            [
                "season",
                "competition_id",
            ],
            as_index=False,
        )
        .agg(
            season_start_date=("date", "min"),
            season_end_date=("date", "max"),
        )
        .rename(
            columns={
                "competition_id":
                    "primary_competition_id",
            }
        )
    )

    # Count games played by each club in each season.
    home_clubs = (
        domestic_games[
            [
                "game_id",
                "season",
                "competition_id",
                "home_club_id",
            ]
        ]
        .rename(
            columns={
                "home_club_id": "club_id",
            }
        )
    )

    away_clubs = (
        domestic_games[
            [
                "game_id",
                "season",
                "competition_id",
                "away_club_id",
            ]
        ]
        .rename(
            columns={
                "away_club_id": "club_id",
            }
        )
    )

    club_games = (
        pd.concat(
            [
                home_clubs,
                away_clubs,
            ],
            ignore_index=True,
        )
        .groupby(
            [
                "season",
                "competition_id",
                "club_id",
            ],
            as_index=False,
        )
        .agg(
            club_games=("game_id", "nunique")
        )
    )

    # This handles leagues with different numbers of matches.
    competition_schedule = (
        club_games
        .groupby(
            [
                "season",
                "competition_id",
            ],
            as_index=False,
        )
        .agg(
            maximum_club_games=(
                "club_games",
                "max",
            )
        )
        .rename(
            columns={
                "competition_id":
                    "primary_competition_id",
            }
        )
    )

    rows = player_seasons.merge(
        season_calendar,
        on=[
            "season",
            "primary_competition_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    rows = rows.merge(
        competition_schedule,
        on=[
            "season",
            "primary_competition_id",
        ],
        how="left",
        validate="many_to_one",
    )

    print(
        "Player-seasons before minutes filter: "
        f"{len(rows):,}"
    )

    rows = rows[
        rows["minutes_played"] >= MIN_MINUTES
    ].copy()

    print(
        f"Player-seasons with at least "
        f"{MIN_MINUTES} minutes: "
        f"{len(rows):,}"
    )

    rows["goals_per_90"] = (
        rows["goals"]
        * 90
        / rows["minutes_played"]
    )

    rows["assists_per_90"] = (
        rows["assists"]
        * 90
        / rows["minutes_played"]
    )

    rows["goal_contributions_per_90"] = (
        (
            rows["goals"]
            + rows["assists"]
        )
        * 90
        / rows["minutes_played"]
    )

    available_minutes = (
        rows["maximum_club_games"]
        * 90
    )

    raw_minutes_share = (
        rows["minutes_played"]
        / available_minutes
    )

    clipped_rows = (
        raw_minutes_share > 1
    ).sum()

    # Transfers can occasionally push the total above the
    # schedule of the primary league, so cap the share at 1.
    rows["minutes_share"] = (
        raw_minutes_share.clip(
            lower=0,
            upper=1,
        )
    )

    print(
        "Minutes-share rows capped at 1.0: "
        f"{clipped_rows:,}"
    )

    return rows


def align_targets(
    player_seasons: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the first valuation after each season ends."""

    targets = (
        valuations[
            [
                "player_id",
                "date",
                "market_value_in_eur",
            ]
        ]
        .dropna(
            subset=[
                "player_id",
                "date",
                "market_value_in_eur",
            ]
        )
        .rename(
            columns={
                "date": "target_valuation_date",
                "market_value_in_eur":
                    "target_market_value_eur",
            }
        )
        .drop_duplicates(
            subset=[
                "player_id",
                "target_valuation_date",
            ],
            keep="last",
        )
    )

    # merge_asof requires the date key itself to be globally
    # sorted, even when matching within player_id.
    left = player_seasons.sort_values(
        [
            "season_end_date",
            "player_id",
        ]
    )

    right = targets.sort_values(
        [
            "target_valuation_date",
            "player_id",
        ]
    )

    aligned = pd.merge_asof(
        left,
        right,
        left_on="season_end_date",
        right_on="target_valuation_date",
        by="player_id",
        direction="forward",
        tolerance=pd.Timedelta(
            days=TARGET_TOLERANCE_DAYS
        ),
        allow_exact_matches=True,
    )

    before_target_filter = len(aligned)

    aligned = aligned.dropna(
        subset=[
            "target_valuation_date",
            "target_market_value_eur",
        ]
    ).copy()

    aligned = aligned[
        aligned["target_market_value_eur"] > 0
    ].copy()

    print(
        "Rows dropped without a timely "
        "post-season target: "
        f"{before_target_filter - len(aligned):,}"
    )

    if (
        aligned["target_valuation_date"]
        < aligned["season_end_date"]
    ).any():
        raise ValueError(
            "A target valuation was recorded "
            "before the season ended."
        )

    return aligned


def add_player_features(
    rows: pd.DataFrame,
    players: pd.DataFrame,
    valuations: pd.DataFrame,
) -> pd.DataFrame:
    """Add age, position, prime-age, and safe contract data."""

    rows = rows.merge(
        players,
        on="player_id",
        how="left",
        validate="many_to_one",
    )

    before_player_filter = len(rows)

    rows = rows[
        rows["position"].isin(
            VALID_POSITIONS
        )
    ].copy()

    rows["age"] = (
        (
            rows["season_start_date"]
            - rows["date_of_birth"]
        ).dt.days
        / 365.25
    )

    rows = rows[
        rows["age"].between(15, 45)
    ].copy()

    print(
        "Rows dropped for missing/invalid "
        "position or age: "
        f"{before_player_filter - len(rows):,}"
    )

    rows["age_squared"] = (
        rows["age"] ** 2
    )

    rows["is_prime_age"] = (
        rows["age"].between(
            23,
            28,
            inclusive="both",
        )
    ).astype("int8")

    # players.csv contains only current contract information.
    # Use it solely for targets close to the current snapshot.
    contract_snapshot_date = (
        valuations["date"].max()
    )

    contract_distance_days = (
        rows["target_valuation_date"]
        - contract_snapshot_date
    ).abs().dt.days

    raw_contract_years = (
        (
            rows["contract_expiration_date"]
            - rows["target_valuation_date"]
        ).dt.days
        / 365.25
    )

    contract_available = (
        rows["contract_expiration_date"].notna()
        & (
            contract_distance_days
            <= CONTRACT_SNAPSHOT_TOLERANCE_DAYS
        )
        & (raw_contract_years >= 0)
    )

    rows["contract_years_remaining"] = (
        raw_contract_years.where(
            contract_available
        )
    )

    rows["contract_info_available"] = (
        contract_available.astype("int8")
    )

    print(
        "Player-seasons with leakage-safe "
        "contract information: "
        f"{contract_available.sum():,}"
    )

    return rows


def add_club_context(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Derive club strength from the previous season."""

    # Aggregate player targets into squad values.
    club_season_values = (
        rows
        .groupby(
            [
                "season",
                "primary_club_id",
            ],
            as_index=False,
        )
        .agg(
            squad_market_value_eur=(
                "target_market_value_eur",
                "sum",
            )
        )
    )

    # Shift the club values forward by one season so each row
    # uses only information from the past.
    previous_club_values = (
        club_season_values.copy()
    )

    previous_club_values["season"] = (
        previous_club_values["season"] + 1
    )

    previous_club_values = (
        previous_club_values.rename(
            columns={
                "squad_market_value_eur":
                    "prior_squad_value_eur",
            }
        )
    )

    rows = rows.merge(
        previous_club_values,
        on=[
            "season",
            "primary_club_id",
        ],
        how="left",
        validate="many_to_one",
    )

    rows["club_strength_available"] = (
        rows["prior_squad_value_eur"]
        .notna()
        .astype("int8")
    )

    # Rank unique clubs within each competition-season.
    club_table = (
        rows[
            [
                "season",
                "primary_competition_id",
                "primary_club_id",
                "prior_squad_value_eur",
            ]
        ]
        .drop_duplicates(
            subset=[
                "season",
                "primary_competition_id",
                "primary_club_id",
            ]
        )
        .copy()
    )

    club_table["club_value_percentile"] = (
        club_table
        .groupby(
            [
                "season",
                "primary_competition_id",
            ]
        )["prior_squad_value_eur"]
        .rank(
            method="average",
            pct=True,
        )
    )

    club_table["club_tier"] = "Unknown"

    has_prior_value = (
        club_table[
            "club_value_percentile"
        ].notna()
    )

    club_table.loc[
        has_prior_value,
        "club_tier",
    ] = pd.cut(
        club_table.loc[
            has_prior_value,
            "club_value_percentile",
        ],
        bins=[
            0,
            0.25,
            0.50,
            0.75,
            1.00,
        ],
        labels=[
            "Lower",
            "Middle",
            "Upper",
            "Elite",
        ],
        include_lowest=True,
    ).astype("string")

    rows = rows.merge(
        club_table[
            [
                "season",
                "primary_competition_id",
                "primary_club_id",
                "club_tier",
            ]
        ],
        on=[
            "season",
            "primary_competition_id",
            "primary_club_id",
        ],
        how="left",
        validate="many_to_one",
    )

    rows["club_tier"] = (
        rows["club_tier"]
        .fillna("Unknown")
    )

    # Impute missing prior-club values with the median for the
    # same competition and season, then the global median.
    competition_median = (
        rows
        .groupby(
            [
                "season",
                "primary_competition_id",
            ]
        )["prior_squad_value_eur"]
        .transform("median")
    )

    global_median = (
        rows["prior_squad_value_eur"].median()
    )

    if pd.isna(global_median):
        global_median = 0

    filled_club_value = (
        rows["prior_squad_value_eur"]
        .fillna(competition_median)
        .fillna(global_median)
    )

    rows["club_strength_log"] = np.log1p(
        filled_club_value
    )

    return rows


def build_output(
    rows: pd.DataFrame,
    competitions: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categories and select final columns."""

    competition_lookup = (
        competitions[
            [
                "competition_id",
                "name",
            ]
        ]
        .drop_duplicates("competition_id")
        .rename(
            columns={
                "competition_id":
                    "primary_competition_id",
                "name": "competition_name",
            }
        )
    )

    rows = rows.merge(
        competition_lookup,
        on="primary_competition_id",
        how="left",
        validate="many_to_one",
    )

    rows["competition_name"] = (
        rows["competition_name"]
        .fillna(
            rows["primary_competition_id"]
        )
    )

    rows["y"] = np.log1p(
        rows["target_market_value_eur"]
    )

    position_features = pd.get_dummies(
        rows["position"],
        prefix="position",
        dtype="int8",
    )

    league_features = pd.get_dummies(
        rows[
            "primary_competition_id"
        ].astype(str),
        prefix="league",
        dtype="int8",
    )

    club_tier_features = pd.get_dummies(
        rows["club_tier"],
        prefix="club_tier",
        dtype="int8",
    )

    numeric_features = [
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
        "contract_years_remaining",
        "contract_info_available",
        "is_prime_age",
        "club_strength_log",
        "club_strength_available",
    ]

    metadata_columns = [
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
    ]

    base_output = rows[
        metadata_columns
        + numeric_features
    ].reset_index(drop=True)

    output = pd.concat(
        [
            base_output,
            position_features.reset_index(
                drop=True
            ),
            league_features.reset_index(
                drop=True
            ),
            club_tier_features.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    output = output.sort_values(
        [
            "season",
            "player_id",
        ]
    ).reset_index(drop=True)

    if output.duplicated(
        subset=[
            "player_id",
            "season",
        ]
    ).any():
        raise ValueError(
            "Duplicate player-season rows detected."
        )

    one_hot_features = (
        position_features.columns.tolist()
        + league_features.columns.tolist()
        + club_tier_features.columns.tolist()
    )

    feature_columns = (
        numeric_features
        + one_hot_features
    )

    return output, feature_columns


def main() -> None:
    """Run the complete feature-engineering pipeline."""

    data = load_inputs()

    domestic_ids = find_domestic_leagues(
        data["competitions"],
        data["games"],
    )

    player_seasons = build_player_seasons(
        data["appearances"],
        data["games"],
        domestic_ids,
    )

    player_seasons = add_season_context(
        player_seasons,
        data["games"],
        domestic_ids,
    )

    player_seasons = align_targets(
        player_seasons,
        data["valuations"],
    )

    player_seasons = add_player_features(
        player_seasons,
        data["players"],
        data["valuations"],
    )

    player_seasons = add_club_context(
        player_seasons
    )

    output, feature_columns = build_output(
        player_seasons,
        data["competitions"],
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 72)
    print(
        f"Final shape: "
        f"{output.shape[0]:,} rows x "
        f"{output.shape[1]:,} columns"
    )
    print(f"Output: {OUTPUT_PATH}")

    print(
        "\nSeason range: "
        f"{output['season'].min()} "
        f"to {output['season'].max()}"
    )

    print(
        "Missing contract feature values: "
        f"{output['contract_years_remaining'].isna().sum():,}"
    )

    print(
        "\nModel feature columns "
        f"({len(feature_columns)}):"
    )

    for number, feature in enumerate(
        feature_columns,
        start=1,
    ):
        print(f"  {number:>3}. {feature}")


if __name__ == "__main__":
    main()
