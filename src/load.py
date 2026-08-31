from pathlib import Path

import pandas as pd


# Find the project root regardless of where the script is executed from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "players": "players.csv",
    "appearances": "appearances.csv",
    "clubs": "clubs.csv",
    "player_valuations": "player_valuations.csv",
}

DATE_COLUMNS = {
    "players": ["date_of_birth", "contract_expiration_date"],
    "appearances": ["date"],
    "clubs": [],
    "player_valuations": ["date"],
}


def load_csv(name: str) -> pd.DataFrame:
    """Load one CSV and give a clear error if it is missing."""
    path = RAW_DATA_DIR / FILES[name]

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Kaggle data before running this script."
        )

    return pd.read_csv(path, low_memory=False)


def print_summary(name: str, frame: pd.DataFrame) -> None:
    """Print basic data-quality information for one DataFrame."""
    print("\n" + "=" * 72)
    print(name.upper())
    print("=" * 72)

    print(f"Shape: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns")

    print("\nDate ranges:")
    available_date_columns = [
        column
        for column in DATE_COLUMNS[name]
        if column in frame.columns
    ]

    if not available_date_columns:
        print("  No date column in this table.")

    for column in available_date_columns:
        parsed_dates = pd.to_datetime(frame[column], errors="coerce")
        valid_dates = parsed_dates.dropna()

        if valid_dates.empty:
            print(f"  {column}: no valid dates")
        else:
            print(
                f"  {column}: "
                f"{valid_dates.min().date()} to {valid_dates.max().date()}"
            )

    print("\nNull counts by column:")
    null_counts = frame.isna().sum().sort_values(ascending=False)
    print(null_counts.to_string())


def print_target_coverage(data: dict[str, pd.DataFrame]) -> None:
    """Show how many players have usable current and historical values."""
    players = data["players"]
    valuations = data["player_valuations"]

    has_current_value = players["market_value_in_eur"].notna()

    print("\n" + "=" * 72)
    print("TARGET COVERAGE")
    print("=" * 72)

    print(
        "Player rows with a current market value: "
        f"{has_current_value.sum():,} / {len(players):,}"
    )

    print(
        "Unique players with a current market value: "
        f"{players.loc[has_current_value, 'player_id'].nunique():,}"
    )

    has_historical_value = (
        valuations["player_id"].notna()
        & valuations["market_value_in_eur"].notna()
    )

    print(
        "Unique players with historical valuations: "
        f"{valuations.loc[has_historical_value, 'player_id'].nunique():,}"
    )


def main() -> None:
    """Load every required table and print its summary."""
    data = {
        name: load_csv(name)
        for name in FILES
    }

    for name, frame in data.items():
        print_summary(name, frame)

    print_target_coverage(data)


if __name__ == "__main__":
    main()
