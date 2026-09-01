"""Build small deployment-safe reference statistics for the Streamlit UI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "ui_reference.json"

POSITION_ORDER = [
    "Attack",
    "Midfield",
    "Defender",
    "Goalkeeper",
]


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {FEATURES_PATH}. Run python src/features.py first."
        )

    data = pd.read_csv(FEATURES_PATH, low_memory=False)

    required = {
        "season",
        "position",
        "target_market_value_eur",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    data["target_market_value_eur"] = pd.to_numeric(
        data["target_market_value_eur"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "season",
            "position",
            "target_market_value_eur",
        ]
    )

    latest_season = int(data["season"].max())
    latest = data.loc[data["season"] == latest_season].copy()

    position_summary = {}

    for position in POSITION_ORDER:
        group = latest.loc[latest["position"] == position]
        values = group["target_market_value_eur"]

        if values.empty:
            continue

        position_summary[position] = {
            "player_count": int(len(group)),
            "mean_value_eur": float(values.mean()),
            "median_value_eur": float(values.median()),
        }

    club_tier_strength = {}

    if {
        "club_tier",
        "club_strength_log",
        "club_strength_available",
    }.issubset(data.columns):
        available = data.loc[
            pd.to_numeric(
                data["club_strength_available"],
                errors="coerce",
            ).fillna(0)
            == 1
        ].copy()

        available["club_strength_log"] = pd.to_numeric(
            available["club_strength_log"],
            errors="coerce",
        )

        available = available.dropna(
            subset=["club_tier", "club_strength_log"]
        )

        tier_medians = available.groupby(
            "club_tier"
        )["club_strength_log"].median()

        club_tier_strength = {
            str(tier): float(value)
            for tier, value in tier_medians.items()
        }

    reference = {
        "reference_season": latest_season,
        "position_summary": position_summary,
        "club_tier_strength_log": club_tier_strength,
        "notes": {
            "position_benchmark": (
                "Mean observed target value in the latest modeled season."
            ),
            "contract": (
                "Contract information is contextual and is not used by "
                "the primary historical model."
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(reference, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("UI REFERENCE CREATED")
    print("=" * 72)
    print(f"Reference season: {latest_season}")
    print(f"Output: {OUTPUT_PATH}")

    for position, values in position_summary.items():
        print(
            f"{position:12} "
            f"players={values['player_count']:,} "
            f"mean=€{values['mean_value_eur'] / 1_000_000:,.2f}M "
            f"median=€{values['median_value_eur'] / 1_000_000:,.2f}M"
        )


if __name__ == "__main__":
    main()
