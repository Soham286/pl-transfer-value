"""Reusable SHAP explanations for player-value predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


FEATURE_LABELS = {
    "goals": "Goals",
    "assists": "Assists",
    "minutes_played": "Minutes",
    "age": "Age",
    "age_squared": "Age²",
    "goals_per_90": "Goals per 90",
    "assists_per_90": "Assists per 90",
    "goal_contributions_per_90": "Goal contributions per 90",
    "appearances": "Appearances",
    "minutes_share": "Season minutes share",
    "is_prime_age": "Prime-age profile",
    "club_strength_log": "Club strength",
    "club_strength_available": "Known club strength",
}


@st.cache_resource(show_spinner=False)
def load_tree_explainer(_model: Any) -> shap.TreeExplainer:
    """Create and cache one explainer for the fitted tree model."""
    return shap.TreeExplainer(_model)


def format_euros(value: float) -> str:
    """Format a value as compact euros."""
    safe_value = max(float(value), 0.0)

    if safe_value >= 1_000_000:
        return f"€{safe_value / 1_000_000:,.1f}M"

    if safe_value >= 1_000:
        return f"€{safe_value / 1_000:,.0f}K"

    return f"€{safe_value:,.0f}"


def friendly_feature_name(column: str) -> str:
    """Convert model-column names into readable labels."""
    if column in FEATURE_LABELS:
        return FEATURE_LABELS[column]

    for prefix, label in (
        ("position_", "Position"),
        ("league_", "League"),
        ("club_tier_", "Club tier"),
    ):
        if column.startswith(prefix):
            category = column.removeprefix(prefix)
            return f"{label}: {category}"

    return column.replace("_", " ").title()


def format_feature_value(
    column: str,
    value: float,
) -> str:
    """Format the observed value displayed in SHAP tooltips."""
    if column.startswith(
        ("position_", "league_", "club_tier_")
    ):
        return "selected" if value >= 0.5 else "not selected"

    if column in {
        "goals",
        "assists",
        "minutes_played",
        "appearances",
        "age",
        "age_squared",
    }:
        return f"{value:,.0f}"

    if column in {
        "is_prime_age",
        "club_strength_available",
    }:
        return "yes" if value >= 0.5 else "no"

    return f"{value:,.2f}"


def build_shap_waterfall(
    *,
    explainer: shap.TreeExplainer,
    model: Any,
    features: pd.DataFrame,
    feature_columns: list[str],
    max_features: int = 8,
    accent: str = "#8B5CF6",
    positive: str = "#22C55E",
    negative: str = "#F97316",
) -> go.Figure:
    """Build a dark-theme SHAP waterfall for one prediction."""
    ordered = features.reindex(
        columns=feature_columns
    ).apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)

    if len(ordered) != 1:
        raise ValueError(
            "SHAP waterfall requires exactly one player."
        )

    explanation = explainer(ordered)

    shap_values = np.asarray(
        explanation.values,
        dtype=float,
    )

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    shap_values = shap_values.reshape(-1)

    if len(shap_values) != len(feature_columns):
        raise ValueError(
            "SHAP output does not match the feature schema."
        )

    base_value = float(
        np.asarray(
            explanation.base_values,
            dtype=float,
        ).reshape(-1)[0]
    )

    prediction_log = float(
        np.asarray(
            model.predict(ordered),
            dtype=float,
        ).reshape(-1)[0]
    )

    reconstructed_log = float(
        base_value + shap_values.sum()
    )

    if not np.isclose(
        reconstructed_log,
        prediction_log,
        atol=1e-3,
        rtol=1e-3,
    ):
        raise ValueError(
            "SHAP contributions do not reconstruct the prediction."
        )

    ranking = np.argsort(
        np.abs(shap_values)
    )[::-1]

    display_count = min(
        max(int(max_features), 1),
        len(feature_columns),
    )

    top_indices = ranking[:display_count]
    remaining_indices = ranking[display_count:]

    labels = ["Model baseline"]
    impacts: list[float] = []
    hover_text = [
        (
            "Model baseline"
            f"<br>{format_euros(np.expm1(base_value))}"
        )
    ]

    for index in top_indices:
        column = feature_columns[int(index)]
        feature_value = float(ordered.iloc[0, int(index)])
        impact = float(shap_values[int(index)])

        labels.append(
            friendly_feature_name(column)
        )
        impacts.append(impact)

        direction = (
            "Pushes the estimate up"
            if impact >= 0
            else "Pushes the estimate down"
        )

        hover_text.append(
            f"{friendly_feature_name(column)}"
            f"<br>Player value: "
            f"{format_feature_value(column, feature_value)}"
            f"<br>SHAP impact: {impact:+.3f}"
            f"<br>{direction}"
        )

    if len(remaining_indices):
        other_impact = float(
            shap_values[remaining_indices].sum()
        )

        labels.append("Other features")
        impacts.append(other_impact)
        hover_text.append(
            "Combined impact of the remaining features"
            f"<br>SHAP impact: {other_impact:+.3f}"
        )

    predicted_euros = float(
        np.expm1(prediction_log)
    )

    labels.append("Final prediction")
    hover_text.append(
        "Final model prediction"
        f"<br>{format_euros(predicted_euros)}"
    )

    measures = (
        ["absolute"]
        + ["relative"] * len(impacts)
        + ["total"]
    )

    x_values = (
        [base_value]
        + impacts
        + [0.0]
    )

    text_values = (
        [format_euros(np.expm1(base_value))]
        + [f"{impact:+.2f}" for impact in impacts]
        + [format_euros(predicted_euros)]
    )

    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=labels,
            x=x_values,
            text=text_values,
            textposition="outside",
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
            connector={
                "line": {
                    "color": "#475569",
                    "width": 1,
                }
            },
            increasing={
                "marker": {
                    "color": positive,
                }
            },
            decreasing={
                "marker": {
                    "color": negative,
                }
            },
            totals={
                "marker": {
                    "color": accent,
                }
            },
        )
    )

    figure.update_layout(
        title={
            "text": "SHAP contribution waterfall",
            "x": 0.01,
        },
        xaxis_title="Contribution to log1p(market value)",
        yaxis={
            "autorange": "reversed",
            "title": None,
        },
        waterfallgap=0.25,
        showlegend=False,
        height=max(
            500,
            46 * len(labels),
        ),
        margin={
            "l": 20,
            "r": 70,
            "t": 75,
            "b": 55,
        },
    )

    return figure
