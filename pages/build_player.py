"""Streamlit interface for the PL Transfer Value model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.explain import (
    build_shap_waterfall,
    load_tree_explainer,
)


# ============================================================================
# BLOCK 1 — PAGE CONFIGURATION AND DESIGN SYSTEM
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
REFERENCE_PATH = PROJECT_ROOT / "models" / "ui_reference.json"

ACCENT = "#8B5CF6"
ACCENT_LIGHT = "#A78BFA"
BACKGROUND = "#090B12"
SURFACE = "#141824"
SURFACE_LIGHT = "#1C2130"
TEXT = "#F4F4F5"
MUTED = "#A1A1AA"
GRID = "#2A3142"

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at 75% 5%,
                    rgba(139, 92, 246, 0.13),
                    transparent 28rem
                ),
                #090B12;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background: #10131D;
            border-right: 1px solid rgba(255, 255, 255, 0.07);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.035em;
        }

        .eyebrow {
            color: #A78BFA;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            margin-bottom: 0.7rem;
            text-transform: uppercase;
        }

        .page-title {
            color: #F4F4F5;
            font-size: clamp(2.5rem, 5vw, 4.8rem);
            font-weight: 800;
            line-height: 0.98;
            margin: 0;
            max-width: 850px;
        }

        .page-subtitle {
            color: #A1A1AA;
            font-size: 1.05rem;
            line-height: 1.7;
            margin-top: 1.15rem;
            max-width: 760px;
        }

        .result-card {
            background:
                linear-gradient(
                    135deg,
                    rgba(139, 92, 246, 0.19),
                    rgba(20, 24, 36, 0.96) 55%
                );
            border: 1px solid rgba(167, 139, 250, 0.32);
            border-radius: 24px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.30);
            margin: 1.8rem 0 1.25rem 0;
            overflow: hidden;
            padding: 2.2rem 2.4rem;
            position: relative;
        }

        .result-card::before {
            background: #8B5CF6;
            content: "";
            height: 100%;
            left: 0;
            position: absolute;
            top: 0;
            width: 5px;
        }

        .result-label {
            color: #C4B5FD;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .result-value {
            color: #FFFFFF;
            font-size: clamp(3.2rem, 8vw, 6rem);
            font-weight: 850;
            letter-spacing: -0.06em;
            line-height: 1;
            margin: 0.6rem 0 1.2rem 0;
        }

        .result-meta {
            color: #D4D4D8;
            display: flex;
            flex-wrap: wrap;
            font-size: 0.96rem;
            gap: 1rem 2rem;
        }

        .result-meta strong {
            color: #FFFFFF;
        }

        .empty-card {
            align-items: center;
            background: rgba(20, 24, 36, 0.72);
            border: 1px dashed rgba(167, 139, 250, 0.38);
            border-radius: 24px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-top: 2rem;
            min-height: 390px;
            padding: 3rem;
            text-align: center;
        }

        .empty-icon {
            align-items: center;
            background: rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(167, 139, 250, 0.28);
            border-radius: 18px;
            display: flex;
            font-size: 2rem;
            height: 72px;
            justify-content: center;
            margin-bottom: 1.4rem;
            width: 72px;
        }

        .empty-title {
            color: #F4F4F5;
            font-size: 1.55rem;
            font-weight: 750;
            margin-bottom: 0.55rem;
        }

        .empty-copy {
            color: #A1A1AA;
            line-height: 1.65;
            max-width: 530px;
        }

        .section-label {
            color: #A78BFA;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            margin-top: 1.6rem;
            text-transform: uppercase;
        }

        .fact-card {
            background: rgba(20, 24, 36, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 18px;
            min-height: 110px;
            padding: 1.2rem 1.3rem;
        }

        .fact-label {
            color: #A1A1AA;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .fact-value {
            color: #F4F4F5;
            font-size: 1.2rem;
            font-weight: 750;
            margin-top: 0.45rem;
        }

        div[data-testid="stForm"] {
            border: 0;
            padding: 0;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            background: #8B5CF6;
            border: 1px solid #8B5CF6;
            border-radius: 12px;
            color: white;
            font-weight: 750;
            min-height: 3rem;
            transition: all 160ms ease;
            width: 100%;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #7C3AED;
            border-color: #A78BFA;
            box-shadow: 0 8px 28px rgba(139, 92, 246, 0.28);
            transform: translateY(-1px);
        }

        hr {
            border-color: rgba(255, 255, 255, 0.08);
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# BLOCK 2 — CACHED MODEL AND REFERENCE LOADING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    """Load the model once and reuse it across Streamlit reruns."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/model.pkl is missing. Run python src/tune.py first."
        )

    loaded = joblib.load(MODEL_PATH)

    required_keys = {
        "model",
        "feature_columns",
        "feature_defaults",
        "metadata",
    }

    missing_keys = required_keys.difference(loaded)

    if missing_keys:
        raise ValueError(
            "The model bundle is missing: "
            + ", ".join(sorted(missing_keys))
        )

    return loaded


@st.cache_data(show_spinner=False)
def load_ui_reference() -> dict:
    """Load small positional benchmarks without loading raw data."""
    if not REFERENCE_PATH.exists():
        return {
            "reference_season": "latest",
            "position_summary": {},
            "club_tier_strength_log": {},
        }

    return json.loads(
        REFERENCE_PATH.read_text(encoding="utf-8")
    )


try:
    bundle = load_model_bundle()
    reference = load_ui_reference()
except Exception as error:
    st.error(
        "The application could not load its model artifacts. "
        f"Details: {error}"
    )
    st.stop()

model = bundle["model"]
feature_columns = list(bundle["feature_columns"])
feature_defaults = dict(bundle["feature_defaults"])
metadata = dict(bundle["metadata"])

evaluation_metrics = metadata.get(
    "evaluation_metrics",
    {},
)

typical_error_eur = float(
    evaluation_metrics.get(
        "mae_eur",
        2_760_000,
    )
)


# ============================================================================
# BLOCK 3 — INPUT DEFINITIONS AND FEATURE CONSTRUCTION
# ============================================================================

LEAGUES = {
    "Premier League": "GB1",
    "La Liga": "ES1",
    "Bundesliga": "L1",
    "Serie A": "IT1",
    "Ligue 1": "FR1",
    "Eredivisie": "NL1",
    "Liga Portugal": "PO1",
    "Belgian Pro League": "BE1",
    "Danish Superliga": "DK1",
    "Greek Super League": "GR1",
    "Russian Premier League": "RU1",
    "Scottish Premiership": "SC1",
    "Süper Lig": "TR1",
    "Ukrainian Premier League": "UKR1",
}

LEAGUE_MATCHES = {
    "GB1": 38,
    "ES1": 38,
    "L1": 34,
    "IT1": 38,
    "FR1": 34,
    "NL1": 34,
    "PO1": 34,
    "BE1": 30,
    "DK1": 22,
    "GR1": 26,
    "RU1": 30,
    "SC1": 38,
    "TR1": 38,
    "UKR1": 30,
}

POSITIONS = [
    "Attack",
    "Midfield",
    "Defender",
    "Goalkeeper",
]

CLUB_TIERS = [
    "Unknown",
    "Lower",
    "Middle",
    "Upper",
    "Elite",
]


def format_euros(value: float) -> str:
    """Format currency as €42.5M rather than a raw float."""
    value = max(float(value), 0.0)

    if value >= 1_000_000_000:
        return f"€{value / 1_000_000_000:,.2f}B"

    if value >= 1_000_000:
        return f"€{value / 1_000_000:,.1f}M"

    if value >= 1_000:
        return f"€{value / 1_000:,.0f}K"

    return f"€{value:,.0f}"


def safe_euro_prediction(log_prediction: float) -> float:
    """Invert log1p safely and guard against numerical overflow."""
    if not np.isfinite(log_prediction):
        raise ValueError("The model returned a non-finite prediction.")

    maximum_log_value = np.log1p(300_000_000)
    guarded_log_value = np.clip(
        log_prediction,
        a_min=0.0,
        a_max=maximum_log_value,
    )

    return float(np.expm1(guarded_log_value))


def build_feature_row(
    *,
    goals: int,
    assists: int,
    minutes: int,
    appearances: int,
    age: int,
    position: str,
    league_code: str,
    club_tier: str,
) -> pd.DataFrame:
    """Convert human-readable inputs into the trained feature schema."""
    row = {
        column: float(feature_defaults.get(column, 0.0))
        for column in feature_columns
    }

    # Reset all one-hot categories before activating selected categories.
    for column in feature_columns:
        if column.startswith(
            ("position_", "league_", "club_tier_")
        ):
            row[column] = 0.0

    row["goals"] = float(goals)
    row["assists"] = float(assists)
    row["minutes_played"] = float(minutes)
    row["age"] = float(age)
    row["age_squared"] = float(age**2)
    row["appearances"] = float(appearances)

    safe_minutes = max(float(minutes), 1.0)

    row["goals_per_90"] = float(
        goals * 90 / safe_minutes
    )
    row["assists_per_90"] = float(
        assists * 90 / safe_minutes
    )
    row["goal_contributions_per_90"] = float(
        (goals + assists) * 90 / safe_minutes
    )

    scheduled_matches = LEAGUE_MATCHES.get(
        league_code,
        38,
    )

    full_season_minutes = scheduled_matches * 90

    row["minutes_share"] = float(
        np.clip(
            minutes / full_season_minutes,
            0.0,
            1.0,
        )
    )

    row["is_prime_age"] = float(
        23 <= age <= 28
    )

    position_column = f"position_{position}"
    league_column = f"league_{league_code}"
    club_tier_column = f"club_tier_{club_tier}"

    if position_column in row:
        row[position_column] = 1.0

    if league_column in row:
        row[league_column] = 1.0

    if club_tier_column in row:
        row[club_tier_column] = 1.0
    elif "club_tier_Unknown" in row:
        row["club_tier_Unknown"] = 1.0

    # Club-strength history is optional. Use a tier-specific training
    # median when available and explicitly flag unknown clubs.
    if club_tier == "Unknown":
        if "club_strength_available" in row:
            row["club_strength_available"] = 0.0

        if "club_strength_log" in row:
            row["club_strength_log"] = 0.0
    else:
        tier_strengths = reference.get(
            "club_tier_strength_log",
            {},
        )

        tier_strength = tier_strengths.get(
            club_tier,
            feature_defaults.get(
                "club_strength_log",
                0.0,
            ),
        )

        if "club_strength_available" in row:
            row["club_strength_available"] = 1.0

        if "club_strength_log" in row:
            row["club_strength_log"] = float(
                tier_strength
            )

    frame = pd.DataFrame(
        [row],
        columns=feature_columns,
    )

    frame = frame.apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)

    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(
            "The generated feature row contains invalid values."
        )

    return frame


def predict_value(features: pd.DataFrame) -> float:
    """Predict one market value in real euros."""
    ordered_features = features[feature_columns]
    log_prediction = float(
        model.predict(ordered_features)[0]
    )
    return safe_euro_prediction(log_prediction)


# ============================================================================
# BLOCK 4 — CHART BUILDERS
# ============================================================================

def apply_chart_style(figure: go.Figure) -> go.Figure:
    """Apply the shared visual design to Plotly charts."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "color": TEXT,
            "family": "Arial, sans-serif",
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 70,
            "b": 30,
        },
        hoverlabel={
            "bgcolor": SURFACE_LIGHT,
            "bordercolor": ACCENT,
            "font_color": TEXT,
        },
    )

    figure.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
    )

    figure.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
    )

    return figure


def build_age_curve(
    original_features: pd.DataFrame,
    player_age: int,
    player_value: float,
) -> go.Figure:
    """Project model value across ages while holding form constant."""
    rows = []

    for projected_age in range(16, 41):
        row = original_features.iloc[0].copy()
        row["age"] = float(projected_age)
        row["age_squared"] = float(projected_age**2)
        row["is_prime_age"] = float(
            23 <= projected_age <= 28
        )
        rows.append(row)

    age_features = pd.DataFrame(
        rows,
        columns=feature_columns,
    )

    age_log_predictions = model.predict(age_features)
    maximum_log_value = np.log1p(300_000_000)

    age_values = np.expm1(
        np.clip(
            age_log_predictions,
            0.0,
            maximum_log_value,
        )
    )

    figure = go.Figure()

    figure.add_vrect(
        x0=23,
        x1=28,
        fillcolor=ACCENT,
        opacity=0.10,
        line_width=0,
        annotation_text="Prime-age window",
        annotation_position="top left",
    )

    figure.add_trace(
        go.Scatter(
            x=list(range(16, 41)),
            y=age_values,
            mode="lines",
            name="Projected value",
            line={
                "color": ACCENT,
                "width": 4,
            },
            hovertemplate=(
                "Age %{x}"
                "<br>Estimated value: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[player_age],
            y=[player_value],
            mode="markers",
            name="Your player",
            marker={
                "color": "#FFFFFF",
                "size": 13,
                "line": {
                    "color": ACCENT,
                    "width": 3,
                },
            },
            hovertemplate=(
                "Your player, age %{x}"
                "<br>Estimated value: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": "Value across the age curve",
            "x": 0.02,
        },
        xaxis_title="Age",
        yaxis_title="Estimated market value",
        height=440,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.10,
            "x": 1,
            "xanchor": "right",
        },
    )

    figure.update_xaxes(
        dtick=4,
        range=[15.5, 40.5],
    )

    figure.update_yaxes(
        tickprefix="€",
        tickformat=".2s",
        rangemode="tozero",
    )

    return apply_chart_style(figure)


def build_position_comparison(
    player_value: float,
    position: str,
) -> go.Figure:
    """Compare the prediction with the latest positional average."""
    position_values = reference.get(
        "position_summary",
        {},
    ).get(
        position,
        {},
    )

    position_average = float(
        position_values.get(
            "mean_value_eur",
            player_value,
        )
    )

    reference_season = reference.get(
        "reference_season",
        "latest",
    )

    labels = [
        "Your player",
        f"{position} average",
    ]

    values = [
        player_value,
        position_average,
    ]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[
                ACCENT,
                "#394155",
            ],
            text=[
                format_euros(value)
                for value in values
            ],
            textposition="outside",
            hovertemplate=(
                "%{x}"
                "<br>Value: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": (
                f"Compared with {position.lower()}s "
                f"in {reference_season}"
            ),
            "x": 0.02,
        },
        yaxis_title="Market value",
        height=440,
        showlegend=False,
    )

    figure.update_yaxes(
        tickprefix="€",
        tickformat=".2s",
        rangemode="tozero",
    )

    return apply_chart_style(figure)


# ============================================================================
# BLOCK 5 — SIDEBAR INPUT FORM
# ============================================================================

available_leagues = {
    name: code
    for name, code in LEAGUES.items()
    if f"league_{code}" in feature_columns
}

if not available_leagues:
    available_leagues = LEAGUES

with st.sidebar:
    st.markdown("## Player profile")
    st.caption(
        "Enter one completed league-season profile. "
        "Inputs are converted to the exact training schema."
    )

    with st.form("player_input_form"):
        position = st.selectbox(
            "Position",
            options=POSITIONS,
            index=0,
        )

        league_name = st.selectbox(
            "League",
            options=list(available_leagues),
            index=0,
        )

        age = st.slider(
            "Age",
            min_value=16,
            max_value=40,
            value=24,
            step=1,
        )

        goals = st.number_input(
            "Goals",
            min_value=0,
            max_value=60,
            value=10,
            step=1,
        )

        assists = st.number_input(
            "Assists",
            min_value=0,
            max_value=40,
            value=6,
            step=1,
        )

        minutes = st.number_input(
            "Minutes played",
            min_value=300,
            max_value=4_500,
            value=2_200,
            step=100,
            help=(
                "The model was trained only on player-seasons "
                "with at least 300 minutes."
            ),
        )

        appearances = st.number_input(
            "Appearances",
            min_value=1,
            max_value=60,
            value=30,
            step=1,
        )

        club_tier = st.selectbox(
            "Club tier",
            options=CLUB_TIERS,
            index=3,
            help=(
                "Club tier was derived from prior-season squad "
                "market strength during training."
            ),
        )

        contract_years = st.slider(
            "Contract years remaining — context only",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help=(
                "Reliable historical contract snapshots were unavailable, "
                "so this input is shown as context but is not passed to "
                "the primary model."
            ),
        )

        submitted = st.form_submit_button(
            "Estimate market value",
            use_container_width=True,
        )


# ============================================================================
# BLOCK 6 — PREDICTION EXECUTION
# ============================================================================

if submitted:
    try:
        selected_league_code = available_leagues[
            league_name
        ]

        input_features = build_feature_row(
            goals=int(goals),
            assists=int(assists),
            minutes=int(minutes),
            appearances=int(appearances),
            age=int(age),
            position=position,
            league_code=selected_league_code,
            club_tier=club_tier,
        )

        predicted_value = predict_value(input_features)

        st.session_state["prediction_result"] = {
            "predicted_value": predicted_value,
            "features": input_features,
            "age": int(age),
            "position": position,
            "league_name": league_name,
            "contract_years": float(contract_years),
            "goals": int(goals),
            "assists": int(assists),
            "minutes": int(minutes),
        }

    except Exception as error:
        st.session_state.pop(
            "prediction_result",
            None,
        )

        st.error(
            "The prediction could not be completed. "
            "Please adjust the inputs and try again. "
            f"Technical detail: {error}"
        )


# ============================================================================
# BLOCK 7 — PAGE HEADER AND RESULTS
# ============================================================================

st.markdown(
    '<div class="eyebrow">Scouting intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="page-title">What is this player really worth?</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-subtitle">
        Estimate a player's market value from age, production,
        playing time, position, league and club context. The model
        was trained on 64,622 historical player-seasons and evaluated
        chronologically on the latest held-out season.
    </div>
    """,
    unsafe_allow_html=True,
)

result = st.session_state.get("prediction_result")

if result is None:
    st.markdown(
        """
        <div class="empty-card">
            <div class="empty-icon">⚽</div>
            <div class="empty-title">Build a player profile</div>
            <div class="empty-copy">
                Use the sidebar to enter a completed season, then select
                <strong>Estimate market value</strong>. Your valuation,
                indicative range, age curve and positional comparison
                will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    predicted_value = float(result["predicted_value"])

    lower_value = max(
        0.0,
        predicted_value - typical_error_eur,
    )

    upper_value = (
        predicted_value + typical_error_eur
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated market value</div>
            <div class="result-value">
                {format_euros(predicted_value)}
            </div>
            <div class="result-meta">
                <span>
                    Indicative range:
                    <strong>
                        {format_euros(lower_value)}
                        –
                        {format_euros(upper_value)}
                    </strong>
                </span>
                <span>
                    Typical error:
                    <strong>
                        ± {format_euros(typical_error_eur)}
                    </strong>
                </span>
                <span>
                    Profile:
                    <strong>
                        {result["position"]} ·
                        {result["league_name"]}
                    </strong>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "The current range is based on held-out MAE, not yet a "
        "statistical prediction interval. Part 7 will replace it "
        "with trained 10th/90th percentile models."
    )


    st.markdown(
        '<div class="eyebrow">Why this price</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### What moved this valuation?")

    try:
        with st.spinner("Explaining this valuation..."):
            shap_explainer = load_tree_explainer(model)

            shap_figure = build_shap_waterfall(
                explainer=shap_explainer,
                model=model,
                features=result["features"],
                feature_columns=feature_columns,
                max_features=8,
                accent=ACCENT,
            )

            shap_figure = apply_chart_style(
                shap_figure
            )

        st.plotly_chart(
            shap_figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            key="custom-player-shap-waterfall",
        )

        st.caption(
            "Green features push the estimate upward; orange "
            "features pull it downward. SHAP values are shown "
            "in log1p-value units because that is the scale on "
            "which the model was trained. Every bar adds to the "
            "baseline and reconstructs the displayed prediction."
        )

    except Exception as error:
        st.warning(
            "The valuation was produced successfully, but its "
            "SHAP explanation is temporarily unavailable."
        )

        with st.expander("Explanation error details"):
            st.code(str(error))

    contribution_rate = (
        (result["goals"] + result["assists"])
        * 90
        / max(result["minutes"], 1)
    )

    if contribution_rate > 2.0:
        st.warning(
            "This is an unusually high goal-contribution rate. "
            "The model can still estimate it, but confidence is lower "
            "because few training examples are this extreme."
        )

    if result["contract_years"] <= 1.0:
        st.info(
            "The player is in the final contract year. This is valuable "
            "scouting context, but it is not included in this historical "
            "model because reliable season-level contract snapshots were "
            "not available."
        )

    st.markdown(
        '<div class="section-label">Player context</div>',
        unsafe_allow_html=True,
    )

    fact_columns = st.columns(4)

    fact_values = [
        (
            "Production",
            f"{result['goals']} G · {result['assists']} A",
        ),
        (
            "Playing time",
            f"{result['minutes']:,} minutes",
        ),
        (
            "Age",
            f"{result['age']} years",
        ),
        (
            "Contract context",
            f"{result['contract_years']:.1f} years",
        ),
    ]

    for column, (label, value) in zip(
        fact_columns,
        fact_values,
    ):
        with column:
            st.markdown(
                f"""
                <div class="fact-card">
                    <div class="fact-label">{label}</div>
                    <div class="fact-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-label">Valuation context</div>',
        unsafe_allow_html=True,
    )

    chart_columns = st.columns(
        2,
        gap="large",
    )

    with chart_columns[0]:
        age_figure = build_age_curve(
            original_features=result["features"],
            player_age=result["age"],
            player_value=predicted_value,
        )

        st.plotly_chart(
            age_figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        st.caption(
            "Form, playing time, position, league and club context "
            "are held constant while age changes."
        )

    with chart_columns[1]:
        position_figure = build_position_comparison(
            player_value=predicted_value,
            position=result["position"],
        )

        st.plotly_chart(
            position_figure,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        st.caption(
            "The benchmark is the observed mean value for that "
            "position in the latest modeled season."
        )


# ============================================================================
# BLOCK 8 — MODEL TRANSPARENCY FOOTER
# ============================================================================

st.divider()

footer_columns = st.columns(4)

footer_items = [
    (
        "Model",
        metadata.get(
            "model_type",
            "XGBoost",
        ),
    ),
    (
        "Held-out MAE",
        format_euros(typical_error_eur),
    ),
    (
        "Held-out R²",
        f"{evaluation_metrics.get('r2', 0.7984):.3f}",
    ),
    (
        "Training window",
        metadata.get(
            "training_window",
            "2012–2025",
        ),
    ),
]

for column, (label, value) in zip(
    footer_columns,
    footer_items,
):
    with column:
        st.caption(label)
        st.markdown(f"**{value}**")

st.caption(
    "Transfermarkt values are estimates rather than confirmed transfer "
    "fees. This tool supports scouting analysis; it does not replace "
    "professional judgment."
)
