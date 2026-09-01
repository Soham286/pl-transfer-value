"""Search and analyze real players using held-out model predictions."""

from __future__ import annotations

import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# PATHS AND DESIGN
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
CATALOG_PATH = PROJECT_ROOT / "models" / "player_catalog.csv"

ACCENT = "#8B5CF6"
ACCENT_LIGHT = "#A78BFA"
TEXT = "#F4F4F5"
MUTED = "#A1A1AA"
SURFACE = "#141824"
SURFACE_LIGHT = "#1C2130"
GRID = "#2A3142"

POSITION_SHORT_NAMES = {
    "Attack": "ATT",
    "Midfield": "MID",
    "Defender": "DEF",
    "Goalkeeper": "GK",
}


# ============================================================================
# CACHED DATA LOADING
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    """Load the deployment model once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/model.pkl is missing."
        )

    bundle = joblib.load(MODEL_PATH)

    required = {
        "model",
        "feature_columns",
        "metadata",
    }

    missing = required.difference(bundle)

    if missing:
        raise ValueError(
            "Model bundle is missing: "
            + ", ".join(sorted(missing))
        )

    return bundle


@st.cache_data(show_spinner=False)
def load_player_catalog() -> pd.DataFrame:
    """Load the compact deployment catalog."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            "models/player_catalog.csv is missing. "
            "Run python src/build_player_catalog.py."
        )

    catalog = pd.read_csv(
        CATALOG_PATH,
        low_memory=False,
    )

    required = {
        "player_id",
        "search_label",
        "player_name",
        "image_url",
        "club_name",
        "position",
        "league_name",
        "actual_value_eur",
        "predicted_value_eur",
        "valuation_gap_eur",
    }

    missing = required.difference(catalog.columns)

    if missing:
        raise ValueError(
            "Player catalog is missing: "
            + ", ".join(sorted(missing))
        )

    catalog["player_id"] = pd.to_numeric(
        catalog["player_id"],
        errors="raise",
    ).astype("int64")

    return catalog


try:
    bundle = load_model_bundle()
    catalog = load_player_catalog()
except Exception as error:
    st.error(
        "The real-player page could not load its artifacts. "
        f"Details: {error}"
    )
    st.stop()

model = bundle["model"]
feature_columns = list(bundle["feature_columns"])
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
# PAGE STYLING
# ============================================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at 78% 4%,
                    rgba(139, 92, 246, 0.15),
                    transparent 30rem
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
            border-right: 1px solid rgba(255,255,255,0.07);
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
            font-size: clamp(2.5rem, 5vw, 4.5rem);
            font-weight: 850;
            line-height: 1;
            margin: 0;
        }

        .page-copy {
            color: #A1A1AA;
            font-size: 1.02rem;
            line-height: 1.7;
            margin: 1rem 0 1.8rem 0;
            max-width: 790px;
        }

        .player-card {
            background:
                radial-gradient(
                    circle at 20% 20%,
                    rgba(139, 92, 246, 0.30),
                    transparent 35%
                ),
                linear-gradient(
                    135deg,
                    #201A3A,
                    #141824 55%,
                    #10131D
                );
            border: 1px solid rgba(167, 139, 250, 0.38);
            border-radius: 28px;
            box-shadow: 0 30px 80px rgba(0,0,0,0.38);
            display: grid;
            grid-template-columns: minmax(260px, 0.72fr) 1.4fr;
            min-height: 470px;
            overflow: hidden;
            position: relative;
            transform:
                perspective(1200px)
                rotateX(0deg)
                rotateY(0deg);
            transition:
                transform 220ms ease,
                box-shadow 220ms ease;
        }

        .player-card:hover {
            box-shadow:
                0 38px 100px rgba(0,0,0,0.48),
                0 0 55px rgba(139,92,246,0.10);
            transform:
                perspective(1200px)
                rotateX(0.6deg)
                rotateY(-1.2deg)
                translateY(-4px);
        }

        .player-visual {
            align-items: flex-end;
            background:
                linear-gradient(
                    180deg,
                    rgba(139,92,246,0.08),
                    rgba(9,11,18,0.50)
                );
            display: flex;
            justify-content: center;
            min-height: 470px;
            overflow: hidden;
            position: relative;
        }

        .photo-fallback {
            align-items: center;
            color: rgba(255,255,255,0.20);
            display: flex;
            font-size: 5rem;
            inset: 0;
            justify-content: center;
            position: absolute;
        }

        .player-photo {
            filter:
                drop-shadow(0 24px 25px rgba(0,0,0,0.45));
            height: 95%;
            max-width: 100%;
            object-fit: contain;
            object-position: bottom center;
            position: relative;
            width: 100%;
            z-index: 2;
        }

        .position-badge {
            background: rgba(9,11,18,0.76);
            border: 1px solid rgba(255,255,255,0.13);
            border-radius: 12px;
            color: #FFFFFF;
            font-size: 0.85rem;
            font-weight: 850;
            left: 1.25rem;
            letter-spacing: 0.12em;
            padding: 0.65rem 0.8rem;
            position: absolute;
            top: 1.25rem;
            z-index: 4;
        }

        .percentile-badge {
            color: #C4B5FD;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            position: absolute;
            right: 1.25rem;
            text-transform: uppercase;
            top: 1.5rem;
            z-index: 4;
        }

        .player-information {
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 2.5rem 2.7rem;
        }

        .player-name {
            color: #FFFFFF;
            font-size: clamp(2rem, 4vw, 3.7rem);
            font-weight: 850;
            letter-spacing: -0.055em;
            line-height: 1;
            margin: 0.5rem 0 0.65rem 0;
        }

        .player-secondary {
            color: #A1A1AA;
            font-size: 1rem;
            line-height: 1.6;
        }

        .valuation-grid {
            display: grid;
            gap: 0.85rem;
            grid-template-columns: repeat(3, 1fr);
            margin-top: 2rem;
        }

        .valuation-item {
            background: rgba(9,11,18,0.46);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 17px;
            padding: 1.1rem;
        }

        .valuation-label {
            color: #A1A1AA;
            font-size: 0.69rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        .valuation-value {
            color: #FFFFFF;
            font-size: clamp(1.25rem, 2vw, 2rem);
            font-weight: 820;
            margin-top: 0.4rem;
        }

        .status-pill {
            align-self: flex-start;
            background: rgba(139,92,246,0.15);
            border: 1px solid rgba(167,139,250,0.30);
            border-radius: 999px;
            color: #D8CCFF;
            font-size: 0.82rem;
            font-weight: 750;
            margin-top: 1.15rem;
            padding: 0.55rem 0.85rem;
        }

        .season-stat-grid {
            display: grid;
            gap: 0.7rem;
            grid-template-columns: repeat(5, 1fr);
            margin-top: 1.35rem;
        }

        .season-stat {
            border-left: 2px solid rgba(167,139,250,0.50);
            padding-left: 0.75rem;
        }

        .season-stat strong {
            color: #FFFFFF;
            display: block;
            font-size: 1.05rem;
        }

        .season-stat span {
            color: #8E95A5;
            font-size: 0.68rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .section-label {
            color: #A78BFA;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            margin: 2rem 0 0.4rem 0;
            text-transform: uppercase;
        }

        .interpretation-card {
            background: rgba(20,24,36,0.74);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            color: #C7CAD1;
            line-height: 1.7;
            padding: 1.3rem 1.4rem;
        }

        @media (max-width: 900px) {
            .player-card {
                grid-template-columns: 1fr;
            }

            .player-visual {
                min-height: 360px;
            }

            .valuation-grid {
                grid-template-columns: 1fr;
            }

            .season-stat-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# FORMATTERS AND CHART HELPERS
# ============================================================================

def format_euros(value: float) -> str:
    """Format a euro amount cleanly."""
    value = abs(float(value))

    if value >= 1_000_000_000:
        return f"€{value / 1_000_000_000:,.2f}B"

    if value >= 1_000_000:
        return f"€{value / 1_000_000:,.1f}M"

    if value >= 1_000:
        return f"€{value / 1_000:,.0f}K"

    return f"€{value:,.0f}"


def format_signed_euros(value: float) -> str:
    """Format positive and negative valuation gaps."""
    prefix = "+" if value > 0 else "−" if value < 0 else ""
    return prefix + format_euros(value)


def chart_style(figure: go.Figure) -> go.Figure:
    """Apply the shared dark chart design."""
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


def build_value_comparison(
    predicted_value: float,
    actual_value: float,
) -> go.Figure:
    """Compare held-out prediction with recorded market value."""
    labels = [
        "Model estimate",
        "Recorded value",
    ]

    values = [
        predicted_value,
        actual_value,
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
            "text": "Model estimate vs recorded value",
            "x": 0.02,
        },
        yaxis_title="Market value",
        height=430,
        showlegend=False,
    )

    figure.update_yaxes(
        tickprefix="€",
        tickformat=".2s",
        rangemode="tozero",
    )

    return chart_style(figure)


def build_age_curve(
    player: pd.Series,
) -> go.Figure:
    """Project value by age while anchoring on the held-out estimate."""
    player_age = int(round(float(player["age"])))
    held_out_value = float(
        player["predicted_value_eur"]
    )

    base_features = {
        column: float(player[column])
        for column in feature_columns
    }

    ages = list(range(16, 41))
    rows = []

    for projected_age in ages:
        row = base_features.copy()
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

    raw_log_predictions = model.predict(
        age_features
    )

    curve_values = np.expm1(
        np.clip(
            raw_log_predictions,
            0.0,
            np.log1p(300_000_000),
        )
    )

    current_index = int(
        np.argmin(
            np.abs(
                np.asarray(ages) - player_age
            )
        )
    )

    current_curve_value = float(
        curve_values[current_index]
    )

    if current_curve_value > 0:
        curve_values = (
            curve_values
            * held_out_value
            / current_curve_value
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
            x=ages,
            y=curve_values,
            mode="lines",
            name="Age projection",
            line={
                "color": ACCENT,
                "width": 4,
            },
            hovertemplate=(
                "Age %{x}"
                "<br>Projected value: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[player_age],
            y=[held_out_value],
            mode="markers",
            name="Selected season",
            marker={
                "color": "#FFFFFF",
                "size": 13,
                "line": {
                    "color": ACCENT,
                    "width": 3,
                },
            },
            hovertemplate=(
                "Age %{x}"
                "<br>Held-out estimate: €%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": "Age-curve projection",
            "x": 0.02,
        },
        xaxis_title="Age",
        yaxis_title="Projected market value",
        height=430,
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

    return chart_style(figure)


# ============================================================================
# PLAYER SEARCH
# ============================================================================

with st.sidebar:
    st.markdown("## Find a player")

    st.caption(
        "Search the latest modeled season. The displayed estimate "
        "was produced without training on that season."
    )

    league_options = [
        "All leagues",
        *sorted(
            catalog["league_name"]
            .dropna()
            .astype(str)
            .unique()
        ),
    ]

    league_filter = st.selectbox(
        "League",
        options=league_options,
    )

    position_options = [
        "All positions",
        *sorted(
            catalog["position"]
            .dropna()
            .astype(str)
            .unique()
        ),
    ]

    position_filter = st.selectbox(
        "Position",
        options=position_options,
    )

    filtered_catalog = catalog.copy()

    if league_filter != "All leagues":
        filtered_catalog = filtered_catalog.loc[
            filtered_catalog["league_name"]
            == league_filter
        ]

    if position_filter != "All positions":
        filtered_catalog = filtered_catalog.loc[
            filtered_catalog["position"]
            == position_filter
        ]

    if filtered_catalog.empty:
        st.warning(
            "No players match these filters."
        )
        st.stop()

    highest_value_index = (
        filtered_catalog["predicted_value_eur"]
        .astype(float)
        .idxmax()
    )

    default_label = filtered_catalog.loc[
        highest_value_index,
        "search_label",
    ]

    player_labels = (
        filtered_catalog["search_label"]
        .astype(str)
        .tolist()
    )

    default_index = player_labels.index(
        str(default_label)
    )

    selected_label = st.selectbox(
        "Player",
        options=player_labels,
        index=default_index,
        help="Start typing a player's name to search.",
    )

    st.caption(
        f"{len(filtered_catalog):,} players available "
        "with the selected filters."
    )

selected_player = filtered_catalog.loc[
    filtered_catalog["search_label"]
    == selected_label
].iloc[0]


# ============================================================================
# PLAYER CARD
# ============================================================================

player_name = html.escape(
    str(selected_player["player_name"])
)

club_name = html.escape(
    str(selected_player["club_name"])
)

league_name = html.escape(
    str(selected_player["league_name"])
)

position = str(selected_player["position"])

sub_position = html.escape(
    str(
        selected_player.get(
            "sub_position",
            position,
        )
    )
)

country = html.escape(
    str(
        selected_player.get(
            "country_of_citizenship",
            "Unknown",
        )
    )
)

photo_url = html.escape(
    str(selected_player["image_url"]),
    quote=True,
)

predicted_value = float(
    selected_player["predicted_value_eur"]
)

actual_value = float(
    selected_player["actual_value_eur"]
)

valuation_gap = float(
    selected_player["valuation_gap_eur"]
)

player_age = int(
    round(float(selected_player["age"]))
)

goals = int(
    round(float(selected_player["goals"]))
)

assists = int(
    round(float(selected_player["assists"]))
)

minutes = int(
    round(float(selected_player["minutes_played"]))
)

appearances = int(
    round(float(selected_player["appearances"]))
)

position_short = POSITION_SHORT_NAMES.get(
    position,
    position[:3].upper(),
)

percentile = float(
    (
        catalog["predicted_value_eur"]
        <= predicted_value
    ).mean()
    * 100
)

position_group = catalog.loc[
    catalog["position"] == position
]

position_rank = int(
    (
        position_group["predicted_value_eur"]
        > predicted_value
    ).sum()
    + 1
)

if valuation_gap > typical_error_eur:
    status_text = (
        "Potential value opportunity — model estimate is "
        "above the recorded value"
    )
elif valuation_gap < -typical_error_eur:
    status_text = (
        "Recorded value carries a premium relative to "
        "the model estimate"
    )
else:
    status_text = (
        "Difference is within the model's typical error"
    )

st.markdown(
    '<div class="eyebrow">Real-player intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="page-title">Search. Compare. Discover value.</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-copy">
        Explore real players from the held-out 2025 season. Every
        estimate on this page was generated by a model trained only
        on earlier seasons, giving us a more honest predicted-versus-recorded
        comparison.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="player-card">
        <div class="player-visual">
            <div class="position-badge">{position_short}</div>
            <div class="percentile-badge">
                {percentile:.0f}th value percentile
            </div>
            <div class="photo-fallback">⚽</div>
            <img
                class="player-photo"
                src="{photo_url}"
                alt="{player_name}"
                referrerpolicy="no-referrer"
                onerror="this.style.display='none';"
            />
        </div>

        <div class="player-information">
            <div class="eyebrow">2025 player valuation</div>
            <div class="player-name">{player_name}</div>

            <div class="player-secondary">
                {club_name} · {league_name}<br>
                {sub_position} · {country}
            </div>

            <div class="valuation-grid">
                <div class="valuation-item">
                    <div class="valuation-label">Model estimate</div>
                    <div class="valuation-value">
                        {format_euros(predicted_value)}
                    </div>
                </div>

                <div class="valuation-item">
                    <div class="valuation-label">Recorded value</div>
                    <div class="valuation-value">
                        {format_euros(actual_value)}
                    </div>
                </div>

                <div class="valuation-item">
                    <div class="valuation-label">Valuation gap</div>
                    <div class="valuation-value">
                        {format_signed_euros(valuation_gap)}
                    </div>
                </div>
            </div>

            <div class="status-pill">{status_text}</div>

            <div class="season-stat-grid">
                <div class="season-stat">
                    <strong>{goals}</strong>
                    <span>Goals</span>
                </div>

                <div class="season-stat">
                    <strong>{assists}</strong>
                    <span>Assists</span>
                </div>

                <div class="season-stat">
                    <strong>{minutes:,}</strong>
                    <span>Minutes</span>
                </div>

                <div class="season-stat">
                    <strong>{appearances}</strong>
                    <span>Apps</span>
                </div>

                <div class="season-stat">
                    <strong>{player_age}</strong>
                    <span>Age</span>
                </div>
            </div>
        </div>
    </div>
    """.strip().replace("\n", " "),
    unsafe_allow_html=True,
)

st.caption(
    "Player images are loaded from the source URLs contained in the "
    "Kaggle Transfermarkt dataset. Market values are estimates, not "
    "confirmed transfer fees."
)


# ============================================================================
# ANALYSIS
# ============================================================================

lower_value = max(
    0.0,
    predicted_value - typical_error_eur,
)

upper_value = (
    predicted_value + typical_error_eur
)

st.markdown(
    '<div class="section-label">Scouting interpretation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="interpretation-card">
        The held-out estimate is
        <strong>{format_euros(predicted_value)}</strong>, with an
        MAE-based indicative range of
        <strong>{format_euros(lower_value)}–{format_euros(upper_value)}</strong>.
        The recorded value is
        <strong>{format_euros(actual_value)}</strong>.
        This player ranks
        <strong>#{position_rank:,} of {len(position_group):,}</strong>
        among modeled {html.escape(position.lower())}s by predicted value.
        A valuation gap is a scouting lead, not proof that a player is
        mispriced.
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
    comparison_figure = build_value_comparison(
        predicted_value,
        actual_value,
    )

    st.plotly_chart(
        comparison_figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

with chart_columns[1]:
    age_figure = build_age_curve(
        selected_player
    )

    st.plotly_chart(
        age_figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

st.caption(
    "The age curve holds performance and context constant. It is a "
    "scenario analysis rather than a forecast of future form."
)

st.divider()

footer_columns = st.columns(4)

footer_values = [
    (
        "Prediction source",
        "Held-out 2025",
    ),
    (
        "Typical error",
        f"± {format_euros(typical_error_eur)}",
    ),
    (
        "Model R²",
        f"{evaluation_metrics.get('r2', 0.7984):.3f}",
    ),
    (
        "Player catalog",
        f"{len(catalog):,} players",
    ),
]

for column, (label, value) in zip(
    footer_columns,
    footer_values,
):
    with column:
        st.caption(label)
        st.markdown(f"**{value}**")
