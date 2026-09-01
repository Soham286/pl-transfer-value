"""Main navigation for the PL Transfer Value application."""

import streamlit as st


st.set_page_config(
    page_title="PL Transfer Value",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

build_player_page = st.Page(
    "pages/build_player.py",
    title="Build a Player",
    icon="⚙️",
    default=True,
)

real_player_page = st.Page(
    "pages/real_player.py",
    title="Search Real Players",
    icon="🔎",
)

navigation = st.navigation(
    {
        "Valuation": [
            build_player_page,
            real_player_page,
        ]
    }
)

navigation.run()