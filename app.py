
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="NBA Archetype Explorer",
    page_icon="🏀",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

FEATURES = [
    "PTS_PER36",
    "FG3A_PER36",
    "FG2A_PER36",
    "AST_PCT",
    "AST_TO",
    "TO_RATIO",
    "TS_PCT",
    "FTA_PER36",
    "OREB_PER36",
]

PERCENT_FEATURES = {"AST_PCT", "TS_PCT"}

LABELS = {
    "PTS_PER36": "Points / 36",
    "FG3A_PER36": "3PA / 36",
    "FG2A_PER36": "2PA / 36",
    "AST_PCT": "Assist %",
    "AST_TO": "AST / TO",
    "TO_RATIO": "Turnover Ratio",
    "TS_PCT": "True Shooting %",
    "FTA_PER36": "FTA / 36",
    "OREB_PER36": "Off. Rebounds / 36",
}

ARCHETYPES = {
    0: "Supporting Perimeter Scorers",
    1: "High-Volume Scoring Creators",
    2: "Interior Finishers & Offensive Rebounders",
    3: "Perimeter-Oriented Secondary Contributors",
    4: "Lower-Scoring Playmakers",
}

DESCRIPTIONS = {
    0: "Moderate perimeter shooting volume and "
       "relatively limited playmaking involvement.",
    1: "High scoring production, free-throw volume, "
       "and substantial offensive creation.",
    2: "Strong shooting efficiency, offensive rebounding, "
       "and limited three-point volume.",
    3: "Perimeter-oriented contributors with high "
       "three-point volume and strong AST/TO.",
    4: "Higher assist involvement with comparatively "
       "modest scoring production.",
}


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data():
    refinement = ROOT / "results" / "refinement"

    assignments = pd.read_csv(
        refinement / "player_assignments_k3_to_k6.csv"
    )

    profiles = pd.read_csv(
        refinement / "cluster_profiles_k5.csv"
    )

    representatives = pd.read_csv(
        refinement / "representative_players_k5.csv"
    )

    assignments = assignments.rename(
        columns={"cluster_k5": "CLUSTER"}
    )

    assignments["CLUSTER"] = assignments["CLUSTER"].astype(int)
    profiles["CLUSTER"] = profiles["CLUSTER"].astype(int)
    representatives["CLUSTER"] = (
        representatives["CLUSTER"].astype(int)
    )

    feature_path = (
        ROOT / "data" / "processed" / "offensive_features.csv"
    )

    players = None

    if feature_path.exists():
        features = pd.read_csv(feature_path)

        players = assignments.merge(
            features,
            on="PLAYER_ID",
            how="left",
            validate="one_to_one",
            suffixes=("", "_FEATURES"),
        )

    comparison_path = refinement / "model_comparison.csv"

    comparison = (
        pd.read_csv(comparison_path)
        if comparison_path.exists()
        else None
    )

    return assignments, profiles, representatives, players, comparison


try:
    assignments, profiles, representatives, players, comparison = (
        load_data()
    )
except (FileNotFoundError, KeyError, ValueError) as exc:
    st.error(f"Unable to load required data: {exc}")
    st.stop()


def display_value(feature, value):
    if pd.isna(value):
        return "N/A"
    if feature in PERCENT_FEATURES:
        return f"{value * 100:.1f}%"
    return f"{value:.2f}"


def archetype_name(cluster):
    return ARCHETYPES.get(int(cluster), f"Cluster {cluster}")


def player_record(name):
    matches = players.loc[players["PLAYER_NAME"] == name]
    return matches.iloc[0]


def radar_chart(rows, names):
    # Use percentile ranks over all players so metrics with
    # different units can share a comparable visual scale.
    valid = players.dropna(subset=FEATURES)
    percentiles = valid[FEATURES].rank(pct=True) * 100

    categories = [LABELS[f] for f in FEATURES]
    fig = go.Figure()

    for row, name in zip(rows, names):
        values = []

        for f in FEATURES:
            value = row[f]
            if pd.isna(value):
                values.append(np.nan)
            else:
                values.append(
                    100 * (valid[f] <= value).mean()
                )

        # Reverse turnover ratio so a lower ratio
        # corresponds to a higher percentile.
        idx = FEATURES.index("TO_RATIO")
        values[idx] = (
            100 * (valid["TO_RATIO"] >= row["TO_RATIO"]).mean()
        )

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            opacity=0.65,
            name=name,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, range=[0, 100]
        )),
        height=520,
        margin=dict(l=55, r=55, t=30, b=30),
        legend=dict(orientation="h", y=-0.15),
    )

    return fig


# --------------------------------------------------
# HEADER / SIDEBAR
# --------------------------------------------------

st.title("🏀 NBA Offensive Archetype Explorer")

st.caption(
    "2025–26 NBA Regular Season | "
    "Unsupervised Machine Learning | K-means Clustering"
)

st.info(
    "This project identifies statistical offensive profiles. "
    "Archetypes are descriptive, not definitive scouting grades. "
    "Defensive metrics are planned for a future version."
)

st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: K-means")
st.sidebar.write("Clusters: 5")
st.sidebar.write(f"Players: {len(assignments)}")
st.sidebar.write("Features: 9")
st.sidebar.write("Silhouette Score: 0.181")
st.sidebar.caption(
    "Five clusters were selected for basketball "
    "interpretability despite a lower silhouette score "
    "than the three-cluster baseline."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Player Explorer",
    "Archetype Analysis",
    "Player Comparisons",
    "Model Insights",
])


# --------------------------------------------------
# TAB 1: PLAYER EXPLORER
# --------------------------------------------------

with tab1:
    st.header("Player Explorer")

    selected_name = st.selectbox(
        "Select an NBA player",
        sorted(assignments["PLAYER_NAME"].unique()),
    )

    selection = assignments.loc[
        assignments["PLAYER_NAME"] == selected_name
    ].iloc[0]

    cluster = int(selection["CLUSTER"])

    st.subheader(selected_name)
    st.success(f"Offensive Archetype: {archetype_name(cluster)}")
    st.write(DESCRIPTIONS[cluster])

    if players is not None:
        player = player_record(selected_name)

        if player[FEATURES].notna().all():
            st.subheader("Offensive Statistics")

            cols = st.columns(3)

            for i, feature in enumerate(FEATURES):
                with cols[i % 3]:
                    st.metric(
                        LABELS[feature],
                        display_value(feature, player[feature]),
                    )

            st.subheader("Player Statistical Percentiles")
            st.plotly_chart(
                radar_chart([player], [selected_name]),
                use_container_width=True,
            )

            st.caption(
                "Radar values are percentiles among the "
                "eligible players in this dataset. "
                "TO Ratio is reversed because lower is better. "
                "A higher percentile does not always mean "
                "a more desirable playing style."
            )
        else:
            st.warning(
                "Some statistics are missing for this player."
            )
    else:
        st.warning(
            "Add data/processed/offensive_features.csv "
            "to enable individual player statistics."
        )

    st.subheader("Representative Players in This Archetype")

    group = representatives[
        representatives["CLUSTER"] == cluster
    ].sort_values("DISTANCE_TO_CENTROID")

    st.dataframe(
        group[["PLAYER_NAME", "DISTANCE_TO_CENTROID"]]
        .rename(columns={
            "PLAYER_NAME": "Player",
            "DISTANCE_TO_CENTROID": "Centroid Distance",
        }),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Representative players are nearest to the "
        "cluster centroid in standardized feature space."
    )


# --------------------------------------------------
# TAB 2: ARCHETYPE ANALYSIS
# --------------------------------------------------

with tab2:
    st.header("Archetype Analysis")

    selected_cluster = st.selectbox(
        "Select an archetype",
        list(ARCHETYPES.keys()),
        format_func=archetype_name,
    )

    profile = profiles.loc[
        profiles["CLUSTER"] == selected_cluster
    ].iloc[0]

    st.subheader(archetype_name(selected_cluster))
    st.write(DESCRIPTIONS[selected_cluster])

    st.metric("Players in Cluster", int(profile["PLAYER_COUNT"]))

    st.subheader("Average Offensive Statistics")

    display_profile = pd.DataFrame({
        "Metric": [LABELS[f] for f in FEATURES],
        "Cluster Average": [
            display_value(f, profile[f]) for f in FEATURES
        ],
    })

    st.dataframe(
        display_profile,
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Cluster Population")

    population = profiles[["CLUSTER", "PLAYER_COUNT"]].copy()
    population["Archetype"] = (
        population["CLUSTER"].map(ARCHETYPES)
    )

    fig = px.bar(
        population,
        x="Archetype",
        y="PLAYER_COUNT",
        color="Archetype",
        title="Player Distribution Across Archetypes",
        labels={"PLAYER_COUNT": "Number of Players"},
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-25,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Compare Archetype Averages")

    selected_feature = st.selectbox(
        "Choose a metric",
        FEATURES,
        format_func=lambda x: LABELS[x],
    )

    comparison_data = profiles[
        ["CLUSTER", selected_feature]
    ].copy()

    comparison_data["Archetype"] = (
        comparison_data["CLUSTER"].map(ARCHETYPES)
    )

    if selected_feature in PERCENT_FEATURES:
        comparison_data[selected_feature] *= 100

    fig = px.bar(
        comparison_data,
        x="Archetype",
        y=selected_feature,
        color="Archetype",
        labels={
            selected_feature: LABELS[selected_feature],
        },
    )

    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-25,
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# TAB 3: PLAYER COMPARISONS
# --------------------------------------------------

with tab3:
    st.header("Player Comparisons")

    if players is None:
        st.warning(
            "Player comparisons require "
            "data/processed/offensive_features.csv."
        )
    else:
        names = sorted(players["PLAYER_NAME"].unique())

        col1, col2 = st.columns(2)

        with col1:
            player_a = st.selectbox(
                "Player A",
                names,
                index=0,
            )

        with col2:
            player_b = st.selectbox(
                "Player B",
                names,
                index=min(1, len(names) - 1),
            )

        a = player_record(player_a)
        b = player_record(player_b)

        st.write(
            f"**{player_a}:** {archetype_name(a['CLUSTER'])}"
        )
        st.write(
            f"**{player_b}:** {archetype_name(b['CLUSTER'])}"
        )

        comparison_table = pd.DataFrame({
            "Metric": [LABELS[f] for f in FEATURES],
            player_a: [
                display_value(f, a[f]) for f in FEATURES
            ],
            player_b: [
                display_value(f, b[f]) for f in FEATURES
            ],
        })

        st.dataframe(
            comparison_table,
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Offensive Profile Comparison")

        if a[FEATURES].notna().all() and b[FEATURES].notna().all():
            st.plotly_chart(
                radar_chart([a, b], [player_a, player_b]),
                use_container_width=True,
            )
        else:
            st.warning("One player has incomplete statistics.")


# --------------------------------------------------
# TAB 4: MODEL INSIGHTS
# --------------------------------------------------

with tab4:
    st.header("Machine Learning Model Insights")

    st.write(
        "The project uses unsupervised K-means clustering "
        "to discover offensive statistical player profiles."
    )

    if comparison is not None:
        st.subheader("Choosing the Number of Clusters")

        fig = px.line(
            comparison,
            x="k",
            y="silhouette_score",
            markers=True,
            labels={
                "k": "Number of Clusters",
                "silhouette_score": "Silhouette Score",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Model Development")

    st.markdown("""
    1. Collect player-level NBA statistics.
    2. Clean data and create per-36-minute features.
    3. Explore correlations and feature distributions.
    4. Standardize selected offensive statistics.
    5. Train K-means models with different cluster counts.
    6. Evaluate silhouette scores and clustering stability.
    7. Interpret clusters using player statistics.
    8. Perform feature sensitivity and outlier diagnostics.
    """)

    st.subheader("Model Limitations")

    st.markdown("""
    - Statistical profiles can overlap across archetypes.
    - Player performance depends on team context.
    - K-means assigns every player to one cluster.
    - The current model includes offensive metrics only.
    - Archetype labels require basketball interpretation.
    """)

    st.subheader("Future Development: Defensive Metrics")

    st.markdown("""
    Future versions will consider:

    - Defensive rebound percentage (DREB%)
    - Steal percentage (STL%)
    - Block percentage (BLK%)
    - Deflections and contested shots
    - Defensive matchup and tracking statistics

    Defensive metrics will help contextualize archetype
    descriptions, but will not be presented as part of
    the current offensive-only clustering model.

    We will test both separate offensive/defensive
    profiles and combined clustering before deciding
    whether defensive metrics should influence
    final archetype naming.
    """)
