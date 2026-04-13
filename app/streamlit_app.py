"""
Forging Line — Piece Travel Time Dashboard

Displays processed pieces with predicted bath time and per-stage
timing detail.

Usage:
    uv run streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vaultech_analysis.inference import Predictor

GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "pieces.parquet"

# Column definitions — process order
PARTIAL_COLS = [
    "partial_furnace_to_2nd_s",
    "partial_2nd_to_3rd_s",
    "partial_3rd_to_4th_s",
    "partial_4th_to_aux_s",
    "partial_aux_to_bath_s",
]

PARTIAL_LABELS = [
    "Furnace → 2nd strike",
    "2nd strike → 3rd strike",
    "3rd strike → 4th strike",
    "4th strike → Aux. press",
    "Aux. press → Bath",
]
CUMULATIVE_COLS = [
    "lifetime_2nd_strike_s",
    "lifetime_3rd_strike_s",
    "lifetime_4th_strike_s",
    "lifetime_auxiliary_press_s",
    "lifetime_bath_s",
]
CUMULATIVE_LABELS = [
    "2nd strike (1st op)",
    "3rd strike (2nd op)",
    "4th strike (drill)",
    "Auxiliary press",
    "Bath",
]


@st.cache_resource
def load_predictor():
    return Predictor()

@st.cache_data
def load_data():
    df = pd.read_parquet(GOLD_FILE)

    # Ensure partial columns exist (older gold exports may be missing them)
    if not all(col in df.columns for col in PARTIAL_COLS):
        df["partial_furnace_to_2nd_s"] = df["lifetime_2nd_strike_s"]
        df["partial_2nd_to_3rd_s"] = df["lifetime_3rd_strike_s"] - df["lifetime_2nd_strike_s"]
        df["partial_3rd_to_4th_s"] = df["lifetime_4th_strike_s"] - df["lifetime_3rd_strike_s"]
        df["partial_4th_to_aux_s"] = df["lifetime_auxiliary_press_s"] - df["lifetime_4th_strike_s"]
        df["partial_aux_to_bath_s"] = df["lifetime_bath_s"] - df["lifetime_auxiliary_press_s"]

    predictor = load_predictor()
    df["predicted_bath_s"] = predictor.predict_batch(df)
    df["prediction_error_s"] = df["predicted_bath_s"] - df["lifetime_bath_s"]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def get_reference(df: pd.DataFrame):
    return df.groupby("die_matrix")[PARTIAL_COLS + CUMULATIVE_COLS].median()


st.set_page_config(page_title="Forging Line Dashboard", layout="wide")
st.title("Forging Line — Piece Travel Time Dashboard")

df = load_data()
reference = get_reference(df)

st.sidebar.header("Filters")

matrices = sorted(df["die_matrix"].unique().tolist())
selected_matrices = st.sidebar.multiselect("Die matrix", matrices, default=matrices)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
date_range = st.sidebar.date_input("Date range", (min_date, max_date))

show_slow_only = st.sidebar.checkbox("Show slow pieces only", value=False)

filtered = df[df["die_matrix"].isin(selected_matrices)].copy()
filtered = filtered[(filtered["timestamp"].dt.date >= date_range[0]) &
                    (filtered["timestamp"].dt.date <= date_range[1])]

if show_slow_only:
    # slow = top 10% bath time per matrix
    p90 = filtered.groupby("die_matrix")["lifetime_bath_s"].quantile(0.90)
    filtered = filtered[filtered.apply(lambda r: r["lifetime_bath_s"] > p90[r["die_matrix"]], axis=1)]

# Summary metrics
total_pieces = len(filtered)
median_bath = filtered["lifetime_bath_s"].median()
median_pred = filtered["predicted_bath_s"].median()
mae = (filtered["prediction_error_s"].abs()).median()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pieces", f"{total_pieces:,}")
c2.metric("Median bath (s)", f"{median_bath:.2f}")
c3.metric("Median predicted (s)", f"{median_pred:.2f}")
c4.metric("Median abs error (s)", f"{mae:.2f}")

table_cols = [
    "timestamp", "piece_id", "die_matrix",
    "lifetime_bath_s", "predicted_bath_s", "prediction_error_s",
    "oee_cycle_time_s"
]

st.subheader("Pieces")
selection = st.dataframe(
    filtered[table_cols].sort_values("timestamp", ascending=False),
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row",
)

selected_idx = None
if hasattr(selection, "selection") and selection.selection.get("rows"):
    selected_idx = selection.selection["rows"][0]


if selected_idx is not None:
    selected_row = filtered.iloc[selected_idx]
    matrix = selected_row["die_matrix"]

    st.subheader("Piece Detail")

    # Reference medians for this matrix
    ref = reference.loc[matrix]

    # Cumulative comparison
    cum_df = pd.DataFrame({
        "Stage": CUMULATIVE_LABELS,
        "Actual": [selected_row[c] for c in CUMULATIVE_COLS],
        "Reference": [ref[c] for c in CUMULATIVE_COLS],
    })
    cum_df["Deviation"] = cum_df["Actual"] - cum_df["Reference"]

    # Partial comparison
    part_df = pd.DataFrame({
        "Segment": PARTIAL_LABELS,
        "Actual": [selected_row[c] for c in PARTIAL_COLS],
        "Reference": [ref[c] for c in PARTIAL_COLS],
    })
    part_df["Deviation"] = part_df["Actual"] - part_df["Reference"]
    part_df["Status"] = part_df["Deviation"].apply(lambda x: "Slow" if x > 0 else "OK")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cumulative Times vs Reference**")
        st.dataframe(cum_df, use_container_width=True)
    with c2:
        st.markdown("**Partial Times vs Reference**")
        st.dataframe(part_df, use_container_width=True)

    st.markdown("**Actual vs Reference Partial Times**")
    chart_df = part_df.set_index("Segment")[["Actual", "Reference"]]
    st.bar_chart(chart_df)
else:
    st.info("Select a piece from the table above to see its per-stage timing detail.")

