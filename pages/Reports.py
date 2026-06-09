from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Reports", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")

st.title("Reports")
st.download_button(
    "Download customer report",
    df.to_csv(index=False),
    file_name="customer_report.csv",
    mime="text/csv",
)
st.dataframe(df.describe(include="all"), width="stretch")
