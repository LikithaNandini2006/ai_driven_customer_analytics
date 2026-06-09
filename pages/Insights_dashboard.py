from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Insights Dashboard", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")

st.title("Insights Dashboard")
best_city = df.groupby("City")["Total_Spending"].sum().idxmax()
st.info(f"Highest revenue city: {best_city}")
st.bar_chart(df.groupby("Gender")["Monthly_Spend"].mean())
