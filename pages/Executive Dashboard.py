from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Executive Dashboard", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")

st.title("Executive Dashboard")
cols = st.columns(4)
cols[0].metric("Customers", len(df))
cols[1].metric("Revenue", f"₹{df['Total_Spending'].sum():,.0f}")
cols[2].metric("Avg Spend", f"₹{df['Monthly_Spend'].mean():,.0f}")
cols[3].metric("Churn", f"{df['Churn'].mean() * 100:.1f}%")

st.bar_chart(df.groupby("City")["Total_Spending"].sum().sort_values(ascending=False))
