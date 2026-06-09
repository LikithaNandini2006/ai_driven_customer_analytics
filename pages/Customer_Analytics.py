from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Customer Analytics", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")

st.title("Customer Analytics")
st.scatter_chart(df, x="Tenure_Months", y="Total_Spending", color="Churn")
st.dataframe(df.sort_values("Total_Spending", ascending=False), width="stretch")
