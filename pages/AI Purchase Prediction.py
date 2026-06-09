from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="AI Purchase Prediction", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")

st.title("AI Purchase Prediction")
st.metric("Purchase Rate", f"{df['Purchased'].mean() * 100:.1f}%")

df["Purchase_Score"] = (
    df["Visits_Last_30_Days"] / df["Visits_Last_30_Days"].max() * 40
    + df["Monthly_Spend"] / df["Monthly_Spend"].max() * 35
    + df["Total_Spending"] / df["Total_Spending"].max() * 25
).clip(0, 100).round(1)

left, right = st.columns(2)
with left:
    st.subheader("Actual Purchase Count")
    st.bar_chart(df["Purchased"].map({0: "No", 1: "Yes"}).value_counts())
with right:
    st.subheader("Average Purchase Score by City")
    st.bar_chart(df.groupby("City")["Purchase_Score"].mean().sort_values(ascending=False))

st.subheader("Customer Purchase Score")
st.bar_chart(df.set_index("Name")["Purchase_Score"].sort_values(ascending=False))

st.dataframe(
    df[["Customer_ID", "Name", "Visits_Last_30_Days", "Monthly_Spend", "Purchase_Score", "Purchased"]],
    width="stretch",
)
