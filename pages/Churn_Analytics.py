from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Churn Analytics", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")

st.title("Churn Analytics")
st.metric("Churn Rate", f"{df['Churn'].mean() * 100:.1f}%")

df["Churn_Risk_Score"] = (
    df["Support_Tickets"] * 18
    + (df["Monthly_Spend"].max() - df["Monthly_Spend"]) / df["Monthly_Spend"].max() * 35
    + (df["Visits_Last_30_Days"].max() - df["Visits_Last_30_Days"]) / df["Visits_Last_30_Days"].max() * 25
).clip(0, 100).round(1)

left, right = st.columns(2)
with left:
    st.subheader("Actual Churn Count")
    st.bar_chart(df["Churn"].map({0: "No", 1: "Yes"}).value_counts())
with right:
    st.subheader("Average Risk by City")
    st.bar_chart(df.groupby("City")["Churn_Risk_Score"].mean().sort_values(ascending=False))

st.subheader("Customer Churn Risk Score")
st.bar_chart(df.set_index("Name")["Churn_Risk_Score"].sort_values(ascending=False))

st.dataframe(
    df[["Customer_ID", "Name", "City", "Support_Tickets", "Total_Spending", "Churn_Risk_Score", "Churn"]],
    width="stretch",
)
