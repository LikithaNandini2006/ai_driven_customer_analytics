from pathlib import Path

import pandas as pd
import streamlit as st


def recommend(row: pd.Series) -> str:
    if row["Support_Tickets"] >= 3:
        return "Priority Support"
    if row["Total_Spending"] >= 25000:
        return "Premium Bundle"
    if row["Visits_Last_30_Days"] >= 8:
        return "Analytics Add-on"
    if row["Total_Spending"] >= 5000:
        return "Loyalty Upgrade"
    return "Starter Offer"


st.set_page_config(page_title="Smart Recommendations", layout="wide")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "customers.csv")
df["Recommendation"] = df.apply(recommend, axis=1)

st.title("Smart Recommendations")

left, right = st.columns(2)
with left:
    st.subheader("Recommendation Count")
    st.bar_chart(df["Recommendation"].value_counts())
with right:
    st.subheader("Revenue by Recommendation")
    st.bar_chart(df.groupby("Recommendation")["Total_Spending"].sum().sort_values(ascending=False))

st.subheader("Customer Spend Behind Recommendations")
st.bar_chart(df.set_index("Name")["Total_Spending"].sort_values(ascending=False))

st.dataframe(df[["Customer_ID", "Name", "City", "Total_Spending", "Recommendation"]], width="stretch")
