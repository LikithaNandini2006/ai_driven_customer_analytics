from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
STYLE_PATH = BASE_DIR / "assets" / "styles.css"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"


st.set_page_config(
    page_title="AI Customer Intelligence",
    page_icon="AI",
    layout="wide",
)


def load_styles() -> None:
    if STYLE_PATH.exists() and STYLE_PATH.stat().st_size > 0:
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    if data.empty:
        return data

    aliases = {
        "Customer_ID": "customer_id",
        "Name": "name",
        "Age": "age",
        "Gender": "gender",
        "City": "city",
        "Monthly_Spend": "monthly_spend",
        "Total_Spending": "total_spending",
        "Visits_Last_30_Days": "visits_last_30_days",
        "Support_Tickets": "support_tickets",
        "Churn": "churn",
        "Purchased": "purchased",
    }
    for source, target in aliases.items():
        if source in data.columns and target not in data.columns:
            data[target] = data[source]
    return data


def numeric_features(df: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    features = df.select_dtypes(include="number").drop(columns=list(excluded), errors="ignore")
    if features.empty:
        raise ValueError("At least one numeric feature column is required.")
    return features.fillna(features.median(numeric_only=True))


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    features = numeric_features(df, {"churn", "purchased"})
    result = df.copy()
    cluster_count = min(3, len(result))
    if cluster_count < 2:
        result["segment"] = "Core"
        return result

    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto").fit_predict(scaled)
    segment_names = {0: "Growth", 1: "Premium", 2: "At Risk"}
    result["segment"] = [segment_names.get(label, f"Segment {label + 1}") for label in labels]
    return result


def train_classifier(df: pd.DataFrame, target: str) -> tuple[float, pd.Series, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"{target} column is missing.")

    features = numeric_features(df, {"churn", "purchased"})
    labels = df[target].astype(int)
    model = RandomForestClassifier(n_estimators=120, random_state=42)

    if len(df) < 6 or labels.nunique() < 2:
        model.fit(features, labels)
        predictions = pd.Series(model.predict(features), index=df.index)
        probabilities = pd.Series(model.predict_proba(features)[:, -1], index=df.index)
        return 1.0, predictions, probabilities

    stratify = labels if labels.value_counts().min() > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    predictions = pd.Series(model.predict(features), index=df.index)
    probabilities = pd.Series(model.predict_proba(features)[:, -1], index=df.index)
    return accuracy, predictions, probabilities


def recommend(row: pd.Series) -> str:
    if row.get("support_tickets", 0) >= 3:
        return "Priority Support"
    if row.get("total_spending", 0) >= 25000:
        return "Premium Bundle"
    if row.get("visits_last_30_days", 0) >= 8:
        return "Analytics Add-on"
    if row.get("total_spending", 0) >= 5000:
        return "Loyalty Upgrade"
    return "Starter Offer"


load_styles()

with st.sidebar:
    if LOGO_PATH.exists() and LOGO_PATH.stat().st_size > 0:
        st.image(str(LOGO_PATH), width=140)
    st.markdown("## AI Customer Intelligence")
    st.caption("Advanced Customer Analytics Platform")
    st.divider()

customers = load_data()
if customers.empty:
    st.error("Customer data empty undi. `data/customers.csv` lo rows add chey.")
    st.stop()

required = {"customer_id", "name", "city", "gender", "monthly_spend", "total_spending", "churn", "purchased"}
missing = sorted(required - set(customers.columns))
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

st.markdown(
    """
    <div class="app-header">
      <div class="app-kicker">AI Driven Project</div>
      <h1>AI Customer Intelligence Dashboard</h1>
      <p class="app-subtitle">Analytics, churn signals, purchase prediction, and recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

view = st.sidebar.radio(
    "View",
    ["Dashboard", "Customer Segmentation", "Churn Prediction", "Purchase Prediction", "Recommendations", "Raw Data"],
)

total_customers = len(customers)
total_revenue = customers["total_spending"].sum()
avg_spend = customers["monthly_spend"].mean()
retention_rate = (1 - customers["churn"].mean()) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{total_customers:,}")
k2.metric("Total Revenue", f"{total_revenue:,.0f}")
k3.metric("Average Monthly Spend", f"{avg_spend:,.0f}")
k4.metric("Retention Rate", f"{retention_rate:.1f}%")

st.divider()

if view == "Dashboard":
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Revenue by City")
        st.bar_chart(customers.groupby("city")["total_spending"].sum().sort_values(ascending=False))
    with right:
        st.subheader("Gender Distribution")
        st.bar_chart(customers["gender"].value_counts())

    trend_left, trend_right = st.columns(2)
    with trend_left:
        st.subheader("Spend vs Visits")
        st.scatter_chart(customers, x="visits_last_30_days", y="monthly_spend", color="churn")
    with trend_right:
        st.subheader("Churn by City")
        st.bar_chart(customers.groupby("city")["churn"].mean().sort_values(ascending=False))

    st.subheader("Top Customers")
    st.dataframe(
        customers[["customer_id", "name", "city", "monthly_spend", "total_spending", "churn"]]
        .sort_values("total_spending", ascending=False)
        .head(10),
        width="stretch",
    )

elif view == "Customer Segmentation":
    segmented = segment_customers(customers)
    st.subheader("AI Customer Segments")
    seg_left, seg_right = st.columns(2)
    with seg_left:
        st.caption("Segment Count")
        st.bar_chart(segmented["segment"].value_counts())
    with seg_right:
        st.caption("Average Spend by Segment")
        st.bar_chart(segmented.groupby("segment")["total_spending"].mean().sort_values(ascending=False))

    st.subheader("Segment Position Map")
    st.scatter_chart(segmented, x="monthly_spend", y="total_spending", color="segment")

    st.dataframe(
        segmented[["customer_id", "name", "city", "total_spending", "monthly_spend", "segment"]],
        width="stretch",
    )

elif view == "Churn Prediction":
    accuracy, predictions, probabilities = train_classifier(customers, "churn")
    output = customers[["customer_id", "name", "city", "support_tickets", "total_spending"]].copy()
    output["predicted_churn"] = predictions.map({0: "No", 1: "Yes"})
    output["churn_risk_score"] = (probabilities * 100).round(1)
    st.subheader("Churn Prediction")
    st.success(f"Model accuracy: {accuracy:.2f}")

    churn_left, churn_right = st.columns(2)
    with churn_left:
        st.caption("Predicted Churn Count")
        st.bar_chart(output["predicted_churn"].value_counts())
    with churn_right:
        st.caption("Average Churn Risk by City")
        st.bar_chart(output.groupby("city")["churn_risk_score"].mean().sort_values(ascending=False))

    st.subheader("Customer Churn Risk Score")
    st.bar_chart(output.set_index("name")["churn_risk_score"].sort_values(ascending=False))

    st.dataframe(output, width="stretch")

elif view == "Purchase Prediction":
    accuracy, predictions, probabilities = train_classifier(customers, "purchased")
    output = customers[["customer_id", "name", "city", "visits_last_30_days", "monthly_spend"]].copy()
    output["predicted_purchase"] = predictions.map({0: "No", 1: "Yes"})
    output["purchase_score"] = (probabilities * 100).round(1)
    st.subheader("Purchase Prediction")
    st.success(f"Model accuracy: {accuracy:.2f}")

    purchase_left, purchase_right = st.columns(2)
    with purchase_left:
        st.caption("Predicted Purchase Count")
        st.bar_chart(output["predicted_purchase"].value_counts())
    with purchase_right:
        st.caption("Average Purchase Score by City")
        st.bar_chart(output.groupby("city")["purchase_score"].mean().sort_values(ascending=False))

    st.subheader("Customer Purchase Score")
    st.bar_chart(output.set_index("name")["purchase_score"].sort_values(ascending=False))

    st.dataframe(output, width="stretch")

elif view == "Recommendations":
    output = customers[["customer_id", "name", "city", "total_spending", "support_tickets"]].copy()
    output["recommended_product"] = customers.apply(recommend, axis=1)
    st.subheader("Smart Recommendations")

    rec_left, rec_right = st.columns(2)
    with rec_left:
        st.caption("Recommendation Count")
        st.bar_chart(output["recommended_product"].value_counts())
    with rec_right:
        st.caption("Revenue Potential by Recommendation")
        st.bar_chart(output.groupby("recommended_product")["total_spending"].sum().sort_values(ascending=False))

    st.subheader("Customer Spending for Recommended Products")
    st.bar_chart(output.set_index("name")["total_spending"].sort_values(ascending=False))

    st.dataframe(output, width="stretch")

else:
    st.subheader("Raw Customer Data")
    raw_left, raw_right = st.columns(2)
    with raw_left:
        st.caption("Monthly Spend by Customer")
        st.bar_chart(customers.set_index("name")["monthly_spend"].sort_values(ascending=False))
    with raw_right:
        st.caption("Total Spending by Customer")
        st.bar_chart(customers.set_index("name")["total_spending"].sort_values(ascending=False))

    st.dataframe(customers, width="stretch")
