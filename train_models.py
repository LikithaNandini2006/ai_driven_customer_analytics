from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(BASE_DIR / "data" / "customers.csv")
    features = df.select_dtypes(include="number").drop(columns=["Churn", "Purchased"], errors="ignore")
    scaled = StandardScaler().fit_transform(features)
    KMeans(n_clusters=3, random_state=42, n_init="auto").fit(scaled)
    RandomForestClassifier(random_state=42).fit(features, df["Churn"])
    RandomForestClassifier(random_state=42).fit(features, df["Purchased"])
    print("Models trained successfully.")


if __name__ == "__main__":
    main()
