from pathlib import Path

import pandas as pd


DATA = [
    ["C001", "Aarav", 24, "Male", "Hyderabad", 8, 1200, 9600, 9, 1, 0, 1],
    ["C002", "Diya", 31, "Female", "Bengaluru", 14, 850, 11900, 5, 2, 0, 1],
    ["C003", "Vikram", 45, "Male", "Chennai", 3, 300, 900, 1, 4, 1, 0],
    ["C004", "Meera", 28, "Female", "Pune", 22, 1500, 33000, 12, 0, 0, 1],
    ["C005", "Rohan", 38, "Male", "Mumbai", 6, 450, 2700, 2, 3, 1, 0],
    ["C006", "Sana", 26, "Female", "Delhi", 18, 980, 17640, 8, 1, 0, 1],
]


def main() -> None:
    columns = [
        "Customer_ID",
        "Name",
        "Age",
        "Gender",
        "City",
        "Tenure_Months",
        "Monthly_Spend",
        "Total_Spending",
        "Visits_Last_30_Days",
        "Support_Tickets",
        "Churn",
        "Purchased",
    ]
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame(DATA, columns=columns).to_csv(data_dir / "customers.csv", index=False)


if __name__ == "__main__":
    main()
