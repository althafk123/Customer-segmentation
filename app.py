
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Mall_Customers.csv")
PLOT_PATH = os.path.join(BASE_DIR, "static", "clusters.png")

INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

# 1. Load the dataset
df = pd.read_csv("Mall_Customers_60_rows.csv", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

# 2. Check and clean the required columns
required_columns = [INCOME_COL, SPENDING_COL]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}. "
            f"Available columns: {list(df.columns)}"
        )

df[required_columns] = df[required_columns].apply(
    pd.to_numeric, errors="coerce"
)

df = df.dropna(subset=required_columns).copy()

if len(df) < 5:
    raise ValueError("At least 5 valid customer records are required.")

# 3. Select features for clustering
X = df[required_columns]

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train the K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Convert cluster centers back to original units
centers = scaler.inverse_transform(kmeans.cluster_centers_)

income_median = df[INCOME_COL].median()
spending_median = df[SPENDING_COL].median()


# 6. Give each cluster an understandable description
def describe_cluster(income, spending):
    if income >= income_median and spending >= spending_median:
        return "High Income, High Spending"
    elif income >= income_median and spending < spending_median:
        return "High Income, Low Spending"
    elif income < income_median and spending >= spending_median:
        return "Low Income, High Spending"
    else:
        return "Low Income, Low Spending"


cluster_names = {}

for cluster_id, center in enumerate(centers):
    cluster_names[cluster_id] = describe_cluster(
        center[0], center[1]
    )

# 7. Create the customer cluster graph
def create_plot():
    plt.figure(figsize=(9, 6))

    for cluster_id in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cluster_id]

        plt.scatter(
            group[INCOME_COL],
            group[SPENDING_COL],
            label=f"Cluster {cluster_id + 1}",
            s=65,
            alpha=0.8
        )

    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        color="black",
        label="Centroids"
    )

    plt.title("Customer Segmentation Using K-Means")
    plt.xlabel("Annual Income (thousands of dollars)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()


os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
create_plot()

# 8. Prepare cluster summaries
def get_summaries():
    summaries = []

    for cluster_id in sorted(df["Cluster"].unique()):
        group = df[df["Cluster"] == cluster_id]

        summaries.append({
            "number": int(cluster_id) + 1,
            "name": cluster_names[cluster_id],
            "count": len(group),
            "income": round(group[INCOME_COL].mean(), 2),
            "spending": round(group[SPENDING_COL].mean(), 2)
        })

    return summaries


# 9. Display the website
@app.route("/")
def home():
    return render_template(
        "index.html",
        summaries=get_summaries(),
        total_customers=len(df),
        result=None,
        income="",
        spending=""
    )


# 10. Predict the cluster for a new customer
@app.route("/predict", methods=["POST"])
def predict():
    try:
        income = float(request.form["income"])
        spending = float(request.form["spending"])

        if not 0 <= income <= 1000:
            raise ValueError("Income must be between 0 and 1000.")

        if not 1 <= spending <= 100:
            raise ValueError("Spending score must be between 1 and 100.")

        new_customer = pd.DataFrame(
            [[income, spending]],
            columns=required_columns
        )

        scaled_customer = scaler.transform(new_customer)
        cluster_id = int(kmeans.predict(scaled_customer)[0])

        result = {
            "number": cluster_id + 1,
            "name": cluster_names[cluster_id],
            "income": income,
            "spending": spending
        }

        return render_template(
            "index.html",
            summaries=get_summaries(),
            total_customers=len(df),
            result=result,
            income=income,
            spending=spending
        )

    except (ValueError, TypeError):
        return render_template(
            "index.html",
            summaries=get_summaries(),
            total_customers=len(df),
            result={"error": "Enter valid numeric values for income and spending."},
            income=request.form.get("income", ""),
            spending=request.form.get("spending", "")
        )


if __name__ == "__main__":
    app.run(debug=True)
