import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any
import time
import re# Added for email validation
from Home_Page import log_activity

os.chdir(r"D:\Project\MeanShift")

# 1. CONFIGURATION & CONSTANTS
CONFIG = {
    "weights": {"recency": 0.2, "frequency": 0.4, "monetary": 0.4, "rfm": 0.8, "income": 0.2},
    "scaler_income_col": "Annual Income (k$)",
    "required_cols": ["RFM Score", "Annual Income (k$)", "Cluster", "Cluster Name"],
    "colors": sns.color_palette("husl", n_colors=6).as_hex(), # 6 clusters max for demo
    "recency_max": 100,
    "quantile_rfm": 0.75,
    "quantile_income": 0.75,
}

# 2. HELPER FUNCTIONS
def require_login() -> None:
    """Stop execution if the user is not logged in."""
    if not st.session_state.get("logged_in", False):
        st.warning("Please log in first!")
        st.stop()

def load_pickle(path: Path) -> Any:
    """Load a pickled object with friendly error handling."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"File not found: `{path}`. Check the `model_files/` folder.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

@st.cache_data
def load_processed_dataset(file) -> pd.DataFrame:
    """Read CSV and validate required columns (cached for speed)."""
    df = pd.read_csv(file)
    missing = [c for c in CONFIG["required_cols"] if c not in df.columns]
    if missing:
        st.error(f"Uploaded file is missing columns: {missing}")
        st.stop()
    return df

def calculate_rfm_score(recency: int, frequency: int, monetary: float) -> float:
    """Weighted RFM score (0-100) – recency is inverted."""
    w = CONFIG["weights"]
    recency_norm = (CONFIG["recency_max"] - recency) / CONFIG["recency_max"] * 100
    return (w["recency"] * recency_norm) + (w["frequency"] * frequency) + (w["monetary"] * monetary)

def weight_features(rfm_score: float, income: float) -> Tuple[float, float]:
    w = CONFIG["weights"]
    return rfm_score * w["rfm"], income * w["income"]

def predict_cluster(scaler, model, rfm_w: float, income_w: float) -> Tuple[int, float]:
    X_new = pd.DataFrame([[rfm_w, income_w]], columns=["RFM Score", CONFIG["scaler_income_col"]])
    X_scaled = scaler.transform(X_new)
    cluster_id = int(model.predict(X_scaled)[0])
    distance = np.linalg.norm(X_scaled - model.cluster_centers_[cluster_id])
    return cluster_id, distance

def calculate_spending_score(recency: int, frequency: int, monetary: float,
                             max_frequency: float, max_monetary: float,
                             income: float, max_income: float) -> int:
    """
    Calculate Spending Score (1-100) blending monetary behavior and income potential.
    Different weighting than RFM to provide unique insights.
    """
    # Normalize monetary (spending behavior)
    monetary_norm = (monetary / max_monetary * 100) if max_monetary > 0 else 0

    # Normalize frequency (purchase frequency)
    frequency_norm = (frequency / max_frequency * 100) if max_frequency > 0 else 0

    # Normalize income (income potential)
    income_norm = (income / max_income * 100) if max_income > 0 else 0

    # Different weights than RFM to distinguish it
    # Emphasize actual spending (monetary) more than income potential
    spending_score = (0.5 * monetary_norm) + (0.3 * frequency_norm) + (0.2 * income_norm)

    # Clamp to 1-100 range
    spending_score = max(1, min(100, int(np.round(spending_score))))

    return spending_score

def recommend_discount(rfm: float, income: float, summary: pd.DataFrame) -> Dict[str, str]:
    high_rfm = summary["RFM Score"].quantile(CONFIG["quantile_rfm"])
    high_inc = summary["Annual Income (k$)"].quantile(CONFIG["quantile_income"])

    if rfm >= high_rfm and income >= high_inc:
        return {"discount": "5-10%", "reason": "Valuable loyal customer – small reward."}
    if rfm >= high_rfm and income < high_inc:
        return {"discount": "15-20%", "reason": "Engaged but price-sensitive – larger incentive."}
    if rfm < high_rfm and income >= high_inc:
        return {"discount": "20-30%", "reason": "High-potential but inactive – strong re-engagement discount."}
    return {"discount": "10-15%", "reason": "Low engagement – modest nudge."}

# 3. PAGE SETUP
st.set_page_config(page_title="Individual Customer Analysis", layout="wide")
st.title("Individual Customer Cluster Analysis")

st.markdown("""Upload the **processed bulk dataset** (exported from the *Customer
Segmentation* page) and then input the metrics of a single customer. The app will assign
the customer to a cluster and suggest a discount strategy.""")

require_login()

# Initialize dataset history
if 'dataset_history' not in st.session_state:
    st.session_state.dataset_history = {}

# 4. LOAD MODELS
@st.cache_resource
def load_models():
    scaler = load_pickle(Path("model_files/scaler.pkl"))
    model = load_pickle(Path("model_files/ms_model.pkl"))
    return scaler, model

scaler, cluster_model = load_models()

# 5. FILE UPLOAD & DATA PREP
uploaded_file = st.file_uploader("Upload Processed Dataset (CSV)", type=["csv"],
help="Must contain RFM, Income, Cluster & Cluster Name")

# --- RESET SESSION IF FILE CLEARED ---
if uploaded_file is None and 'df_clusters' in st.session_state:
    # Clear previous dataset & dependent session keys
    for key in ['df_clusters', 'spending_col']:
        if key in st.session_state:
            del st.session_state[key]
    st.warning("Dataset cleared. Please upload a new processed dataset to continue.")
    st.stop() # Stop the app until new file is uploaded

if uploaded_file:
    new_df = load_processed_dataset(uploaded_file)
    key = f"{uploaded_file.name}_{int(time.time())}"
    st.session_state.dataset_history[key] = new_df
    st.session_state.df_clusters = new_df

    # Add Spending Score if not present
    spending_col = 'Spending Score (1-100)' if 'Spending Score (1-100)' in new_df.columns else 'Spending Score'
    if spending_col not in new_df.columns:
        max_freq = new_df['Frequency (visits)'].max()
        max_monetary = new_df['Monetary ($)'].max()
        max_income = new_df['Annual Income (k$)'].max()

        new_df[spending_col] = new_df.apply(
            lambda row: calculate_spending_score(
                row['Recency (days)'],
                row['Frequency (visits)'],
                row['Monetary ($)'],
                max_freq,
                max_monetary,
                row['Annual Income (k$)'],
                max_income
            ), axis=1
        )
    st.session_state.spending_col = spending_col

    # Log the upload
    log_activity(st.session_state["username"], "dataset_upload", {
        "file_name": uploaded_file.name,
        "records": len(new_df)
    })

# Load current dataset from session
if 'df_clusters' not in st.session_state:
    st.warning("Please upload the processed dataset to continue.")
    st.stop()

df_clusters = st.session_state.df_clusters
spending_col = st.session_state.spending_col

# Extract Cluster → Name mapping
cluster_names: Dict[int, str] = df_clusters[["Cluster", "Cluster Name"]].drop_duplicates().set_index("Cluster")["Cluster Name"].to_dict()

@st.cache_data
def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Cluster")[["RFM Score", "Annual Income (k$)"]].mean().reset_index()

cluster_summary_df = cluster_summary(df_clusters)

# 6. USER INPUTS
st.subheader("Customer Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    default_id = int(df_clusters['CustomerID'].max() + 1) if 'CustomerID' in df_clusters.columns else 0
    st.markdown("CustomerID *")
    customer_id = st.number_input("CustomerID Input", min_value=0, value=default_id, step=1, label_visibility="collapsed")
    st.markdown("Recency (days since last purchase) *")
    recency = st.number_input("Recency Input", min_value=0, max_value=CONFIG["recency_max"], value=50, step=1, label_visibility="collapsed")

with col2:
    st.markdown("Gender *")
    gender = st.selectbox("Gender Input", ["Male", "Female"], label_visibility="collapsed")
    st.markdown("Frequency (total purchases) *")
    frequency = st.number_input("Frequency Input", min_value=0, value=10, step=1, label_visibility="collapsed")
    st.markdown("Monetary (avg spend per visit, $) *")
    monetary = st.number_input("Monetary Input", min_value=0.0, value=30.0, step=0.5, label_visibility="collapsed")

with col3:
    st.markdown("Mail ID *")
    mail_id = st.text_input("Mail ID Input", label_visibility="collapsed")
    st.markdown("Age *")
    age = st.number_input("Age Input", min_value=0, value=30, step=1, label_visibility="collapsed")
    st.markdown("Annual Income (k$) *")
    annual_income = st.number_input("Annual Income Input", min_value=0.0, value=75.0, step=5.0, label_visibility="collapsed")

# 7. ANALYSIS BUTTON
if st.button("Analyze Customer", type="primary"):
    # Validation checks
    valid = True
    if customer_id <= 0:
        st.error("CustomerID must be greater than 0.")
        valid = False
    if recency < 0:
        st.error("Recency must be at least 0.")
        valid = False
    if frequency < 0:
        st.error("Frequency must be at least 0.")
        valid = False
    if monetary < 0:
        st.error("Monetary must be at least 0.")
        valid = False
    if annual_income < 0:
        st.error("Annual Income must be at least 0.")
        valid = False
    if not mail_id:
        st.error("Mail ID is required.")
        valid = False
    elif not re.match(r"[^@]+@[^@]+\.[^@]+", mail_id): # Basic email validation
        st.error("Please enter a valid Mail ID (e.g., example@domain.com).")
        valid = False
    if age < 18: 
        st.error("Age must be 18 or above.")
        valid = False
    if not gender:
        st.error("Gender is required.")
        valid = False

    if valid:
        rfm_raw = calculate_rfm_score(recency, frequency, monetary)
        rfm_w, income_w = weight_features(rfm_raw, annual_income)

        # Spending Score
        max_freq = df_clusters['Frequency (visits)'].max()
        max_monetary = df_clusters['Monetary ($)'].max()
        max_income = df_clusters['Annual Income (k$)'].max()

        spending_score = calculate_spending_score(
            recency, frequency, monetary,
            max_freq, max_monetary,
            annual_income, max_income
        )

        cluster_id, dist = predict_cluster(scaler, cluster_model, rfm_w, income_w)
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        # Display
        st.subheader("Customer Analysis Result")
        st.success(f"**Assigned Cluster:** {cluster_name} (ID: {cluster_id})")

        if "Passive" in cluster_name:
            profile = "High income but low activity – valuable but needs reactivation."
        elif "Loyal" in cluster_name:
            profile = "Engaged and high-value – consistent purchasing behavior."
        elif "Price-Sensitive" in cluster_name:
            profile = "Frequent buyer but responds strongly to discounts."
        else:
            profile = "Typical customer profile for this cluster."

        st.markdown(f"**Profile Insight:** {profile}")
        st.markdown(f"**RFM Score:** {rfm_raw}")
        st.markdown(f"**Spending Score:** {spending_score}")

        with st.expander("View detailed metrics"):
            st.info(
                f"Weighted RFM: `{rfm_w:.2f}` | Weighted Income: `{income_w:.2f}` | "
                f"Distance to centroid: `{dist:.4f}`"
            )

        # Cluster Plot
        st.subheader("Cluster Visualisation")
        fig, ax = plt.subplots(figsize=(10, 6))
        for cid, name in cluster_names.items():
            data = df_clusters[df_clusters["Cluster"] == cid]
            ax.scatter(data["RFM Score"] * 0.8, data["Annual Income (k$)"] * 0.2, label=name, color=CONFIG["colors"][cid % len(CONFIG["colors"])], alpha=0.6, s=80)
        ax.scatter(rfm_w, income_w, c="black", s=250, marker="X", label="New Customer", edgecolors="white", linewidth=1.5)
        ax.set_xlabel("Weighted RFM Score")
        ax.set_ylabel("Weighted Annual Income (k$)")
        ax.set_title("Customer clusters with the new entry")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

        # Similar Customers
        st.subheader("Similar Customers")
        df_similar = df_clusters.copy()
        df_similar['Weighted RFM'] = df_similar['RFM Score'] * 0.8
        df_similar['Weighted Income'] = df_similar['Annual Income (k$)'] * 0.2

        # Calculate distance including Spending Score similarity
        spending_score_norm = spending_score / 100.0
        df_similar['Spending Score Norm'] = df_similar[spending_col] / 100.0

        df_similar['Distance'] = np.sqrt(
            (df_similar['Weighted RFM'] - rfm_w)**2 +
            (df_similar['Weighted Income'] - income_w)**2 +
            (df_similar['Spending Score Norm'] - spending_score_norm)**2
        )
        df_similar = df_similar.sort_values('Distance')
        st.dataframe(df_similar.head(5).drop(columns=['Weighted RFM', 'Weighted Income', 'Spending Score Norm', 'Distance']))

        # Discount Recommendation
        st.subheader("Discount Recommendation")
        recommendation = recommend_discount(rfm_raw, annual_income, cluster_summary_df)
        st.success(
            f"Suggested Discount: **{recommendation['discount']}** \nReason: "
            f"{recommendation['reason']}"
        )

        with st.expander("See cluster averages for context"):
            st.dataframe(cluster_summary_df.style.format({"RFM Score": "{:.2f}", "Annual Income (k$)": "{:.2f}"}))

        # Log the analysis
        log_activity(st.session_state["username"], "customer_analyzed", {
            "customer_id": customer_id,
            "cluster": cluster_name
        })

# Add Customer to Dataset
if st.button("Add Customer to Dataset"):
    # Reuse the same validation as above
    valid = True
    if customer_id <= 0:
        st.error("CustomerID must be greater than 0.")
        valid = False
    if recency < 0:
        st.error("Recency must be at least 0.")
        valid = False
    if frequency < 0:
        st.error("Frequency must be at least 0.")
        valid = False
    if monetary < 0:
        st.error("Monetary must be at least 0.")
        valid = False
    if annual_income < 0:
        st.error("Annual Income must be at least 0.")
        valid = False
    if not mail_id:
        st.error("Mail ID is required.")
        valid = False
    elif not re.match(r"[^@]+@[^@]+\.[^@]+", mail_id):
        st.error("Please enter a valid Mail ID (e.g., example@domain.com).")
        valid = False
    if age < 18:
        st.error("Age must be 18 or above.")
        valid = False
    if not gender:
        st.error("Gender is required.")
        valid = False

    if valid:
        rfm_raw = calculate_rfm_score(recency, frequency, monetary)
        rfm_w, income_w = weight_features(rfm_raw, annual_income)
        cluster_id, _ = predict_cluster(scaler, cluster_model, rfm_w, income_w)
        cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        # Spending Score
        max_freq = st.session_state.df_clusters['Frequency (visits)'].max()
        max_monetary = st.session_state.df_clusters['Monetary ($)'].max()
        max_income = st.session_state.df_clusters['Annual Income (k$)'].max()

        spending_score = calculate_spending_score(
            recency, frequency, monetary,
            max_freq, max_monetary,
            annual_income, max_income
        )

        new_row = pd.DataFrame(index=[0])
        for col, val in {'CustomerID': customer_id, 'Gender': gender, 'Mail id': mail_id, 'Age': age,
                         'Recency (days)': recency, 'Frequency (visits)': frequency, 'Monetary ($)': monetary,
                         'Annual Income (k$)': annual_income}.items():
            if col in df_clusters.columns:
                new_row[col] = val
        new_row['RFM Score'] = rfm_raw
        new_row['Cluster'] = cluster_id
        new_row['Cluster Name'] = cluster_name
        new_row[spending_col] = spending_score

        st.session_state.df_clusters = pd.concat([st.session_state.df_clusters, new_row],ignore_index=True)

        # Recalculate Spending Score for entire dataset with updated max values
        max_freq = st.session_state.df_clusters['Frequency (visits)'].max()
        max_monetary = st.session_state.df_clusters['Monetary ($)'].max()
        max_income = st.session_state.df_clusters['Annual Income (k$)'].max()

        st.session_state.df_clusters[spending_col] = st.session_state.df_clusters.apply(
            lambda row: calculate_spending_score(
                row['Recency (days)'],
                row['Frequency (visits)'],
                row['Monetary ($)'],
                max_freq,
                max_monetary,
                row['Annual Income (k$)'],
                max_income
            ), axis=1
        )

        st.success("Customer added to the dataset. You can download the updated dataset below.")

        # Log the addition
        log_activity(st.session_state["username"], "customer_added", {
            "customer_id": customer_id
        })

# Download updated dataset
st.subheader("Download Updated Dataset")
updated_csv = st.session_state.df_clusters.to_csv(index=False).encode('utf-8')
st.download_button(label="Download Updated Processed Data", data=updated_csv,
file_name="updated_processed_customer_data.csv", mime="text/csv")