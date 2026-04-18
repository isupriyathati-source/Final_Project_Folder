import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any
import time
import os
import re  # Added for email validation
try:
    from Home_Page import log_activity
except Exception:
    # Fallback when running the page in isolation (or if Home_Page isn't on PYTHONPATH)
    def log_activity(*args, **kwargs):
        return None
from sklearn.neighbors import NearestNeighbors

# Replace chdir with safe path - FIXED absolute path (models are in root Model_Files)
MODEL_DIR = Path(r"D:\Project\DBSCAN\Model_Files")

# 1. CONFIGURATION & CONSTANTS
CONFIG = {
    "weights": {"recency": 0.2, "frequency": 0.4, "monetary": 0.4, "rfm": 0.8, "income": 0.2},
    "scaler_income_col": "Annual Income (k$)",
    "required_cols": ["RFM Score", "Annual Income (k$)", "Cluster", "Cluster Name"],
    "colors": sns.color_palette("husl", n_colors=6).as_hex(), # 6 clusters max for demo
    "recency_max": 100,
    "quantile_rfm": 0.75,
    "quantile_income": 0.75,
    "dbscan_eps": 0.5, # Default eps value for DBSCAN
    "dbscan_min_samples": 5, # Default min_samples for DBSCAN
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
        st.error(f"File not found: `{path}`. Check the `Model_Files/` folder.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()
@st.cache_data
def load_processed_dataset(file) -> pd.DataFrame:
    """Read CSV and validate required columns (cached for speed)."""
    df = pd.read_csv(file)
    # Drop unnamed index columns that appear when saving from pandas
    unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    # Attempt to normalize and map common alternate column names
    df, _rename_map = normalize_and_map_columns(df)
    # If required columns are still missing, show available columns clearly
    missing = [c for c in CONFIG["required_cols"] if c not in df.columns]
    if missing:
        avail = df.columns.tolist()
        st.error(
            "Uploaded file is missing required columns: %s.\nAvailable columns: %s"
            % (missing, avail)
        )
        st.stop()
    return df
def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to find and rename common alternate column names to the canonical names.
    This helps when uploaded CSVs use slightly different headings like
    'Annual Income', 'Annual Income (k)', 'annual income (k$)' etc.
    """
    # Create a helper that normalizes a column string for matching
    def keyify(s: str) -> str:
        return re.sub(r'[^a-z0-9]', '', str(s).lower())
    col_keys = {col: keyify(col) for col in df.columns}
    # Expanded mapping of canonical_name -> list of keywords to match
    candidates = {
        "CustomerID": ["customer", "id"],
        "Gender": ["gender"],
        "Age": ["age"],
        "Annual Income (k$)": ["annual", "income"],
        "Recency (days)": ["recency", "day"],
        "Frequency (visits)": ["frequency", "visit"],
        "Monetary ($)": ["monetary", "money", "spend"],
        "Spending Score (1-100)": ["spending", "score"],
        "Mail id": ["mail", "email"],
        "RFM Score": ["rfm", "score"],
        "Cluster": ["cluster"],
        "Cluster Name": ["cluster", "name"],
    }
    rename_map = {}
    for canon, keys in candidates.items():
        # if canonical already present, skip
        if canon in df.columns:
            continue
        for col, k in col_keys.items():
            if all(key in k for key in keys):
                rename_map[col] = canon
                break
    # Special-case common alternate cluster column names
    if "DBSCAN_Cluster" in df.columns and "Cluster" not in df.columns:
        rename_map["DBSCAN_Cluster"] = "Cluster"
    if "cluster" in df.columns and "Cluster" not in df.columns:
        rename_map["cluster"] = "Cluster"
    if "ClusterLabel" in df.columns and "Cluster" not in df.columns:
        rename_map["ClusterLabel"] = "Cluster"
    # Normalize spending score name to the longer variant if possible
    if "Spending Score" in df.columns and "Spending Score (1-100)" not in df.columns:
        rename_map["Spending Score"] = "Spending Score (1-100)"
    if "spending_score" in df.columns and "Spending Score (1-100)" not in df.columns:
        rename_map["spending_score"] = "Spending Score (1-100)"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, rename_map
def build_column_map(df: pd.DataFrame) -> Dict[str, str]:
    """Return a mapping of canonical keys to actual dataframe column names (or None)."""
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        # try fuzzy keyify match
        keys = {str(col).lower(): col for col in df.columns}
        for c in cands:
            kc = re.sub(r'[^a-z0-9]', '', c.lower())
            for k, orig in keys.items():
                if kc == re.sub(r'[^a-z0-9]', '', k) or kc in re.sub(r'[^a-z0-9]', '', k) or re.sub(r'[^a-z0-9]', '', k) in kc:
                    return orig
        return None
    return {
        'CustomerID': pick('CustomerID', 'customerid'),
        'Recency': pick('Recency (days)', 'recency', 'Recency'),
        'Frequency': pick('Frequency (visits)', 'frequency', 'Frequency'),
        'Monetary': pick('Monetary ($)', 'monetary', 'Monetary'),
        'Income': pick('Annual Income (k$)', 'Annual Income', 'annual income'),
        'RFM Score': pick('RFM Score', 'RFMScore', 'rfm_score'),
        'Spending': pick('Spending Score (1-100)', 'Spending Score', 'spending_score'),
        'Mail id': pick('Mail id', 'mail_id', 'Mail ID', 'mail'),
        'Cluster': pick('Cluster', 'DBSCAN_Cluster', 'cluster', 'ClusterLabel'),
        'Cluster Name': pick('Cluster Name', 'cluster_name', 'ClusterName'),
        'Gender': pick('Gender', 'gender'),
        'Age': pick('Age', 'age')
    }
def calculate_rfm_score(recency: int, frequency: int, monetary: float) -> float:
    """Weighted RFM score (0-100) – recency is inverted."""
    w = CONFIG["weights"]
    recency_norm = (CONFIG["recency_max"] - recency) / CONFIG["recency_max"] * 100
    return (w["recency"] * recency_norm) + (w["frequency"] * frequency) + (w["monetary"] * monetary)
def weight_features(rfm_score: float, income: float) -> Tuple[float, float]:
    w = CONFIG["weights"]
    return rfm_score * w["rfm"], income * w["income"]
def predict_cluster(scaler, model, rfm_w: float, income_w: float) -> Tuple[int, float]:
    """
    For DBSCAN, we need to find the nearest core point to determine the cluster.
    Returns the cluster ID and distance to nearest core point.
    """
    X_new = pd.DataFrame([[rfm_w, income_w]], columns=["RFM Score", CONFIG["scaler_income_col"]])
    X_scaled = scaler.transform(X_new)
   
    # For DBSCAN, we need to find the nearest neighbor in the dataset
    # First, we need to get the core points of the DBSCAN model
    if hasattr(model, 'core_sample_indices_'):
        core_points = model.components_
        if len(core_points) > 0:
            # Find the nearest core point
            distances = np.linalg.norm(core_points - X_scaled, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_core_point = core_points[nearest_idx]
            distance = distances[nearest_idx]
           
            # Get the cluster label of this core point
            # We need to find which cluster this core point belongs to
            core_labels = model.labels_[model.core_sample_indices_]
            cluster_id = core_labels[nearest_idx]
           
            return cluster_id, distance
   
    # If no core points or other issues, return noise (-1)
    return -1, float('inf')
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
    scaler = load_pickle(MODEL_DIR / "scaler.pkl")
    model = load_pickle(MODEL_DIR / "dbscan_model.pkl")
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
    # Build a column map for robust access
    colmap = build_column_map(new_df)
    spending_col = colmap.get('Spending') or 'Spending Score (1-100)'
    if spending_col not in new_df.columns:
        # Resolve required columns with fallbacks
        freq_col = colmap.get('Frequency')
        mon_col = colmap.get('Monetary')
        rec_col = colmap.get('Recency')
        inc_col = colmap.get('Income')
        if not all([freq_col, mon_col, rec_col, inc_col]):
            st.error(f"Cannot compute Spending Score: missing columns among Recency/Frequency/Monetary/Income. Available: {new_df.columns.tolist()}")
            st.stop()
        max_freq = new_df[freq_col].max()
        max_monetary = new_df[mon_col].max()
        max_income = new_df[inc_col].max()
        new_df[spending_col] = new_df.apply(
            lambda row: calculate_spending_score(
                row[rec_col],
                row[freq_col],
                row[mon_col],
                max_freq,
                max_monetary,
                row[inc_col],
                max_income
            ), axis=1
        )
    st.session_state.spending_col = spending_col
    # Log the upload
    log_activity(st.session_state["username"], "dataset_upload", {
        "file_name": uploaded_file.name,
        "records": len(new_df)
    })
    # If uploaded file doesn't contain clustering, offer to run clustering
    missing_cluster = "Cluster" not in new_df.columns or "Cluster Name" not in new_df.columns
    if missing_cluster:
        st.warning("Uploaded file does not appear to contain clustering results.")
        if st.button("Run Clustering (DBSCAN) on this dataset"):
            try:
                with st.spinner("Running clustering — this may take a moment..."):
                    clustered = run_clustering_on_df(new_df)
                    st.session_state.df_clusters = clustered
                    st.success("Clustering complete — dataset updated in session.")
                    # Update cluster_names variable for the rest of the page
                    cluster_names_local = clustered[["Cluster", "Cluster Name"]].drop_duplicates().set_index("Cluster")["Cluster Name"].to_dict()
                    st.session_state.cluster_names = cluster_names_local
            except Exception as e:
                st.error(f"Clustering failed: {e}")
# Load current dataset from session
if 'df_clusters' not in st.session_state:
    st.warning("Please upload the processed dataset to continue.")
    st.stop()
df_clusters = st.session_state.df_clusters
spending_col = st.session_state.spending_col
# Build column map for this dataset so later code uses the actual column names
colmap = build_column_map(df_clusters)
# Validate essential columns early and show friendly message if missing
essential = ['RFM Score', 'Income', 'Frequency']
if colmap.get('RFM Score') is None or colmap.get('Income') is None:
    st.warning("The uploaded dataset is missing essential columns (RFM Score or Annual Income). Use a processed clustered dataset or run clustering.")
    # Allow user to continue but many features will be disabled
# Extract Cluster → Name mapping (use session override if clustering was run)
cluster_names: Dict[int, str] = {}
if 'cluster_names' in st.session_state:
    cluster_names = st.session_state.cluster_names
else:
    if "Cluster" in df_clusters.columns:
        if "Cluster Name" in df_clusters.columns:
            cluster_names = df_clusters[["Cluster", "Cluster Name"]].drop_duplicates().set_index("Cluster")["Cluster Name"].to_dict()
        else:
            unique_clusters = df_clusters["Cluster"].dropna().unique().tolist()
            for cid in unique_clusters:
                try:
                    cid_int = int(cid)
                except Exception:
                    cid_int = cid
                cluster_names[cid_int] = f"Cluster {cid_int}"
# Ensure we have a name for noise points
if -1 not in cluster_names:
    cluster_names[-1] = "Noise/Outliers"
def try_derive_cluster_names(df: pd.DataFrame) -> Dict[int, str]:
    """Try to derive human-friendly cluster names from available columns.
    Looks for common columns like 'Segment', 'Label', 'ClusterName', or
    any column containing 'segment' or 'label' and maps unique values to
    cluster ids if a mapping exists (e.g., same length, or uses cluster id column).
    """
    names = {}
    # If there's a Cluster Name column already, use it
    if "Cluster Name" in df.columns and "Cluster" in df.columns:
        return df[["Cluster", "Cluster Name"]].drop_duplicates().set_index("Cluster")["Cluster Name"].to_dict()
    # Look for a column whose values look like cluster names (Segment, Label)
    candidate_cols = [c for c in df.columns if any(k in c.lower() for k in ("segment", "label", "clustername", "cluster_name"))]
    if candidate_cols:
        col = candidate_cols[0]
        # If column has same number of unique values as cluster ids, map by value
        if "Cluster" in df.columns:
            for cid in df["Cluster"].dropna().unique():
                vals = df[df["Cluster"] == cid][col].dropna().unique()
                if len(vals) > 0:
                    names[int(cid) if isinstance(cid, (int, float)) and not np.isnan(cid) else cid] = str(vals[0])
    return names
def run_clustering_on_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run a lightweight clustering on the provided dataframe and return updated df with 'Cluster' and 'Cluster Name'."""
    # Ensure features available (use column mapping for robustness)
    features = []
    colmap_local = build_column_map(df)
    if "RFM Score" in df.columns:
        features.append("RFM Score")
    else:
        # try to compute RFM Score if Recency/Frequency/Monetary present (mapped names)
        rec_col = colmap_local.get('Recency')
        freq_col = colmap_local.get('Frequency')
        mon_col = colmap_local.get('Monetary')
        if all([rec_col, freq_col, mon_col]):
            df["RFM Score"] = df.apply(lambda r: calculate_rfm_score(r[rec_col], r[freq_col], r[mon_col]), axis=1)
            features.append("RFM Score")
    inc_col_local = colmap_local.get('Income')
    if inc_col_local:
        features.append(inc_col_local if inc_col_local in df.columns else 'Annual Income (k$)')
    if len(features) < 2:
        raise ValueError("Not enough features available to run clustering. Need RFM and Annual Income.")
    X = df[features].fillna(0).values
    # Try to use loaded scaler if shape matches, else fallback to StandardScaler
    try:
        scaler_local = scaler
        X_scaled = scaler_local.transform(pd.DataFrame(X, columns=features))
    except Exception:
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        X_scaled = ss.fit_transform(X)
    # Run DBSCAN with config parameters
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=CONFIG.get("dbscan_eps", 0.5), min_samples=CONFIG.get("dbscan_min_samples", 5))
    labels = db.fit_predict(X_scaled)
    df = df.copy()
    df["Cluster"] = labels
    # Derive cluster names
    derived = try_derive_cluster_names(df)
    for cid in df["Cluster"].dropna().unique():
        if cid not in derived:
            derived[cid] = f"Cluster {int(cid) if isinstance(cid, (int, float)) and not np.isnan(cid) else cid}"
    derived[-1] = "Noise/Outliers"
    df["Cluster Name"] = df["Cluster"].map(derived)
    return df
@st.cache_data
def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    # For DBSCAN, we need to exclude noise points from the summary
    return df[df["Cluster"] != -1].groupby("Cluster")[["RFM Score", "Annual Income (k$)"]].mean().reset_index()
cluster_summary_df = cluster_summary(df_clusters)
# 6. USER INPUTS
st.subheader("Customer Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    default_id = int(df_clusters['CustomerID'].max() + 1) if 'CustomerID' in df_clusters.columns else 1
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
        with st.spinner("Analyzing customer — please wait..."):
            rfm_raw = calculate_rfm_score(recency, frequency, monetary)
            rfm_w, income_w = weight_features(rfm_raw, annual_income)
            # Spending Score (use mapped columns)
            freq_col = colmap.get('Frequency')
            mon_col = colmap.get('Monetary')
            inc_col = colmap.get('Income')
            if not all([freq_col, mon_col, inc_col]):
                st.error("Dataset missing Frequency/Monetary/Income columns required to compute spending score.")
                st.stop()
            max_freq = df_clusters[freq_col].max()
            max_monetary = df_clusters[mon_col].max()
            max_income = df_clusters[inc_col].max()
            spending_score = calculate_spending_score(
                recency, frequency, monetary,
                max_freq, max_monetary,
                annual_income, max_income
            )
            cluster_id, dist = predict_cluster(scaler, cluster_model, rfm_w, income_w)
            cluster_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        # Display
        st.subheader("Customer Analysis Result")
       
        # Handle the case where the customer is classified as noise
        if cluster_id == -1:
            st.warning(f"**Customer is classified as Noise/Outlier**")
            profile = "This customer doesn't fit well into any existing cluster and may represent a unique pattern or anomaly."
        else:
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
            if cluster_id == -1:
                st.info(
                    f"Weighted RFM: `{rfm_w:.2f}` | Weighted Income: `{income_w:.2f}` | "
                    f"Customer classified as noise (doesn't fit any cluster)"
                )
            else:
                st.info(
                    f"Weighted RFM: `{rfm_w:.2f}` | Weighted Income: `{income_w:.2f}` | "
                    f"Distance to nearest core point: `{dist:.4f}`"
                )
        # Cluster Plot
        st.subheader("Cluster Visualisation")
        fig, ax = plt.subplots(figsize=(10, 6))
       
        # Determine actual column names for plotting
        rfm_col = colmap.get('RFM Score') or 'RFM Score'
        inc_col = colmap.get('Income') or 'Annual Income (k$)'
        # Plot regular clusters
        for cid, name in cluster_names.items():
            if cid == -1: # Skip noise for now
                continue
            data = df_clusters[df_clusters[colmap.get('Cluster','Cluster')] == cid]
            ax.scatter(data[rfm_col] * 0.8, data[inc_col] * 0.2,
                      label=name, color=CONFIG["colors"][cid % len(CONFIG["colors"])],
                      alpha=0.6, s=80)
       
        # Plot noise points if any
        if -1 in cluster_names and -1 in df_clusters["Cluster"].values:
            noise_data = df_clusters[df_clusters[colmap.get('Cluster','Cluster')] == -1]
            ax.scatter(noise_data[rfm_col] * 0.8, noise_data[inc_col] * 0.2,
                      label="Noise/Outliers", color="gray",
                      alpha=0.6, s=80, marker="x")
       
        # Plot the new customer
        if cluster_id == -1:
            ax.scatter(rfm_w, income_w, c="red", s=250, marker="X", label="New Customer (Noise)",
                      edgecolors="white", linewidth=1.5)
        else:
            ax.scatter(rfm_w, income_w, c="black", s=250, marker="X", label="New Customer",
                      edgecolors="white", linewidth=1.5)
           
        ax.set_xlabel("Weighted RFM Score")
        ax.set_ylabel("Weighted Annual Income (k$)")
        ax.set_title("Customer clusters with the new entry")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        # Similar Customers
        st.subheader("Similar Customers")
        df_similar = df_clusters.copy()
        rfm_col = colmap.get('RFM Score') or 'RFM Score'
        inc_col = colmap.get('Income') or 'Annual Income (k$)'
        df_similar['Weighted RFM'] = df_similar[rfm_col] * 0.8
        df_similar['Weighted Income'] = df_similar[inc_col] * 0.2
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
        # For noise points, use a different recommendation approach
        if cluster_id == -1:
            st.success(
                f"Suggested Discount: **25-35%** \nReason: "
                f"Customer doesn't fit existing patterns - use strong incentive to establish relationship."
            )
        else:
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
            "cluster": cluster_name if cluster_id != -1 else "Noise/Outlier"
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
        # Spending Score (use mapped columns)
        st_colmap = build_column_map(st.session_state.df_clusters)
        freq_col = st_colmap.get('Frequency')
        mon_col = st_colmap.get('Monetary')
        inc_col = st_colmap.get('Income')
        if not all([freq_col, mon_col, inc_col]):
            st.error("Dataset missing Frequency/Monetary/Income columns required to compute spending score.")
            st.stop()
        max_freq = st.session_state.df_clusters[freq_col].max()
        max_monetary = st.session_state.df_clusters[mon_col].max()
        max_income = st.session_state.df_clusters[inc_col].max()
        spending_score = calculate_spending_score(
            recency, frequency, monetary,
            max_freq, max_monetary,
            annual_income, max_income
        )
        # Build a new row using the dataset's actual column names (via column map)
        new_row = pd.DataFrame(index=[0])
        st_colmap = build_column_map(st.session_state.df_clusters)
        field_map = {
            'CustomerID': customer_id,
            'Gender': gender,
            'Mail id': mail_id,
            'Age': age,
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'Income': annual_income,
        }
        # Map canonical keys to actual column names and set values
        for canon_key, val in field_map.items():
            actual = None
            if canon_key == 'Recency':
                actual = st_colmap.get('Recency')
            elif canon_key == 'Frequency':
                actual = st_colmap.get('Frequency')
            elif canon_key == 'Monetary':
                actual = st_colmap.get('Monetary')
            elif canon_key == 'Income':
                actual = st_colmap.get('Income')
            else:
                actual = st_colmap.get(canon_key)
            # If the dataset already has the column, use it; otherwise create it with the canonical name
            col_name = actual or canon_key if isinstance(actual, str) else canon_key
            new_row[col_name] = val
        # Add computed fields using mapped names
        rfm_col_name = st_colmap.get('RFM Score') or 'RFM Score'
        cluster_col_name = st_colmap.get('Cluster') or 'Cluster'
        cluster_name_col = st_colmap.get('Cluster Name') or 'Cluster Name'
        spending_col_name = st.session_state.get('spending_col', spending_col)
        new_row[rfm_col_name] = rfm_raw
        new_row[cluster_col_name] = cluster_id
        new_row[cluster_name_col] = cluster_name
        new_row[spending_col_name] = spending_score
        # Append to session dataset and persist
        df_existing = st.session_state.df_clusters.copy()
        updated = pd.concat([df_existing, new_row], ignore_index=True, sort=False)
        st.session_state.df_clusters = updated
        st.success(f"Customer {int(customer_id)} added to dataset.")
        # Log addition
        log_activity(st.session_state.get("username"), "customer_added", {"customer_id": customer_id, "cluster": cluster_name})
        # Recalculate Spending Score for entire dataset with updated max values
        # Use mapped column names where available
        st_colmap = build_column_map(st.session_state.df_clusters)
        r_col = st_colmap.get('Recency') or 'Recency (days)'
        f_col = st_colmap.get('Frequency') or 'Frequency (visits)'
        m_col = st_colmap.get('Monetary') or 'Monetary ($)'
        inc_col = st_colmap.get('Income') or 'Annual Income (k$)'
        max_freq = st.session_state.df_clusters[f_col].max()
        max_monetary = st.session_state.df_clusters[m_col].max()
        max_income = st.session_state.df_clusters[inc_col].max()
        st.session_state.df_clusters[st.session_state.get('spending_col', spending_col)] = st.session_state.df_clusters.apply(
            lambda row: calculate_spending_score(
                row[r_col],
                row[f_col],
                row[m_col],
                max_freq,
                max_monetary,
                row[inc_col],
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