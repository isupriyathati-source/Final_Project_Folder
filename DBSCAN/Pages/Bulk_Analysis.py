# D:\Project\DBSCAN\Pages\Bulk_Analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import asyncio
import requests
import os
import warnings
import re
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

os.chdir(r"D:\Project\DBSCAN")
st.set_page_config(layout="wide", page_title="Customer Segmentation App")

# ACTIVITY LOGGING
try:
    from Home_Page import log_activity
except ImportError:
    def log_activity(username, action, details=None):
        print(f"ACTIVITY LOG (Dummy): User={username}, Action={action}")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("Please log in first!")
    st.stop()

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ==================== FIXED MODEL LOADING ====================
BASE_DIR = r"D:\Project\DBSCAN\Model_Files"
try:
    with open(os.path.join(BASE_DIR, "dbscan_model.pkl"), "rb") as f:
        dbscan_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model files not found in {BASE_DIR}")
    st.stop()

# ==================== GEMINI HELPER (ROBUST) ====================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

from gemini_helper import get_cluster_names_api, get_cluster_names_fallback

def get_cluster_info(cluster_summary: pd.DataFrame):
    """ONE SINGLE API CALL for all clusters with full fallback"""
    if not GEMINI_API_KEY:
        st.info("🔄 Using fallback cluster names (Gemini key not found)")
        return get_cluster_names_fallback(cluster_summary)

    df_api = get_cluster_names_api(cluster_summary, GEMINI_API_KEY)
    if df_api.empty:
        st.warning("Gemini unavailable – using smart fallback names")
        return get_cluster_names_fallback(cluster_summary)
    return df_api

# ==================== HELPERS ====================
def normalize_columns(df):
    def clean_name(col):
        col = re.sub(r'\([^)]*\)', '', col)
        col = col.lower().strip()
        col = re.sub(r'[^\w]+', '_', col)
        return col.strip('_')
    df.rename(columns={c: clean_name(c) for c in df.columns}, inplace=True)
    return df

def calculate_optimal_eps(X_scaled, min_samples=4):
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, min_samples - 1])
    x = np.arange(len(k_distances))
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (k_distances - k_distances.min()) / (k_distances.max() - k_distances.min())
    vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    vec /= np.linalg.norm(vec)
    vec_perp = np.array([-vec[1], vec[0]])
    deltas = np.dot(np.stack((x_norm, y_norm), axis=1) - np.array([x_norm[0], y_norm[0]]), vec_perp)
    knee_index = np.argmax(np.abs(deltas))
    return k_distances[knee_index]

def plot_dbscan_clusters_3d(X_scaled, labels, noise_mask, df_original):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    z_col = 'age' if 'age' in df_original.columns else 'frequency'
    local_scaler = MinMaxScaler()
    z_values = local_scaler.fit_transform(df_original[[z_col]]).flatten()

    if np.any(noise_mask):
        ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], z_values[noise_mask],
                   c='red', s=50, label='Noise/Outliers', edgecolors='k', alpha=0.6)

    unique_labels = set(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            continue
        class_member_mask = (labels == k)
        xy = X_scaled[class_member_mask]
        z = z_values[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], z, c=[col], s=50, label=f"Cluster {k}", edgecolors='k', alpha=0.8)

    ax.set_title('3D Customer Clusters (RFM, Income, Frequency)', fontsize=14, fontweight='bold')
    ax.set_xlabel('RFM Score (scaled)')
    ax.set_ylabel('Annual Income (scaled)')
    ax.set_zlabel(f'{z_col.title()} (scaled)')
    ax.view_init(elev=20, azim=45)
    plt.legend(loc='upper right')
    plt.tight_layout()
    return fig

# ==================== MAIN UI ====================
st.title("Customer Segmentation App")
st.markdown("### Bulk Analysis: Upload CSV or Excel")

with st.sidebar:
    st.header("Configuration")
    clean_data = st.checkbox("Remove rows with missing values", value=True)
    st.markdown("---")

uploaded_file = st.file_uploader("Upload a Data File", type=["csv", "xlsx", "xls"])

REQUIRED_COLS = ['recency', 'frequency', 'monetary', 'annual_income']

if uploaded_file:
    MAX_SIZE_MB = 200
    if uploaded_file.size > MAX_SIZE_MB * 1024 * 1024:
        st.error(f"❌ File size exceeds {MAX_SIZE_MB}MB limit.")
        st.stop()

    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext == 'csv':
            df_original = pd.read_csv(uploaded_file)
        else:
            df_original = pd.read_excel(uploaded_file)

        df = normalize_columns(df_original.copy())

        if clean_data:
            before = len(df)
            df.dropna(inplace=True)
            if before != len(df):
                st.toast(f"Cleaned data: Removed {before - len(df)} rows.", icon="🧹")

        # RFM Calculation
        df['RFM Score'] = (0.2 * (100 - df['recency'])) + (0.4 * df['frequency']) + (0.4 * df['monetary'])

        X = df[['RFM Score', 'annual_income']].copy()
        X['RFM Score'] *= 0.8
        X['annual_income'] *= 0.2
        X.rename(columns={'annual_income': 'Annual Income (k$)'}, inplace=True)

        X_scaled = scaler.transform(X)

        # DBSCAN Prediction
        labels = dbscan_model.fit_predict(X_scaled)
        df['Cluster'] = labels

        # Cluster Summary
        cluster_summary_df = df[df['Cluster'] != -1].groupby("Cluster")[["RFM Score", "Annual Income (k$)"]].mean().reset_index()

        # SMART CLUSTER NAMES (ONE API CALL)
        cluster_info_df = get_cluster_info(cluster_summary_df)

        st.success(f"✅ Clustering complete! Estimated clusters: {len(cluster_summary_df)}")
        st.dataframe(cluster_info_df, use_container_width=True)

        # 3D Plot
        noise_mask = labels == -1
        fig_3d = plot_dbscan_clusters_3d(X_scaled, labels, noise_mask, df_original)
        st.pyplot(fig_3d)

        # Download
        output = df.copy()
        output['Cluster Name'] = output['Cluster'].map(
            dict(zip(cluster_info_df['cluster_id'], cluster_info_df['name'])) 
            if not cluster_info_df.empty else {c: f"Cluster {c}" for c in output['Cluster'].unique()}
        )
        csv = output.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Data", csv, "processed_customer_data.csv", "text/csv")

        log_activity(st.session_state["username"], "bulk_analysis", {"records": len(df)})

    except Exception as e:
        st.error(f"Error processing file: {e}")