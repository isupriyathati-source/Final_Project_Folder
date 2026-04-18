# D:\Project\KMeans\Pages\Bulk_Analysis.py
import os
os.chdir(r"D:\Project\KMeans")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import asyncio
import requests
import warnings
import re
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.cluster import KMeans

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

# ==================== MODEL LOADING ====================
try:
    with open("model_files/kmeans_model.pkl", "rb") as f:
        kmeans_model = pickle.load(f)
    with open("model_files/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'kmeans_model.pkl' and 'scaler.pkl' are in 'model_files/'.")
    st.stop()

# ==================== GEMINI HELPER ====================
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

# ==================== MAIN APP ====================
st.set_page_config(layout="wide", page_title="Customer Segmentation App")
st.title("Customer Segmentation App")
st.markdown("### Upload a CSV file to automatically segment your customers using K-Means")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

REQUIRED_COLS_NORMALIZED = ['recency', 'frequency', 'monetary', 'annual_income']

if uploaded_file:
    if 'processed_data_hash' not in st.session_state or st.session_state['processed_data_hash'] != hash(uploaded_file.getvalue()):
        df_original = pd.read_csv(uploaded_file)
        
        # Column Normalization
        def normalize_columns(df):
            def clean_name(col):
                col = re.sub(r'\([^)]*\)', '', col)
                col = col.lower().strip()
                col = re.sub(r'[^\w]+', '_', col)
                return col.strip('_')
            df.rename(columns={c: clean_name(c) for c in df.columns}, inplace=True)
            return df
        
        df = normalize_columns(df_original.copy())
        
        if not all(col in df.columns for col in REQUIRED_COLS_NORMALIZED):
            st.error(f"Missing required columns. Found: {df.columns.tolist()}")
            st.stop()
        
        # RFM Calculation
        df['RFM Score'] = (0.2 * (100 - df['recency'])) + (0.4 * df['frequency']) + (0.4 * df['monetary'])
        
        # Prepare for scaling
        X = df[['RFM Score', 'annual_income']].copy()
        X['RFM Score'] *= 0.8
        X['annual_income'] *= 0.2
        X.rename(columns={'annual_income': 'Annual Income (k$)'}, inplace=True)
        
        X_scaled = scaler.transform(X)
        
        # KMeans Prediction
        labels = kmeans_model.predict(X_scaled)
        df['Cluster'] = labels
        
        st.success(f"✅ Clustering complete! Number of clusters: {kmeans_model.n_clusters}")
        
        # Cluster Summary
        cluster_summary_df = df.groupby("Cluster")[["RFM Score", "Annual Income (k$)"]].mean().reset_index()
        
        # Get Smart Cluster Names (ONE API CALL)
        cluster_info_df = get_cluster_info(cluster_summary_df)
        
        st.subheader("Cluster Insights")
        st.dataframe(cluster_info_df, use_container_width=True)
        
        # Simple 2D Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for cid in df['Cluster'].unique():
            data = df[df['Cluster'] == cid]
            ax.scatter(data['RFM Score'], data['Annual Income (k$)'], label=f"Cluster {cid}", alpha=0.7)
        ax.set_xlabel("Weighted RFM Score")
        ax.set_ylabel("Weighted Annual Income (k$)")
        ax.set_title("K-Means Customer Segments")
        ax.legend()
        st.pyplot(fig)
        
        # Add Cluster Name to DataFrame
        name_map = dict(zip(cluster_info_df['cluster_id'], cluster_info_df['name'])) if not cluster_info_df.empty else {}
        df['Cluster Name'] = df['Cluster'].map(name_map).fillna("Unknown")
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Data with Cluster Names", csv, "kmeans_processed_data.csv", "text/csv")
        
        # Store in session for Individual Analysis / Email
        st.session_state.df_clusters = df
        
        log_activity(st.session_state["username"], "bulk_analysis", {"records": len(df), "clusters": kmeans_model.n_clusters})

    else:
        st.info("Data already processed. Scroll down to download.")