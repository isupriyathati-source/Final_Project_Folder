import streamlit as st
import subprocess
import webbrowser
import time
import socket

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ================= STYLE =================
st.markdown("""
<style>
.stApp { background-color: #0E1117; color: white; }

h1 {
    text-align: center;
}

.card {
    background: #1E1E1E;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    transition: 0.3s;
}

.card:hover {
    background: #2C2C2C;
    transform: translateY(-5px);
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #6C63FF, #4B47C8);
    color: white;
    padding: 12px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("Customer Segmentation")
st.write("Select the method based on your business goal")

# ================= FUNCTIONS =================

def is_port_used(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def run_app(path, port, key):
    # Prevent duplicate execution
    if key not in st.session_state:
        st.session_state[key] = False

    if not st.session_state[key]:
        # Start app only if not already running
        if not is_port_used(port):
            subprocess.Popen([
                "streamlit", "run", path,
                "--server.port", str(port)
            ])
            time.sleep(2)

        # Open browser once
        webbrowser.open(f"http://localhost:{port}")

        # Mark as launched
        st.session_state[key] = True

# ================= UI =================

col1, col2, col3 = st.columns(3)

# ---------- KMEANS ----------
with col1:
    st.markdown("<div class='card'><h3>Fixed Groups</h3><p>Use when you know number of customer groups</p></div>", unsafe_allow_html=True)
    if st.button("Open Fixed Groups"):
        run_app(r"D:\Project\KMeans\Home_Page.py", 8601, "kmeans")

# ---------- MEANSHIFT ----------
with col2:
    st.markdown("<div class='card'><h3>Automatic Grouping</h3><p>No need to define number of groups</p></div>", unsafe_allow_html=True)
    if st.button("Open Automatic Grouping"):
        run_app(r"D:\Project\MeanShift\Home_Page.py", 8602, "meanshift")

# ---------- DBSCAN ----------
with col3:
    st.markdown("<div class='card'><h3>Outlier Detection</h3><p>Find unusual or extreme customers</p></div>", unsafe_allow_html=True)
    if st.button("Open Outlier Detection"):
        run_app(r"D:\Project\DBSCAN\Home_Page.py", 8603, "dbscan")

# ================= FOOTER =================
st.markdown("---")
st.caption("Click a method to launch its dashboard in a new tab.")