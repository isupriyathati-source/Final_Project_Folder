import streamlit as st
from pymongo import MongoClient
import hashlib
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from datetime import datetime
import os

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# 🔥 CHANGED: Title from MeanShift → Customer Segmentation
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

# --- Custom CSS (UNCHANGED) ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3 {
        text-align: center;
        color: #FFFFFF;
    }

    .card {
        background: #1E1E1E;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
        text-align: center;
        transition: all 0.3s ease;
    }
    .card:hover {
        background: #2C2C2C;
        transform: translateY(-5px);
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.1);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #6C63FF, #4B47C8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #7F78FF, #5A55D6);
        box-shadow: 0 0 15px rgba(108, 99, 255, 0.5);
    }

    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: white;
        border-radius: 8px;
    }

    label {
        color: #CCCCCC;
    }

    #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- MongoDB Setup (UNCHANGED) ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ismaster")
    db = client["Business_Owners"]
    users_collection = db["users"]
    activities_collection = db["activities"]
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

def make_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    return users_collection.find_one({"username": username, "password": make_hash(password)})

def register_user(username, password):
    if users_collection.find_one({"username": username}):
        return False, "Username already taken"
    users_collection.insert_one({"username": username, "password": make_hash(password)})
    return True, "Registration successful!"

def log_activity(username, action, details=None):
    if details is None:
        details = {}
    activity = {
        "username": username,
        "action": action,
        "details": details,
        "timestamp": datetime.now()
    }
    try:
        activities_collection.insert_one(activity)
    except Exception as e:
        st.error(f"Failed to log activity: {e}")

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Login"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --- LOGIN / SIGNUP ---
if not st.session_state.logged_in:
    # 🔥 CHANGED: Removed MeanShift wording
    st.title("Welcome to K-Means Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            st.session_state.app_mode = "Login"
            st.rerun()
    with col2:
        if st.button("Sign Up"):
            st.session_state.app_mode = "Signup"
            st.rerun()

    st.markdown("---")

    if st.session_state.app_mode == "Login":
        st.subheader("Login to Continue")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    log_activity(username, "login")
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    else:
        st.subheader("Create an Account")
        with st.form("signup_form"):
            username = st.text_input("Choose a Username")
            password = st.text_input("Set Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if not username or not password:
                    st.error("Fields cannot be empty.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(username, password)
                    if success:
                        log_activity(username, "signup")
                        st.success(msg)
                        st.session_state.app_mode = "Login"
                        st.rerun()
                    else:
                        st.error(msg)

# --- DASHBOARD ---
else:
    st.title(f"Welcome, {st.session_state.username}!")
    # 🔥 CHANGED: More generic (works for K-Means + MeanShift)
    st.subheader("Choose a section below to continue your customer segmentation analysis:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("<div class='card'><h3>Bulk Analysis</h3><p>Upload and analyze customer datasets using trained clustering model.</p>", unsafe_allow_html=True)
        if st.button("Open Bulk Analysis", key="bulk_btn"):
            st.switch_page("pages/Bulk_Analysis.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><h3>Email Send</h3><p>Send analysis results directly to your clients.</p>", unsafe_allow_html=True)
        if st.button("Open Email", key="email_btn"):
            st.switch_page("pages/Email_Send.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'><h3>Individual Analysis</h3><p>View insights for a single customer profile.</p>", unsafe_allow_html=True)
        if st.button("Open Individual Analysis", key="individual_btn"):
            st.switch_page("pages/Individual_Analysis.py")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='card'><h3>Activity History</h3><p>View your past activities and logs.</p>", unsafe_allow_html=True)
        if st.button("Open Activity History", key="history_btn"):
            st.switch_page("pages/Activity_History.py")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Logout"):
        log_activity(st.session_state.username, "logout")
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()