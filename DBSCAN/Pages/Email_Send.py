# D:\Project\DBSCAN\Pages\Email_Send.py
import streamlit as st
import pandas as pd
import asyncio
import smtplib
from email.message import EmailMessage
import ssl
import os

# ACTIVITY LOGGING
try:
    from Home_Page import log_activity
except ImportError:
    def log_activity(username, action, details=None):
        print(f"ACTIVITY LOG (Dummy): User={username}, Action={action}")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("Please log in first!")
    st.stop()

st.set_page_config(layout="wide", page_title="Send Email")
st.title("Send Email to Customers")
st.markdown("### Upload a CSV and select recipients")

# GEMINI API KEY
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

from gemini_helper import generate_all_email_templates, generate_email_fallback

# Email Settings
SMTP_SERVER = st.text_input("SMTP Server (e.g., smtp.gmail.com)", "smtp.gmail.com")
SMTP_PORT = st.number_input("SMTP Port (e.g., 587)", 587)
SENDER_EMAIL = st.text_input("Sender Email Address", "")
SENDER_PASSWORD = st.text_input("Sender Password (or App Password)", "", type="password")

# Discount Selection
discount_options = ["5-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%"]
selected_discount = st.selectbox("Select Discount Range:", discount_options)

# Generate Templates (ONE API CALL)
if st.button("Generate Mail Suggestions", use_container_width=True):
    with st.spinner(f"Generating professional email templates for {selected_discount}..."):
        # Use session state or uploaded data for cluster summary
        if 'df_clusters' in st.session_state and not st.session_state.df_clusters.empty:
            cluster_summary = st.session_state.df_clusters.groupby("Cluster")[["RFM Score", "Annual Income (k$)"]].mean().reset_index()
        else:
            cluster_summary = pd.DataFrame(columns=["Cluster", "RFM Score", "Annual Income (k$)"])

        suggestions = asyncio.run(
            generate_all_email_templates(cluster_summary, selected_discount, GEMINI_API_KEY)
        )
        if not suggestions:
            suggestions = generate_email_fallback(cluster_summary, selected_discount)

        st.session_state['mail_suggestions'] = suggestions
        st.success(f"✅ Generated {len(suggestions)} email templates!")

# Display Templates
if 'mail_suggestions' in st.session_state and st.session_state['mail_suggestions']:
    suggestions = st.session_state['mail_suggestions']
    option_names = [f"Option {i+1}: {s.get('subject', '')[:50]}..." for i, s in enumerate(suggestions)]
    selected_idx = st.radio("Choose template:", range(len(suggestions)), format_func=lambda x: option_names[x])
    
    selected = suggestions[selected_idx]
    st.subheader("Selected Email")
    st.markdown(f"**Subject:** {selected['subject']}")
    st.write(selected['body'])

# File Upload for Recipients
uploaded_file = st.file_uploader("Upload CSV with customer emails", type=["csv"], key="email_file")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    possible_email_cols = [col for col in df.columns if any(k in col.lower() for k in ["email", "mail", "mail_id"])]
    email_column = st.selectbox("Select email column:", possible_email_cols or df.columns)
    
    if email_column:
        email_addresses = df[email_column].dropna().unique().tolist()
        selected_recipients = st.multiselect("Choose recipients:", email_addresses, default=email_addresses)

        if st.button("Send Emails", type="primary"):
            if selected_recipients:
                context = ssl.create_default_context()
                with st.spinner("Sending emails..."):
                    try:
                        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                            server.starttls(context=context)
                            server.login(SENDER_EMAIL, SENDER_PASSWORD)
                            for recipient in selected_recipients:
                                msg = EmailMessage()
                                msg.set_content(selected['body'])
                                msg['Subject'] = selected['subject']
                                msg['From'] = SENDER_EMAIL
                                msg['To'] = recipient
                                server.send_message(msg)
                                log_activity(st.session_state["username"], "email_sent", {"to": recipient})
                        st.success(f"✅ Emails sent to {len(selected_recipients)} customers!")
                    except Exception as e:
                        st.error(f"Failed to send: {e}")
            else:
                st.warning("Please select at least one recipient.")