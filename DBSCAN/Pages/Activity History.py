import streamlit as st
from datetime import datetime
from Home_Page import client, activities_collection  # Import the global client and collection

# Ensure user is logged in
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.warning("Please log in first!")
    st.stop()

st.title("Activity History")

# Fetch user's activities (latest first for performance)
username = st.session_state["username"]
activities = list(activities_collection.find({"username": username}).sort("timestamp", -1).limit(50))

# Display activities
st.subheader("Recent Activities")
if activities:
    for activity in activities:
        timestamp = activity["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        action = activity["action"]
        # Extract details (customize based on what you log; fallback to JSON string)
        details_str = " | ".join([f"{k}: {v}" for k, v in activity.get("details", {}).items()]) if activity.get("details") else "No details"
        st.write(f"**{timestamp}** - {action.capitalize()}: {details_str}")
else:
    st.info("No activities recorded yet.")

# Optional: Clear history button (deletes only this user's logs)
if st.button("Clear My History"):
    result = activities_collection.delete_many({"username": username})
    st.success(f"Cleared {result.deleted_count} activities.")
    st.rerun()