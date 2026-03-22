import streamlit as st
import requests
from frontend.api.api_client import client

def render_admin_dashboard():
    st.title("⚙️ Admin Dashboard & Scheduler")

    health = client._get("/")

    if not health:
        st.error("Cannot connect to Backend API.")
        return

    # 1. System Status
    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Status", health.get("status", "OFFLINE").upper())
        st.metric("ML Engine", health.get("ml_engine", "FAILED").upper())

    with col2:
        sched = health.get("scheduler", {})
        st.metric("Scheduler", sched.get("status", "INACTIVE").upper())
        st.metric("Active Jobs", len(sched.get("jobs", [])))

    st.divider()

    # 2. Jobs Table
    st.subheader("📅 Scheduled Tasks")
    if sched.get("jobs"):
        st.table(sched["jobs"])
    else:
        st.info("No active jobs in scheduler.")

    # 3. Quick Actions
    st.subheader("🚀 Manual Triggers")
    c1, c2 = st.columns(2)

    def trigger_job(job_id):
        try:
            # We use the internal API client or direct requests
            url = f"{client.base_url}/jobs/{job_id}/run"
            resp = requests.post(url)
            if resp.status_code == 200:
                st.toast(f"✅ Job '{job_id}' triggered successfully!")
            else:
                st.error(f"Failed to trigger job: {resp.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    with c1:
        if st.button("Trigger Daily ETL Now"):
            trigger_job("daily_etl")
    with c2:
        if st.button("Refresh Live Odds"):
             trigger_job("live_odds")

    st.divider()

    # 4. Logs
    st.subheader("📝 Recent System Logs")
    logs_data = client._get("/logs")
    if logs_data and "logs" in logs_data:
        st.code("\n".join(logs_data["logs"]), language="log")
    else:
        st.warning("No logs available from API.")
