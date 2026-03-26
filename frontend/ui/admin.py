import streamlit as st
import requests
import pandas as pd
from api.api_client import client, fetch_model_metrics

def render_admin_dashboard():
    st.title("⚙️ Admin Dashboard & ML Center")

    health = client._get("/")
    if not health:
        st.error("Cannot connect to Backend API.")
        return

    # --- TABS ---
    tab_models, tab_seasonal, tab_discipline, tab_backtest, tab_jobs = st.tabs([
        "🧠 Model Synthesis", "📅 Seasonal Trends", "🏇 Discipline Metrics", "🔍 Back-Analysis", "📅 Scheduler"
    ])

    # Load All Metrics
    metrics_df = fetch_model_metrics()

    # ------------------
    # TAB 1: MODEL SYNTHESIS
    # ------------------
    with tab_models:
        st.subheader("Active Inference Models")
        available_models = health.get("available_models", [])
        
        if available_models:
            summary_data = []
            for m in available_models:
                m_low = m.lower()
                if not metrics_df.empty:
                    m_stats = metrics_df[(metrics_df['model_name'] == m_low) & (metrics_df['segment_type'] == 'discipline_overall')]
                    avg_loss = m_stats['logloss'].mean() if not m_stats.empty else None
                    avg_auc = m_stats['auc'].mean() if not m_stats.empty else None
                    avg_roi = m_stats['roi'].mean() if not m_stats.empty else None
                else:
                    avg_loss, avg_auc, avg_roi = None, None, None

                # Score Logic
                score = "⚪ N/A"
                if avg_loss:
                    if avg_loss < 0.25: score = "⭐⭐⭐⭐⭐"
                    elif avg_loss < 0.27: score = "⭐⭐⭐"
                    else: score = "⭐"

                summary_data.append({
                    "Specialty": m.upper(),
                    "Status": "🟢 Active",
                    "Score": score,
                    "Avg LogLoss": round(avg_loss, 4) if avg_loss else "N/A",
                    "Avg AUC": round(avg_auc, 3) if avg_auc else "N/A",
                    "Avg ROI (%)": round(avg_roi, 1) if avg_roi else "N/A"
                })
            
            st.table(summary_data)
        else:
            st.warning("No ML models loaded in the backend.")

    # ------------------
    # TAB 2: SEASONAL TRENDS
    # ------------------
    with tab_seasonal:
        st.subheader("Seasonal Trends (ROI % per Month)")
        if metrics_df.empty:
            st.info("No back-analysis data available. Run 'trainer.py' to generate metrics.")
        else:
            seasonal_df = metrics_df[metrics_df['segment_type'] == 'discipline_month'].copy()
            if not seasonal_df.empty:
                # Pivot table to match requested format
                pivot_seasonal = seasonal_df.pivot_table(
                    index='test_month', 
                    columns='segment_value', 
                    values='roi'
                )
                pivot_seasonal.index.name = 'Month'
                st.dataframe(pivot_seasonal.style.format("{:.1f}%").background_gradient(cmap='RdYlGn', axis=None))
            else:
                st.info("No seasonal data found. Ensure disciplines and months are tracked.")

    # ------------------
    # TAB 3: DISCIPLINE METRICS
    # ------------------
    with tab_discipline:
        st.subheader("Overall Performance by Discipline")
        if not metrics_df.empty:
            discipline_df = metrics_df[metrics_df['segment_type'] == 'discipline_overall'].copy()
            if not discipline_df.empty:
                # Format for display
                display_df = discipline_df[['segment_value', 'num_races', 'win_rate', 'roi', 'avg_odds']].copy()
                display_df.columns = ['Discipline', 'Races', 'Win Rate (%)', 'ROI (%)', 'Avg Odds']
                st.dataframe(
                    display_df.sort_values('ROI (%)', ascending=False),
                    column_config={
                        "Win Rate (%)": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                        "ROI (%)": st.column_config.NumberColumn("ROI", format="%.1f%%"),
                        "Avg Odds": st.column_config.NumberColumn("Avg Odds", format="%.2f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )

    # ------------------
    # TAB 4: BACK-ANALYSIS (TRACK LEVEL)
    # ------------------
    with tab_backtest:
        st.subheader("Detailed Track Performance")
        if metrics_df.empty:
            st.info("No back-analysis data available.")
        else:
            track_df = metrics_df[metrics_df['segment_type'] == 'track_month'].copy()
            # Filter UI
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                sel_model = st.selectbox("Select Model", ["all"] + sorted(list(track_df['model_name'].unique())))
            with col_f2:
                sel_month = st.multiselect("Select Months", sorted(list(track_df['test_month'].unique())), default=[])

            filtered_df = track_df.copy()
            if sel_model != "all":
                filtered_df = filtered_df[filtered_df['model_name'] == sel_model]
            if sel_month:
                filtered_df = filtered_df[filtered_df['test_month'].isin(sel_month)]

            st.dataframe(
                filtered_df.sort_values(['roi'], ascending=False),
                column_config={
                    "model_name": "Model",
                    "segment_value": "Track Code",
                    "test_month": "Month",
                    "num_races": "Count",
                    "logloss": st.column_config.NumberColumn("LogLoss", format="%.4f"),
                    "roi": st.column_config.NumberColumn("ROI", format="%.1f%%"),
                    "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                },
                use_container_width=True,
                hide_index=True
            )

    # ------------------
    # TAB 5: SCHEDULER
    # ------------------
    with tab_jobs:
        sched = health.get("scheduler", {})
        if sched.get("jobs"):
            st.table(sched["jobs"])
        else:
            st.info("No active jobs in scheduler.")

        st.subheader("🚀 Manual Triggers")
        c1, c2 = st.columns(2)
        def trigger_job(job_id):
            try:
                url = f"{client.base_url}/jobs/{job_id}/run"
                resp = requests.post(url)
                if resp.status_code == 200:
                    st.toast(f"✅ Job '{job_id}' triggered successfully!")
                else: st.error(f"Failed to trigger job: {resp.text}")
            except Exception as e: st.error(f"Error: {e}")

        with c1:
            if st.button("Trigger Daily ETL Now"): trigger_job("daily_etl")
        with c2:
            if st.button("Refresh Live Odds"): trigger_job("live_odds")

    st.divider()

    # Logs
    st.subheader("📝 System Logs")
    logs_data = client._get("/logs")
    if logs_data and "logs" in logs_data:
        st.code("\n".join(logs_data["logs"]), language="log")
    else:
        st.warning("No logs available from API.")
