import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from api.api_client import client, fetch_model_metrics

def render_admin_dashboard():
    st.title("⚙️ Admin Dashboard & ML Tournament")

    health = client._get("/")
    if not health:
        st.error("Cannot connect to Backend API.")
        return

    # Load All Metrics
    metrics_df = fetch_model_metrics()

    # --- TABS ---
    tab_models, tab_tournament, tab_seasonal, tab_discipline, tab_jobs = st.tabs([
        "🧠 Model Synthesis", "🏆 Tournament Results", "📅 Seasonal Trends", "🏇 Discipline Metrics", "📅 Scheduler"
    ])

    # ------------------
    # TAB 1: MODEL SYNTHESIS
    # ------------------
    with tab_models:
        st.subheader("Active Production Models")
        available_models = health.get("available_models", [])
        
        if available_models:
            summary_data = []
            for m in available_models:
                m_low = m.lower()
                # Find the winner (best ROI or lowest LogLoss) from the latest tournament
                if not metrics_df.empty:
                    m_stats = metrics_df[(metrics_df['model_name'] == m_low) & (metrics_df['segment_type'] == 'discipline_overall')]
                    # In production, we usually use the best one
                    best_algo = m_stats.sort_values('logloss').iloc[0] if not m_stats.empty else None
                else: best_algo = None

                summary_data.append({
                    "Specialty": m.upper(),
                    "Status": "🟢 Active",
                    "Winner Algo": best_algo['algorithm'].upper() if best_algo is not None else "UNKNOWN",
                    "Best LogLoss": round(best_algo['logloss'], 4) if best_algo is not None else "N/A",
                    "Current ROI": f"{best_algo['roi']:.1f}%" if best_algo is not None else "N/A",
                    "Win Rate": f"{best_algo['win_rate']:.1%}" if best_algo is not None else "N/A"
                })
            
            st.table(summary_data)
        else:
            st.warning("No ML models loaded in the backend.")

    # ------------------
    # TAB 2: TOURNAMENT COMPARISON (NEW)
    # ------------------
    with tab_tournament:
        st.subheader("Algorithm Battle: LogLoss vs ROI")
        if metrics_df.empty:
            st.info("No tournament data yet. Run the new MultiModelTrainer.")
        else:
            overall_perf = metrics_df[metrics_df['segment_type'] == 'discipline_overall'].copy()
            
            # Filter by Target
            target_list = sorted(list(overall_perf['model_name'].unique()))
            sel_target = st.selectbox("Select Target Discipline", target_list)
            
            battle_df = overall_perf[overall_perf['model_name'] == sel_target]
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_loss = px.bar(battle_df, x='algorithm', y='logloss', color='algorithm', 
                                title="LogLoss Comparison (Lower is Better)",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col_b2:
                fig_roi = px.bar(battle_df, x='algorithm', y='roi', color='algorithm',
                               title="ROI Comparison (%)",
                               color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_roi, use_container_width=True)

            st.dataframe(battle_df[['algorithm', 'logloss', 'auc', 'roi', 'win_rate', 'num_races']].sort_values('logloss'), hide_index=True)

    # ------------------
    # TAB 3: SEASONAL TRENDS
    # ------------------
    with tab_seasonal:
        st.subheader("Seasonal Trends (ROI % per Month)")
        if metrics_df.empty:
            st.info("No data.")
        else:
            # For trends, we filter by the 'production' algorithm (best overall for each model)
            seasonal_df = metrics_df[metrics_df['segment_type'] == 'discipline_month'].copy()
            if not seasonal_df.empty:
                pivot_seasonal = seasonal_df.pivot_table(index='test_month', columns='segment_value', values='roi')
                st.dataframe(pivot_seasonal.style.format("{:.1f}%").background_gradient(cmap='RdYlGn', axis=None))

    # ------------------
    # TAB 4: DISCIPLINE METRICS
    # ------------------
    with tab_discipline:
        st.subheader("Overall Performance Matrix")
        if not metrics_df.empty:
            discipline_df = metrics_df[metrics_df['segment_type'] == 'discipline_overall'].copy()
            st.dataframe(
                discipline_df[['model_name', 'algorithm', 'num_races', 'win_rate', 'roi', 'avg_odds']],
                column_config={
                    "model_name": "Target",
                    "algorithm": "Algo",
                    "roi": st.column_config.NumberColumn("ROI", format="%.1f%%"),
                    "win_rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                },
                hide_index=True,
                use_container_width=True
            )

    # ------------------
    # TAB 5: SCHEDULER
    # ------------------
    with tab_jobs:
        sched = health.get("scheduler", {})
        if sched.get("jobs"): st.table(sched["jobs"])
        st.subheader("🚀 Manual Triggers")
        c1, c2 = st.columns(2)
        def trigger_job(job_id):
            try:
                resp = requests.post(f"{client.base_url}/jobs/{job_id}/run")
                if resp.status_code == 200: st.toast(f"✅ Job '{job_id}' triggered!")
                else: st.error(f"Failed: {resp.text}")
            except Exception as e: st.error(f"Error: {e}")
        with c1: 
            if st.button("Trigger Daily ETL"): trigger_job("daily_etl")
        with c2:
            if st.button("Refresh Live Odds"): trigger_job("live_odds")

    st.divider()
    # Logs
    st.subheader("📝 System Logs")
    logs_data = client._get("/logs")
    if logs_data and "logs" in logs_data: st.code("\n".join(logs_data["logs"]), language="log")
