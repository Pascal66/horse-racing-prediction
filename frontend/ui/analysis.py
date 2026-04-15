import streamlit as st
import pandas as pd
from api.api_client import fetch_predictions, fetch_participants, fetch_backtest_results
pd.set_option('display.max_columns', None)

def get_horse_color(p_num: int) -> str:
    """Attribue une couleur fixe par numéro de programme pour uniformiser la vue."""
    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
        "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
        "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#000000"
    ]
    return colors[p_num % len(colors)]

def find_best_roi_match(model_version: str, trainer_stats: dict) -> float:
    if not model_version or not trainer_stats: return -99.0
    if model_version in trainer_stats: return trainer_stats[model_version].get("roi", -99.0)
    m_parts = set(model_version.lower().split('_'))
    for k, v in trainer_stats.items():
        k_parts = set(k.lower().split('_'))
        if m_parts.issubset(k_parts) or k_parts.issubset(m_parts): return v.get("roi", -99.0)
    return -99.0

def render_analysis_view(race_id: int):
    with st.spinner("Analyse multi-modèles..."):
        backtest = fetch_backtest_results()
        participant_data = fetch_participants(race_id)
        preds = {
            "tabnet": fetch_predictions(race_id, algo="tabnet"),
            "ltr": fetch_predictions(race_id, algo="ltr"),
            "hyperstack": fetch_predictions(race_id, algo="hyperstack")
        }

    if participant_data.empty:
        st.warning("No participant data available.")
        return

    # print(participant_data.head())

    # Mappings
    participant_data['program_number'] = pd.to_numeric(participant_data['program_number'], errors='coerce').fillna(0).astype(int)
    horse_map = participant_data.set_index('program_number')['horse_name'].to_dict()

    def format_actual_val(val, incident):
        if pd.notnull(val) and val > 0: return str(int(val))
        if pd.notnull(incident) and str(incident).strip(): return str(incident)
        return ""

    result_map = {row['program_number']: format_actual_val(row.get('finish_rank'), row.get('incident_code')) for _, row in participant_data.iterrows()}

    trainer_stats = backtest.get("trainers", {})
    model_order = []
    for algo_key in ["tabnet", "ltr", "hyperstack"]:
        df_p = preds[algo_key]
        if not df_p.empty and 'model_version' in df_p.columns:
            m_ver = df_p['model_version'].iloc[0]
            roi = find_best_roi_match(m_ver, trainer_stats)
            df_p['program_number'] = pd.to_numeric(df_p['program_number'], errors='coerce').fillna(0).astype(int)
            model_order.append({"algo": algo_key, "roi": roi, "df": df_p, "version": m_ver})
    model_order.sort(key=lambda x: x["roi"], reverse=True)

    # --- LAYOUT PRINCIPAL ---
    col_main, col_side = st.columns([2.5, 1])

    with col_main:
        st.subheader("🏆 Comparaison des Pronostics")
        if not model_order:
            st.info("Aucune prédiction disponible.")
        else:
            cols = st.columns(len(model_order))
            for idx, m in enumerate(model_order):
                with cols[idx]:
                    roi_val = m['roi']
                    roi_color = "#2e7d32" if roi_val > 0 else ("#9e9e9e" if roi_val == -99.0 else "#d32f2f")
                    st.markdown(f"""<div style="text-align:center; border-bottom:3px solid {roi_color}; margin-bottom:10px;">
                        <b style="font-size:1.1em;">{m['algo'].upper()}</b><br>
                        <span style="color:{roi_color}; font-weight:bold;">ROI SG: {roi_val:.1f}%</span>
                    </div>""", unsafe_allow_html=True)

                    top_df = m['df'].sort_values('win_probability', ascending=False).head(3)
                    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
                        p_num = row['program_number']
                        h_name = horse_map.get(p_num, f"n°{p_num}")
                        prob = row['win_probability'] * 100
                        actual = result_map.get(p_num, "")

                        h_color = get_horse_color(p_num)
                        bg = "#fff9c4" if rank == 1 else "#ffffff"

                        st.html(f"""
                            <div style="background:{bg}; border:1px solid #ddd; border-left:6px solid {h_color}; padding:6px; border-radius:4px; margin-bottom:4px; font-size:0.85em; position:relative;">
                                {f'<span style="float:right; background:#4CAF50; color:white; padding:0px 5px; border-radius:3px; font-weight:bold;">{actual}</span>' if actual else ""}
                                <b>#{p_num}</b> {h_name[:14]}<br>
                                <span style="color:#555;">Prob: <b>{prob:.1f}%</b></span>
                            </div>
                        """)

    with col_side:
        # --- TRACKING HIER/AUJOURD'HUI ---
        st.subheader("⏱️ Tracking")

        today_data = backtest.get("today_live", {}).get("trainers", {})
        yesterday_data = backtest.get("yesterday_bilan", {}).get("trainers", {})

        def get_best_model_perf(period_data):
            if not period_data: return None, None
            best_m = None
            best_roi = -999
            for m, d in period_data.items():
                if d["roi"] > best_roi:
                    best_roi = d["roi"]
                    best_m = m
            return best_m, period_data[best_m] if best_m else None

        t_best_m, t_perf = get_best_model_perf(today_data)
        y_best_m, y_perf = get_best_model_perf(yesterday_data)

        c1, c2 = st.columns(2)
        with c1:
            if t_perf:
                st.metric("Aujourd'hui", f"{t_perf['roi']:.1f}%", f"{t_perf['win_rate']:.1f}% WR", help=f"Best: {t_best_m}")
            else:
                st.metric("Aujourd'hui", "N/A")
        with c2:
            if y_perf:
                st.metric("Hier", f"{y_perf['roi']:.1f}%", f"{y_perf['win_rate']:.1f}% WR", help=f"Best: {y_best_m}")
            else:
                st.metric("Hier", "N/A")

        st.divider()

        st.subheader("📊 Normes")
        # On récupère le contexte de la course (discipline + mois)
        if not participant_data.empty:
            # Note: il nous faudrait la discipline de la course ici.
            # Comme analysis.py ne reçoit que race_id, on va tricher un peu en la déduisant du nom du modèle ou en passant l'info.
            # Pour l'instant, on affiche les meilleures perfs du backtest pour le mois en cours.
            cur_month = pd.Timestamp.now().month # TODO: Utiliser le mois de la course si possible
            st.info(f"Performances historiques pour le mois {cur_month}")

            if model_order:
                best = model_order[0]
                m_stats = trainer_stats.get(best['version'], {}).get("seasonal_analysis", {})
                # On cherche la discipline dans les clés de seasonal_analysis
                for disc, months in m_stats.items():
                    if str(cur_month) in str(months) or cur_month in months:
                        s = months.get(cur_month) or months.get(str(cur_month))
                        st.markdown(f"**Discipline: {disc}**")
                        st.write(f"🎯 ROI SG: `{s['roi']:.1f}%`")
                        st.write(f"🥈 ROI Placé: `{s['roi_sp']:.1f}%`")
                        st.write(f"🔗 ROI Couplé: `{s['roi_cg']:.1f}%`")
                        st.write(f"🔢 ROI Trio: `{s['roi_trio']:.1f}%`")
                        st.caption(f"Basé sur {s['count']} courses")
                        break

    st.divider()

    # --- Table Détaillée ---
    if model_order:
        best = model_order[0]
        st.markdown(f"### 🏇 Détails via {best['algo'].upper()} ({best['version']})")
        full_data = pd.merge(participant_data, best['df'][['program_number', 'win_probability', 'predicted_rank']], on='program_number', how='left').sort_values('win_probability', ascending=False)
        full_data['actual_result'] = full_data.apply(lambda r: format_actual_val(r.get('finish_rank'), r.get('incident_code')), axis=1)
        st.dataframe(full_data[['predicted_rank', 'actual_result', 'program_number', 'horse_name', 'jockey_name', 'reference_odds', 'live_odds_30mn', 'live_odds', 'win_probability']], width="stretch", hide_index=True,
            column_config={
                "win_probability": st.column_config.ProgressColumn("Prob.", format="%.1f%%", min_value=0, max_value=1),
                "live_odds": st.column_config.NumberColumn("Live", format="%.1f"),
                "live_odds_30mn": st.column_config.NumberColumn("30 mn", format="%.1f"),
                "predicted_rank": st.column_config.NumberColumn("IA", format="%d")
            })
