import streamlit as st
import pandas as pd
from api.api_client import fetch_predictions, fetch_participants, fetch_backtest_results

pd.set_option('display.max_columns', None)

def get_horse_color(p_num: int) -> str:
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
    with st.spinner("Chargement de l'analyse..."):
        backtest = fetch_backtest_results()
        participant_data = fetch_participants(race_id)
        preds = {
            "tabnet": fetch_predictions(race_id, algo="tabnet"),
            "ltr": fetch_predictions(race_id, algo="ltr"),
            "hyperstack": fetch_predictions(race_id, algo="hyperstack")
        }

    if participant_data.empty:
        st.warning("Données participants indisponibles.")
        return

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
            df_p['horse_name'] = df_p['program_number'].map(horse_map)
            model_order.append({"algo": algo_key, "roi": roi, "df": df_p, "version": m_ver})
    model_order.sort(key=lambda x: x["roi"], reverse=True)

    # --- LAYOUT ---
    col_main, col_side = st.columns([2.5, 1])

    with col_main:
        st.subheader("🏆 Pronostics Comparés")
        if not model_order:
            st.info("Aucune prédiction.")
        else:
            cols = st.columns(len(model_order))
            for idx, m in enumerate(model_order):
                with cols[idx]:
                    roi_val = m['roi']
                    roi_color = "#2e7d32" if roi_val > 0 else ("#9e9e9e" if roi_val == -99.0 else "#d32f2f")
                    st.markdown(f"""<div style="text-align:center; border-bottom:3px solid {roi_color}; margin-bottom:10px;">
                        <b style="font-size:1.1em;">{m['algo'].upper()}</b><br>
                        <span style="color:{roi_color}; font-weight:bold;">ROI: {roi_val:.1f}%</span>
                    </div>""", unsafe_allow_html=True)
                    
                    top_df = m['df'].sort_values('win_probability', ascending=False).head(3)
                    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
                        p_num = row['program_number']
                        h_name = horse_map.get(p_num, f"n°{p_num}")
                        prob = row['win_probability'] * 100
                        actual = result_map.get(p_num, "")
                        h_color = get_horse_color(p_num)
                        bg = "#fff9c4" if rank == 1 else "#ffffff"
                        st.html(f"""<div style="background:{bg}; border:1px solid #ddd; border-left:6px solid {h_color}; padding:6px; border-radius:4px; margin-bottom:4px; font-size:0.85em; position:relative;">
                                {f'<span style="float:right; background:#4CAF50; color:white; padding:0px 5px; border-radius:3px; font-weight:bold;">{actual}</span>' if actual else ""}
                                <b>#{p_num}</b> {h_name[:14]}<br>
                                <span style="color:#555;">Prob: <b>{prob:.1f}%</b></span>
                            </div>""")

    with col_side:
        st.subheader("⏱️ Live Tracking")
        
        def render_period_stats(title, period_data):
            st.markdown(f"**{title}**")
            t_data = period_data.get("trainers", period_data) if period_data else {}
            if not t_data:
                st.caption("Aucune donnée")
                return
            
            for btype in ['SG', 'SP', 'CG', 'TRIO']:
                pk = f"profit_{btype.lower()}" if btype!='SG' else "profit"
                nk = f"nb_bets_{btype.lower()}" if btype!='SG' else "count"
                wk = f"nb_wins_{btype.lower()}" if btype!='SG' else "nb_wins"
                
                valid_models = [k for k in t_data.keys() if t_data[k].get(nk, 0) > 0]
                if not valid_models: continue
                
                # On cherche l'expert par type de pari
                best_m = max(valid_models, key=lambda k: t_data[k].get(pk, -9999))
                d = t_data[best_m]
                profit, nb, wins = d.get(pk, 0), d.get(nk, 0), d.get(wk, 0)
                
                if nb > 0:
                    courses = int(nb / (3 if btype in ['SG', 'SP', 'CG'] else 1))
                    color = "#2e7d32" if profit > 0 else "#d32f2f"
                    st.markdown(f"""
                        <div style='font-size:0.75em; margin-bottom:5px; line-height:1.2;'>
                            {btype} : <span style='color:{color}; font-weight:bold;'>{profit:+.1f}€</span> ({wins}/{courses})<br>
                            <code style='font-size:0.8em; color:#666;'>via {best_m}</code>
                        </div>
                    """, unsafe_allow_html=True)

        render_period_stats("Aujourd'hui", backtest.get("today", {}))
        st.write("")
        render_period_stats("Hier", backtest.get("yesterday", {}))

        st.divider()
        st.subheader("📊 Normes")
        if model_order:
            best = model_order[0]
            st.info(f"Historique {best['algo'].upper()}")
            m_stats = trainer_stats.get(best['version'], {}).get("seasonal_analysis", {})
            for disc, months in m_stats.items():
                cur_month = pd.Timestamp.now().month
                s = months.get(cur_month) or months.get(str(cur_month))
                if s:
                    st.write(f"ROI SG: `{s['roi']:.1f}%` ({s['count']} courses)")
                    break

    st.divider()
    if model_order:
        best = model_order[0]
        st.markdown(f"### 🏇 Détails : {best['algo'].upper()} ({best['version']})")
        full_data = pd.merge(participant_data, best['df'][['program_number', 'win_probability', 'predicted_rank']], on='program_number', how='left').sort_values('win_probability', ascending=False)
        full_data['actual_result'] = full_data.apply(lambda r: format_actual_val(r.get('finish_rank'), r.get('incident_code')), axis=1)
        st.dataframe(full_data[['predicted_rank', 'actual_result', 'program_number', 'horse_name', 'jockey_name', 'reference_odds', 'live_odds_30mn', 'live_odds', 'win_probability']], use_container_width=True, hide_index=True,
            column_config={
                "win_probability": st.column_config.ProgressColumn("Prob.", format="%.1f%%", min_value=0, max_value=1),
                "live_odds": st.column_config.NumberColumn("Live", format="%.1f"),
                "live_odds_30mn": st.column_config.NumberColumn("30 mn", format="%.1f"),
                "predicted_rank": st.column_config.NumberColumn("IA", format="%d")
            })
