import streamlit as st
import pandas as pd
import plotly.express as px
from api.api_client import fetch_backtest_results

def render_backtest_tab():
    st.header("📊 Backtesting & Rentability Analysis")
    
    # Bouton de rafraîchissement
    col_ref, col_info = st.columns([1, 4])
    
    # On utilise un flag dans st.session_state pour savoir si on doit forcer l'update
    if 'force_backtest' not in st.session_state:
        st.session_state.force_backtest = False

    with col_ref:
        if st.button("🔄 Actualiser (LENT)"):
            st.session_state.force_backtest = True

    with col_info:
        st.info("Les résultats sont mis en cache sur le disque pour un affichage instantané. L'actualisation peut prendre 2-3 minutes.")

    with st.spinner("Chargement des données de backtesting..."):
        # L'API client gère maintenant le paramètre force_update
        results = fetch_backtest_results(force_update=st.session_state.force_backtest)
        # On remet le flag à False après l'appel
        st.session_state.force_backtest = False

    if not results:
        st.error("Aucune réponse du serveur (API injoignable ou timeout).")
        return

    if "error" in results:
        st.error(f"Erreur API : {results.get('error')}")
        with st.expander("Détails techniques (Raw JSON)"):
            st.json(results)
        return

    if "last_updated" in results:
        st.caption(f"📅 Dernière mise à jour complète : {results['last_updated']}")

    # --- Metrics Summary ---
    st.subheader("🚀 Stratégies de Paris")
    strategies = results.get("strategies", {})
    sniper = strategies.get("sniper", {})
    kelly = strategies.get("kelly", {})

    col_s1, col_s2, col_s3 = st.columns(3)
    if True: #sniper:
        col_s1.metric("ROI Sniper", f"{sniper.get('roi', 0):.2f}%")
        col_s2.metric("Taux Réussite Sniper", f"{sniper.get('win_rate', 0):.2f}%")
        col_s3.metric("Paris Sniper", sniper.get('total_bets', 0))

    col_k1, col_k2, col_k3 = st.columns(3)
    if True: #kelly:
        col_k1.metric("ROI Kelly (Sim)", f"{kelly.get('roi', 0):.2f}%")
        col_k2.metric("Profit Kelly", f"{kelly.get('total_profit', 0):.2f}€")
        col_k3.metric("Paris Kelly", kelly.get('total_bets', 0))

    composite = strategies.get("composite", {})
    if True: #composite:
        st.subheader("🌟 Stratégie Composite (Auto-Selection)")
        c1, c2, c3 = st.columns(3)
        c1.metric("ROI SG", f"{composite.get('roi', 0):.2f}%", help="Simple Gagnant")
        c2.metric("Win Rate", f"{composite.get('win_rate', 0):.2f}%")
        c3.metric("Total Paris", composite.get('total_bets', 0))

        c4, c5, c6 = st.columns(3)
        c4.metric("ROI CG", f"{composite.get('roi_cg', 0):.2f}%")
        c5.metric("ROI CP", f"{composite.get('roi_cp', 0):.2f}%")
        c6.metric("ROI Trio", f"{composite.get('roi_trio', 0):.2f}%")

    st.divider()

    # --- Trainers Performance ---
    st.subheader("🧠 Performance des Trainers")
    trainers = results.get("trainers", {})
    if trainers:
        trainer_list = []
        for name, data in trainers.items():
            trainer_list.append({
                "Trainer": name,
                "ROI SG": data.get("roi", 0),
                "ROI SP": data.get("roi_place", 0),
                "ROI CG": data.get("roi_cg", 0),
                "ROI CP": data.get("roi_cp", 0),
                "ROI Trio": data.get("roi_trio", 0),
                "Win Rate": data.get("win_rate", 0),
                "Bets": data.get("total_bets", 0),
                "Avg Odds": data.get("avg_odds", 0)
            })

        df_trainers = pd.DataFrame(trainer_list)
        st.dataframe(
            df_trainers,
            hide_index=True,
            column_config={
                "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.2f%%"),
                # "ROI SG": st.column_config.NumberColumn("ROI SG", format="%.2f%%"),
                # "ROI SP": st.column_config.NumberColumn("ROI SP", format="%.2f%%"),
                # "ROI CG": st.column_config.NumberColumn("ROI CG", format="%.2f%%"),
                # "ROI CP": st.column_config.NumberColumn("ROI CP", format="%.2f%%"),
                # "ROI Trio": st.column_config.NumberColumn("ROI Trio", format="%.2f%%"),
                # "Avg Odds": st.column_config.NumberColumn("Avg Odds", format="%.2f"),
                "ROI SG": st.column_config.NumberColumn("ROI SG", format="%.1f%%"),
                "ROI SP": st.column_config.NumberColumn("ROI SP", format="%.1f%%"),
                "ROI CG": st.column_config.NumberColumn("ROI CG", format="%.1f%%"),
                "ROI CP": st.column_config.NumberColumn("ROI CP", format="%.1f%%"),
                "ROI Trio": st.column_config.NumberColumn("ROI Trio", format="%.1f%%"),
                "Avg Odds": st.column_config.NumberColumn("Avg Odds", format="%.2f"),
                "Bets": st.column_config.NumberColumn("Nombre de paris"),
            },
            width='stretch'
        )

        # Export CSV
        csv_trainers = df_trainers.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Télécharger Performance Trainers (CSV)", csv_trainers, 'backtest_trainers.csv', 'text/csv')

        fig = px.bar(df_trainers, x="Trainer", y="ROI SG", color="ROI SG", title="ROI SG par Trainer", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, width='stretch')
        st.divider()

        # --- Detailed Model Analysis ---
        st.subheader("🔍 Analyse Détaillée par Modèle")
        selected_trainer = st.selectbox("Sélectionner un modèle pour analyse approfondie :", list(trainers.keys()))

        if selected_trainer:
            t_data = trainers[selected_trainer]

            # --- ROI Evolution Trend ---
            trend_data = t_data.get("daily_trend", [])
            if trend_data:
                df_trend = pd.DataFrame(trend_data)
                df_trend['date'] = pd.to_datetime(df_trend['date'])
                fig_trend = px.line(df_trend, x='date', y='profit',
                                    title=f"Évolution du Profit (SG) : {selected_trainer}",
                                    labels={'profit': 'Profit Cumulé (€)', 'date': 'Date'})
                fig_trend.update_traces(line_color='#00CC96')
                st.plotly_chart(fig_trend, width='stretch') # use_container_width=True)
            else:
                st.info("Données de tendance non disponibles pour ce modèle.")

            # --- ROI by Discipline ---
            disc_data = t_data.get("discipline_analysis", {})
            if disc_data:
                df_disc = pd.DataFrame(
                    [{"Discipline": d, "ROI": s["roi"], "Bets": s["count"]} for d, s in disc_data.items()])
                fig_disc = px.bar(df_disc, x="Discipline", y="ROI", color="ROI", text="Bets",
                                  title=f"ROI par Discipline : {selected_trainer}",
                                  color_continuous_scale="RdYlGn",
                                  labels={"ROI": "ROI %", "Bets": "Nombre de paris"})
                fig_disc.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig_disc, width='stretch') # use_container_width=True)

        # --- Analyse Saisonnière ---
        st.subheader("📅 Matrice de Performance Saisonnière")
        # selected_trainer = st.selectbox("Sélectionner un modèle :", list(trainers.keys()))
        
        if selected_trainer:
            seasonal_data = trainers[selected_trainer].get("seasonal_analysis", {})
            if seasonal_data:
                rows = []
                for disc, months in seasonal_data.items():
                    for mon, stats in months.items():
                        rows.append({"Discipline": disc, "Mois": int(mon), "ROI": stats["roi"], "Volume": stats["count"]})
                
                df_seasonal = pd.DataFrame(rows)
                if not df_seasonal.empty:
                    pivot_roi = df_seasonal.pivot(index="Discipline", columns="Mois", values="ROI")
                    fig_heat = px.imshow(pivot_roi, labels=dict(x="Mois", y="Discipline", color="ROI %"), color_continuous_scale="RdYlGn", text_auto=".0f", title=f"Saisonnalité : {selected_trainer}")
                    st.plotly_chart(fig_heat, width='stretch')
                else:
                    st.info("Pas assez de données pour l'analyse saisonnière.")
    else:
        st.warning("Aucune donnée de performance pour les trainers.")

    st.divider()
    st.caption("Données basées sur les rapports PMU définitifs quand disponibles.")
    
    with st.expander("🛠 Zone de Debug"):
        st.write("Dernière réponse brute de l'API :")
        st.json(results)
