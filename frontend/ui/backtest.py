import streamlit as st
import pandas as pd
import plotly.express as px
from api.api_client import fetch_backtest_results


def render_backtest_tab():
    st.header("📊 Backtesting & Rentability Analysis")
    st.info("Cette section reproduit l'analyse de rentabilité basée sur les données historiques.")

    with st.spinner("Chargement des données de backtesting..."):
        results = fetch_backtest_results()

    if not results or "error" in results:
        st.error(results.get("error", "Erreur lors de la récupération des données de backtesting."))
        return

    # --- Metrics Summary ---
    st.subheader("🚀 Stratégies de Paris")
    strategies = results.get("strategies", {})
    sniper = strategies.get("sniper", {})
    kelly = strategies.get("kelly", {})

    col_s1, col_s2, col_s3 = st.columns(3)
    if sniper:
        col_s1.metric("ROI Sniper", f"{sniper['roi']:.2f}%")
        col_s2.metric("Taux Réussite Sniper", f"{sniper['win_rate']:.2f}%")
        col_s3.metric("Paris Sniper", sniper['total_bets'])
    else:
        col_s1.info("Sniper: Pas de données")

    col_k1, col_k2, col_k3 = st.columns(3)
    if kelly:
        col_k1.metric("ROI Kelly (Sim)", f"{kelly['roi']:.2f}%")
        col_k2.metric("Profit Kelly", f"{kelly['total_profit']:.2f}€")
        col_k3.metric("Paris Kelly", kelly['total_bets'])
    else:
        col_k1.info("Kelly: Pas de données")

    st.divider()

    # --- Trainers Performance ---
    st.subheader("🧠 Performance des Trainers")
    trainers = results.get("trainers", {})
    if trainers:
        trainer_list = []
        for name, data in trainers.items():
            trainer_list.append({
                "Trainer": name,
                "ROI": data["roi"],
                "Success Rate": data["win_rate"],
                "Bets": data["total_bets"],
                "Avg Odds": data["avg_odds"]
            })

        df_trainers = pd.DataFrame(trainer_list)
        
        st.dataframe(
            df_trainers,
            hide_index=True,
            column_config={
                "Success Rate": st.column_config.NumberColumn("Success Rate", format="%.2f%%"),
                "ROI": st.column_config.NumberColumn("ROI", format="%.2f%%"),
                "Avg Odds": st.column_config.NumberColumn("Avg Odds", format="%.2f"),
            },
            width='stretch' #use_container_width=True
        )

        # Export CSV des performances globales
        csv_trainers = df_trainers.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger Performance Trainers (CSV)",
            data=csv_trainers,
            file_name='backtest_trainers_performance.csv',
            mime='text/csv',
        )

        # Chart ROI
        fig = px.bar(df_trainers, x="Trainer", y="ROI", color="ROI",
                     title="ROI par Trainer",
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, width='stretch') #use_container_width=True)

        # --- Analyse Saisonnière Avancée ---
        st.subheader("📅 Matrice de Performance Saisonnière")
        selected_trainer = st.selectbox("Sélectionner un modèle pour l'analyse temporelle :", list(trainers.keys()))
        
        if selected_trainer:
            seasonal_data = trainers[selected_trainer].get("seasonal_analysis", {})
            
            if seasonal_data:
                # Transformation des données pour le heatmap
                rows = []
                for discipline, months in seasonal_data.items():
                    for month, stats in months.items():
                        rows.append({
                            "Discipline": discipline,
                            "Mois": int(month),
                            "ROI": stats["roi"],
                            "ROI_Place": stats.get("roi_place", 0.0),
                            "Win_Rate": stats["win_rate"],
                            "Volume": stats["count"],
                            "Avg_Odds": stats["avg_odds"]
                        })
                
                df_seasonal = pd.DataFrame(rows)
                
                if not df_seasonal.empty:
                    # Heatmap du ROI par Mois et Discipline
                    pivot_roi = df_seasonal.pivot(index="Discipline", columns="Mois", values="ROI")
                    fig_heat = px.imshow(pivot_roi, 
                                        labels=dict(x="Mois (1-12)", y="Discipline", color="ROI %"),
                                        color_continuous_scale="RdYlGn",
                                        text_auto=".0f",
                                        title=f"Saisonnalité du ROI : {selected_trainer}")
                    st.plotly_chart(fig_heat, width='stretch') # use_container_width=True)
                    
                    # Export CSV des données saisonnières
                    csv_seasonal = df_seasonal.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Télécharger Analyse Saisonnière {selected_trainer} (CSV)",
                        data=csv_seasonal,
                        file_name=f'backtest_seasonal_{selected_trainer}.csv',
                        mime='text/csv',
                    )
                    
                    st.write("💡 *Conseil : Ciblez les cellules vert foncé avec un volume de paris suffisant.*")
                else:
                    st.info("Pas assez de données pour l'analyse saisonnière de ce trainer.")


    else:
        st.warning("Aucune donnée de performance pour les trainers.")

    st.divider()
    st.caption("Données basées sur les rapports PMU définitifs quand disponibles, sinon sur les cotes de référence.")
