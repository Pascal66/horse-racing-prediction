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
        st.table(df_trainers.style.format({
            "ROI": "{:.2f}%",
            "Success Rate": "{:.2f}%",
            "Avg Odds": "{:.2f}"
        }))

        # Chart ROI
        fig = px.bar(df_trainers, x="Trainer", y="ROI", color="ROI",
                     title="ROI par Trainer",
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

        # Discipline breakdown for selected trainer
        selected_trainer = st.selectbox("Détail par discipline pour :", list(trainers.keys()))
        if selected_trainer:
            disc_data = trainers[selected_trainer]["by_discipline"]
            disc_list = [{"Discipline": d, "ROI": v["roi"], "Count": v["count"]} for d, v in disc_data.items()]
            df_disc = pd.DataFrame(disc_list)
            st.dataframe(df_disc.style.format({"ROI": "{:.2f}%"}))

            fig_disc = px.bar(df_disc, x="Discipline", y="ROI", title=f"ROI par Discipline ({selected_trainer})")
            st.plotly_chart(fig_disc, use_container_width=True)

    else:
        st.warning("Aucune donnée de performance pour les trainers.")

    st.divider()
    st.caption("Données basées sur les rapports PMU définitifs quand disponibles, sinon sur les cotes de référence.")
