import streamlit as st
import pandas as pd
import json
import numpy as np

from api.api_client import fetch_portfolio_data


def calculate_drawdown(series):
    """Calculates the maximum drawdown of a cumulative profit series."""
    if series.empty:
        return 0
    peak = series.cummax()
    drawdown = (series - peak)
    return drawdown.min()


def render_portfolio_tab():
    st.header("💼 Portfolio & Equity Curves")

    with st.spinner("Chargement des données du portefeuille..."):
        # Fetch only since 01 january of last year
        last_year = pd.Timestamp.today().year - 1
        start_date = pd.Timestamp(f'{last_year}-01-01')
        nb_days = (pd.Timestamp.today() - start_date).days + 1
        data = fetch_portfolio_data(days=nb_days)

    if not data:
        st.warning("Aucune donnée de performance historique trouvée dans la base de données.")
        return

    df = pd.DataFrame(data)
    df['performance_date'] = pd.to_datetime(df['performance_date'])
    # Calculate profit per entry: nb_bets * (roi / 100)
    df['daily_profit'] = df['nb_bets'] * (df['roi'] / 100.0)

    # --- 1. FILTERS ---
    with st.expander("🔍 Filtres et Configuration", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            models = sorted(df['model_version'].unique())
            selected_models = st.multiselect("Modèles", models,
                                             default=models[:2] if len(models) > 1 else models)
        with c2:
            bet_types = sorted(df['bet_type'].unique())
            selected_bet_types = st.multiselect("Types de Paris", bet_types,
                                                default=["SG", "SP"] if "SG" in bet_types and "SP" in bet_types else bet_types[:1])
        with c3:
            disciplines = sorted(df['discipline'].unique()) if 'discipline' in df.columns else []
            if disciplines:
                selected_disciplines = st.multiselect("Disciplines", disciplines, default=disciplines)
            else:
                selected_disciplines = []

    # Filtering
    mask = df['model_version'].isin(selected_models) & df['bet_type'].isin(selected_bet_types)
    if disciplines and selected_disciplines:
        mask &= df['discipline'].isin(selected_disciplines)
    
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        st.info("Veuillez ajuster les filtres pour afficher des données.")
        return

    # --- 2. KPI METRICS ---
    total_bets = filtered_df['nb_bets'].sum()
    total_wins = filtered_df['nb_wins'].sum()
    total_profit = filtered_df['daily_profit'].sum()
    global_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Profit Total", f"{total_profit:,.2f} €")
    m2.metric("ROI Global", f"{global_roi:.2f} %")
    m3.metric("Paris Totaux", f"{total_bets}")
    m4.metric("Taux de Réussite", f"{win_rate:.2f} %")

    # --- 3. DATA PROCESSING FOR CHART ---
    daily_stats = filtered_df.groupby(['model_version', 'performance_date']).agg({
        'daily_profit': 'sum',
        'nb_bets': 'sum',
        'nb_wins': 'sum'
    }).reset_index().sort_values('performance_date')

    daily_stats['cum_profit'] = daily_stats.groupby('model_version')['daily_profit'].cumsum()

    # --- 4. EQUITY CHART (Chart.js) ---
    st.subheader("📈 Courbes d'Équité (Cumul de Profit)")

    datasets = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
    all_dates = sorted(daily_stats['performance_date'].unique())
    labels = [d.strftime('%Y-%m-%d') for d in all_dates]

    for i, model in enumerate(selected_models):
        model_data = daily_stats[daily_stats['model_version'] == model]
        if model_data.empty:
            continue
            
        model_series = pd.Series(index=all_dates, data=np.nan)
        for _, row in model_data.iterrows():
            model_series[row['performance_date']] = row['cum_profit']

        model_series = model_series.ffill().fillna(0)

        datasets.append({
            "label": model,
            "data": model_series.tolist(),
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "33",
            "fill": False,
            "tension": 0.1,
            "pointRadius": 2
        })

    chart_json = json.dumps({"labels": labels, "datasets": datasets})
    html_code = f"""
    <div style="width: 100%; height: 450px;">
        <canvas id="equityChart"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
        new Chart(document.getElementById('equityChart').getContext('2d'), {{
            type: 'line',
            data: {chart_json},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ mode: 'index', intersect: false }},
                scales: {{
                    y: {{ title: {{ display: true, text: 'Profit Cumulé (€)' }}, grid: {{ color: '#f0f0f0' }} }},
                    x: {{ grid: {{ display: false }} }}
                }},
                plugins: {{ legend: {{ position: 'bottom' }} }}
            }}
        }});
    </script>
    """
    st.iframe(html_code, height=460)

    # --- 5. SUMMARY TABLE ---
    st.subheader("📋 Analyse Détaillée par Modèle")
    
    summary_list = []
    for model in selected_models:
        m_df = daily_stats[daily_stats['model_version'] == model]
        if m_df.empty: continue
        
        m_profit = m_df['daily_profit'].sum()
        m_bets = m_df['nb_bets'].sum()
        m_wins = m_df['nb_wins'].sum()
        m_dd = calculate_drawdown(m_df['cum_profit'])
        
        summary_list.append({
            "Modèle": model,
            "Profit (€)": round(m_profit, 2),
            "ROI (%)": round((m_profit / m_bets * 100), 2) if m_bets > 0 else 0,
            "Max Drawdown (€)": round(m_dd, 2),
            "Taux R. (%)": round((m_wins / m_bets * 100), 2) if m_bets > 0 else 0,
            "Paris": m_bets
        })
    
    st.dataframe(pd.DataFrame(summary_list), hide_index=True, width='stretch')
