import streamlit as st
import pandas as pd
import json
import streamlit.components.v1 as components
from api.api_client import fetch_portfolio_data

def render_portfolio_tab():
    st.header("💼 Portfolio & Equity Curves")

    with st.spinner("Chargement des données du portefeuille..."):
        data = fetch_portfolio_data(days=365)

    if not data:
        st.warning("Aucune donnée de performance historique trouvée dans la base de données.")
        return

    df = pd.DataFrame(data)
    df['performance_date'] = pd.to_datetime(df['performance_date'])

    # 1. Filters
    cols = st.columns(2)
    with cols[0]:
        models = sorted(df['model_version'].unique())
        selected_models = st.multiselect("Sélectionner les Modèles", models, default=models[:2] if len(models) > 1 else models)

    with cols[1]:
        bet_types = sorted(df['bet_type'].unique())
        selected_bet_types = st.multiselect("Sélectionner les Types de Paris", bet_types, default=["SG", "SP"] if "SG" in bet_types and "SP" in bet_types else bet_types[:1])

    if not selected_models or not selected_bet_types:
        st.info("Veuillez sélectionner au moins un modèle et un type de pari.")
        return

    # 2. Data Processing
    filtered_df = df[df['model_version'].isin(selected_models) & df['bet_type'].isin(selected_bet_types)].copy()

    if filtered_df.empty:
        st.warning("Aucune donnée pour la sélection actuelle.")
        return

    # Calculate daily profit: nb_bets * (roi / 100)
    filtered_df['daily_profit'] = filtered_df['nb_bets'] * (filtered_df['roi'] / 100.0)

    # Group by model and date to get daily totals (summing across disciplines for selected bet types)
    daily_stats = filtered_df.groupby(['model_version', 'performance_date']).agg({
        'daily_profit': 'sum',
        'nb_bets': 'sum'
    }).reset_index().sort_values('performance_date')

    # Calculate cumulative profit per model
    daily_stats['cum_profit'] = daily_stats.groupby('model_version')['daily_profit'].cumsum()

    # 3. Chart.js Visualization
    st.subheader("📈 Courbes d'Équité (Cumul de Profit)")

    # Prepare data for Chart.js
    datasets = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']

    all_dates = sorted(daily_stats['performance_date'].unique())
    labels = [d.strftime('%Y-%m-%d') for d in all_dates]

    for i, model in enumerate(selected_models):
        model_data = daily_stats[daily_stats['model_version'] == model]
        # Align with all dates to avoid gaps
        model_series = pd.Series(index=all_dates, data=float('nan'))
        for _, row in model_data.iterrows():
            model_series[row['performance_date']] = row['cum_profit']

        # Forward fill to show continuous line even if no bets on some days
        model_series = model_series.ffill().fillna(0)

        datasets.append({
            "label": model,
            "data": model_series.tolist(),
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "33",
            "fill": False,
            "tension": 0.1
        })

    chart_data = {
        "labels": labels,
        "datasets": datasets
    }

    chart_json = json.dumps(chart_data)

    html_code = f"""
    <div style="width: 100%; height: 500px;">
        <canvas id="equityChart"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {chart_json},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Date'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Profit Cumulé (€)'
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    tooltip: {{
                        enabled: true
                    }}
                }}
            }}
        }});
    </script>
    """

    components.html(html_code, height=550)

    # 4. Summary Table
    st.subheader("📋 Résumé des Performances")
    summary = filtered_df.groupby('model_version').agg({
        'nb_bets': 'sum',
        'nb_wins': 'sum',
        'daily_profit': 'sum'
    }).reset_index()

    summary['ROI Global'] = (summary['daily_profit'] / summary['nb_bets'] * 100).round(2)
    summary['Taux de Réussite'] = (summary['nb_wins'] / summary['nb_bets'] * 100).round(2)
    summary.rename(columns={'daily_profit': 'Profit Total (€)', 'nb_bets': 'Total Paris', 'nb_wins': 'Total Gagnants'}, inplace=True)

    st.dataframe(summary, hide_index=True, use_container_width=True)
