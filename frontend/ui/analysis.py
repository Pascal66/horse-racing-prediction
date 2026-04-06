import streamlit as st
import pandas as pd
from api.api_client import fetch_predictions, fetch_participants
pd.set_option('display.max_columns', None)

def render_analysis_view(race_id: int):
    """
    Renders the detailed prediction tables and cards for a specific race.
    """
    with st.spinner("Loading race data..."):
        prediction_data = fetch_predictions(race_id)
        participant_data = fetch_participants(race_id)
    
    if participant_data.empty:
        st.warning("No participant data available for this race.")
        return

    # Merge Logic
    if not prediction_data.empty:
        full_race_data = pd.merge(
            participant_data,
            prediction_data[['program_number', 'win_probability', 'predicted_rank']],
            on='program_number', 
            how='left'
        )
    else:
        full_race_data = participant_data.copy()
        full_race_data['win_probability'] = None
        full_race_data['predicted_rank'] = None

    full_race_data.sort_values(by=['win_probability'], ascending=False, inplace=True)

    # Top 3 Cards
    if not prediction_data.empty:
        st.subheader("🏆 AI Forecast")
        col1, col2, col3 = st.columns(3)
        top_3 = full_race_data.head(3)
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"] 
        
        for i, (idx_r, row) in enumerate(top_3.iterrows()):
            if i < len(top_3):
                with [col1, col2, col3][i]:
                    st.markdown(
                        f"""<div style="background:white; border-top:5px solid {colors[i]}; padding:15px; border-radius:8px; text-align:center; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                            <h2 style="margin:0; color:#333;">#{row.get('program_number', '-')}</h2>
                            <div style="font-weight:bold; color:#555;">{row.get('horse_name', 'Unknown')}</div>
                            <div style="color:{colors[i]}; font-size:1.4em; font-weight:bold; margin-top:5px;">{row.get('win_probability', 0)*100:.1f}%</div>
                            {f'<div style="color:red; font-weight:bold; margin-top:5px;">Result: {int(row["finish_rank"])}</div>' if 'finish_rank' in row and pd.notnull(row['finish_rank']) 
                            else '<div></div>'}
                        """, unsafe_allow_html=True
                    )

    # 2. Detailed Table
    st.markdown("### 🏇 Participants & Analysis")
    
    # Mapping and columns selection
    # Back-compatibility: ensure we have columns or defaults
    if 'reference_odds' not in full_race_data.columns: full_race_data['reference_odds'] = None
    if 'live_odds' not in full_race_data.columns: full_race_data['live_odds'] = None

    # display_cols = ['predicted_rank', 'program_number', 'horse_name', 'jockey_name', 'trainer_name', 'reference_odds', 'live_odds', 'win_probability']
    display_cols = ['predicted_rank', 'finish_rank', 'program_number', 'horse_name', 'jockey_name', 'trainer_name', 'reference_odds', 'live_odds', 'win_probability']
    display_cols = [c for c in display_cols if c in full_race_data.columns]

    st.dataframe(
        full_race_data[display_cols],
        width=1200,
        hide_index=True,
        column_config={
            # "predicted_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "predicted_rank": st.column_config.NumberColumn("AI Pred", format="%d"),
            "finish_rank": st.column_config.NumberColumn("Actual", format="%d 🏁"),
            "program_number": "No.",
            "horse_name": "Horse",
            "jockey_name": "Jockey/Driver",
            "trainer_name": "Trainer",
            "reference_odds": st.column_config.NumberColumn("Ref. Odds", format="%.1f 🏁"),
            "live_odds": st.column_config.NumberColumn("Live Odds", format="%.1f 🔥"),
            "win_probability": st.column_config.ProgressColumn(
                "Win Prob.",
                format="%.1f%%",
                min_value=0, 
                max_value=1
            )
        }
    )
