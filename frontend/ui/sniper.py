import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from api.api_client import get_sniper_bets
import state.store as store

def render_sniper_section():
    date_code = store.get_date_code()
    
    st.markdown("## 🎯 AI Market Scanner")
    
    with st.spinner(f"Scanning market for {date_code}..."):
        # sniper_bets is a list of dictionaries (contains both Sniper and Kelly)
        all_recs = get_sniper_bets(date_code)
        
        # We need race details (status, start_timestamp, timezone_offset) to display time/status
        races_df = store.get_races_data()
    
    if all_recs:
        # Separate strategies
        sniper_recs = [r for r in all_recs if r.get('strategy') == 'Sniper']
        kelly_recs = [r for r in all_recs if r.get('strategy', '').startswith('Kelly')]

        tab_sniper, tab_kelly = st.tabs([f"🎯 Sniper ({len(sniper_recs)})", f"⚖️ Kelly ({len(kelly_recs)})"])

        with tab_sniper:
            if sniper_recs:
                render_recommendation_table(sniper_recs, races_df)
            else:
                st.info("No Sniper opportunities found.")

        with tab_kelly:
            if kelly_recs:
                render_recommendation_table(kelly_recs, races_df)
            else:
                st.info("No Kelly opportunities found.")
    else:
        st.info("ℹ️ No high-value opportunities identified for this date.")

def render_recommendation_table(recommendations, races_df):
    bet_rows = []
    for bet in recommendations:
        # We need to find the corresponding race to get status and time
        race_info = None
        if not races_df.empty:
            # Try to find the race
            if 'race_id' in bet:
                race_match = races_df[races_df['race_id'] == bet['race_id']]
            else:
                # Fallback to meeting/race num match
                race_match = races_df[
                    (races_df['meeting_number'] == bet.get('meeting_num')) & 
                    (races_df['race_number'] == bet.get('race_num'))
                ]
            
            if not race_match.empty:
                race_info = race_match.iloc[0]

        # Determine Race Status String
        race_label = f"R{bet.get('meeting_num', '?')}C{bet.get('race_num', '?')}"
        
        race_col_val = race_label # Default
        outcome_str = ""

        if race_info is not None:
            status = race_info.get('race_status', 'DRAFT')
            
            if status in ['FIN', 'ANN', 'ABN']: # Finished, Annulé, Abandonné
                # Show status category if available, else status
                status_display = race_info.get('race_status_category', status)
                race_col_val = f"{race_label} ({status_display})"
                
                actual_pos = bet.get('actual_position')
                if actual_pos:
                    outcome_str = f" 🏁 **{actual_pos}**"
            else:
                # Show Time
                start_ts = race_info.get('start_timestamp')
                
                if pd.notnull(start_ts):
                    # Handle conversion if not already datetime
                    if not pd.api.types.is_datetime64_any_dtype(pd.Series([start_ts])):
                         start_ts = pd.to_datetime(start_ts, unit='ms')

                    # Add offset if exists
                    offset = race_info.get('timezone_offset', 0)
                    if pd.isna(offset):
                        offset = 0

                    if offset != 0:
                        local_time = start_ts + pd.Timedelta(milliseconds=offset)
                    else:
                        local_time = start_ts
                    
                    time_str = local_time.strftime('%H:%M')
                    race_col_val = f"{race_label} ({time_str})"
        
        # Construct row
        horse_display = f"#{bet.get('program_number', '?')} {bet.get('horse_name', 'Unknown')}"
        if outcome_str:
            horse_display += horse_display + outcome_str

        bet_rows.append({
            "Race": race_col_val,
            "Horse": horse_display,
            "Odds": f"{bet.get('odds', 0):.1f}",
            "AI Prob": bet.get('win_probability', 0), # Keep raw for column_config
            "Edge": f"+{bet.get('edge', 0)*100:.1f}%",
            "Model": bet.get('model_version', 'N/A'),
            "Strategy": bet.get('strategy', 'N/A')
        })
    
    # Display
    df_display = pd.DataFrame(bet_rows)
    # CORRECTION: use_container_width=True -> width="stretch"
    st.dataframe(
        df_display,
        width="stretch", #use_container_width=True,
        hide_index=True,
        column_config={
            "Race": st.column_config.TextColumn("Race", width="medium"),
            "Horse": st.column_config.TextColumn("Horse", width="large"),
            "AI Prob": st.column_config.ProgressColumn("Win Probability", min_value=0, max_value=1, format="%.2f%%"),
            "Edge": st.column_config.TextColumn("Edge", help="Diff AI vs Market"),
            "Model": st.column_config.TextColumn("Model", width="small"),
        }
    )
