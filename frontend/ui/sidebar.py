import streamlit as st
from datetime import datetime, timezone
import pandas as pd
from api.api_client import fetch_daily_races, client
import state.store as store

def render_sidebar():
    with st.sidebar:
        st.title("🏇 Turf Analytics")
        st.markdown("---")
        
        # System Health (Short)
        health = client._get("/")
        if health:
            cols = st.columns(2)
            with cols[0]:
                st.caption(f"🧠 AI: {'🟢' if health.get('ml_engine') == 'loaded' else '🔴'}")
            with cols[1]:
                st.caption(f"📅 Job: {'🟢' if health.get('scheduler', {}).get('status') == 'running' else '🔴'}")

        st.markdown("---")
        st.subheader("📅 Schedule")
        
        # 1. Date Selection
        current_date = store.get_date_obj()
        new_date = st.date_input("Select a date", current_date)
        
        # Update state immediately if changed
        if new_date != current_date:
            store.set_date(new_date)
            st.rerun() # Force reload to fetch new data

        # 2. Fetch Data based on State
        date_code = store.get_date_code()
        
        with st.spinner("Loading schedule..."):
            races_df = fetch_daily_races(date_code)
            
            # Ensure start_timestamp is datetime for comparisons
            if not races_df.empty and 'start_timestamp' in races_df.columns:
                 # Convert milliseconds timestamp to datetime (UTC)
                 # JSON numbers are usually interpreted as int, so we specify unit='ms'
                 races_df['start_timestamp'] = pd.to_datetime(races_df['start_timestamp'], unit='ms')
                 
                 # Localize to UTC if naive
                 if races_df['start_timestamp'].dt.tz is None:
                     races_df['start_timestamp'] = races_df['start_timestamp'].dt.tz_localize('UTC')

            store.set_races_data(races_df)

        # 3. Meeting Selection
        if not races_df.empty:
            unique_meetings = sorted(races_df['meeting_number'].unique())
            
            # Helper to create label with next race time
            def format_meeting(m_num):
                m_races = races_df[races_df['meeting_number'] == m_num]
                if m_races.empty:
                    return f"R{m_num} - Unknown"
                
                track = m_races.iloc[0]['racetrack_code']
                
                # Find next race time
                now = pd.Timestamp.now(tz=timezone.utc)
                upcoming_races = m_races[m_races['start_timestamp'] > now]
                
                if not upcoming_races.empty:
                    # Get the next start time (UTC)
                    next_race_row = upcoming_races.sort_values('start_timestamp').iloc[0]
                    next_race_time_utc = next_race_row['start_timestamp']
                    
                    # Calculate display time (Local)
                    # Use timezone_offset if available (in milliseconds)
                    if 'timezone_offset' in next_race_row and pd.notnull(next_race_row['timezone_offset']):
                        offset_ms = next_race_row['timezone_offset']
                        local_time = next_race_time_utc + pd.Timedelta(milliseconds=offset_ms)
                    else:
                        local_time = next_race_time_utc

                    time_str = local_time.strftime('%H:%M')
                    return f"R{m_num} - {track} (Next: {time_str})"
                else:
                    return f"R{m_num} - {track} (Finished)"

            st.subheader("📍 Meeting")
            
            current_meeting = store.get_selected_meeting()
            
            # Logic to automatically select the meeting with the NEXT upcoming race globally
            if (current_meeting is None or current_meeting not in unique_meetings) and len(unique_meetings) > 0:
                # current_meeting = unique_meetings[0]
                now = pd.Timestamp.now(tz=timezone.utc)
                # Filter for all upcoming races across all meetings
                upcoming_all = races_df[races_df['start_timestamp'] > now]

                if not upcoming_all.empty:
                    # Select meeting of the very next race
                    next_race = upcoming_all.sort_values('start_timestamp').iloc[0]
                    current_meeting = next_race['meeting_number']
                else:
                    # If all finished, select the last meeting (likely the most recent one)
                    current_meeting = unique_meetings[-1]

                store.set_selected_meeting(current_meeting)

            selected_meeting = st.radio(
                "Choose a meeting:",
                unique_meetings,
                format_func=format_meeting,
                index=unique_meetings.index(current_meeting) if current_meeting in unique_meetings else 0
            )
            
            # Write to state
            store.set_selected_meeting(selected_meeting)

            st.markdown("---")
            st.caption("v3.2.1 • Fix Time Parsing")
        else:
            st.warning("No races available for this date.")