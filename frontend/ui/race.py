import streamlit as st
import state.store as store
from ui.analysis import render_analysis_view

def render_race_grid():
    races_df = store.get_races_data()
    selected_meeting = store.get_selected_meeting()

    if races_df is None or races_df.empty or selected_meeting is None:
        st.info("👈 Please select a date and meeting from the sidebar.")
        return

    # Filter races for current meeting
    meeting_races = races_df[
        races_df['meeting_number'] == selected_meeting
    ].sort_values('race_number')

    if meeting_races.empty:
        st.warning("No races found for this meeting.")
        return

    # Header
    # print(meeting_races.head())

    meeting_name = meeting_races.iloc[0]['meeting_libelle']
    date_str = store.get_date_code()
    # Format date for better display (DD/MM/YYYY)
    formatted_date = f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"

    st.markdown(f"## 🏟️ Meeting {selected_meeting} : {meeting_name} ({formatted_date})")

    # Create Tabs
    # race_labels = [f"C{r['race_number']}" for _, r in meeting_races.iterrows()]
    # tabs = st.tabs(race_labels)
    # Determine "Current" Race logic: first upcoming, OR last finished if all over.
    from datetime import timezone
    import pandas as pd
    now = pd.Timestamp.now(tz=timezone.utc)

    # Ensure start_timestamp is datetime (it was processed in sidebar.py and stored in races_df)
    upcoming_races = meeting_races[meeting_races['start_timestamp'] > now].sort_values('race_number')

    if not upcoming_races.empty:
        default_race_id = upcoming_races.iloc[0]['race_id']
    else:
        # All finished, pick the last one
        default_race_id = meeting_races.sort_values('race_number', ascending=False).iloc[0]['race_id']

    # Create Pills for Race Selection
    race_options = {r['race_id']: f"C{r['race_number']}" for _, r in meeting_races.iterrows()}

    # Get current index for the pill selection
    race_ids_list = list(race_options.keys())
    default_idx = race_ids_list.index(default_race_id)

    # Render Tabs
    # for (idx, race_row), tab in zip(meeting_races.iterrows(), tabs):
    #     with tab:
    # Use st.radio with a horizontal layout or st.pills if available in recent streamlit
    # Fallback to radio horizontal for compatibility
    selected_race_id = st.pills(
        "Select Race",
        options=race_ids_list,
        format_func=lambda x: race_options[x],
        selection_mode="single",
        default=default_race_id,
        label_visibility="collapsed"
    )

    if selected_race_id:
        store.set_selected_race(selected_race_id)
        # Render Content for selected race
        race_row = meeting_races[meeting_races['race_id'] == selected_race_id].iloc[0]
        render_race_tab_content(race_row)

def render_race_tab_content(race_row):
    """Renders the content inside a single race tab."""
    col_info, col_action = st.columns([3, 1])
    
    with col_info:
        st.markdown(f"### 🚩 C{race_row['race_number']} - {race_row['discipline']}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Distance", f"{race_row['distance_m']} m")
        m2.metric("Runners", f"{race_row.get('declared_runners_count', '-')}")
        m3.metric("Racetrack", f"PRIX {race_row['racetrack_libelle']}")

        render_analysis_view(race_row['race_id'])
