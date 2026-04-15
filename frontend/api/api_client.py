import logging
import requests
import pandas as pd
import streamlit as st
import os
from typing import List, Dict, Any, Optional

# Logger Setup
logger = logging.getLogger(__name__)

class APIClient:
    """
    Singleton-style API client to handle all backend communication.
    """
    def __init__(self):
        self.base_url = os.getenv("API_URL", "http://192.168.1.156:8000").rstrip("/")
        self.session = requests.Session()
        self.timeout = 300

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": f"Timeout après {self.timeout} secondes."}
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

client = APIClient()

@st.cache_data(ttl=300)
def fetch_daily_races(date_code: str) -> pd.DataFrame:
    data = client._get(f"/races/{date_code}")
    return pd.DataFrame(data) if data and not isinstance(data, dict) else pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_predictions(race_id: int, algo: str = "auto") -> pd.DataFrame:
    """Retrieves predictions for a race, allowing algorithm selection."""
    params = {"algo": algo}
    data = client._get(f"/races/{race_id}/predict", params=params)
    return pd.DataFrame(data) if data and not isinstance(data, dict) else pd.DataFrame()

@st.cache_data(ttl=300)
def get_sniper_bets(date_str: str) -> List[Dict[str, Any]]:
    data = client._get(f"/bets/sniper/{date_str}")
    return data if data and not isinstance(data, dict) else []

@st.cache_data(ttl=300)
def fetch_participants(race_id: int) -> pd.DataFrame:
    data = client._get(f"/races/{race_id}/participants")
    return pd.DataFrame(data) if data and not isinstance(data, dict) else pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_model_metrics(model_name: Optional[str] = None) -> pd.DataFrame:
    params = {}
    if model_name: params["model_name"] = model_name
    data = client._get("/metrics", params=params)
    return pd.DataFrame(data) if data and not isinstance(data, dict) else pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_backtest_results(force_update: bool = False) -> Dict[str, Any]:
    params = {"force": force_update}
    data = client._get("/backtest", params=params)
    return data if data is not None else {"error": "Réponse vide"}
