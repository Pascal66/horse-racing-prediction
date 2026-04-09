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
    Uses requests.Session for connection pooling.
    """
    def __init__(self):
        # Allow environment override for Docker (backend:8000)
        self.base_url = os.getenv("API_URL", "http://192.168.1.156:8000").rstrip("/")
        self.session = requests.Session()
        self.timeout = 10

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Internal helper for GET requests with error handling."""
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            logger.debug(f"GET {url} with params {params}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request Failed: {e}")
            return None

# Instantiate a global client
client = APIClient()

@st.cache_data(ttl=300)
def fetch_daily_races(date_code: str) -> pd.DataFrame:
    data = client._get(f"/races/{date_code}")
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_predictions(race_id: int) -> pd.DataFrame:
    data = client._get(f"/races/{race_id}/predict")
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=300)
def get_sniper_bets(date_str: str) -> List[Dict[str, Any]]:
    data = client._get(f"/bets/sniper/{date_str}")
    return data if data else []

@st.cache_data(ttl=300)
def fetch_participants(race_id: int) -> pd.DataFrame:
    data = client._get(f"/races/{race_id}/participants")
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_model_metrics(model_name: Optional[str] = None) -> pd.DataFrame:
    """Retrieves Back-Analysis metrics from the backend."""
    params = {}
    if model_name:
        params["model_name"] = model_name
    data = client._get("/metrics", params=params)
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=7200)
def fetch_backtest_results() -> Dict[str, Any]: # -> pd.DataFrame: #
    """Retrieves detailed backtesting results."""
    data = client._get("/backtest")
    return data if data else {}
    # return pd.DataFrame(data) if data else pd.DataFrame()