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
        # On passe à 300s (5 minutes) pour les calculs lourds de backtesting
        self.timeout = 300

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """Internal helper for GET requests with error handling."""
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            logger.debug(f"GET {url} with params {params}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"API Request Timeout (>{self.timeout}s) on {endpoint}")
            return {"error": f"Timeout après {self.timeout} secondes."}
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request Failed: {e}")
            return {"error": str(e)}

# Instantiate a global client
client = APIClient()

@st.cache_data(ttl=300)
def fetch_daily_races(date_code: str) -> pd.DataFrame:
    data = client._get(f"/races/{date_code}")
    if isinstance(data, dict) and "error" in data: return pd.DataFrame()
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_predictions(race_id: int) -> pd.DataFrame:
    data = client._get(f"/races/{race_id}/predict")
    if isinstance(data, dict) and "error" in data: return pd.DataFrame()
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=300)
def get_sniper_bets(date_str: str) -> List[Dict[str, Any]]:
    data = client._get(f"/bets/sniper/{date_str}")
    if isinstance(data, dict) and "error" in data: return []
    return data if data else []

@st.cache_data(ttl=300)
def fetch_participants(race_id: int) -> pd.DataFrame:
    data = client._get(f"/races/{race_id}/participants")
    if isinstance(data, dict) and "error" in data: return pd.DataFrame()
    return pd.DataFrame(data) if data else pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_model_metrics(model_name: Optional[str] = None) -> pd.DataFrame:
    params = {}
    if model_name:
        params["model_name"] = model_name
    data = client._get("/metrics", params=params)
    if isinstance(data, dict) and "error" in data: return pd.DataFrame()
    return pd.DataFrame(data) if data else pd.DataFrame()

# On réduit le TTL du cache pour pouvoir tester plus facilement
@st.cache_data(ttl=60)
def fetch_backtest_results(force_update: bool = False) -> Dict[str, Any]:
    """Retrieves detailed backtesting results."""
    params = {"force": force_update}
    data = client._get("/backtest", params=params)
    # On s'assure de renvoyer au moins un dictionnaire d'erreur si None
    return data if data is not None else {"error": "Réponse vide de l'API"}
