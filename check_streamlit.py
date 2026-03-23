import requests
import json

def test_api():
    base_url = "http://localhost:8501" # Streamlit doesn't have the API, but I can't start the FastAPI backend easily without a real DB.
    # Actually, I can't really verify the UI easily without a running backend.
    # But I can check if the streamlit app at least loads.
    print("Checking if Streamlit is up...")
    try:
        resp = requests.get(base_url)
        print(f"Status Code: {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
