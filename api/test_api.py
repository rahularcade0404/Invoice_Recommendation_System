"""import os
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests
from tabulate import tabulate

CONTACT_NAME = "A M Auto Sales"
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
URL = f"{BASE_URL}/recommendations/state/{quote(CONTACT_NAME, safe='')}"

deadline = time.monotonic() + 120
while True:
    try:
        health_response = requests.get(f"{BASE_URL}/", timeout=3)
        health_response.raise_for_status()
        break
    except requests.RequestException:
        if time.monotonic() >= deadline:
            print(
                "API did not become ready within 120 seconds. "
                "Start it from the project root with `python -m uvicorn api.api:app --reload` and wait for "
                "'Application startup complete.'"
            )
            sys.exit(1)
        time.sleep(1)

try:
    response = requests.get(
        URL,
        params={"top_n": 3},
        timeout=30,
    )
except requests.RequestException as exc:
    print(f"API request failed: {exc}")
    sys.exit(1)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data)

    # Print pretty table directly to terminal console
    print(f"\n=== State Recommendations for {CONTACT_NAME} ===")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
else:
    print(f"Error {response.status_code}: {response.text}") """

import pandas as pd
from urllib.parse import quote
from tabulate import tabulate
from fastapi.testclient import TestClient

from api.api import app

CONTACT_NAME = "A M Auto Sales"

client = TestClient(app)

def test_state_recommendations():
    health_response = client.get("/health")
    assert health_response.status_code == 200, health_response.text

    health_data = health_response.json()
    assert health_data.get("status") == "healthy"

    url = f"/recommendations/state/{quote(CONTACT_NAME, safe='')}"
    response = client.get(url, params={"top_n": 3})
    assert response.status_code == 200, response.text

    data = response.json()
    assert isinstance(data, list), data

    df = pd.DataFrame(data)
    assert not df.empty, "No recommendations returned"

    print(f"\n=== State Recommendations for {CONTACT_NAME} ===")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
