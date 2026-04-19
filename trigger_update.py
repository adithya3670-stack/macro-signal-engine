import requests
import json
import time

url = 'http://127.0.0.1:8002/api/update_latest'

try:
    print("Triggering Update...")
    resp = requests.post(url)
    print(f"Status: {resp.status_code}")
    print(f"Response: {resp.json()}")
except Exception as e:
    print(f"Error: {e}")
