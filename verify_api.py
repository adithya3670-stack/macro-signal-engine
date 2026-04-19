
import requests
import json
import os

BASE_URL = "http://127.0.0.1:8002/api/ensembling"

def verify():
    # 1. Generate Accuracy
    print("Generating Accuracy Winner...")
    resp = requests.post(f"{BASE_URL}/generate_winner_ensemble", json={"folder": "2019", "metric": "accuracy"})
    print(resp.json())
    
    # 2. Generate Precision
    print("Generating Precision Winner...")
    resp = requests.post(f"{BASE_URL}/generate_winner_ensemble", json={"folder": "2019", "metric": "precision"})
    print(resp.json())
    
    # 3. Scan Architectures
    print("Scanning Architectures for 2019...")
    resp = requests.get(f"{BASE_URL}/scan_folder_architectures?folder=2019")
    data = resp.json()
    print(json.dumps(data, indent=2))
    
    # Verify we see two winner options
    archs = data.get('architectures', [])
    winners = [a for a in archs if a['value'].startswith('winner_ensemble_')]
    print(f"Found {len(winners)} winner configs.")
    
    if len(winners) >= 2:
        print("SUCCESS: Found multiple winner configs.")
    else:
        print("FAILURE: Did not find expected winner configs.")

if __name__ == "__main__":
    verify()
