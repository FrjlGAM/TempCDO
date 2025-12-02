"""Test daily forecast to see lag feature values"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import create_features, load_model, forecast_daily
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from main import app

def test_daily():
    """Test daily forecast endpoint"""
    model = load_model()
    client = TestClient(app)
    
    response = client.post("/api/forecast/daily", json={"date": "2024-01-15"})
    data = response.json()
    
    print("="*80)
    print("DAILY FORECAST ANALYSIS - January 15, 2024")
    print("="*80)
    
    temps = [f['temperature'] for f in data['forecast']]
    print(f"\nTemperature range: {min(temps):.2f}°C to {max(temps):.2f}°C")
    print(f"Range: {max(temps) - min(temps):.2f}°C")
    print(f"Mean: {np.mean(temps):.2f}°C")
    print(f"Std: {np.std(temps):.2f}°C")
    
    # Check first few hours
    print("\nFirst 5 hours:")
    for i in range(5):
        hour = data['forecast'][i]['hour']
        temp = data['forecast'][i]['temperature']
        date_obj = datetime(2024, 1, 15, hour)
        # Create features to see what lag_1 would be
        if i == 0:
            features = create_features(date_obj, hour, None)
        else:
            # Use previous hour's temp
            hist = {1: temps[i-1]}
            features = create_features(date_obj, hour, hist)
        lag_1 = features['temperature_2m_lag_1'].values[0]
        print(f"  Hour {hour:02d}: temp={temp:.2f}°C, lag_1={lag_1:.2f}°C")

if __name__ == "__main__":
    test_daily()

