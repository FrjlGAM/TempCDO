"""Test script to check feature values and predictions"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import create_features, load_model
from datetime import datetime
import pandas as pd
import numpy as np

def test_predictions():
    """Test predictions with different dates and hours"""
    print("Loading model...")
    model = load_model()
    
    # Test different scenarios
    test_cases = [
        (datetime(2024, 1, 15, 4), "Winter night (4 AM)"),
        (datetime(2024, 1, 15, 14), "Winter day (2 PM)"),
        (datetime(2024, 6, 15, 4), "Summer night (4 AM)"),
        (datetime(2024, 6, 15, 14), "Summer day (2 PM)"),
        (datetime(2024, 12, 15, 4), "December night (4 AM)"),
        (datetime(2024, 12, 15, 14), "December day (2 PM)"),
    ]
    
    print("\n" + "="*80)
    print("FEATURE VALUES AND PREDICTIONS")
    print("="*80)
    
    predictions = []
    
    for date_obj, description in test_cases:
        hour = date_obj.hour
        features_df = create_features(date_obj, hour)
        
        # Get key lag features
        temp_lag_1 = features_df['temperature_2m_lag_1'].values[0]
        temp_lag_3 = features_df['temperature_2m_lag_3'].values[0]
        temp_lag_24 = features_df['temperature_2m_lag_24'].values[0]
        rolling_mean = features_df['temperature_2m_rolling_mean_24'].values[0]
        rolling_std = features_df['temperature_2m_rolling_std_24'].values[0]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        predictions.append(prediction)
        
        print(f"\n{description}")
        print(f"  Date: {date_obj.date()}, Hour: {hour:02d}:00")
        print(f"  Lag features:")
        print(f"    temp_lag_1: {temp_lag_1:.2f}°C")
        print(f"    temp_lag_3: {temp_lag_3:.2f}°C")
        print(f"    temp_lag_24: {temp_lag_24:.2f}°C")
        print(f"    rolling_mean: {rolling_mean:.2f}°C")
        print(f"    rolling_std: {rolling_std:.2f}°C")
        print(f"  Prediction: {prediction:.2f}°C")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Min prediction: {min(predictions):.2f}°C")
    print(f"Max prediction: {max(predictions):.2f}°C")
    print(f"Range: {max(predictions) - min(predictions):.2f}°C")
    print(f"Mean: {np.mean(predictions):.2f}°C")
    print(f"Std: {np.std(predictions):.2f}°C")

if __name__ == "__main__":
    test_predictions()

