"""
Test script to verify the model can make predictions with the new feature creation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import create_features, load_model, EXPECTED_FEATURES
from datetime import datetime
import pandas as pd
import numpy as np

def test_prediction():
    """Test that the model can make predictions"""
    print("Testing model prediction with new feature creation...")
    
    # Load model
    try:
        model = load_model()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Create test features
    test_date = datetime(2024, 6, 15, 12, 0)  # June 15, 2024, 12:00 PM
    test_hour = 12
    
    try:
        features = create_features(test_date, test_hour)
        print(f"✓ Features created: shape {features.shape}")
        print(f"  Expected {len(EXPECTED_FEATURES)} features, got {features.shape[1]}")
        
        if features.shape[1] != len(EXPECTED_FEATURES):
            print(f"✗ Feature count mismatch!")
            return False
        
        # Create DataFrame with feature names
        features_df = pd.DataFrame(features, columns=EXPECTED_FEATURES)
        print("✓ DataFrame created with feature names")
        
        # Make prediction
        prediction = model.predict(features_df)
        temp = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
        print(f"✓ Prediction successful: {temp:.2f}°C")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Tests failed!")
        sys.exit(1)

