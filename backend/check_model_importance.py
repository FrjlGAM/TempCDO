"""Check model feature importance"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main import load_model
import joblib
import numpy as np

def check_importance():
    """Check which features the model considers most important"""
    model = load_model()
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = model.feature_name_
        
        # Get top 20 most important features
        indices = np.argsort(importances)[::-1][:20]
        
        print("="*80)
        print("TOP 20 MOST IMPORTANT FEATURES")
        print("="*80)
        for idx in indices:
            print(f"{feature_names[idx]:40s} {importances[idx]:.4f}")
        
        # Check if lag features are in top 10
        lag_features = ['temperature_2m_lag_1', 'temperature_2m_lag_3', 'temperature_2m_lag_24', 
                       'temperature_2m_rolling_mean_24', 'temperature_2m_rolling_std_24']
        print("\n" + "="*80)
        print("LAG FEATURE IMPORTANCE")
        print("="*80)
        for lag_feat in lag_features:
            if lag_feat in feature_names:
                idx = list(feature_names).index(lag_feat)
                rank = list(np.argsort(importances)[::-1]).index(idx) + 1
                print(f"{lag_feat:40s} Importance: {importances[idx]:.4f} (Rank: {rank})")
    else:
        print("Model does not have feature_importances_ attribute")

if __name__ == "__main__":
    check_importance()

